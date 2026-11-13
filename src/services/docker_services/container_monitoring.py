"""
Container Monitoring and Logging Module

Handles container monitoring, health checks, logging, and file operations.
Extracted from docker_service.py to improve modularity and maintainability.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import Any

import docker
import docker.errors
import docker.types

from coding_client.utils.retry_helper import retry_docker
from src.core.log_safety import redact_secrets, summarize_command

logger = logging.getLogger(__name__)


class ContainerMonitor:
    """Monitors containers and handles logging operations."""

    def __init__(self, docker_client: Any):
        """
        Initialize the container monitor.

        Args:
            docker_client: Docker client instance
        """
        self.client = docker_client

    async def wait_for_analysis_completion(self, docker_id: str, timeout: int = 300) -> bool:
        """
        Wait for container analysis to complete by checking for completion marker.

        Args:
            docker_id: Container ID
            timeout: Timeout in seconds

        Returns:
            bool: True if analysis completed, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if not self.client:
                    logger.error("Docker client is not available")
                    return False

                docker_container = self.client.containers.get(docker_id)
                result = docker_container.exec_run(
                    "test -f /app/data/.analysis_complete", stdout=True, stderr=True
                )

                if result.exit_code == 0:
                    logger.info(f"Container {docker_id} analysis marked as complete")
                    return True

            except docker.errors.DockerException as e:
                logger.warning(f"Error checking completion status for {docker_id}: {e}")

            await asyncio.sleep(2)

        logger.warning(f"Container {docker_id} analysis completion timeout after {timeout}s")
        return False

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    async def exec_in_container(
        self,
        docker_id: str,
        command: list[str],
        timeout: int = 30,
        environment: dict[str, str] | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute command in container and return structured result with retry logic.

        Args:
            docker_id: Container ID
            command: Command to execute
            timeout: Timeout in seconds
            environment: Environment variables
            user: User to run command as

        Returns:
            Dictionary with execution results
        """
        if not self.client:
            raise Exception("Docker not available")

        start_time = time.time()
        try:
            container = self.client.containers.get(docker_id)

            exec_params: dict[str, Any] = {
                "cmd": command,
                "environment": environment or {},
                "stdout": True,
                "stderr": True,
            }

            if user:
                exec_params["user"] = user

            exec_result = container.exec_run(**exec_params)
            execution_time = time.time() - start_time
            timeout_occurred = execution_time >= timeout

            return {
                "exit_code": exec_result.exit_code,
                "stdout": (
                    exec_result.output.decode("utf-8", errors="replace")
                    if exec_result.output
                    else ""
                ),
                "stderr": "",
                "execution_time": execution_time,
                "timeout": timeout_occurred,
            }

        except docker.errors.DockerException as e:
            execution_time = time.time() - start_time
            # Free-form exception text from the same exec path `_create_file`
            # uses, so a docker-py APIError that renders the failed `Cmd` back
            # carries the same base64 blob. Best-effort denylist layer — no
            # structural projection is possible on an arbitrary message.
            logger.error(
                "Command execution failed in container %s: %s",
                docker_id,
                redact_secrets(str(e)),
            )
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": execution_time,
                "timeout": False,
            }

    @retry_docker(max_attempts=2, min_wait=0.5, max_wait=3.0)
    async def get_container_stats(self, docker_id: str) -> dict[str, Any]:
        """
        Get resource usage statistics for container with retry logic.

        Args:
            docker_id: Container ID

        Returns:
            Dictionary with container statistics
        """
        if not self.client:
            return {}

        try:
            container = self.client.containers.get(docker_id)
            stats = container.stats(stream=False)

            memory_usage = 0
            cpu_usage = 0.0

            if "memory" in stats:
                memory_usage = stats["memory"].get("usage", 0) // (1024 * 1024)  # Convert to MB

            if "cpu" in stats:
                cpu_delta = stats["cpu"]["cpu_usage"]["total_usage"]
                system_delta = stats["cpu"]["system_cpu_usage"]
                if system_delta > 0:
                    cpu_usage = cpu_delta / system_delta * 100.0

            return {
                "memory_mb": memory_usage,
                "cpu_percent": cpu_usage,
                "docker_id": docker_id,
            }

        except docker.errors.DockerException as e:
            logger.warning(f"Failed to get stats for container {docker_id}: {e}")
            return {}

    @retry_docker(max_attempts=2, min_wait=0.5, max_wait=3.0)
    def get_container_logs(self, docker_id: str, tail: int = 100, since: str | None = None) -> str:
        """
        Get container logs with retry logic.

        Args:
            docker_id: Container ID
            tail: Number of lines to retrieve from end
            since: Timestamp or relative time (e.g., '1h', '30m')

        Returns:
            Container logs as string
        """
        if not self.client:
            return "Docker not available"

        try:
            container = self.client.containers.get(docker_id)

            logs_params: dict[str, int | bool | str] = {
                "tail": tail,
                "stdout": True,
                "stderr": True,
            }

            if since:
                logs_params["since"] = since

            logs = container.logs(**logs_params)
            return logs.decode("utf-8", errors="replace")

        except docker.errors.NotFound:
            return f"Container {docker_id} not found"
        except docker.errors.DockerException as e:
            logger.error(f"Error getting logs for container {docker_id}: {e}")
            return f"Error retrieving logs: {str(e)}"

    async def create_file_in_container(
        self, docker_id: str, file_path: str, content: str, mode: str = "w"
    ) -> bool:
        """
        Create file with content in container.

        Args:
            docker_id: Container ID
            file_path: Path to file in container
            content: File content
            mode: File mode (w, a, etc.)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            raise Exception("Docker not available")

        try:
            encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
            dir_path = os.path.dirname(file_path) or "."

            commands = [
                f"mkdir -p {dir_path}",
                f"echo '{encoded_content}' | base64 -d > {file_path}",
            ]

            for cmd in commands:
                result = await self.exec_in_container(docker_id, ["sh", "-c", cmd], timeout=10)
                if result["exit_code"] != 0:
                    # ⛔ Previously `f"Failed to execute: {cmd} - ..."` with the
                    # raw command. `cmd` is the `echo '<base64>' | base64 -d`
                    # line built just above, and `content` is whatever the
                    # caller is writing into the container — for the git flows
                    # that is a credentials file, so the blob DECODES to
                    # a github.com URL whose userinfo is a live installation
                    # token (`x-access-token:ghs_...`). A base64
                    # payload is not obfuscation: it is a REVERSIBLE credential
                    # in an ERROR record, and it was called "out of scope"
                    # precisely because the record does not look like a secret.
                    # The projection drops the blob (a value) and keeps `echo`
                    # / `base64 -d` / the destination path.
                    logger.error(
                        "Failed to execute: %s - %s",
                        summarize_command(cmd),
                        redact_secrets(str(result["stderr"])),
                    )
                    return False

            logger.debug(f"Created file {file_path} in container {docker_id}")
            return True

        except docker.errors.DockerException as e:
            logger.error(f"Failed to create file {file_path} in container {docker_id}: {e}")
            return False

    @retry_docker(max_attempts=3, min_wait=1.0, max_wait=5.0)
    async def read_container_file(
        self,
        container_name: str,
        file_path: str,
        offset: int = 0,
        length: int | None = None,
    ) -> dict[str, Any]:
        """
        Read file from container using docker exec with retry logic.

        Args:
            container_name: Container name
            file_path: Path to file in container
            offset: Line offset to start reading from
            length: Number of lines to read

        Returns:
            Dictionary with file content and metadata
        """
        if not self.client:
            raise Exception("Docker not available")

        try:
            docker_container = self.client.containers.get(container_name)
            result = docker_container.exec_run(f"cat /app/data/repo/{file_path}")

            if result.exit_code != 0:
                if b"No such file" in result.output:
                    raise FileNotFoundError(f"File not found: {file_path}")
                raise RuntimeError(f"Failed to read file: {result.output.decode()}")

            content = result.output.decode("utf-8")

            # Apply offset and length if specified
            if offset > 0 or length is not None:
                lines = content.split("\n")
                if offset > 0:
                    lines = lines[offset:]
                if length is not None:
                    lines = lines[:length]
                content = "\n".join(lines)

            # Get file metadata
            stat_result = docker_container.exec_run(f"stat -c '%s %Y' /app/data/repo/{file_path}")
            size, mtime = (0, 0)
            if stat_result.exit_code == 0:
                try:
                    size, mtime = map(int, stat_result.output.decode().strip().split())
                except (ValueError, AttributeError) as parse_err:
                    logger.debug("Failed to parse stat output: %s", parse_err)

            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "size": size,
                "modified_time": mtime,
                "lines_total": len(content.split("\n")) if content else 0,
                "lines_returned": len(content.split("\n")) if content else 0,
            }

        except (docker.errors.DockerException, FileNotFoundError, RuntimeError) as e:
            logger.error(f"Failed to read file {file_path} from container {container_name}: {e}")
            raise

    async def list_container_directory(
        self, container_name: str, path: str = "", recursive: bool = False
    ) -> list[dict[str, Any]]:
        """
        List directory contents in container.

        Args:
            container_name: Container name
            path: Directory path
            recursive: Whether to list recursively

        Returns:
            List of directory items
        """
        if not self.client:
            raise Exception("Docker not available")

        try:
            docker_container = self.client.containers.get(container_name)

            if recursive:
                cmd = f"find /app/data/repo/{path} -type f -o -type d"
            else:
                cmd = f"ls -la /app/data/repo/{path}"

            result = docker_container.exec_run(cmd)

            if result.exit_code != 0:
                if b"No such file" in result.output:
                    raise FileNotFoundError(f"Directory not found: {path}")
                raise RuntimeError(f"Failed to list directory: {result.output.decode()}")

            items = []
            for line in result.output.decode().strip().split("\n"):
                if not line.strip() or line.startswith("total"):
                    continue

                if recursive:
                    item_path = line.strip().replace("/app/data/repo/", "").lstrip("/")
                    if item_path:
                        items.append(
                            {
                                "name": os.path.basename(item_path),
                                "path": item_path,
                                "type": "unknown",
                            }
                        )
                else:
                    parts = line.split()
                    if len(parts) >= 9:
                        permissions = parts[0]
                        size = parts[4] if parts[4].isdigit() else 0
                        name = " ".join(parts[8:])

                        if name in [".", ".."]:
                            continue

                        item_type = "directory" if permissions.startswith("d") else "file"
                        item_path = f"{path}/{name}".strip("/")

                        items.append(
                            {
                                "name": name,
                                "path": item_path,
                                "type": item_type,
                                "size": int(size) if item_type == "file" else None,
                                "permissions": permissions,
                            }
                        )

            return items

        except (docker.errors.DockerException, FileNotFoundError, RuntimeError) as e:
            logger.error(f"Failed to list directory {path} in container {container_name}: {e}")
            raise

    async def get_container_repository_tree(self, container_name: str) -> dict[str, Any]:
        """
        Get complete repository tree structure.

        Args:
            container_name: Container name

        Returns:
            Tree structure dictionary
        """
        if not self.client:
            raise Exception("Docker not available")

        try:
            docker_container = self.client.containers.get(container_name)
            result = docker_container.exec_run(
                ["/bin/sh", "-c", "find /app/data/repo -type f -o -type d | head -1000"]
            )

            if result.exit_code != 0:
                raise RuntimeError(f"Failed to get repository tree: {result.output.decode()}")

            tree = {"type": "directory", "name": "repo", "children": {}}

            for line in result.output.decode().strip().split("\n"):
                if line.strip():
                    path = line.replace("/app/data/repo", "").strip("/")
                    if path:
                        self._add_to_tree(tree, path)

            return tree

        except (docker.errors.DockerException, RuntimeError) as e:
            logger.error(f"Failed to get repository tree from container {container_name}: {e}")
            raise

    def _add_to_tree(self, tree: dict[str, Any], path: str) -> None:
        """
        Helper to build tree structure.

        Args:
            tree: Tree dictionary to add to
            path: File/directory path
        """
        parts = path.split("/")
        current = tree["children"]

        for i, part in enumerate(parts):
            if part not in current:
                is_file = i == len(parts) - 1 and "." in part
                current[part] = {
                    "type": "file" if is_file else "directory",
                    "name": part,
                    "children": {} if not is_file else None,
                }
            current = current[part]["children"] if current[part]["children"] is not None else {}

    async def search_container_files(
        self, container_name: str, pattern: str, max_results: int = 100
    ) -> list[dict[str, Any]]:
        """
        Search for files by name pattern in container.

        Args:
            container_name: Container name
            pattern: Search pattern
            max_results: Maximum number of results

        Returns:
            List of matching files
        """
        if not self.client:
            raise Exception("Docker not available")

        try:
            docker_container = self.client.containers.get(container_name)
            cmd = f"find /app/data/repo -name '{pattern}' -type f | head -{max_results}"
            result = docker_container.exec_run(["/bin/sh", "-c", cmd])

            if result.exit_code != 0:
                logger.warning(f"Find command failed: {result.output.decode()}")
                return []

            files = []
            for line in result.output.decode().strip().split("\n"):
                if line.strip():
                    file_path = line.replace("/app/data/repo/", "").strip("/")
                    if file_path:
                        files.append(
                            {
                                "name": os.path.basename(file_path),
                                "path": file_path,
                                "type": "file",
                            }
                        )

            return files

        except docker.errors.DockerException as e:
            logger.error(f"Failed to search files in container {container_name}: {e}")
            return []

    async def search_container_content(
        self,
        container_name: str,
        pattern: str,
        file_pattern: str | None = None,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search for content pattern in container files.

        Args:
            container_name: Container name
            pattern: Content pattern to search for
            file_pattern: Optional file pattern filter
            case_sensitive: Whether search is case sensitive
            max_results: Maximum number of results

        Returns:
            List of content matches
        """
        if not self.client:
            raise Exception("Docker not available")

        try:
            docker_container = self.client.containers.get(container_name)

            grep_cmd = ["grep", "-r", "-n"]
            if not case_sensitive:
                grep_cmd.append("-i")
            grep_cmd.append(pattern)
            grep_cmd.append("/app/data/repo/")

            if file_pattern:
                grep_cmd.extend(["--include", file_pattern])

            cmd = " ".join(grep_cmd) + f" | head -{max_results}"
            result = docker_container.exec_run(["/bin/sh", "-c", cmd])

            if result.exit_code not in [0, 1]:  # 1 is normal for grep when no matches
                logger.warning(f"Grep command failed: {result.output.decode()}")
                return []

            matches = []
            for line in result.output.decode().strip().split("\n"):
                if line.strip() and ":" in line:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        file_path = parts[0].replace("/app/data/repo/", "").strip("/")
                        line_number = parts[1]
                        content = parts[2]

                        matches.append(
                            {
                                "file": file_path,
                                "line": (int(line_number) if line_number.isdigit() else 0),
                                "content": content.strip(),
                                "match_pattern": pattern,
                            }
                        )

            return matches

        except docker.errors.DockerException as e:
            logger.error(f"Failed to search content in container {container_name}: {e}")
            return []

    async def install_tools_in_container(
        self, docker_id: str, tools: list[str], language: str = "python"
    ) -> bool:
        """
        Install testing/quality tools in container.

        Args:
            docker_id: Container ID
            tools: List of tools to install
            language: Programming language

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            raise Exception("Docker not available")

        try:
            if language == "python":
                if tools:
                    pip_install_cmd = ["pip", "install", "--no-cache-dir"] + tools
                    result = await self.exec_in_container(docker_id, pip_install_cmd, timeout=1200)
                    if result["exit_code"] != 0:
                        logger.warning(f"Tool installation failed: {result['stderr']}")
                        return False

            elif language in ["javascript", "typescript"]:
                if tools:
                    npm_install_cmd = ["npm", "install", "-g"] + tools
                    result = await self.exec_in_container(docker_id, npm_install_cmd, timeout=1200)
                    if result["exit_code"] != 0:
                        logger.warning(f"Tool installation failed: {result['stderr']}")
                        return False

            logger.info(f"Successfully installed tools {tools} for {language}")
            return True

        except docker.errors.DockerException as e:
            logger.error(f"Failed to install tools in container {docker_id}: {e}")
            return False
