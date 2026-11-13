"""Tests for services/docker_services/image_management.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import docker.errors
import pytest

from src.services.docker_services.image_management import ImageManager


class TestImageManagerInit:
    def test_stores_client(self):
        client = MagicMock()
        mgr = ImageManager(client)
        assert mgr.client is client


class TestImageExists:
    def test_no_client(self):
        mgr = ImageManager(None)
        assert mgr.image_exists("python:3.11") is False

    def test_image_found(self):
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        mgr = ImageManager(client)
        assert mgr.image_exists("python:3.11") is True
        client.images.get.assert_called_once_with("python:3.11")

    def test_image_not_found(self):
        client = MagicMock()
        client.images.get.side_effect = docker.errors.ImageNotFound("not found")
        mgr = ImageManager(client)
        assert mgr.image_exists("missing:latest") is False

    def test_generic_exception(self):
        client = MagicMock()
        client.images.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = ImageManager(client)
        assert mgr.image_exists("broken:latest") is False


class TestPullImage:
    @pytest.fixture
    def mgr(self):
        client = MagicMock()
        return ImageManager(client)

    async def test_no_client(self):
        mgr = ImageManager(None)
        result = await mgr.pull_image("python")
        assert result is False

    async def test_success(self, mgr):
        result = await mgr.pull_image("python", tag="3.11")
        assert result is True
        mgr.client.images.pull.assert_called_once_with("python", tag="3.11")

    async def test_default_tag(self, mgr):
        await mgr.pull_image("python")
        mgr.client.images.pull.assert_called_once_with("python", tag="latest")

    async def test_image_not_found(self, mgr):
        mgr.client.images.pull.side_effect = docker.errors.ImageNotFound("not found")
        result = await mgr.pull_image("nonexistent")
        assert result is False

    async def test_api_error(self, mgr):
        mgr.client.images.pull.side_effect = docker.errors.APIError("api error")
        result = await mgr.pull_image("broken")
        assert result is False

    async def test_generic_exception(self, mgr):
        mgr.client.images.pull.side_effect = docker.errors.DockerException("unexpected")
        result = await mgr.pull_image("broken")
        assert result is False


class TestBuildImage:
    @pytest.fixture
    def mgr(self):
        client = MagicMock()
        return ImageManager(client)

    async def test_no_client(self):
        mgr = ImageManager(None)
        result = await mgr.build_image("/path", "myimage:latest")
        assert result is False

    async def test_success(self, mgr):
        mock_image = MagicMock()
        build_logs = [
            {"stream": "Step 1/3 : FROM python:3.11\n"},
            {"stream": "Step 2/3 : COPY . /app\n"},
        ]
        mgr.client.images.build.return_value = (mock_image, build_logs)

        result = await mgr.build_image("/path", "myimage:latest")
        assert result is True
        mgr.client.images.build.assert_called_once_with(
            path="/path",
            tag="myimage:latest",
            dockerfile="Dockerfile",
            buildargs={},
            rm=True,
            forcerm=True,
        )

    async def test_custom_dockerfile_and_buildargs(self, mgr):
        mgr.client.images.build.return_value = (MagicMock(), [])
        result = await mgr.build_image(
            "/path", "img:v1", dockerfile="Dockerfile.dev", build_args={"ENV": "test"}
        )
        assert result is True
        call_kwargs = mgr.client.images.build.call_args[1]
        assert call_kwargs["dockerfile"] == "Dockerfile.dev"
        assert call_kwargs["buildargs"] == {"ENV": "test"}

    async def test_build_logs_with_errors(self, mgr):
        build_logs = [{"error": "some build error"}]
        mgr.client.images.build.return_value = (MagicMock(), build_logs)
        # Still returns True since build completed; error is just logged
        result = await mgr.build_image("/path", "img:v1")
        assert result is True

    async def test_build_error(self, mgr):
        err = docker.errors.BuildError("build failed", [{"stream": "error line\n"}])
        mgr.client.images.build.side_effect = err
        result = await mgr.build_image("/path", "img:v1")
        assert result is False

    async def test_api_error(self, mgr):
        mgr.client.images.build.side_effect = docker.errors.APIError("api error")
        result = await mgr.build_image("/path", "img:v1")
        assert result is False

    async def test_generic_exception(self, mgr):
        mgr.client.images.build.side_effect = docker.errors.DockerException("unexpected")
        result = await mgr.build_image("/path", "img:v1")
        assert result is False


class TestListImages:
    def test_no_client(self):
        mgr = ImageManager(None)
        assert mgr.list_images() == []

    def test_success(self):
        client = MagicMock()
        mock_img1 = MagicMock()
        mock_img1.id = "sha256:abc"
        mock_img1.tags = ["python:3.11"]
        mock_img1.attrs = {
            "Created": "2024-01-01T00:00:00Z",
            "Size": 900_000_000,
            "Config": {"Labels": {"maintainer": "test"}},
        }
        mock_img2 = MagicMock()
        mock_img2.id = "sha256:def"
        mock_img2.tags = []
        mock_img2.attrs = {"Created": "", "Size": 0, "Config": {}}

        client.images.list.return_value = [mock_img1, mock_img2]
        mgr = ImageManager(client)

        result = mgr.list_images(filters={"dangling": False})
        assert len(result) == 2
        assert result[0]["id"] == "sha256:abc"
        assert result[0]["tags"] == ["python:3.11"]
        assert result[0]["labels"] == {"maintainer": "test"}
        assert result[1]["labels"] == {}
        client.images.list.assert_called_once_with(filters={"dangling": False})

    def test_exception(self):
        client = MagicMock()
        client.images.list.side_effect = docker.errors.DockerException("list fail")
        mgr = ImageManager(client)
        assert mgr.list_images() == []

    def test_no_filters(self):
        client = MagicMock()
        client.images.list.return_value = []
        mgr = ImageManager(client)
        result = mgr.list_images()
        assert result == []
        client.images.list.assert_called_once_with(filters=None)


class TestRemoveImage:
    async def test_no_client(self):
        mgr = ImageManager(None)
        result = await mgr.remove_image("python:3.11")
        assert result is False

    async def test_success(self):
        client = MagicMock()
        mgr = ImageManager(client)
        result = await mgr.remove_image("python:3.11", force=True)
        assert result is True
        client.images.remove.assert_called_once_with("python:3.11", force=True)

    async def test_image_not_found_returns_true(self):
        client = MagicMock()
        client.images.remove.side_effect = docker.errors.ImageNotFound("not found")
        mgr = ImageManager(client)
        result = await mgr.remove_image("missing:latest")
        assert result is True

    async def test_api_error(self):
        client = MagicMock()
        client.images.remove.side_effect = docker.errors.APIError("api error")
        mgr = ImageManager(client)
        result = await mgr.remove_image("img:latest")
        assert result is False

    async def test_generic_exception(self):
        client = MagicMock()
        client.images.remove.side_effect = docker.errors.DockerException("unexpected")
        mgr = ImageManager(client)
        result = await mgr.remove_image("img:latest")
        assert result is False


class TestGetImageInfo:
    def test_no_client(self):
        mgr = ImageManager(None)
        assert mgr.get_image_info("python:3.11") is None

    def test_success(self):
        client = MagicMock()
        mock_image = MagicMock()
        mock_image.id = "sha256:abc123"
        mock_image.tags = ["python:3.11"]
        mock_image.attrs = {
            "Created": "2024-01-01T00:00:00Z",
            "Size": 900_000_000,
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {"Labels": {"ver": "1.0"}},
        }
        client.images.get.return_value = mock_image
        mgr = ImageManager(client)

        result = mgr.get_image_info("python:3.11")
        assert result is not None
        assert result["id"] == "sha256:abc123"
        assert result["architecture"] == "amd64"
        assert result["os"] == "linux"
        assert result["labels"] == {"ver": "1.0"}

    def test_not_found(self):
        client = MagicMock()
        client.images.get.side_effect = docker.errors.ImageNotFound("not found")
        mgr = ImageManager(client)
        assert mgr.get_image_info("missing:latest") is None

    def test_exception(self):
        client = MagicMock()
        client.images.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = ImageManager(client)
        assert mgr.get_image_info("broken:latest") is None


class TestGetLanguageContainerImage:
    @pytest.fixture
    def mgr(self):
        return ImageManager(MagicMock())

    def test_python(self, mgr):
        assert mgr.get_language_container_image("python") == "code_graph_helper-api:latest"

    def test_javascript(self, mgr):
        assert mgr.get_language_container_image("javascript") == "node:18-slim"

    def test_typescript(self, mgr):
        assert mgr.get_language_container_image("typescript") == "node:18-slim"

    def test_java(self, mgr):
        assert mgr.get_language_container_image("java") == "openjdk:17-slim"

    def test_go(self, mgr):
        assert mgr.get_language_container_image("go") == "golang:1.20-alpine"

    def test_rust(self, mgr):
        assert mgr.get_language_container_image("rust") == "rust:1.70-slim"

    def test_cpp(self, mgr):
        assert mgr.get_language_container_image("cpp") == "gcc:latest"

    def test_c(self, mgr):
        assert mgr.get_language_container_image("c") == "gcc:latest"

    def test_csharp(self, mgr):
        assert mgr.get_language_container_image("csharp") == "mcr.microsoft.com/dotnet/sdk:7.0"

    def test_unknown_returns_default(self, mgr):
        assert mgr.get_language_container_image("brainfuck") == "code_graph_helper-api:latest"

    def test_case_insensitive(self, mgr):
        assert mgr.get_language_container_image("Python") == "code_graph_helper-api:latest"
        assert mgr.get_language_container_image("JAVA") == "openjdk:17-slim"


class TestEnsureImageAvailable:
    async def test_image_exists_locally(self):
        client = MagicMock()
        mgr = ImageManager(client)
        with patch.object(mgr, "image_exists", return_value=True):
            result = await mgr.ensure_image_available("python:3.11")
        assert result is True

    async def test_auto_pull_disabled(self):
        client = MagicMock()
        mgr = ImageManager(client)
        with patch.object(mgr, "image_exists", return_value=False):
            result = await mgr.ensure_image_available("python:3.11", auto_pull=False)
        assert result is False

    async def test_pull_with_tag(self):
        client = MagicMock()
        mgr = ImageManager(client)
        with (
            patch.object(mgr, "image_exists", return_value=False),
            patch.object(mgr, "pull_image", return_value=True) as mock_pull,
        ):
            result = await mgr.ensure_image_available("python:3.11")
        assert result is True
        mock_pull.assert_called_once_with("python", "3.11")

    async def test_pull_without_tag(self):
        client = MagicMock()
        mgr = ImageManager(client)
        with (
            patch.object(mgr, "image_exists", return_value=False),
            patch.object(mgr, "pull_image", return_value=True) as mock_pull,
        ):
            result = await mgr.ensure_image_available("python")
        assert result is True
        mock_pull.assert_called_once_with("python", "latest")

    async def test_pull_fails(self):
        client = MagicMock()
        mgr = ImageManager(client)
        with (
            patch.object(mgr, "image_exists", return_value=False),
            patch.object(mgr, "pull_image", return_value=False),
        ):
            result = await mgr.ensure_image_available("missing:latest")
        assert result is False


class TestPruneImages:
    def test_no_client(self):
        mgr = ImageManager(None)
        result = mgr.prune_images()
        assert result == {"images_deleted": [], "space_reclaimed": 0}

    def test_success(self):
        client = MagicMock()
        client.images.prune.return_value = {
            "ImagesDeleted": [{"Deleted": "sha256:abc"}],
            "SpaceReclaimed": 50000,
        }
        mgr = ImageManager(client)
        result = mgr.prune_images(filters={"dangling": True})
        assert result["space_reclaimed"] == 50000
        assert len(result["images_deleted"]) == 1
        client.images.prune.assert_called_once_with(filters={"dangling": True})

    def test_none_deleted(self):
        client = MagicMock()
        client.images.prune.return_value = {
            "ImagesDeleted": None,
            "SpaceReclaimed": 0,
        }
        mgr = ImageManager(client)
        result = mgr.prune_images()
        assert result["images_deleted"] == []

    def test_exception(self):
        client = MagicMock()
        client.images.prune.side_effect = docker.errors.DockerException("prune fail")
        mgr = ImageManager(client)
        result = mgr.prune_images()
        assert result == {"images_deleted": [], "space_reclaimed": 0}


class TestTagImage:
    def test_no_client(self):
        mgr = ImageManager(None)
        assert mgr.tag_image("src:v1", "dst") is False

    def test_success(self):
        client = MagicMock()
        mock_image = MagicMock()
        mock_image.tag.return_value = True
        client.images.get.return_value = mock_image
        mgr = ImageManager(client)

        result = mgr.tag_image("python:3.11", "myrepo/python", tag="latest")
        assert result is True
        mock_image.tag.assert_called_once_with("myrepo/python", "latest")

    def test_tag_returns_false(self):
        client = MagicMock()
        mock_image = MagicMock()
        mock_image.tag.return_value = False
        client.images.get.return_value = mock_image
        mgr = ImageManager(client)

        result = mgr.tag_image("python:3.11", "myrepo/python")
        assert result is False

    def test_image_not_found(self):
        client = MagicMock()
        client.images.get.side_effect = docker.errors.ImageNotFound("not found")
        mgr = ImageManager(client)
        assert mgr.tag_image("missing:v1", "dst") is False

    def test_generic_exception(self):
        client = MagicMock()
        client.images.get.side_effect = docker.errors.DockerException("unexpected")
        mgr = ImageManager(client)
        assert mgr.tag_image("broken:v1", "dst") is False
