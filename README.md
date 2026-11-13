# coding-containers

Owns repository workspaces: cloned GitHub repositories plus their isolated Docker or Kubernetes runtime. Other services delegate here to create a checkout, run commands, edit files, install dependencies, execute coding agents, or push changes.

Port: **8691**

## Why This Service Exists

The code-graph-helper platform analyzes and modifies code inside GitHub repositories. Each checkout is a workspace backed by an isolated runtime. This service owns that lifecycle and sits behind `coding-gateway` (port 8687).

`/api/workspaces` is the canonical public API. `/api/containers` remains callable
as a compatibility alias but is excluded from OpenAPI, preventing generated
clients from seeing two names for the same resource.

`container_name` is the stable cross-service join key. Since 2026-05-27 the strategic-layer `requirements_management` service binds ArchitectureElements to coding-side artifacts by `(tenant_id, container_name)` via its `architecture_element_realisation_bindings` table; `fr`'s `project_repositories` join (operational layer) consumes the same key. No change is required here — the existing `ContainerInfo` registry remains the source of truth for `container_name ⇄ repo_url`.

## Architecture

```
coding-gateway (8687)
  |
  +---> coding-containers (8691)
            |
            +--- Docker Engine (local containers)
            +--- Kubernetes API (in-cluster pods, when KUBERNETES_IN_CLUSTER=true)
            +--- Agent-Host API (when AGENT_HOST_URL is set)
            +--- Redis (caching, container metadata)
```

**Key components:**

| Component | Location | Purpose |
|-----------|----------|---------|
| `ContainerManager` | `src/core/container_manager.py` | Facade over core + GitHub provisioning |
| `CoreContainerManager` | `src/core/container_manager_core.py` | Docker client, container create/remove, port allocation |
| `GitHubContainerProvisioner` | `src/core/github_container_provisioner.py` | Clones repos, handles forks, PR provisioning |
| `K8sContainerProvisioner` | `src/core/kubernetes_container_provisioner.py` | Creates K8s Deployments/Services instead of Docker containers |
| `DockerService` | `src/services/docker_service.py` | Modular facade: lifecycle, images, networks, monitoring |
| `AgentService` | `src/services/agent_service.py` | Executes Claude Code / Codex CLI inside containers |
| `ContainerInstaller` | `src/core/container_installer.py` | Detects and installs dependencies (pip, npm, etc.) |
| `RedisService` | `src/core/redis_service.py` | Caching and container metadata |
| `ContainerMetadataRegistry` | `src/core/container_metadata_registry.py` | JSON-file-based container registry persistence |

**Execution tiers** (for `exec` and agent endpoints):
1. **Kubernetes** -- if container has a `.svc.cluster.local` endpoint, exec via K8s pod exec
2. **Agent-host** -- if `AGENT_HOST_URL` is set and container is tracked there, delegate via HTTP
3. **Docker** -- direct `docker exec` fallback

## Authentication

Auth is provided by `coding_client.auth.middleware.AuthMiddleware` (from the `coding-client` pip package), gated behind `ENABLE_AUTH` (default `true`).

**How it works:**
1. **Istio-forwarded headers** -- `x-user-id`, `x-tenant-id`, `x-user-roles`, `x-user-name`, `x-user-email`. Set by the Istio mesh / gateway after license-authz validation.
2. **License token introspection** -- if no Istio headers, a `Bearer` token in the `Authorization` header is introspected against `licenses.licenses.svc.cluster.local:8544/license/introspect` to extract tenant context.
3. **Trusted-network bypass** -- requests from RFC1918/loopback IPs with no headers get a default tenant context (`DEFAULT_TENANT_ID`, default `n8n-dev`).
4. **Dev/test API key** -- when `ENVIRONMENT=development` or `testing`, a `Bearer` token matching `DEV_API_KEY` or `TEST_API_KEY` is accepted.

Public (unauthenticated) paths: `/`, `/health`, `/ready`, `/metrics`, `/openapi.json`, `/docs`, `/redoc`.

Every route handler uses `Depends(require_tenant)` or `Depends(get_current_user_with_tenant)` for tenant-scoped data access. No direct Keycloak interaction; no JWT validation in this service.

## API Routes

### Service Root

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe. Returns `{"status": "ok"}` |
| `GET` | `/` | Service info with links to `/docs` and `/health` |

### Workspace Lifecycle (`/api/workspaces`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/workspaces/` | List workspaces. Query: `status_filter` (running/stopped/error/creating/removing), `limit` (1-1000, default 100). Tenant-filtered. |
| `POST` | `/api/workspaces/` | Create a workspace from a GitHub repo. Body: `GitHubRepoRequest` (repo_url, branch, container_name, access_token, pr_number, etc.). Returns 201. |
| `GET` | `/api/workspaces/{container_name}` | Get workspace details (runtime status, repo URL, code graph ID, etc.). |
| `GET` | `/api/workspaces/{container_name}/status` | Get lightweight runtime status. |
| `DELETE` | `/api/workspaces/{container_name}` | Remove workspace/runtime resources and its associated code graph. Query: `force` (bool). |
| `DELETE` | `/api/workspaces/all` | Remove all workspaces for the current tenant. |

### File Operations (`/api/workspaces/{container_name}/...`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/workspaces/{container_name}/mkdir` | Create directory inside the workspace. Body: `{"path": "/app/src/new_module"}`. |
| `POST` | `/api/workspaces/{container_name}/touch` | Create empty file. Body: `{"path": "/app/src", "filename": "service", "extension": "py"}`. |
| `POST` | `/api/workspaces/{container_name}/write-file` | Write content to file (base64-encodes internally). Body: `{"file_path": "...", "content": "..."}`. |
| `GET` | `/api/workspaces/{container_name}/files` | List directory or read file. Query: `path` (default `/workspace`). Returns entries for directories and content for files. |
| `POST` | `/api/workspaces/{container_name}/move` | Move or rename a file or directory tree. Body: `{"source_path": "src/old", "destination_path": "src/new", "overwrite": false}`. |
| `DELETE` | `/api/workspaces/{container_name}/files` | Delete a file, symlink, or directory. Query: `path`; set `recursive=true` for a non-empty directory. |

Move and delete paths are confined to the cloned repository under `/workspace`.
Relative paths are resolved from `/workspace`, and the workspace root itself cannot
be moved or deleted.

### Git Operations (`/api/workspaces/{container_name}/git/...`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/workspaces/{container_name}/git/status` | Git status (porcelain parse): modified, new, deleted files; staged/unstaged counts; current branch. |
| `POST` | `/api/workspaces/{container_name}/git/commit` | Commit changes. Body: `{"message": "...", "files": ["*"], "author_name": "...", "author_email": "..."}`. |
| `POST` | `/api/workspaces/{container_name}/git/push` | Push to GitHub fork. Body: `{"branch": "...", "force": false, "create_fork": true}`. |

### Command Execution (`/api/workspaces/{container_name}/exec`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/workspaces/{container_name}/exec` | Execute a whitelisted command in the workspace runtime. |

### Agent Execution (`/api/workspaces/{container_name}/...`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/workspaces/{container_name}/execute-agent` | Run Claude Code or Codex CLI in the workspace runtime. |
| `GET` | `/api/workspaces/{container_name}/agent-capabilities` | Check which CLI tools are available. |

### Dependency Management (`/api/workspaces/{container_name}/...`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/workspaces/{container_name}/install-dependencies` | Auto-detect and install dependencies. |
| `GET` | `/api/workspaces/{container_name}/dependency-status` | Analyze project type, dependency files, and installation plan. |

### Container Health Monitoring (`/api/health/...`)

Proxied from the gateway for container-specific health checks.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health/container/{name}` | Health status of a single container. |
| `GET` | `/api/health/containers/unhealthy` | List all containers with ERROR status. |
| `GET` | `/api/health/container/{name}/ready` | Readiness probe (status == RUNNING). Returns 503 if not ready. |
| `GET` | `/api/health/container/{name}/live` | Liveness probe (not STOPPED or ERROR). Returns 503 if not alive. |
| `GET` | `/api/health/container/{name}/statistics` | Container metadata: language, codebase path, created_at. |

### Code Validation (`/api/validation/...`)

Proxied from the gateway for validation that requires Docker access.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/validation/multi-line-code-snippet` | Validate Python code: AST syntax check + optional runtime execution in a container. Body: `{"code_snippet": "...", "container_name": "...", "include_execution_validation": true, "execution_timeout": 10}`. |

## Configuration

Environment variables used by this service (prefix `CODING_CONTAINERS_` for pydantic-settings fields, or set directly for plain `os.getenv` variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | `coding-containers` | Service identifier |
| `VERSION` | `1.0.0` | Service version |
| `ENVIRONMENT` | `production` | Environment name (`development`/`testing` enables API key auth) |
| `ENABLE_AUTH` | `true` | Enable `AuthMiddleware` from coding-client |
| `ENABLE_DOCS` | `false` | Expose Swagger UI at `/docs` |
| `DEFAULT_TENANT_ID` | `n8n-dev` | Default tenant for trusted-network requests |
| `CODING_CONTAINERS_DOCKER_BASE_PORT` | `8700` | Starting port for provisioned containers |
| `CODING_CONTAINERS_MAIN_API_ENDPOINT` | `http://localhost:8100` | Main API endpoint URL |
| `CODING_CONTAINERS_TEMP_DIR` | `/tmp` | Temporary directory |
| `CODE_GRAPH_DB_DIR` | `{temp_dir}/code_graphs` | Code graph database directory |
| `CONTAINER_DATA_DIR` | `/app/data` | Base directory for container metadata registry JSON |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | (none) | Redis password |
| `AGENT_HOST_URL` | (none) | Delegate container ops to agent-host service |
| `KUBERNETES_IN_CLUSTER` | `false` | Use K8s provisioning instead of Docker |
| `PROVISIONER_NAMESPACE` | `code-graph-helper` | K8s namespace for provisioned pods |
| `PROVISIONER_IMAGE` | `ghcr.io/nonagenticai/codegraph-runtime:latest` | Container image for K8s pods |
| `PROVISIONER_NODE_ARCH` | (none) | Node architecture selector for K8s pods |
| `PROVISIONED_CONTAINERS_VOLUME_NAME` | (auto) | Explicit shared volume name |
| `COMPOSE_PROJECT_NAME` | `code_graph_helper` | Docker Compose project name (used for volume discovery) |
| `GITHUB_ACCESS_TOKEN` | (none) | Fallback GitHub token for clone/push operations |
| `GIT_CLONE_TIMEOUT` | `1200` | Git clone timeout in seconds |
## Directory Structure

```
src/
  main.py                          # FastAPI app, middleware, routers, exception handlers
  auth/                            # (empty -- auth is in coding-client package)
  config/
    settings.py                    # Pydantic-settings: docker port, Redis, temp dir
  core/
    container_manager.py           # High-level facade
    container_manager_core.py      # Docker client, port allocation, agent-host tier
    container_installer.py         # Dependency detection and installation
    container_metadata_registry.py # JSON-file persistence for container registry
    container_spec.py              # Container specification models
    github_container_provisioner.py# Clone, fork, PR provisioning
    graph_operations.py            # Code graph DB operations
    kubernetes_container_provisioner.py # K8s Deployment/Service creation
    redis_service.py               # Redis cache wrapper
  clients/
    github_client.py               # GitHub API client
    realtime_client.py             # Realtime service client
  models/
    agent_models.py                # Agent execution request/response
    common_api_models.py           # Shared response status enum
    container_api_models.py        # Container CRUD request/response models
    container_models.py            # ContainerInfo, ContainerStatus, GitHubRepoRequest
    error_models.py                # Standardized error response models
    pydantic_ai_models.py          # Pydantic-AI integration models
    security_models.py             # Security-related models
    testing_models.py              # Testing-related models
  processing/
    multi_language_factory.py      # Multi-language support factory
  routers/
    agents.py                      # POST execute-agent, GET agent-capabilities
    container_dependencies.py      # POST install-dependencies, GET dependency-status
    container_exec.py              # POST exec (whitelisted command execution)
    containers.py                  # CRUD, file ops, git ops, file listing
    health.py                      # Container health, readiness, liveness, statistics
    validation.py                  # Multi-line code snippet validation
  services/
    agent_service.py               # Claude Code / Codex CLI execution logic
    container_health_service.py    # Container health checking
    container_health_history_service.py # Health history tracking
    docker_service.py              # DockerService facade
    docker_services/
      container_lifecycle.py       # Container start/stop/remove
      container_monitoring.py      # Container stats and monitoring
      docker_api_integration.py    # Low-level Docker API wrapper
      image_management.py          # Image pull/build/tag
      network_volume_management.py # Network and volume operations
    agents/
      pydantic_ai_agent.py         # Single Pydantic-AI agent
      pydantic_ai_agents.py        # Multi-agent orchestration
      pydantic_ai_logger.py        # Agent execution logging
  utils/
    cache_manager.py               # Cache management utilities
    cache_utils.py                 # Cache decorator/helpers
    retry_helper.py                # Retry logic with backoff
tests/                             # pytest test suite
```

## Dependencies on Other Services

| Service | Purpose |
|---------|---------|
| **Docker Engine** | Container lifecycle (create, exec, remove). Optional -- degrades gracefully. |
| **Redis** | Caching, metadata. Non-fatal if unavailable at startup. |
| **coding-gateway** (8687) | Proxies external requests to this service. |
| **coding-client** (pip package) | Auth middleware, tenant extraction, response helpers, exception types. |
| **GitHub API** | Repository cloning, fork creation, PR operations. |
| **Kubernetes API** | Pod/Deployment management when `KUBERNETES_IN_CLUSTER=true`. |

## Development

```bash
# Install dependencies
uv sync --dev

# Run locally
uv run uvicorn src.main:app --host 0.0.0.0 --port 8691 --reload

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run ty check
```

## CI/CD

GitHub Actions workflow: `secret scan + private-refs guard → ruff + ty + pytest → build amd64 → smoke test → push`

Image pushed to `ghcr.io/nonagenticai/coding-containers:sha-<commit>`, tagged `:latest` on a main build.
The two runtime images (`codegraph-runtime`, `codegraph-runtime-rust`) have their own workflow,
`.github/workflows/runtime-image.yaml`. A deployment watches the image through its own GitOps repo —
this repository holds no deploy credentials.

## Docker

```bash
docker compose build
docker compose up -d
# Health check: curl http://localhost:8691/health
```
