"""Cross-service clients for coding-containers.

Only the services coding-containers actually calls are exported here:
GitHub (token minting for provisioning) and realtime (lifecycle broadcasts).

The former ``testing_client``/``security_client`` stubs were removed: their sole
consumer was ``CodeExecutionService``, which had no importer anywhere in the
``coding/`` bundle. coding-testing and coding-security are driven by the gateway
and by coding-security calling *into* this service, not the other way round.
"""

from .github_client import GitHubTokenManager, get_github_token_manager
from .realtime_client import (
    BroadcastEventType,
    BroadcastPriority,
    BroadcastResult,
    BroadcastService,
    get_broadcast_service,
)

__all__ = [
    # github
    "GitHubTokenManager",
    "get_github_token_manager",
    # realtime
    "BroadcastEventType",
    "BroadcastPriority",
    "BroadcastResult",
    "BroadcastService",
    "get_broadcast_service",
]
