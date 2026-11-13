"""Background services inside a workspace pod (the P7 multi-repo probe).

A provider repository is started as a local process in the SAME workspace pod as
its consumer, probed on ``127.0.0.1``, and stopped. Nothing crosses a pod
boundary: no Service, no port exposure, no NetworkPolicy change.
"""

from .service import (
    DEFAULT_MAX_LIFETIME_SECONDS,
    MAX_SERVICES_PER_WORKSPACE,
    SERVICE_ROOT,
    ServiceSpecError,
    build_launch_spec,
    pod_program_argv,
    validate_env,
    validate_port,
    validate_service_id,
)

__all__ = [
    "DEFAULT_MAX_LIFETIME_SECONDS",
    "MAX_SERVICES_PER_WORKSPACE",
    "SERVICE_ROOT",
    "ServiceSpecError",
    "build_launch_spec",
    "pod_program_argv",
    "validate_env",
    "validate_port",
    "validate_service_id",
]
