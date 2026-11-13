"""Value types shared by every workspace runtime backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "WORKSPACE_EXEC_USER",
    "WORKSPACE_UID",
    "ExecutionProfile",
    "RunOnceResult",
    "RuntimeCapabilities",
    "RuntimeKind",
    "UnsupportedCapabilityError",
]


#: The unprivileged identity every workspace command runs as.
#:
#: Three backends previously gave three answers: Kubernetes pins numeric uid
#: 1000 via the pod security context, the agent host was sent the NAME "coder",
#: and Docker sent nothing at all and inherited whatever the image's USER was.
#: For an interface whose entire purpose is that the backend does not change
#: the behaviour, that is the wrong number of answers.
#:
#: Both forms are kept because the backends genuinely differ in what they
#: accept: the Kubernetes exec API cannot set an identity at all (it is fixed by
#: the pod spec, hence the numeric value here is a cross-check rather than an
#: instruction), while the agent host's HTTP body takes a name. They MUST
#: describe the same user — the runtime image creates `coder` as uid 1000, and
#: `test_the_identity_constants_agree` pins that.
WORKSPACE_UID = 1000
WORKSPACE_EXEC_USER = "coder"


class RuntimeKind(StrEnum):
    """Which backend actually runs a workspace's commands."""

    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    AGENT_HOST = "agent_host"


class ExecutionProfile(StrEnum):
    """Whether a command came from a caller or from this service.

    Makes explicit something that was previously only true by accident: the
    public ``/exec`` route validates its command against an allowlist, while
    ``ContainerManager.execute_command`` — reached by internal callers — does
    not. Both funnel into the same backend, so the distinction has to travel
    with the request rather than live in whichever function happened to be
    entered first.
    """

    #: Built by this service (git plumbing, tree walks, manifest reads). The
    #: argv is not caller-controlled, so the allowlist does not apply.
    SERVER_AUTHORED = "server_authored"
    #: Supplied by a caller. Must pass the command allowlist.
    CALLER_SUPPLIED = "caller_supplied"


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """What a backend can actually do.

    Probed rather than assumed: the Kubernetes exec stream can carry stdin, the
    Docker ``exec_run`` convenience wrapper cannot, and the agent host has no
    field for it at all. Code that needs stdin asks instead of discovering the
    difference as a silently-dropped payload.
    """

    kind: RuntimeKind
    #: Backend can feed bytes to the command's stdin.
    stdin: bool = False
    #: Backend bounds output as it reads, rather than after buffering it whole.
    incremental_output_limit: bool = False


@dataclass(frozen=True, slots=True)
class RunOnceResult:
    """The outcome of one bounded command.

    ``stdout_truncated`` / ``stderr_truncated`` are separate from
    ``bytes_omitted`` on purpose: a caller usually only needs to know *whether*
    to distrust a stream, and reporting a single combined flag would make a
    clipped stdout look like a clipped stderr.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration: float
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    bytes_omitted: int = 0
    timed_out: bool = False

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


class UnsupportedCapabilityError(RuntimeError):
    """Raised when a request needs something the selected backend lacks.

    An explicit failure, never a silent degradation: dropping an unsupported
    stdin payload and running the command anyway produces a plausible result
    for a command that never received its input.
    """

    def __init__(self, kind: RuntimeKind, capability: str) -> None:
        super().__init__(f"{kind.value} runtime does not support {capability}")
        self.kind = kind
        self.capability = capability
