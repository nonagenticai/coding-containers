"""Client for coding-realtime service (port 8694).

Wraps CodingRealtimeClient from coding-client. Broadcast is fire-and-forget:
a publish failure never crashes the caller — but it is never silent either.
Every call returns a :class:`BroadcastResult` whose ``ok`` is DERIVED from the
response (or from the raised ServiceError's status code), never assumed, and
every non-2xx is logged with the status code and the container name. A 404 is
logged at ERROR and flagged separately, because 404 is the signature of the
ownership check failing to resolve the container (coding-realtime's
``validate_and_get_container_for_tenant`` maps every miss to 404) — i.e. the
event was emitted before the container was registered, or under the wrong
tenant. That class of bug used to be completely invisible.

Fire-and-forget applies to TRANSPORT failures only. A programming error — a
wrong call signature, an event type coding-realtime rejects, a broadcast with no
owning tenant — is a bug, not a delivery problem, and is re-raised. A blanket
``except Exception`` here is what kept the original ``broadcast(json={...})``
``TypeError`` invisible for weeks while every lifecycle event was dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from coding_client.clients import CodingRealtimeClient, ServiceError
from src.clients._tenant import service_token, tenant_headers

logger = logging.getLogger(__name__)

#: coding-realtime returns 404 both for "container unknown" and for
#: "container not owned by this tenant" — the publish-side BOLA check.
_NOT_FOUND = 404

_http_client: CodingRealtimeClient | None = None


def _get_http_client() -> CodingRealtimeClient:
    global _http_client
    if _http_client is None:
        _http_client = CodingRealtimeClient()
    return _http_client


class BroadcastTenantMissingError(Exception):
    """A lifecycle broadcast had no owning tenant to attribute it to.

    Deliberately NOT a subclass of ``ValueError`` / ``RuntimeError`` /
    ``DockerException``, so none of the broad ``except`` clauses on the container
    lifecycle path can quietly downgrade it to a generic "container creation
    failed".

    Filling this in with ``DEFAULT_TENANT_ID``, a literal, or "whatever tenant we
    can find" is precisely the cross-tenant bug class the ``X-Tenant-Id`` header
    exists to close: the publish would then be authorised against a tenant that
    does not own the container. A missing tenant is a bug in the caller, so it is
    loud.
    """


class BroadcastEventType(StrEnum):
    """Event types coding-realtime's publish endpoint actually accepts.

    This enum MIRRORS ``coding-realtime``'s ``BroadcastEventType``
    (``src/services/websocket_broadcast.py``). ``POST /api/broadcast/publish``
    validates ``event_type`` against that enum and answers ``422 Invalid
    event_type`` for anything outside it.

    ``container_status`` / ``container_created`` / ``container_removed`` used to
    be declared here. They exist in no coding-realtime enum, are emitted by no
    call site in this repo, and were verified live to 422 on the wire against the
    running service — so they were pure drift and are removed, rather than being
    added upstream to legitimise values nothing sends. Everything this service
    genuinely emits (``container_started`` / ``container_stopped`` /
    ``container_health``) is already valid there.

    Adding a value here that coding-realtime does not know re-creates the
    silent-422 bug; ``tests/unit/test_realtime_client_contract.py`` pins the set.
    """

    CONTAINER_STARTED = "container_started"
    CONTAINER_STOPPED = "container_stopped"
    CONTAINER_HEALTH = "container_health"
    ANALYSIS_PROGRESS = "analysis_progress"


class BroadcastPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class BroadcastResult:
    """Outcome of a publish attempt. ``ok`` is DERIVED, never assumed.

    ``ok`` means the publish endpoint accepted the event (2xx). It says nothing
    about fan-out: ``delivered=False`` / ``subscriber_count=0`` is a successful
    publish that nobody was listening for, which is normal and not an error.
    """

    ok: bool
    status_code: int | None = None
    delivered: bool | None = None
    subscriber_count: int | None = None
    error: str | None = None

    @property
    def not_found(self) -> bool:
        """True when the publish was rejected 404.

        coding-realtime resolves the container for the calling tenant before it
        publishes, so 404 means "not registered yet, or not yours" — never a
        transport problem.
        """
        return self.status_code == _NOT_FOUND


class BroadcastService:
    """Broadcast service - delegates to coding-realtime. Fire-and-forget."""

    async def broadcast(
        self,
        event_type: BroadcastEventType,
        data: dict[str, Any],
        priority: BroadcastPriority = BroadcastPriority.NORMAL,
        *,
        tenant_id: str | None,
    ) -> BroadcastResult:
        """Publish a container lifecycle broadcast (fire-and-forget on transport).

        Delegates to CodingRealtimeClient.broadcast, which POSTs to
        /api/broadcast/publish (D9 — the old helper hit the WS-only /subscribe
        route, 404, and the event was silently dropped). The container is carried
        inside ``data['container_name']``.

        ``tenant_id`` is REQUIRED and keyword-only. It is the OWNING tenant of the
        container — the authenticated caller's tenant at provisioning time
        (``User.effective_tenant_id``, threaded down as
        ``docker_service.create_container(tenant_id=...)`` and stamped onto
        ``ContainerInfo.tenant_id``). It is sent as ``X-Tenant-Id`` and
        coding-realtime resolves the container against it (BOLA) before fanning
        out. It used to be optional, and an absent tenant silently sent *no*
        header — which only worked because coding-realtime still trusts the pod
        CIDR. That fallback is being removed, at which point a header-less
        publish 401s.

        Does not raise for transport/service failures (coding-realtime down,
        slow, or answering non-2xx): those are logged and reported through the
        returned :class:`BroadcastResult`, because a notification must never
        break the container operation that emitted it. It DOES raise for
        programming errors — a missing tenant, a wrong call signature — since
        those are bugs that must not be able to hide.

        Raises:
            BroadcastTenantMissingError: no owning tenant was resolvable.
        """
        container_name = data.get("container_name")
        if not tenant_id:
            raise BroadcastTenantMissingError(
                f"Refusing to broadcast {event_type!s} for container "
                f"{container_name!r} with no owning tenant. The tenant must be "
                "threaded from the authenticated provisioning request; it is never "
                "defaulted, because publishing under a guessed tenant is a "
                "cross-tenant authorisation bug."
            )
        try:
            client = _get_http_client()
            result = await client.broadcast(
                event_type=str(event_type),
                data=data,
                priority=str(priority),
                container_name=container_name,
                # X-Tenant-Id names the container's OWNING tenant; Authorization
                # carries this service's own licence so coding-realtime can vouch
                # the assertion once it stops trusting a bare header. The two must
                # stay separate: a token in X-Tenant-Id would resolve every
                # publish to coding-containers' own tenant and the BOLA check
                # would then pass against the wrong owner.
                headers=tenant_headers(
                    tenant_id,
                    operation=f"broadcast {event_type!s} for {container_name!r}",
                    service_token=service_token(),
                ),
            )
        except ServiceError as e:
            status = getattr(e, "status_code", None)
            if status == _NOT_FOUND:
                # The single most important line in this module: a 404 here is
                # NOT a transport hiccup, it is the ownership check refusing to
                # resolve the container. Emitting a lifecycle event before the
                # container is registered (or under the wrong tenant) lands here
                # and used to be swallowed without a trace.
                logger.error(
                    "Broadcast %s REJECTED 404 for container %s (tenant=%s): the container is "
                    "not resolvable by the publish ownership check — it is either not registered "
                    "yet or not owned by this tenant. Event dropped: %s",
                    event_type,
                    container_name,
                    tenant_id,
                    e,
                )
            else:
                logger.warning(
                    "Broadcast %s failed with status %s for container %s (fire-and-forget): %s",
                    event_type,
                    status,
                    container_name,
                    e,
                )
            return BroadcastResult(ok=False, status_code=status, error=str(e))
        except (httpx.HTTPError, OSError) as e:
            # Fire-and-forget, but only for TRANSPORT failures: connect/read
            # timeouts, DNS, connection resets, coding-realtime being down. Those
            # must never break provisioning, and are logged at ERROR (not
            # warning) with the exception so a permanently-broken path is visible
            # in the logs instead of being a debug-level shrug.
            #
            # Everything else propagates ON PURPOSE. A TypeError from a wrong
            # call signature, an AttributeError, a BroadcastTenantMissingError —
            # those are bugs. The bare `except Exception` that used to be here is
            # the reason `broadcast(json={...})` looked healthy while 100% of
            # container lifecycle events were being dropped.
            logger.error(
                "Broadcast %s for container %s failed to reach coding-realtime (fire-and-forget)",
                event_type,
                container_name,
                exc_info=True,
            )
            return BroadcastResult(ok=False, error=str(e))

        # 2xx. Derive fan-out from the body rather than assuming it was seen.
        delivered: bool | None = None
        subscriber_count: int | None = None
        if isinstance(result, dict):
            raw_delivered = result.get("delivered")
            if isinstance(raw_delivered, bool):
                delivered = raw_delivered
            raw_count = result.get("subscriber_count")
            if isinstance(raw_count, int) and not isinstance(raw_count, bool):
                subscriber_count = raw_count

        # Surface a no-op fan-out: delivered=false means nobody was
        # subscribed. Not an error, but observable (the whole point of D9 —
        # no more pretending a dropped broadcast succeeded).
        if delivered is False:
            logger.debug(
                "Broadcast %s reached no subscribers for container %s",
                event_type,
                container_name,
            )
        return BroadcastResult(
            ok=True,
            status_code=200,
            delivered=delivered,
            subscriber_count=subscriber_count,
        )


_instance: BroadcastService | None = None


def get_broadcast_service() -> BroadcastService:
    global _instance
    if _instance is None:
        _instance = BroadcastService()
    return _instance
