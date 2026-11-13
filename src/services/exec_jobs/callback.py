"""Completion callback for async workspace exec jobs.

Semantics copied from coding-analysis #118 (``bug_hunter.callback_url_allowed`` /
``_send_completion_callback``), with the setting renamed to
``EXEC_CALLBACK_URL_PREFIXES``:

* Allowlist, fail closed. No configured prefix refuses every ``callback_url``.
  The origin (scheme + host:port) must match a prefix EXACTLY and only the path
  is a prefix; after it the path must be plain segments (``_CALLBACK_TAIL``) —
  the traversal guard, because httpx collapses ``..`` before sending. No
  userinfo, no fragment, no whitespace/control characters, at most 2048 chars.
  Re-checked at send time, so narrowing the prefixes also stops queued jobs.
* Delivery: one round of up to 3 attempts (2 s / 4 s backoff) per call; the
  response is classified by :func:`classify_response` against n8n 2.42.6's
  ``WaitingWebhooks.executeWebhook``. A round that ends retryable returns
  ``RETRY`` and the caller persists it (``callback_state = 'retrying'`` with a
  capped ``callback_next_attempt_at``) for the reaper to resend — indefinitely,
  never a give-up: a parked n8n cell has no other way to be resumed. Only a
  response proving no parked cell can still take it, or a misconfiguration, is
  final. No redirects followed. Never raises.
"""

from __future__ import annotations

import asyncio
import logging
import re
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

CALLBACK_TIMEOUT_S = 10.0
CALLBACK_ATTEMPTS = 3
CALLBACK_URL_MAX_LEN = 2048

# What may follow the configured prefix: plain path segments only (an n8n resume
# URL is ``<prefix><executionId>`` with an optional ``/<suffix>``). No dots, no
# percent-encoding, no backslashes, no empty segments.
_CALLBACK_TAIL = re.compile(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*")


#: Durable retry backoff: 5 s doubling per round, capped. The cap is an interval
#: ceiling, not a give-up — a retryable callback is retried for as long as it
#: stays retryable (George 2026-10-11: no time budgets).
RETRY_BASE_S = 5
RETRY_CAP_S = 300


class CallbackOutcome(StrEnum):
    """Delivery state, persisted as ``exec_jobs.callback_state``."""

    DELIVERED = "delivered"
    REFUSED = "refused"  # final non-2xx: no parked cell can take it, or misconfigured
    RETRY = "retrying"  # not deliverable YET: the reaper resends it later
    DISALLOWED = "disallowed"  # no longer under an allowed prefix at send time


#: n8n's ConflictError text while the execution has not parked on its Wait node
#: yet (``execution.status === 'running'``) — a fast job's callback can arrive
#: before the cell that POSTed the job reaches its Wait node.
N8N_RUNNING_ALREADY = "is running already"


def retry_delay_seconds(total_attempts: int) -> int:
    """Delay before the next durable round, from attempts made so far."""
    rounds = max(0, total_attempts // CALLBACK_ATTEMPTS - 1)
    return min(RETRY_CAP_S, RETRY_BASE_S * 2 ** min(rounds, 10))


def classify_response(status: int, text: str) -> tuple[CallbackOutcome, bool]:
    """Map one resume-POST response to ``(outcome, misconfigured)``.

    Derived from n8n 2.42.6 ``dist/webhooks/waiting-webhooks.js``
    (``executeWebhook``) — what each answer means for the parked cell:

    * 2xx — resumed: DELIVERED.
    * 409 "… is running already." — the execution has not parked on its Wait
      node yet (``status === 'running'``): RETRY.
    * 409 "… has finished with error." / "… has finished already." — the
      execution is over; no parked cell remains: final.
    * 404 "… does not exist." — execution deleted/pruned: final.
    * 404 "… does not contain a waiting webhook with a matching path/method." /
      "Could not find node to process webhook." — wrong suffix/method: final,
      misconfigured.
    * 401 "Invalid token" (the per-execution ``resumeToken`` /signature does not
      match), 400/403/422 and any other 4xx, 3xx (form-waiting redirect): final,
      misconfigured.
    * 408 / 429 / 5xx — transient: RETRY.
    """
    if 200 <= status < 300:
        return CallbackOutcome.DELIVERED, False
    if status == 409:
        if N8N_RUNNING_ALREADY in text:
            return CallbackOutcome.RETRY, False
        return CallbackOutcome.REFUSED, False
    if status in (408, 429) or status >= 500:
        return CallbackOutcome.RETRY, False
    if status == 404 and "does not exist" in text:
        return CallbackOutcome.REFUSED, False
    return CallbackOutcome.REFUSED, True


def _response_text(resp: Any) -> str:
    try:
        return str(resp.text)[:2000]
    except Exception:
        return ""


def callback_url_allowed(url: str | None) -> bool:
    """True iff ``url`` is an http(s) URL under a configured callback prefix."""
    if not url or len(url) > CALLBACK_URL_MAX_LEN:
        return False
    if any(ord(c) <= 0x20 or ord(c) == 0x7F for c in url):
        return False
    try:
        parts = urlsplit(url)
    except ValueError:  # e.g. "http://[::1/x" — malformed is a refusal, never a 500
        return False
    if (
        parts.scheme not in ("http", "https")
        or not parts.netloc
        or "@" in parts.netloc
        or parts.fragment
    ):
        return False
    for prefix in get_settings().exec_callback_prefixes:
        allowed = urlsplit(prefix)
        base = allowed.path if allowed.path.endswith("/") else allowed.path + "/"
        if (
            parts.scheme == allowed.scheme
            and parts.netloc.lower() == allowed.netloc.lower()
            and parts.path.startswith(base)
            and _CALLBACK_TAIL.fullmatch(parts.path[len(base) :])
        ):
            return True
    return False


async def send_callback(
    url: str, body: dict[str, Any], *, job_id: str
) -> tuple[CallbackOutcome, int]:
    """POST ``body`` to ``url``: one round of up to 3 attempts. Never raises.

    Returns ``(outcome, attempts_made)``. ``RETRY`` means the round ended on a
    retryable answer; the caller persists it for a later round.
    """
    if not callback_url_allowed(url):
        logger.warning("Exec job %s: callback_url no longer allowed; not sent", job_id)
        return CallbackOutcome.DISALLOWED, 0
    attempt = 0
    try:
        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT_S, follow_redirects=False) as client:
            for attempt in range(1, CALLBACK_ATTEMPTS + 1):
                try:
                    resp = await client.post(url, json=body)
                    text = _response_text(resp)
                    outcome, misconfigured = classify_response(resp.status_code, text)
                    if outcome is CallbackOutcome.DELIVERED:
                        logger.info(
                            "Exec job %s: callback delivered (HTTP %s, attempt %s)",
                            job_id,
                            resp.status_code,
                            attempt,
                        )
                        return outcome, attempt
                    if outcome is CallbackOutcome.REFUSED:
                        log = logger.error if misconfigured else logger.warning
                        log(
                            "Exec job %s: callback refused (HTTP %s%s): %s",
                            job_id,
                            resp.status_code,
                            "; callback_url misconfigured" if misconfigured else "",
                            text[:300],
                        )
                        return outcome, attempt
                    logger.warning(
                        "Exec job %s: callback not accepted yet (HTTP %s, attempt %s): %s",
                        job_id,
                        resp.status_code,
                        attempt,
                        text[:300],
                    )
                except httpx.HTTPError as exc:
                    logger.warning(
                        "Exec job %s: callback failed (attempt %s): %s", job_id, attempt, exc
                    )
                if attempt < CALLBACK_ATTEMPTS:
                    await asyncio.sleep(2**attempt)
    except Exception:  # never let a callback failure escape into the job path
        # Retried, not dropped: giving up would park the cell forever.
        logger.exception("Exec job %s: callback errored; will retry", job_id)
        return CallbackOutcome.RETRY, max(attempt, 1)
    logger.warning(
        "Exec job %s: callback round of %s attempts not accepted; will retry", job_id, attempt
    )
    return CallbackOutcome.RETRY, attempt
