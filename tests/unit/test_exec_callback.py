"""Async exec completion callback: allowlist (fail closed), delivery and n8n classification.

Cases copied from coding-analysis tests/unit/test_bug_hunt_completion_callback.py
(#118), against coding-containers' EXEC_CALLBACK_URL_PREFIXES.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.services.exec_jobs import callback as cb
from src.services.exec_jobs.callback import (
    CallbackOutcome,
    callback_url_allowed,
    retry_delay_seconds,
    send_callback,
)

PREFIX = "http://203.0.113.10:5678/webhook-waiting/"
GOOD = PREFIX + "4711"


def _settings(prefixes: str) -> SimpleNamespace:
    return SimpleNamespace(
        exec_callback_prefixes=tuple(p.strip() for p in prefixes.split(",") if p.strip())
    )


@pytest.fixture
def allow_prefix():
    with patch.object(cb, "get_settings", return_value=_settings(PREFIX)):
        yield


class Recorder:
    """Stands in for httpx.AsyncClient; records POSTs and returns scripted results."""

    def __init__(self, results: list):
        self.results = list(results)
        self.posts: list[tuple[str, dict]] = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json):
        self.posts.append((url, json))
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        status, text = r if isinstance(r, tuple) else (r, "")
        return SimpleNamespace(status_code=status, text=text)


# ── allowlist ─────────────────────────────────────────────────────────


def test_no_prefixes_configured_refuses_everything():
    with patch.object(cb, "get_settings", return_value=_settings("")):
        assert callback_url_allowed(GOOD) is False


def test_settings_property_parses_the_env_value(monkeypatch):
    from src.config.settings import Settings

    monkeypatch.setenv("EXEC_CALLBACK_URL_PREFIXES", f" {PREFIX} , http://h:1/x/ ,")
    assert Settings().exec_callback_prefixes == (PREFIX, "http://h:1/x/")
    monkeypatch.setenv("EXEC_CALLBACK_URL_PREFIXES", "")
    assert Settings().exec_callback_prefixes == ()


@pytest.mark.parametrize(
    "url",
    [
        GOOD,
        PREFIX + "4711?signature=abc123",  # n8n appends its resume token as a query param
        PREFIX + "4711/my-suffix",  # Wait node with a webhook suffix
    ],
)
def test_allowed_urls(allow_prefix, url):
    assert callback_url_allowed(url) is True


@pytest.mark.parametrize(
    "tail",
    [
        "../webhook/x",  # httpx collapses this and POSTs /webhook/x (unauthenticated webhook)
        "./../rest/x",
        "4711/../../rest/x",
        "%2e%2e/webhook/x",
        "4711%2F..%2Fx",
        "..\\webhook\\x",
        "4711//x",  # empty segment
        "",  # the bare prefix, no execution id
        "4711/",
    ],
)
def test_tail_after_prefix_must_be_plain_segments(allow_prefix, tail):
    assert callback_url_allowed(PREFIX + tail) is False


def test_fragment_is_refused(allow_prefix):
    assert callback_url_allowed(GOOD + "#frag") is False


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "http://203.0.113.10:8080/internal",  # off-prefix: an in-cluster SSRF target
        "http://203.0.113.10:5678/rest/workflows",  # same host, other path
        "http://203.0.113.10:5678@evil.example/webhook-waiting/1",  # userinfo trick
        "file:///etc/passwd",
        "ftp://203.0.113.10:5678/webhook-waiting/1",
        "http://[::1/x",  # malformed: a refusal, never an exception
        GOOD + " x",  # whitespace that urlsplit would strip
        GOOD + "\x7f",
        PREFIX + "a" * 2048,  # over the length cap
    ],
)
def test_refused_urls(allow_prefix, url):
    assert callback_url_allowed(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://203.0.113.10:5678@evil.example/x",  # userinfo: the real host is evil.example
        "http://203.0.113.10:56789/x",  # port extended past the prefix
        "http://203.0.113.10:5678.evil.example/x",  # host extended past the prefix
        "https://203.0.113.10:5678/x",  # scheme differs
    ],
)
def test_origin_only_prefix_cannot_be_extended(url):
    with patch.object(cb, "get_settings", return_value=_settings("http://203.0.113.10:5678")):
        assert callback_url_allowed(url) is False
        assert callback_url_allowed("http://203.0.113.10:5678/webhook-waiting/1") is True


def test_httpx_really_collapses_dot_segments():
    """Pins WHY the tail check exists: the URL actually sent is the collapsed one."""
    assert httpx.URL(PREFIX + "../webhook/x").path == "/webhook/x"


# ── delivery ──────────────────────────────────────────────────────────


# n8n 2.42.6 error bodies (ResponseError -> {"code", "message"}), verbatim messages
# from dist/webhooks/waiting-webhooks.js executeWebhook / getWebhookExecutionData.
N8N_RUNNING = (409, '{"code":409,"message":"The execution \\"4711\\" is running already."}')
N8N_FINISHED_ERR = (
    409,
    '{"code":409,"message":"The execution \\"4711\\" has finished with error."}',
)
N8N_FINISHED = (409, '{"code":409,"message":"The execution \\"4711 has finished already."}')
N8N_GONE = (404, '{"code":404,"message":"The execution \\"4711\\" does not exist."}')
N8N_NO_WAIT_PATH = (
    404,
    '{"code":404,"message":"The workflow for execution \\"4711\\" does not contain a '
    'waiting webhook with a matching path/method."}',
)
N8N_BAD_TOKEN = (401, '{"error":"Invalid token"}')


async def _send(results: list, url: str = GOOD) -> tuple[CallbackOutcome, Recorder]:
    rec = Recorder(results)
    with (
        patch.object(cb.httpx, "AsyncClient", new=rec),
        patch.object(cb.asyncio, "sleep", new=AsyncMock()),
    ):
        res = await send_callback(url, {"job_id": "j1"}, job_id="j1")
    # (outcome, attempts); tolerate a bare outcome so a regression reds on the
    # assertion that matters rather than on unpacking.
    outcome = res[0] if isinstance(res, tuple) else res
    return outcome, rec


async def test_2xx_is_delivered_once(allow_prefix):
    outcome, rec = await _send([204])
    assert outcome is CallbackOutcome.DELIVERED
    assert rec.posts == [(GOOD, {"job_id": "j1"})]


async def test_send_reports_attempts_made(allow_prefix):
    rec = Recorder([503, 503, 200])
    with (
        patch.object(cb.httpx, "AsyncClient", new=rec),
        patch.object(cb.asyncio, "sleep", new=AsyncMock()),
    ):
        assert await send_callback(GOOD, {}, job_id="j1") == (CallbackOutcome.DELIVERED, 3)


async def test_4xx_is_final_not_retried(allow_prefix):
    outcome, rec = await _send([404, 200])
    assert outcome is CallbackOutcome.REFUSED
    assert len(rec.posts) == 1


async def test_3xx_is_not_delivered(allow_prefix):
    outcome, rec = await _send([302, 200])
    assert outcome is CallbackOutcome.REFUSED
    assert len(rec.posts) == 1


async def test_5xx_and_transport_errors_are_retried(allow_prefix):
    outcome, rec = await _send([503, httpx.ConnectError("refused"), 200])
    assert outcome is CallbackOutcome.DELIVERED
    assert len(rec.posts) == 3


async def test_n8n_running_already_is_retried_until_the_cell_parks(allow_prefix):
    """D1: a fast job's callback lands before the cell reaches its Wait node.

    n8n answers 409 "is running already"; treating that as final loses the
    callback and parks the cell forever.
    """
    outcome, rec = await _send([N8N_RUNNING, N8N_RUNNING, 200])
    assert outcome is CallbackOutcome.DELIVERED
    assert len(rec.posts) == 3


async def test_round_of_retryable_answers_ends_retry_not_given_up(allow_prefix):
    """D2: a round that never got a final answer is RETRY (persisted), never a give-up."""
    outcome, rec = await _send([500, httpx.ConnectError("down"), N8N_RUNNING])
    assert outcome is CallbackOutcome.RETRY
    assert len(rec.posts) == 3


@pytest.mark.parametrize(
    ("response", "expected", "misconfigured"),
    [
        ((200, ""), CallbackOutcome.DELIVERED, False),
        (N8N_RUNNING, CallbackOutcome.RETRY, False),
        (N8N_FINISHED_ERR, CallbackOutcome.REFUSED, False),
        (N8N_FINISHED, CallbackOutcome.REFUSED, False),
        (N8N_GONE, CallbackOutcome.REFUSED, False),
        (N8N_NO_WAIT_PATH, CallbackOutcome.REFUSED, True),
        (N8N_BAD_TOKEN, CallbackOutcome.REFUSED, True),
        ((400, "bad"), CallbackOutcome.REFUSED, True),
        ((403, ""), CallbackOutcome.REFUSED, True),
        ((422, ""), CallbackOutcome.REFUSED, True),
        ((307, ""), CallbackOutcome.REFUSED, True),  # form-waiting redirect
        ((409, "some other conflict"), CallbackOutcome.REFUSED, False),
        ((408, ""), CallbackOutcome.RETRY, False),
        ((429, ""), CallbackOutcome.RETRY, False),
        ((500, ""), CallbackOutcome.RETRY, False),
        ((503, ""), CallbackOutcome.RETRY, False),
    ],
)
def test_classify_n8n_responses(response, expected, misconfigured):
    assert cb.classify_response(*response) == (expected, misconfigured)


@pytest.mark.parametrize("response", [N8N_FINISHED, N8N_FINISHED_ERR, N8N_GONE])
async def test_no_parked_cell_left_is_final(allow_prefix, response):
    outcome, rec = await _send([response, 200])
    assert outcome is CallbackOutcome.REFUSED
    assert len(rec.posts) == 1


async def test_misconfigured_refusal_is_logged_as_error(allow_prefix, caplog):
    outcome, _rec = await _send([N8N_BAD_TOKEN])
    assert outcome is CallbackOutcome.REFUSED
    assert any(r.levelname == "ERROR" and "misconfigured" in r.getMessage() for r in caplog.records)


def test_retry_backoff_doubles_per_round_and_is_capped():
    delays = [retry_delay_seconds(n) for n in range(3, 3 * 40, 3)]
    assert delays[:4] == [5, 10, 20, 40]
    assert max(delays) == cb.RETRY_CAP_S == 300
    assert delays == sorted(delays)  # never shrinks, never stops


async def test_prefix_narrowed_after_queue_is_not_sent():
    with patch.object(cb, "get_settings", return_value=_settings("http://other:1/x/")):
        outcome, rec = await _send([200])
    assert outcome is CallbackOutcome.DISALLOWED
    assert rec.posts == []


async def test_unexpected_error_never_escapes_and_is_retried(allow_prefix):
    outcome, _rec = await _send([RuntimeError("boom")])
    assert outcome is CallbackOutcome.RETRY
