"""A network fence for "this code path must not talk to GitHub" assertions.

Patching a module's ``Github``/``Auth`` names only proves that ONE import was not
used; code that reached GitHub any other way (another client, a re-import) passes.
This fence sits under every HTTP client in the process instead: PyGithub and
``requests`` go through ``requests.Session.send``, and anything else opens a socket
with ``socket.create_connection`` or ``socket.socket.connect``. Every attempt is
recorded and refused, so a test asserts ``fence.attempts == []``.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import requests


@dataclass
class Fence:
    attempts: list[str] = field(default_factory=list)


@contextmanager
def no_outbound_network() -> Iterator[Fence]:
    fence = Fence()

    def _send(_self: Any, request: Any, **_: Any) -> Any:
        fence.attempts.append(f"requests:{getattr(request, 'url', '?')}")
        raise requests.exceptions.ConnectionError("outbound network fenced in this test")

    def _create_connection(address: Any, *_: Any, **__: Any) -> Any:
        fence.attempts.append(f"socket:{address!r}")
        raise OSError("outbound network fenced in this test")

    def _connect(_self: Any, address: Any) -> Any:
        fence.attempts.append(f"connect:{address!r}")
        raise OSError("outbound network fenced in this test")

    with (
        patch.object(requests.Session, "send", _send),
        patch.object(socket, "create_connection", _create_connection),
        patch.object(socket.socket, "connect", _connect),
    ):
        yield fence
