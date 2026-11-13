"""Source service A of the py2rust multi-service fixture: calls B over HTTP.

``B_URL`` is B's loopback base URL.
"""

from __future__ import annotations

import json
import os
import urllib.request

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/chain/{key}")
def chain(key: str) -> dict:
    base = os.environ["B_URL"].rstrip("/")
    with urllib.request.urlopen(f"{base}/items/{key}", timeout=5) as resp:  # noqa: S310  # nosec B310
        b = json.loads(resp.read())
    return {"service": "a", "b": b}
