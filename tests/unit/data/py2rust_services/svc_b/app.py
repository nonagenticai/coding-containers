"""Source service B of the py2rust multi-service fixture: an in-process counter.

Run as ``python3 -m uvicorn app:app --host 127.0.0.1 --port <port>`` from this
directory.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI()
_local: dict[str, int] = {}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/items/{key}")
def item(key: str) -> dict:
    _local[key] = _local.get(key, 0) + 1
    return {"service": "b", "key": key, "count": _local[key]}
