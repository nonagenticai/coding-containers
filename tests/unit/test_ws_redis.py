"""py2rust gap #4 full: ``ws-redis`` is admitted by the command policy, raw servers are not.

Its behaviour against a real Valkey (loopback bind, no persistence, the memory cap,
set / get) is exercised in the runtime-image CI smoke.
"""

from __future__ import annotations

from src.runtime.policy import validate_command


def test_policy_admits_the_wrappers_not_the_raw_servers():
    assert validate_command(["ws-redis", "start"])
    assert validate_command(["ws-redis", "start"], toolchain="python+rust")
    for raw in (["valkey-server", "--bind", "0.0.0.0"], ["valkey-cli"], ["redis-server"]):  # noqa: S104
        assert not validate_command(raw, toolchain="python+rust")
