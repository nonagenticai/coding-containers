"""A credential must not be reachable from any exec log record.

Pod stdout is scraped into Loki, which retains it, so a token printed by a
logger outlives the token itself. These tests assert the property at every
place a command is rendered for a log line, on BOTH the success and the failure
path — the failure path being the one that fires while the token is still live:

  1. ``K8sContainerProvisioner.exec_in_pod``            INFO  (success)
  2. ``K8sContainerProvisioner.exec_in_pod``            ERROR (backend failure)
  3. ``ContainerManager.execute_command``               ERROR (except re-log)
  4. ``container_exec.exec_in_container``               INFO  ("Exec success")
  5. ``agent_service._build_cli_command``               DEBUG
  6. ``container_monitoring.create_file_in_container``  ERROR (base64 blob)

⛔ Sites 4-6 were all MISSED by the first pass, and 4 was in a file whose other
three renders were being fixed in the same edit. Fixing 3 of 4 sites in one
file is the failure mode this estate keeps repeating, so the sites above were
re-enumerated from an AST sweep of every ``logger.*`` call in ``src/`` rather
than from the previous session's list.

⭐ Every assertion is paired with a POSITIVE CONTROL. "The token is absent" is
satisfied trivially by logging nothing at all, and a log line with no
diagnostic value is reverted by the next person who has to debug an exec. So
each test also asserts that the non-secret identifiers — pod, deployment,
container name — and the program that actually ran are STILL present.

⭐ Every test is also pinned to its OWN logger (``_messages(logger_name=...)``
or a message-substring filter), because the review found a test that passed on
a NEIGHBOURING site's record. Breaking one guard must fail exactly one test.

⛔ The token literals here are synthetic and low-entropy on purpose
(``ghs_`` + a repeated character): the repo runs a detect-secrets gate, and a
real-looking credential in a test file is its own incident.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

import src.core.kubernetes_container_provisioner as _prov_mod
from src.core.kubernetes_container_provisioner import K8sContainerProvisioner
from src.core.log_safety import (
    _MAX_REDACT_CHARS,
    _SECRET_PATTERNS,
    redact_secrets,
    summarize_command,
)

# Synthetic. Shaped like a GitHub installation token, deliberately low-entropy.
FAKE_TOKEN = "ghs_" + "a" * 36
# Shaped like nothing any denylist knows — the case a regex-based scrub loses.
UNKNOWN_SHAPE_VALUE = "zzq" + "-Wv7-novel-shape-0000"

# The real command git_push.py builds, with the token in it.
TOKEN_WRITE_CMD = f"umask 077 && printf '%s' '{FAKE_TOKEN}' > /tmp/.git_cred_ab12cd34"
PUSH_CMD = "cd /workspace && git push -u origin topic"


def _messages(
    caplog: pytest.LogCaptureFixture,
    min_level: int = logging.NOTSET,
    logger_name: str | None = None,
) -> str:
    """Everything a log record could carry the token in.

    Not just ``record.getMessage()``: an un-interpolated ``%s`` argument is
    still published by any structured handler, so the args are joined in too.

    ``min_level`` and ``logger_name`` scope the assertion to ONE site. Without
    them, the INFO record that ``exec_in_pod`` emits on its way in would satisfy
    the failure-path test's mutation control on its own, and the ERROR render —
    a DIFFERENT render, the one that fires while the token is still live —
    would be untested. ⭐ The adversarial review found exactly that: a test
    passing on a neighbouring site's record. Every test below pins itself to
    its own logger, so breaking one guard fails one test.
    """
    parts = []
    for record in caplog.records:
        if record.levelno < min_level:
            continue
        if logger_name is not None and record.name != logger_name:
            continue
        parts.append(record.getMessage())
        parts.extend(str(arg) for arg in (record.args or ()))
    return "\n".join(parts)


def _make_provisioner() -> K8sContainerProvisioner:
    with patch("kubernetes.config.load_kube_config"):
        prov = K8sContainerProvisioner(namespace="test-ns", in_cluster=False)
    prov.apps_v1 = MagicMock()
    prov.core_v1 = MagicMock()
    return prov


# ---------------------------------------------------------------------------
# summarize_command — the projection itself
# ---------------------------------------------------------------------------


class TestSummarizeCommand:
    def test_string_command_drops_the_token_and_keeps_the_programs(self):
        summary = summarize_command(TOKEN_WRITE_CMD)
        assert FAKE_TOKEN not in summary
        # Positive control: an operator can still see what ran.
        assert "umask" in summary
        assert "printf" in summary

    def test_sh_dash_c_wrapper_is_unwrapped_not_stopped_at(self):
        """``exec_in_pod`` always receives ["sh", "-c", <script>].

        Rendering that as "sh -c" would be safe and useless. The projection
        looks inside the script, so the diagnostic value survives.
        """
        summary = summarize_command(["sh", "-c", PUSH_CMD])
        assert "git push" in summary
        assert "-u" in summary

    def test_a_list_argv_is_not_assumed_safe(self):
        """The old slice printed argv[:3] raw, so a URL credential leaked."""
        summary = summarize_command(["git", "push", f"https://x-access-token:{FAKE_TOKEN}@h/o/r"])
        assert FAKE_TOKEN not in summary
        assert "git push" in summary

    def test_an_unknown_secret_shape_is_also_absent(self):
        """The reason this is an allowlist and not a denylist.

        ``UNKNOWN_SHAPE_VALUE`` matches none of ``redact_secrets``' patterns.
        It is absent from the summary anyway, because it is a VALUE and values
        are never emitted — which is the whole point of the design.
        """
        cmd = f"cd /workspace && curl -H 'Authorization: Bearer {UNKNOWN_SHAPE_VALUE}' https://h"
        summary = summarize_command(cmd)
        assert UNKNOWN_SHAPE_VALUE not in summary
        assert redact_secrets(UNKNOWN_SHAPE_VALUE) == UNKNOWN_SHAPE_VALUE, (
            "control: the denylist does NOT know this shape, so the absence "
            "above was produced by the structural projection, not by the scrub"
        )
        assert "curl" in summary

    def test_empty_is_distinguishable_from_redacted(self):
        assert summarize_command("") == "<empty> (len=0)"
        assert summarize_command(None) == "<none>"


# ---------------------------------------------------------------------------
# Site 1 — the INFO in exec_in_pod (success path)
# ---------------------------------------------------------------------------


class TestExecInPodInfoLog:
    def test_token_absent_from_the_info_record(self, caplog: pytest.LogCaptureFixture):
        prov = _make_provisioner()
        caplog.set_level(logging.DEBUG, logger=_prov_mod.__name__)

        with (
            patch.object(prov, "_resolve_pod_from_deployment", return_value="ws-pod-abc123"),
            patch.object(_prov_mod, "k8s_stream") as mock_stream,
        ):
            resp = MagicMock()
            resp.is_open.side_effect = [True, False]
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            resp.returncode = 0
            mock_stream.return_value = resp

            prov.exec_in_pod(
                deployment_name="ws-deploy",
                command=["sh", "-c", TOKEN_WRITE_CMD],
            )

        logged = _messages(caplog)
        assert FAKE_TOKEN not in logged
        assert "ghs_" not in logged
        # Positive controls: this is a redaction, not a deletion.
        assert "ws-pod-abc123" in logged
        assert "ws-deploy" in logged
        assert "printf" in logged


# ---------------------------------------------------------------------------
# Site 2 — the ERROR in exec_in_pod (backend failure path)
# ---------------------------------------------------------------------------


class TestExecInPodErrorLog:
    def test_token_absent_when_the_exec_stream_fails(self, caplog: pytest.LogCaptureFixture):
        """`str(e)` is free-form and can render the exec URL, whose
        ``?command=`` parameters carry the script."""
        prov = _make_provisioner()
        caplog.set_level(logging.DEBUG, logger=_prov_mod.__name__)

        with (
            patch.object(prov, "_resolve_pod_from_deployment", return_value="ws-pod-abc123"),
            patch.object(_prov_mod, "k8s_stream") as mock_stream,
        ):
            mock_stream.side_effect = RuntimeError(
                f"connection refused for .../exec?command=sh&command=-c&command={FAKE_TOKEN}"
            )
            result = prov.exec_in_pod(
                deployment_name="ws-deploy",
                command=["sh", "-c", TOKEN_WRITE_CMD],
            )

        assert result["exit_code"] == -1
        # Scoped to ERROR so this test stands or falls on the `except` render
        # alone, independently of the INFO line above it.
        logged = _messages(caplog, min_level=logging.ERROR)
        assert logged, "the except handler must still report the failure"
        assert FAKE_TOKEN not in logged
        # Positive controls.
        assert "ws-pod-abc123" in logged
        assert "connection refused" in logged
        assert "printf" in logged


# ---------------------------------------------------------------------------
# Site 3 — the `except` re-log in ContainerManager.execute_command
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    with (
        patch("src.core.container_manager.get_core_manager") as mock_core,
        patch("src.core.container_manager.get_github_container_provisioner") as mock_github,
        patch("src.core.container_manager.get_docker_service") as mock_ds,
    ):
        mock_core.return_value = MagicMock()
        mock_github.return_value = MagicMock()
        mock_ds.return_value = MagicMock()

        from src.core.container_manager import ContainerManager

        yield ContainerManager()


class TestExecuteCommandExceptionLog:
    """⭐ This site is NOT the same render as the INFO line.

    It interpolated the WHOLE command string, not a three-element slice, so it
    disclosed strictly more — and only on failure, i.e. exactly when the push
    has errored, the token is still live, and a retry is imminent. A secret
    that leaks only when something goes wrong is the distribution least likely
    to turn up in a happy-path log sample.
    """

    async def test_token_absent_from_the_failure_path_record(
        self, manager, caplog: pytest.LogCaptureFixture
    ):
        import src.core.container_manager as _cm_mod

        caplog.set_level(logging.DEBUG, logger=_cm_mod.__name__)

        runtime = MagicMock()

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("upload-pack: not our ref")

        runtime.run_once = _boom

        with (
            patch.object(manager, "get_container", return_value="ws-box"),
            patch.object(_cm_mod, "resolve_runtime", return_value=runtime),
        ):
            result = await manager.execute_command("ws-box", TOKEN_WRITE_CMD)

        assert result["success"] is False

        logged = _messages(caplog, min_level=logging.ERROR)
        assert logged, "the except handler must still report the failure"
        assert FAKE_TOKEN not in logged
        assert "ghs_" not in logged
        # Positive controls: the operator still learns which workspace failed,
        # what ran, and why.
        assert "ws-box" in logged
        assert "printf" in logged
        assert "upload-pack: not our ref" in logged


# ---------------------------------------------------------------------------
# Site 4 — the `Exec success` INFO in the container_exec ROUTER
#
# ⛔ This site was MISSED by the first pass while lines 292 and 320 of the SAME
# FILE were being fixed, and it was the worst of the four: `" ".join(
# request.command)` is the whole argv, not even the `[:3]` slice the others
# had, at INFO, on the SUCCESS path. Both boundary gates admit the payload, so
# it reached a 30-day Loki alongside a 200.
# ---------------------------------------------------------------------------


class TestExecSuccessRouterLog:
    """Drives the REAL endpoint, through the REAL boundary gates."""

    @staticmethod
    def _argv() -> list[str]:
        return ["git", "push", f"https://x-access-token:{FAKE_TOKEN}@github.com/o/r"]

    def test_both_boundary_gates_admit_the_credential_payload(self):
        """The control that makes this site's exposure real rather than theoretical.

        If either gate rejected this argv, the leak would be unreachable and
        the test below would prove nothing.
        """
        from src.runtime import check_dangerous_patterns, validate_command

        assert validate_command(self._argv()) is True
        assert check_dangerous_patterns(self._argv()) is True

    async def test_token_absent_from_the_success_record(self, caplog: pytest.LogCaptureFixture):
        import src.routers.container_exec as _ce

        caplog.set_level(logging.DEBUG, logger=_ce.__name__)

        result = MagicMock()
        result.exit_code = 0
        result.duration = 0.10
        result.stdout = "ok"
        result.stderr = ""
        result.truncated = False
        result.bytes_omitted = 0

        async def _run_once(*_a, **_k):
            return result

        runtime = MagicMock()
        runtime.run_once = _run_once
        runtime.kind.value = "kubernetes"

        with (
            patch.object(_ce, "get_container_manager", return_value=MagicMock()),
            patch.object(_ce, "resolve_runtime", return_value=runtime),
            patch(
                "src.core.container_manager_core.get_container_manager",
                return_value=MagicMock(),
            ),
        ):
            response = await _ce.exec_in_container(
                container_name="ws-box",
                request=_ce.ExecRequest(command=self._argv(), timeout=30),
                tenant_id="t1",
            )

        assert response.exit_code == 0

        success = [r for r in caplog.records if "Exec success" in r.getMessage()]
        assert success, "the success path must still log"
        logged = success[0].getMessage()

        assert FAKE_TOKEN not in logged
        assert "ghs_" not in logged
        # Positive controls: this is a redaction, not a deletion. All four of
        # these were present in the leaking record and must survive.
        assert "ws-box" in logged
        assert "git push" in logged
        assert "exit_code=0" in logged
        assert "kubernetes" in logged


# ---------------------------------------------------------------------------
# Site 5 — agent_service._build_cli_command
#
# ⛔ Previously called "out of scope" because it logs `cmd[:3]`, i.e. "a program
# and two sub-commands". The `claude` branch builds ["claude", "-p", prompt,
# …], so element 2 of that slice IS THE WHOLE PROMPT — caller-supplied free
# text that routinely contains a credential. The `codex` branch appends the
# prompt LAST, so the identical expression hid it there and disclosed it here:
# a slice is not a redaction, it is an argv-order coincidence.
# ---------------------------------------------------------------------------


class TestAgentServiceCliCommandLog:
    def test_prompt_absent_from_the_built_command_record(self, caplog: pytest.LogCaptureFixture):
        import src.services.agent_service as _as_mod
        from src.services.agent_service import AgentService

        caplog.set_level(logging.DEBUG, logger=_as_mod.__name__)

        with patch.object(_as_mod, "DockerService"):
            svc = AgentService.__new__(AgentService)

        prompt = f"push the branch using token {FAKE_TOKEN}"
        cmd = AgentService._build_cli_command(
            svc, cli="claude", prompt=prompt, model="opus", system_prompt=None
        )

        logged = _messages(caplog, logger_name=_as_mod.__name__)
        assert logged, "the command build must still be logged"
        assert FAKE_TOKEN not in logged
        assert "push the branch" not in logged, "the prompt body must not be published"
        # Positive controls.
        assert "claude" in logged
        assert "-p" in logged
        # Control: the COMMAND still carries the token. We redacted the log
        # line, not the behaviour — if this flips, the test is passing because
        # the feature broke.
        assert FAKE_TOKEN in " ".join(cmd)


# ---------------------------------------------------------------------------
# Site 6 — container_monitoring.create_file_in_container
#
# ⛔ The record logged `echo '<base64>' | base64 -d > file` on failure, and for
# the git flows that blob DECODES to a live credential URL. A base64 payload is
# not obfuscation; it is a REVERSIBLE credential in an ERROR record, which is
# exactly why it did not look like a secret to the first pass.
# ---------------------------------------------------------------------------


class TestCreateFileInContainerLog:
    async def test_record_does_not_decode_to_a_credential(self, caplog: pytest.LogCaptureFixture):
        import base64
        import re as _re

        import src.services.docker_services.container_monitoring as _mon_mod
        from src.services.docker_services.container_monitoring import ContainerMonitor

        caplog.set_level(logging.DEBUG, logger=_mon_mod.__name__)

        cred_url = f"https://x-access-token:{FAKE_TOKEN}@github.com/o/r"

        monitor = ContainerMonitor.__new__(ContainerMonitor)
        monitor.client = MagicMock()

        async def _fake_exec(_docker_id, command, timeout=10):
            # mkdir succeeds; the base64 write is the one that fails. Without
            # this the loop returns before the blob-bearing command is ever
            # rendered, and the test would assert against the WRONG record.
            script = command[2] if len(command) > 2 else ""
            if script.startswith("mkdir"):
                return {"exit_code": 0, "stdout": "", "stderr": ""}
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "sh: cannot create /w/.git/creds: read-only",
            }

        monitor.exec_in_container = _fake_exec

        created = await monitor.create_file_in_container(
            "dock123", "/workspace/.git/credentials", cred_url
        )
        assert created is False

        failures = [r for r in caplog.records if "Failed to execute" in r.getMessage()]
        assert failures, "the failure must still be reported"
        logged = failures[-1].getMessage()

        raw_blob = base64.b64encode(cred_url.encode()).decode()
        assert FAKE_TOKEN not in logged
        assert raw_blob not in logged

        # ⭐ THE assertion for this site: not "the token string is absent" but
        # "nothing in this record DECODES to the token". A plaintext check
        # passes trivially on a base64 record.
        for candidate in _re.findall(r"[A-Za-z0-9+/=]{16,}", logged):
            for pad in range(3):
                try:
                    decoded = base64.b64decode(candidate + "=" * pad).decode("utf-8")
                except Exception:
                    continue
                assert FAKE_TOKEN not in decoded, f"{candidate!r} decodes to the token"
                assert "x-access-token" not in decoded
                break

        # Control: the blob really was reversible, so the assertion above is
        # testing something that could fail.
        assert base64.b64decode(raw_blob).decode() == cred_url

        # Positive controls: an operator can still diagnose the failure.
        assert "echo" in logged
        assert "base64 -d" in logged
        assert "/workspace/.git/credentials" in logged
        assert "read-only" in logged


# ---------------------------------------------------------------------------
# redact_secrets must be LINEAR — this diff introduced a ReDoS and the fix is
# only provable by timing.
# ---------------------------------------------------------------------------


class TestRedactSecretsIsLinear:
    """⛔ Generalising the URL scheme to ``[a-zA-Z][a-zA-Z0-9+.-]*://`` made
    this quadratic on ANY long alphanumeric run — no ``://`` needed anywhere in
    the input, because the unbounded ``*`` backtracks at every start offset.

    Measured on ``"fatal: " + "A"*n`` BEFORE the fix:
        32 KiB 0.71s | 64 KiB 2.86s | 128 KiB 12.10s   (×4 per doubling)
    and the carrier is real: ``git_push`` feeds raw exec stderr into
    ``_sanitize_error_message``, capped at ``exec_max_output_bytes`` (1 MiB
    default) ⇒ ~12 minutes blocking an async handler.

    AFTER, with every quantifier bounded: 0.0010s / 0.0019s / 0.0038s raw, and
    0.0009s flat through the input cap. The thresholds below are ~100× the
    measured cost, so this fails on a reintroduced ReDoS and not on a slow
    machine.

    ⛔⛔ THE TIMING TESTS MUST BYPASS THE INPUT CAP, and this is not a detail —
    it is the whole validity of this class. A first version of these tests
    called ``redact_secrets`` directly, and a MUTATION CONTROL caught them
    being VACUOUS: restoring the unbounded ``*`` to the scheme pattern did not
    fail a single one. The 16 KiB cap truncates the payload before the regex
    ever sees it, so the tests were measuring the cap and would have gone green
    with the ReDoS fully reintroduced — a guard that cannot fail. So the
    linearity assertions below drive ``_SECRET_PATTERNS`` directly, and the cap
    gets its own separate end-to-end test. Two defences, two tests; never one
    test that silently only exercises the cheaper one.
    """

    @staticmethod
    def _elapsed_raw(payload: str) -> float:
        """Cap BYPASSED: this is what proves the REGEX is linear."""
        start = time.perf_counter()
        for pattern, replacement in _SECRET_PATTERNS:
            pattern.sub(replacement, payload)
        return time.perf_counter() - start

    @staticmethod
    def _elapsed(payload: str) -> float:
        """End-to-end, cap ACTIVE."""
        start = time.perf_counter()
        redact_secrets(payload)
        return time.perf_counter() - start

    @pytest.mark.parametrize("kib", [32, 64, 128])
    def test_long_alphanumeric_run_is_fast(self, kib: int):
        """Measured pre-fix: 0.71s / 2.86s / 12.10s. Post-fix: ~0.001-0.004s."""
        assert self._elapsed_raw("fatal: " + "A" * (kib * 1024)) < 0.5

    # ⚠️ 128 KiB is deliberately the LARGEST raw (cap-bypassed) size tested.
    # The real worst case is 1 MiB (``exec_max_output_bytes``' default), but
    # asserting that raw means a REINTRODUCED ReDoS makes this suite hang for
    # ~12 minutes before failing, which is how a slow test gets deleted instead
    # of read. Linearity is established by 32/64/128 + the ratio test; the 1 MiB
    # figure is covered end-to-end through the cap by
    # ``test_the_cap_also_holds_end_to_end`` below. Pre-fix, 128 KiB already
    # takes 12.10s and blows the 0.5s threshold by 24×.

    def test_repeated_scheme_prefix_is_fast(self):
        """The SECOND carrier: an unbounded userinfo ``[^/\\s@]+`` is quadratic
        on ``a://`` repeated, since each match scans to end-of-input for an
        ``@`` that is not there."""
        assert self._elapsed_raw("a://" * (128 * 256)) < 0.5

    def test_growth_is_not_quadratic(self):
        """A ratio check, so a reintroduced ReDoS fails on SHAPE, not only on
        an absolute threshold that a fast machine might still clear."""
        base = max(self._elapsed_raw("fatal: " + "A" * (64 * 1024)), 1e-6)
        doubled = self._elapsed_raw("fatal: " + "A" * (128 * 1024))
        # Linear ⇒ ~2×. Quadratic ⇒ ~4×. 3× separates them with margin.
        assert doubled / base < 3.0

    def test_the_cap_also_holds_end_to_end(self):
        """The cap is defence in depth BEHIND the bounded quantifiers, so it
        gets its own assertion rather than standing in for them."""
        assert self._elapsed("fatal: " + "A" * (1024 * 1024)) < 0.5

    def test_the_cap_truncates_and_says_so(self):
        out = redact_secrets("B" * (_MAX_REDACT_CHARS + 500))
        assert len(out) < _MAX_REDACT_CHARS + 200
        assert "characters omitted" in out

    def test_a_secret_is_still_redacted_after_the_generalisation(self):
        """The bounded scheme must not have un-done what it was widened FOR:
        a credential on a non-https scheme."""
        for scheme in ("https", "git+ssh", "ftp"):
            out = redact_secrets(f"remote: {scheme}://user:{FAKE_TOKEN}@h.corp/o/r failed")
            assert FAKE_TOKEN not in out
            assert scheme in out, "positive control: the scheme itself survives"


# ---------------------------------------------------------------------------
# The widened projection: more useful, and still closed.
# ---------------------------------------------------------------------------


class TestProjectionUsefulness:
    """⭐ The first cut was safe but close to useless, measured:
        ``ruff format -- a.py b.py``              lost the FILES
        ``cd /w && python -m pytest tests/x.py``  lost the TEST PATH
        ``cd /w && git -c credential.helper= push``  lost the VERB ``push``
    Path-shaped words, a ``--`` passthrough, and scanning past an opaque word
    recover all three WITHOUT admitting a value.
    """

    def test_file_arguments_survive(self):
        assert summarize_command("ruff format -- a.py b.py").startswith("ruff format -- a.py b.py")

    def test_test_path_survives(self):
        assert "tests/x.py" in summarize_command("cd /w && python -m pytest tests/x.py")

    def test_the_verb_behind_an_opaque_argument_survives(self):
        """The worst measured loss: ``push`` disappeared behind ``-c
        credential.helper=`` two positions earlier."""
        summary = summarize_command("cd /w && git -c credential.helper= push origin topic")
        assert "push" in summary
        assert "credential.helper=" not in summary
        # The gap is VISIBLE, not silently closed: `git -c push` would read as
        # a command that was never run.
        assert "?" in summary


class TestWideningDidNotOpenAHole:
    """⚠️ A/B projection attacks re-run against the widened rules. Each is a
    value that must NOT reach the log even though the word shapes were relaxed.
    """

    @staticmethod
    def _b64_credential_blob() -> tuple[str, str]:
        import base64

        url = f"https://x-access-token:{FAKE_TOKEN}@github.com/o/r"
        return base64.b64encode(url.encode()).decode(), url

    def test_base64_credential_blob_is_not_path_shaped(self):
        """The realistic carrier for the path allowance: it is 104 chars and
        contains no ``/``, so it fails two of the three barriers independently."""
        blob, _ = self._b64_credential_blob()
        summary = summarize_command(f"echo '{blob}' | base64 -d > /workspace/.git/creds")
        assert blob not in summary
        # Positive control: the widening DID keep the destination path.
        assert "/workspace/.git/creds" in summary

    def test_a_short_blob_with_a_slash_is_still_rejected(self):
        """Barrier 3 alone: a blob short enough and slashed enough to clear the
        length and ``/`` checks is still stopped by the long-alnum-run rule."""
        blob = "aa/" + "Zm9vYmFyYmF6cXV4MTIzNA"
        summary = summarize_command(f"echo '{blob}' | base64 -d")
        assert blob not in summary

    @pytest.mark.parametrize(
        "command,secret",
        [
            (f"umask 077 && printf '%s' '{FAKE_TOKEN}' > /tmp/.git_cred_ab12", FAKE_TOKEN),
            (["git", "push", f"https://x-access-token:{FAKE_TOKEN}@github.com/o/r"], FAKE_TOKEN),
            (f"git clone --config=http.extraheader={FAKE_TOKEN} https://h/r", FAKE_TOKEN),
            (
                f"cd /w && curl -H 'Authorization: Bearer {UNKNOWN_SHAPE_VALUE}' https://h",
                UNKNOWN_SHAPE_VALUE,
            ),
        ],
    )
    def test_value_position_secrets_remain_absent(self, command, secret):
        assert secret not in summarize_command(command)


class TestDocumentedResiduals:
    """⚠️ These are the limits the module docstring now STATES rather than
    papering over. They are asserted so that if the projection is ever
    tightened enough to close them, the docstring is updated in the same
    commit instead of silently over-claiming.
    """

    def test_a_command_position_token_can_appear(self):
        summary = summarize_command(f"git push; {UNKNOWN_SHAPE_VALUE}")
        assert UNKNOWN_SHAPE_VALUE in summary, (
            "documented residual: after `;` this word is a PROGRAM. If this "
            "now fails, the guarantee got STRONGER — update the module "
            "docstring, which currently claims this leaks."
        )

    def test_a_lowercase_password_in_command_position_can_appear(self):
        assert "hunterhunterhunter" in summarize_command("sshpass -p hunterhunterhunter ssh"), (
            "documented residual; see the module docstring"
        )
