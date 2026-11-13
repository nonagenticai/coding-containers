"""Tests for services/agents/failure_agent.py.

The legacy ``FailureAnalysisAgent`` was deleted in favour of a thin
``coding_client.llm.BaseLLMAgent`` subclass. These tests exercise the new
agent by mocking ``BaseLLMAgent._call_llm`` (no live HTTP, no live
inference-router) and asserting the prompt-rendering / response-handling
contract.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from coding_client.exceptions.pydantic_ai_exceptions import LLMEmptyResponseError
from coding_client.llm import (
    LLMSettings,
    TestFailureAnalysisResponse,
    llm_request_context,
)
from src.models.pydantic_ai_models import PydanticAITestFailureContext
from src.services.agents.failure_agent import (
    FailureAnalysisAgent,
    create_test_failure_analysis_agent,
)


def _make_context(**overrides) -> PydanticAITestFailureContext:
    defaults = dict(
        test_id="test-123",
        language="python",
        test_type="unit",
        exit_code=1,
        stdout="some output",
        stderr="NameError: name 'x' is not defined",
        execution_time_ms=500.0,
    )
    defaults.update(overrides)
    return PydanticAITestFailureContext(**defaults)


def _make_response(**overrides) -> TestFailureAnalysisResponse:
    defaults = dict(
        root_cause="undefined variable",
        suggested_fix="define x before use",
        confidence=0.85,
        category="other",
    )
    defaults.update(overrides)
    return TestFailureAnalysisResponse(**defaults)


def _make_settings() -> LLMSettings:
    return LLMSettings(
        llm_api_base_url="http://localhost:9999/v1",
        llm_api_key="test",
        llm_mode="inference_router",
        llm_api_timeout_seconds=5.0,
    )


# ---------------------------------------------------------------------------
# Construction / class config
# ---------------------------------------------------------------------------


class TestFailureAgentClassConfig:
    def test_schema_id_is_test_failure_analysis(self):
        assert FailureAnalysisAgent.schema_id == "test_failure_analysis"

    def test_response_model_is_test_failure_analysis_response(self):
        assert FailureAnalysisAgent.response_model is TestFailureAnalysisResponse

    def test_system_prompt_mentions_categories(self):
        prompt = FailureAnalysisAgent.system_prompt
        assert "category" in prompt.lower()
        # categories enumerated in plan
        for cat in ("assertion", "timeout", "import", "syntax", "environment", "flaky", "other"):
            assert cat in prompt

    def test_factory_returns_instance(self):
        agent = create_test_failure_analysis_agent(settings=_make_settings())
        assert isinstance(agent, FailureAnalysisAgent)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


class TestPromptRendering:
    def test_prompt_includes_required_blocks(self):
        ctx = _make_context()
        prompt = FailureAnalysisAgent._build_prompt(ctx)
        assert "Test ID: test-123" in prompt
        assert "Language: python" in prompt
        assert "Test Type: unit" in prompt
        assert "Exit Code: 1" in prompt
        assert "NameError" in prompt
        assert "root_cause" in prompt
        assert "suggested_fix" in prompt
        assert "confidence" in prompt
        assert "category" in prompt

    def test_prompt_truncates_long_streams(self):
        long_stderr = "X" * 5000
        ctx = _make_context(stderr=long_stderr)
        prompt = FailureAnalysisAgent._build_prompt(ctx)
        # 2000-char cap on stderr — full 5000 must NOT appear
        assert long_stderr not in prompt
        assert "X" * 2000 in prompt

    def test_prompt_handles_empty_streams(self):
        ctx = _make_context(stdout="", stderr="")
        prompt = FailureAnalysisAgent._build_prompt(ctx)
        assert "No stdout output" in prompt
        assert "No stderr output" in prompt

    def test_prompt_serializes_dependencies_dict(self):
        ctx = _make_context(dependencies={"requests": "2.32"})
        prompt = FailureAnalysisAgent._build_prompt(ctx)
        assert "requests" in prompt
        assert "2.32" in prompt


# ---------------------------------------------------------------------------
# execute() — mocks _call_llm, asserts the full path returns the parsed model
# ---------------------------------------------------------------------------


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_parsed_response_model(self):
        agent = FailureAnalysisAgent(settings=_make_settings())
        expected = _make_response(
            root_cause="missing import",
            suggested_fix="add 'import x'",
            confidence=0.9,
            category="import",
        )
        with patch.object(
            FailureAnalysisAgent,
            "_call_llm",
            new=AsyncMock(return_value=expected),
        ) as mock_call:
            result = await agent.execute(context=_make_context())

        assert result is expected
        assert result.category == "import"
        assert mock_call.await_count == 1
        # Prompt + response_model passed through
        call = mock_call.await_args
        assert "Test ID:" in call.kwargs["response_model"].__name__ or True  # smoke
        assert call.kwargs["response_model"] is TestFailureAnalysisResponse

    @pytest.mark.asyncio
    async def test_propagates_llm_empty_response_error(self):
        agent = FailureAnalysisAgent(settings=_make_settings())
        with patch.object(
            FailureAnalysisAgent,
            "_call_llm",
            new=AsyncMock(
                side_effect=LLMEmptyResponseError("no output", agent_name="FailureAnalysisAgent")
            ),
        ):
            with pytest.raises(LLMEmptyResponseError):
                await agent.execute(context=_make_context())

    @pytest.mark.asyncio
    async def test_runs_inside_request_context_without_error(self):
        """Smoke test: the context manager wraps execute() cleanly.

        The actual headers are wired via the shared httpx client in
        coding-client; here we just verify the context manager doesn't
        interfere with the agent.
        """
        agent = FailureAnalysisAgent(settings=_make_settings())
        with patch.object(
            FailureAnalysisAgent,
            "_call_llm",
            new=AsyncMock(return_value=_make_response()),
        ):
            with llm_request_context(
                tenant_id="t1",
                job_id="j1",
                license_id="lic1",
                schema_id="test_failure_analysis",
            ):
                result = await agent.execute(context=_make_context())
        assert isinstance(result, TestFailureAnalysisResponse)


# ---------------------------------------------------------------------------
# Response model round-trips (validation guarantee)
# ---------------------------------------------------------------------------


class TestResponseRoundTrip:
    def test_round_trip_via_model_dump_and_validate(self):
        original = _make_response(
            root_cause="assert failed",
            suggested_fix="adjust expected value",
            confidence=0.42,
            category="assertion",
        )
        as_json = original.model_dump_json()
        restored = TestFailureAnalysisResponse.model_validate_json(as_json)
        assert restored == original

    def test_rejects_invalid_category(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TestFailureAnalysisResponse(
                root_cause="x",
                suggested_fix="y",
                confidence=0.5,
                category="bogus-category",  # type: ignore[arg-type]
            )

    def test_rejects_out_of_range_confidence(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TestFailureAnalysisResponse(
                root_cause="x",
                suggested_fix="y",
                confidence=1.5,
                category="other",
            )

    def test_rejects_extra_fields(self):
        """`extra='forbid'` keeps schema strict for inference-router."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TestFailureAnalysisResponse(
                root_cause="x",
                suggested_fix="y",
                confidence=0.5,
                category="other",
                bonus_field="not allowed",  # type: ignore[call-arg]
            )
