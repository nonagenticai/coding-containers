"""Test-failure analysis agent backed by inference-router.

Replaces the legacy ``FailureAnalysisAgent`` (deleted) which built provider
strings and called OpenAI/Anthropic/Groq directly. All LLM traffic now flows
through inference-router via ``coding_client.llm.BaseLLMAgent``; provider
selection + fallback are server-side.

The agent's structured output is :class:`TestFailureAnalysisResponse` from
``coding_client.llm`` — fields: ``root_cause``, ``suggested_fix``,
``confidence``, ``category``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from coding_client.llm import (
    BaseLLMAgent,
    LLMSettings,
    TestFailureAnalysisResponse,
    last_failover_chain_id,
    last_inference_id,
)

if TYPE_CHECKING:
    from src.models.pydantic_ai_models import PydanticAITestFailureContext

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a senior test engineer. Diagnose the failed test and return a "
    "structured response with: root cause, suggested fix, confidence in [0, 1], "
    "and a single category from "
    "{assertion, timeout, import, syntax, environment, flaky, other}. "
    "Be concrete and actionable."
)


class FailureAnalysisAgent(BaseLLMAgent):
    """Inference-router-backed test-failure analysis agent."""

    schema_id: ClassVar[str] = "test_failure_analysis"
    response_model: ClassVar[type[BaseModel] | None] = TestFailureAnalysisResponse
    system_prompt: ClassVar[str] = _SYSTEM_PROMPT

    def __init__(self, settings: LLMSettings | None = None) -> None:
        super().__init__(settings=settings)

    async def execute(self, **kwargs: Any) -> TestFailureAnalysisResponse:
        """Analyse a single failing test context.

        Args:
            context (PydanticAITestFailureContext): required keyword.

        Caller is expected to wrap this in
        ``coding_client.llm.llm_request_context(...)`` so that
        X-Tenant-Id / X-Job-Id / X-License-Id / X-Schema-Id headers are
        attached for inference-router routing + provenance.

        Returns the parsed pydantic response model. After the call, the
        caller can read ``last_inference_id()`` / ``last_failover_chain_id()``
        to record provenance.
        """
        try:
            context: PydanticAITestFailureContext = kwargs["context"]
        except KeyError as exc:
            raise TypeError(
                "FailureAnalysisAgent.execute() requires keyword argument 'context'"
            ) from exc

        prompt = self._build_prompt(context)
        response = await self._call_llm(
            prompt,
            response_model=TestFailureAnalysisResponse,
        )

        inference_id = last_inference_id()
        failover_chain_id = last_failover_chain_id()
        logger.info(
            "test_failure_analysis_completed",
            extra={
                "test_id": context.test_id,
                "category": response.category,
                "confidence": response.confidence,
                "inference_id": inference_id,
                "failover_chain_id": failover_chain_id,
            },
        )

        return response

    @staticmethod
    def _build_prompt(context: PydanticAITestFailureContext) -> str:
        """Render the failure context into a prompt for the LLM plane."""
        dependencies = (
            json.dumps(context.dependencies, indent=2)
            if context.dependencies
            else "No dependencies listed"
        )
        resource_usage = (
            json.dumps(context.resource_usage, indent=2)
            if context.resource_usage
            else "No resource-usage data"
        )
        stdout = (context.stdout or "")[:2000] or "No stdout output"
        stderr = (context.stderr or "")[:2000] or "No stderr output"
        source = (context.source_code or "")[:1000] or "No source code provided"
        test_code = (context.test_code or "")[:1000] or "No test code provided"

        return f"""Analyze the following failing test.

**Test Information:**
- Test ID: {context.test_id}
- Language: {context.language}
- Test Type: {context.test_type}
- Exit Code: {context.exit_code}
- Execution Time: {context.execution_time_ms}ms
- Container: {context.container_name or "N/A"}

**STDOUT:**
{stdout}

**STDERR:**
{stderr}

**Source Code:**
{source}

**Test Code:**
{test_code}

**Dependencies:**
{dependencies}

**Resource Usage:**
{resource_usage}

Return JSON conforming to the configured response schema with:
1. root_cause - concise root-cause explanation
2. suggested_fix - concrete, actionable fix
3. confidence - float in [0.0, 1.0]
4. category - one of: assertion, timeout, import, syntax, environment, flaky, other
"""


def create_test_failure_analysis_agent(
    settings: LLMSettings | None = None,
) -> FailureAnalysisAgent:
    """Factory for :class:`FailureAnalysisAgent`."""
    return FailureAnalysisAgent(settings=settings)
