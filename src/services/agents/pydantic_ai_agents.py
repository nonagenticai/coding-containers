"""
PydanticAI Agent Models for Intelligent Test Execution Analysis
Type-safe AI agents for test orchestration, scheduling, and optimization.
"""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

# Try to import pydantic_ai, but make it optional
# Use TYPE_CHECKING to provide type hints even when pydantic_ai is not installed
if TYPE_CHECKING:
    from pydantic_ai import Agent, RunContext

    HAS_PYDANTIC_AI: bool
else:
    try:
        from pydantic_ai import Agent, RunContext

        HAS_PYDANTIC_AI = True
    except ImportError:
        HAS_PYDANTIC_AI = False
        # pydantic_ai integration is optional - application will work without AI features
        Agent: Any = None
        RunContext: Any = None

from src.models.testing_models import TestRequest, TestResult

logger = logging.getLogger(__name__)


class ExecutionMetrics(BaseModel):
    """Comprehensive metrics for test execution analysis."""

    test_id: str
    language: str
    test_type: str
    execution_time_ms: float
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None
    exit_code: int
    success: bool
    error_count: int = 0
    warning_count: int = 0
    test_count: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    coverage_percentage: float | None = None
    container_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(validate_assignment=True)

    @field_serializer("timestamp")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()


class ExecutionPattern(BaseModel):
    """Test execution pattern analysis model."""

    pattern_id: str
    pattern_type: str  # "failure", "performance", "resource", "timing"
    frequency: int
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_languages: list[str]
    affected_test_types: list[str]
    first_occurrence: datetime
    last_occurrence: datetime
    recommendations: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v


class SchedulingContext(BaseModel):
    """Context for test scheduling decisions."""

    pending_tests: list[TestRequest]
    active_tests: list[str]  # test_ids
    available_resources: dict[str, Any]
    historical_performance: dict[str, ExecutionMetrics]
    current_load: float = Field(ge=0.0, le=1.0)
    priority_queue: list[str] = Field(default_factory=list)
    resource_constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("current_load")
    @classmethod
    def validate_load(cls, v):
        """Validate current load is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Current load must be between 0.0 and 1.0")
        return v


class PydanticAIRecommendation(BaseModel):
    """Test execution recommendation model."""

    __test__: ClassVar[bool] = False

    recommendation_id: str
    recommendation_type: str  # "optimization", "scheduling", "configuration", "resource"
    priority: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    impact_assessment: str
    implementation_steps: list[str]
    estimated_improvement: float | None = None
    confidence_level: float = Field(ge=0.0, le=1.0)
    applicable_languages: list[str] = Field(default_factory=list)
    applicable_test_types: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence level is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence level must be between 0.0 and 1.0")
        return v


class FailureAnalysis(BaseModel):
    """Test failure analysis model."""

    analysis_id: str
    test_id: str
    failure_type: str  # "compilation", "runtime", "assertion", "timeout", "resource"
    error_message: str
    stack_trace: str | None = None
    root_cause: str | None = None
    suggested_fixes: list[str] = Field(default_factory=list)
    similar_failures: list[str] = Field(default_factory=list)
    frequency_score: float = Field(ge=0.0, le=1.0)
    complexity_score: float = Field(ge=0.0, le=1.0)
    fix_confidence: float = Field(ge=0.0, le=1.0)
    language: str
    test_type: str
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("frequency_score", "complexity_score", "fix_confidence")
    @classmethod
    def validate_scores(cls, v):
        """Validate scores are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Scores must be between 0.0 and 1.0")
        return v


class ExecutionPatternAnalyzer:
    """PydanticAI Agent for analyzing test execution patterns."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the pattern analyzer agent."""
        self.model = model
        self.system_prompt = """
            You are an expert test execution pattern analyzer. Analyze test execution
            metrics to identify patterns, trends, and anomalies. Focus on:

            1. Performance patterns (slow tests, resource usage)
            2. Failure patterns (common failure modes, error clustering)
            3. Resource utilization patterns
            4. Timing patterns (peak usage, scheduling conflicts)
            5. Language-specific patterns

            Provide actionable insights with high confidence scores for reliable patterns.
            Be conservative with confidence scores - only use high confidence (>0.8) for
            well-established patterns with sufficient data.
            """

    async def analyze_execution_patterns(
        self,
        metrics: list[ExecutionMetrics],
        context: dict[str, Any] | None = None,
    ) -> list[ExecutionPattern]:
        """Analyze test execution metrics to identify patterns."""
        try:
            # Prepare analysis context
            analysis_context = {
                "total_executions": len(metrics),
                "time_range": self._calculate_time_range(metrics),
                "languages": list(set(m.language for m in metrics)),
                "test_types": list(set(m.test_type for m in metrics)),
                "success_rate": self._calculate_success_rate(metrics),
                "average_execution_time": self._calculate_average_execution_time(metrics),
                "context": context or {},
            }

            # For now, use rule-based pattern detection
            # In a real implementation, this would use the PydanticAI agent
            patterns = self._detect_patterns_rule_based(metrics, analysis_context)

            logger.info(f"Analyzed {len(metrics)} test executions, found {len(patterns)} patterns")
            return patterns

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Pattern analysis failed: {e}")
            return []

    def _calculate_time_range(self, metrics: list[ExecutionMetrics]) -> str:
        """Calculate time range of metrics."""
        if not metrics:
            return "No data"

        timestamps = [m.timestamp for m in metrics]
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = end_time - start_time

        return f"{start_time.isoformat()} to {end_time.isoformat()} ({duration})"

    def _calculate_success_rate(self, metrics: list[ExecutionMetrics]) -> float:
        """Calculate overall success rate."""
        if not metrics:
            return 0.0

        successful = sum(1 for m in metrics if m.success)
        return successful / len(metrics)

    def _calculate_average_execution_time(self, metrics: list[ExecutionMetrics]) -> float:
        """Calculate average execution time."""
        if not metrics:
            return 0.0

        return sum(m.execution_time_ms for m in metrics) / len(metrics)

    def _detect_patterns_rule_based(
        self, metrics: list[ExecutionMetrics], analysis_context: dict[str, Any]
    ) -> list[ExecutionPattern]:
        """Detect patterns using rule-based analysis."""
        patterns: list[ExecutionPattern] = []

        if not metrics:
            return patterns

        # Pattern 1: Performance issues
        slow_tests = [m for m in metrics if m.execution_time_ms > 5000]  # >5 seconds
        if len(slow_tests) > len(metrics) * 0.2:  # More than 20% are slow
            patterns.append(
                ExecutionPattern(
                    pattern_id=f"slow_tests_{datetime.now(UTC).timestamp()}",
                    pattern_type="performance",
                    frequency=len(slow_tests),
                    severity=("high" if len(slow_tests) > len(metrics) * 0.5 else "medium"),
                    description=f"Detected {len(slow_tests)} slow test executions (>5s)",
                    affected_languages=list(set(m.language for m in slow_tests)),
                    affected_test_types=list(set(m.test_type for m in slow_tests)),
                    first_occurrence=min(m.timestamp for m in slow_tests),
                    last_occurrence=max(m.timestamp for m in slow_tests),
                    recommendations=[
                        "Optimize slow tests by reducing setup time",
                        "Consider parallelizing test execution",
                        "Review test data size and complexity",
                    ],
                    confidence_score=0.8,
                )
            )

        # Pattern 2: High failure rate
        failed_tests = [m for m in metrics if not m.success]
        if len(failed_tests) > len(metrics) * 0.1:  # More than 10% failures
            patterns.append(
                ExecutionPattern(
                    pattern_id=f"high_failure_rate_{datetime.now(UTC).timestamp()}",
                    pattern_type="failure",
                    frequency=len(failed_tests),
                    severity=("high" if len(failed_tests) > len(metrics) * 0.3 else "medium"),
                    description=f"High failure rate detected: {len(failed_tests)}/{len(metrics)} tests failed",
                    affected_languages=list(set(m.language for m in failed_tests)),
                    affected_test_types=list(set(m.test_type for m in failed_tests)),
                    first_occurrence=min(m.timestamp for m in failed_tests),
                    last_occurrence=max(m.timestamp for m in failed_tests),
                    recommendations=[
                        "Review failing tests for common issues",
                        "Check test environment stability",
                        "Consider adding test retry logic",
                    ],
                    confidence_score=0.9,
                )
            )

        # Pattern 3: Resource usage patterns
        high_memory_tests = [m for m in metrics if m.memory_usage_mb and m.memory_usage_mb > 512]
        if len(high_memory_tests) > len(metrics) * 0.15:  # More than 15% use high memory
            patterns.append(
                ExecutionPattern(
                    pattern_id=f"high_memory_usage_{datetime.now(UTC).timestamp()}",
                    pattern_type="resource",
                    frequency=len(high_memory_tests),
                    severity="medium",
                    description=f"High memory usage detected in {len(high_memory_tests)} tests (>512MB)",
                    affected_languages=list(set(m.language for m in high_memory_tests)),
                    affected_test_types=list(set(m.test_type for m in high_memory_tests)),
                    first_occurrence=min(m.timestamp for m in high_memory_tests),
                    last_occurrence=max(m.timestamp for m in high_memory_tests),
                    recommendations=[
                        "Monitor memory usage in tests",
                        "Consider increasing container memory limits",
                        "Review test data and object lifecycle",
                    ],
                    confidence_score=0.7,
                )
            )

        return patterns


class SchedulingOptimizer:
    """PydanticAI Agent for optimizing test execution scheduling."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the scheduling optimizer agent."""
        self.model = model
        self.system_prompt = """
            You are an expert test scheduling optimizer. Optimize test execution order
            and resource allocation based on:

            1. Test execution time predictions
            2. Resource requirements and constraints
            3. Test dependencies and priorities
            4. Historical performance data
            5. Current system load

            Prioritize:
            - Fast tests first for quick feedback
            - Resource-intensive tests during low load
            - Critical tests before optional ones
            - Tests with high failure rates for early detection

            Provide an optimized execution order that maximizes throughput and minimizes
            total execution time while respecting resource constraints.
            """

    async def optimize_test_schedule(self, context: SchedulingContext) -> list[str]:
        """Optimize test execution schedule based on context."""
        try:
            # Use rule-based scheduling optimization
            optimized_order = self._optimize_schedule_rule_based(context)

            logger.info(f"Optimized schedule for {len(context.pending_tests)} pending tests")
            return optimized_order

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Schedule optimization failed: {e}")
            # Fallback to simple FIFO scheduling
            return [str(req.test_id) for req in context.pending_tests if hasattr(req, "test_id")]

    def _optimize_schedule_rule_based(self, context: SchedulingContext) -> list[str]:
        """Optimize schedule using rule-based approach."""
        # Get test requests with their estimated execution times
        test_info: list[dict[str, Any]] = []
        for req in context.pending_tests:
            test_id = getattr(req, "test_id", None)
            if not test_id:
                continue

            # Get historical performance if available
            historical_metrics = context.historical_performance.get(test_id)
            estimated_time = 60000.0  # Default 1 minute
            if historical_metrics:
                estimated_time = historical_metrics.execution_time_ms

            test_info.append(
                {
                    "test_id": test_id,
                    "estimated_time": estimated_time,
                    "language": getattr(req, "language", "unknown"),
                    "request": req,
                }
            )

        # Sort by estimated execution time (fastest first)
        test_info.sort(key=lambda x: x["estimated_time"])

        # Return ordered test IDs
        return [info["test_id"] for info in test_info]


class RecommendationEngine:
    """PydanticAI Agent for generating test execution recommendations."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the recommendation engine agent."""
        self.model = model
        self.system_prompt = """
            You are an expert test execution advisor. Generate actionable recommendations
            for improving test execution based on:

            1. Performance bottlenecks and optimization opportunities
            2. Resource utilization improvements
            3. Test reliability and stability issues
            4. Configuration optimizations
            5. Infrastructure improvements

            Focus on:
            - High-impact, low-effort improvements
            - Measurable performance gains
            - Reliability improvements
            - Resource optimization
            - Developer experience enhancements

            Provide specific, actionable recommendations with clear implementation steps
            and expected benefits.
            """

    async def generate_recommendations(
        self,
        patterns: list[ExecutionPattern],
        metrics: list[ExecutionMetrics],
        context: dict[str, Any] | None = None,
    ) -> list[PydanticAIRecommendation]:
        """Generate test execution recommendations based on patterns and metrics."""
        try:
            # Use rule-based recommendation generation
            recommendations = self._generate_recommendations_rule_based(patterns, metrics, context)

            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []

    def _generate_recommendations_rule_based(
        self,
        patterns: list[ExecutionPattern],
        metrics: list[ExecutionMetrics],
        context: dict[str, Any] | None = None,
    ) -> list[PydanticAIRecommendation]:
        """Generate recommendations using rule-based approach."""
        recommendations: list[PydanticAIRecommendation] = []

        # Analyze patterns to generate recommendations
        for pattern in patterns:
            if pattern.pattern_type == "performance" and pattern.severity in [
                "high",
                "critical",
            ]:
                recommendations.append(
                    PydanticAIRecommendation(
                        recommendation_id=f"perf_opt_{datetime.now(UTC).timestamp()}",
                        recommendation_type="optimization",
                        priority="high",
                        title="Address Performance Issues",
                        description=f"Performance pattern detected: {pattern.description}",
                        impact_assessment="High impact on test execution time",
                        implementation_steps=[
                            "Identify bottlenecks in slow tests",
                            "Optimize test setup and teardown",
                            "Consider parallel test execution",
                            "Review resource allocation",
                        ],
                        estimated_improvement=0.3,
                        confidence_level=0.8,
                        applicable_languages=pattern.affected_languages,
                        applicable_test_types=pattern.affected_test_types,
                    )
                )

            elif pattern.pattern_type == "failure" and pattern.severity in [
                "high",
                "critical",
            ]:
                recommendations.append(
                    PydanticAIRecommendation(
                        recommendation_id=f"reliability_{datetime.now(UTC).timestamp()}",
                        recommendation_type="reliability",
                        priority="high",
                        title="Improve Test Reliability",
                        description=f"High failure rate pattern: {pattern.description}",
                        impact_assessment="High impact on test reliability",
                        implementation_steps=[
                            "Analyze common failure patterns",
                            "Add retry logic for flaky tests",
                            "Improve test environment stability",
                            "Review test data dependencies",
                        ],
                        estimated_improvement=0.25,
                        confidence_level=0.85,
                        applicable_languages=pattern.affected_languages,
                        applicable_test_types=pattern.affected_test_types,
                    )
                )

            elif pattern.pattern_type == "resource":
                recommendations.append(
                    PydanticAIRecommendation(
                        recommendation_id=f"resource_{datetime.now(UTC).timestamp()}",
                        recommendation_type="resource",
                        priority="medium",
                        title="Optimize Resource Usage",
                        description=f"Resource usage pattern: {pattern.description}",
                        impact_assessment="Medium impact on resource efficiency",
                        implementation_steps=[
                            "Monitor resource usage patterns",
                            "Adjust container resource limits",
                            "Optimize test data size",
                            "Review memory leaks in tests",
                        ],
                        estimated_improvement=0.15,
                        confidence_level=0.7,
                        applicable_languages=pattern.affected_languages,
                        applicable_test_types=pattern.affected_test_types,
                    )
                )

        # General recommendations based on metrics
        if metrics:
            total_executions = len(metrics)
            failed_executions = sum(1 for m in metrics if not m.success)

            if failed_executions > 0:
                failure_rate = failed_executions / total_executions
                if failure_rate > 0.1:  # More than 10% failure rate
                    recommendations.append(
                        PydanticAIRecommendation(
                            recommendation_id=f"general_reliability_{datetime.now(UTC).timestamp()}",
                            recommendation_type="configuration",
                            priority="medium",
                            title="Improve Overall Test Reliability",
                            description=f"Overall failure rate is {failure_rate:.1%}",
                            impact_assessment="Medium impact on development velocity",
                            implementation_steps=[
                                "Implement test health monitoring",
                                "Add automated test failure reporting",
                                "Review test environment setup",
                                "Consider test flakiness analysis",
                            ],
                            estimated_improvement=0.2,
                            confidence_level=0.75,
                        )
                    )

        return recommendations


class ServiceTestFailureAnalyzer:
    """Service-level PydanticAI Agent for analyzing test failures and suggesting fixes."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the failure analyzer agent."""
        self.model = model
        self.system_prompt = """
            You are an expert test failure analyzer. Analyze test failures to:

            1. Identify root causes of failures
            2. Classify failure types (compilation, runtime, assertion, timeout, etc.)
            3. Suggest specific fixes and improvements
            4. Identify similar failures and patterns
            5. Assess fix confidence and complexity

            Focus on:
            - Accurate failure classification
            - Actionable fix suggestions
            - Root cause analysis
            - Pattern recognition across similar failures
            - Confidence assessment for suggested fixes

            Provide detailed analysis with specific, implementable solutions.
            """

    async def analyze_test_failure(
        self,
        test_result: TestResult,
        historical_failures: list[FailureAnalysis] | None = None,
        context: dict[str, Any] | None = None,
    ) -> FailureAnalysis:
        """Analyze a test failure and provide detailed analysis."""
        try:
            # Use rule-based failure analysis
            analysis = self._analyze_failure_rule_based(test_result, historical_failures, context)

            logger.info(f"Analyzed test failure for {test_result.test_id}")
            return analysis

        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Failure analysis failed: {e}")
            # Return basic failure analysis
            return FailureAnalysis(
                analysis_id=f"fallback_{test_result.test_id}_{datetime.now(UTC).timestamp()}",
                test_id=test_result.test_id,
                failure_type="unknown",
                error_message=getattr(test_result, "error_message", "Unknown error"),
                root_cause="Analysis failed due to system error",
                suggested_fixes=["Manual investigation required"],
                frequency_score=0.0,
                complexity_score=0.5,
                fix_confidence=0.0,
                language=test_result.language,
                test_type=test_result.test_type or "unknown",
            )

    def _analyze_failure_rule_based(
        self,
        test_result: TestResult,
        historical_failures: list[FailureAnalysis] | None = None,
        context: dict[str, Any] | None = None,
    ) -> FailureAnalysis:
        """Analyze failure using rule-based approach."""
        # Extract error information from test result
        error_message = getattr(
            test_result,
            "error_message",
            getattr(test_result, "output", "Unknown error"),
        )
        stack_trace = getattr(test_result, "stack_trace", "")

        # Classify failure type based on error message
        failure_type = self._classify_failure_type(error_message, stack_trace)

        # Generate suggested fixes based on failure type
        suggested_fixes = self._generate_fixes_for_failure_type(failure_type, error_message)

        # Determine root cause
        root_cause = self._determine_root_cause(failure_type, error_message, stack_trace)

        # Calculate scores
        frequency_score = self._calculate_frequency_score(test_result, historical_failures)
        complexity_score = self._calculate_complexity_score(failure_type, error_message)
        fix_confidence = self._calculate_fix_confidence(failure_type, error_message)

        # Find similar failures
        similar_failures = self._find_similar_failures(test_result, historical_failures)

        return FailureAnalysis(
            analysis_id=f"analysis_{test_result.test_id}_{datetime.now(UTC).timestamp()}",
            test_id=test_result.test_id,
            failure_type=failure_type,
            error_message=error_message,
            stack_trace=stack_trace,
            root_cause=root_cause,
            suggested_fixes=suggested_fixes,
            similar_failures=similar_failures,
            frequency_score=frequency_score,
            complexity_score=complexity_score,
            fix_confidence=fix_confidence,
            language=test_result.language,
            test_type=test_result.test_type or "unknown",
        )

    def _classify_failure_type(self, error_message: str, stack_trace: str) -> str:
        """Classify failure type based on error message."""
        error_lower = error_message.lower()

        if "assertion" in error_lower or "assert" in error_lower:
            return "assertion"
        elif "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "compilation" in error_lower or "syntax" in error_lower:
            return "compilation"
        elif "import" in error_lower or "module" in error_lower:
            return "import"
        elif "null" in error_lower or "none" in error_lower:
            return "null_pointer"
        elif "memory" in error_lower or "out of memory" in error_lower:
            return "memory"
        elif "network" in error_lower or "connection" in error_lower:
            return "network"
        else:
            return "runtime"

    def _generate_fixes_for_failure_type(self, failure_type: str, error_message: str) -> list[str]:
        """Generate suggested fixes based on failure type."""
        fixes = {
            "assertion": [
                "Review test assertions for correctness",
                "Check expected vs actual values",
                "Verify test data setup",
            ],
            "timeout": [
                "Increase timeout limit",
                "Optimize test performance",
                "Check for blocking operations",
            ],
            "compilation": [
                "Fix syntax errors",
                "Check import statements",
                "Verify language version compatibility",
            ],
            "import": [
                "Check module installation",
                "Verify import paths",
                "Check virtual environment setup",
            ],
            "null_pointer": [
                "Add null checks",
                "Initialize variables properly",
                "Check object lifecycle",
            ],
            "memory": [
                "Increase memory limits",
                "Optimize memory usage",
                "Check for memory leaks",
            ],
            "network": [
                "Check network connectivity",
                "Verify service availability",
                "Add retry logic",
            ],
            "runtime": [
                "Review stack trace for clues",
                "Check runtime dependencies",
                "Verify environment setup",
            ],
        }

        return fixes.get(failure_type, ["Manual investigation required"])

    def _determine_root_cause(self, failure_type: str, error_message: str, stack_trace: str) -> str:
        """Determine root cause of failure."""
        causes = {
            "assertion": "Test expectation mismatch",
            "timeout": "Test execution exceeds time limit",
            "compilation": "Code compilation error",
            "import": "Module import failure",
            "null_pointer": "Null reference error",
            "memory": "Memory resource exhaustion",
            "network": "Network connectivity issue",
            "runtime": "Runtime execution error",
        }

        return causes.get(failure_type, "Unknown error condition")

    def _calculate_frequency_score(
        self,
        test_result: TestResult,
        historical_failures: list[FailureAnalysis] | None,
    ) -> float:
        """Calculate frequency score based on historical failures."""
        if not historical_failures:
            return 0.0

        # Count similar failures
        similar_count = sum(
            1
            for failure in historical_failures
            if failure.language == test_result.language
            and failure.test_type == (test_result.test_type or "unknown")
        )

        return min(similar_count / 10.0, 1.0)  # Normalize to 0-1

    def _calculate_complexity_score(self, failure_type: str, error_message: str) -> float:
        """Calculate complexity score for fixing the failure."""
        complexity_scores = {
            "assertion": 0.2,
            "timeout": 0.4,
            "compilation": 0.3,
            "import": 0.5,
            "null_pointer": 0.3,
            "memory": 0.7,
            "network": 0.6,
            "runtime": 0.5,
        }

        return complexity_scores.get(failure_type, 0.5)

    def _calculate_fix_confidence(self, failure_type: str, error_message: str) -> float:
        """Calculate confidence in suggested fixes."""
        confidence_scores = {
            "assertion": 0.9,
            "timeout": 0.8,
            "compilation": 0.95,
            "import": 0.85,
            "null_pointer": 0.8,
            "memory": 0.6,
            "network": 0.7,
            "runtime": 0.5,
        }

        return confidence_scores.get(failure_type, 0.5)

    def _find_similar_failures(
        self,
        test_result: TestResult,
        historical_failures: list[FailureAnalysis] | None,
    ) -> list[str]:
        """Find similar failures from historical data."""
        if not historical_failures:
            return []

        similar: list[str] = []
        for failure in historical_failures:
            if (
                failure.language == test_result.language
                and failure.test_type == (test_result.test_type or "unknown")
                and len(similar) < 5
            ):  # Limit to 5 similar failures
                similar.append(failure.test_id)

        return similar


class PydanticAITestCoordinator:
    """Coordinated system of PydanticAI agents for intelligent test execution."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the test coordinator with all agents."""
        self.pattern_analyzer = ExecutionPatternAnalyzer(model)
        self.scheduling_optimizer = SchedulingOptimizer(model)
        self.recommendation_engine = RecommendationEngine(model)
        self.failure_analyzer = ServiceTestFailureAnalyzer(model)

        # Internal state
        self.execution_history: list[ExecutionMetrics] = []
        self.failure_history: list[FailureAnalysis] = []
        self.patterns_cache: list[ExecutionPattern] = []
        self.recommendations_cache: list[PydanticAIRecommendation] = []

        logger.info("PydanticAI Test Coordinator initialized")

    async def record_execution(self, metrics: ExecutionMetrics) -> None:
        """Record test execution metrics for analysis."""
        self.execution_history.append(metrics)

        # Maintain rolling window of recent executions
        max_history = 1000
        if len(self.execution_history) > max_history:
            self.execution_history = self.execution_history[-max_history:]

        logger.debug(f"Recorded execution metrics for {metrics.test_id}")

    async def analyze_patterns(self, force_refresh: bool = False) -> list[ExecutionPattern]:
        """Analyze execution patterns using the pattern analyzer agent."""
        if not force_refresh and self.patterns_cache:
            return self.patterns_cache

        if len(self.execution_history) < 10:
            logger.info("Insufficient execution history for pattern analysis")
            return []

        try:
            patterns = await self.pattern_analyzer.analyze_execution_patterns(
                self.execution_history
            )
            self.patterns_cache = patterns
            return patterns

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Pattern analysis failed: {e}")
            return []

    async def optimize_schedule(
        self,
        pending_tests: list[TestRequest],
        active_tests: list[str],
        available_resources: dict[str, Any],
        current_load: float = 0.0,
    ) -> list[str]:
        """Optimize test execution schedule using the scheduling optimizer agent."""
        try:
            # Build historical performance context
            historical_performance = {}
            for metrics in self.execution_history:
                historical_performance[metrics.test_id] = metrics

            # Create scheduling context
            context = SchedulingContext(
                pending_tests=pending_tests,
                active_tests=active_tests,
                available_resources=available_resources,
                historical_performance=historical_performance,
                current_load=current_load,
            )

            # Get optimized schedule
            optimized_order = await self.scheduling_optimizer.optimize_test_schedule(context)
            return optimized_order

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Schedule optimization failed: {e}")
            # Fallback to simple FIFO
            return [str(req.test_id) for req in pending_tests if hasattr(req, "test_id")]

    async def generate_recommendations(
        self, force_refresh: bool = False
    ) -> list[PydanticAIRecommendation]:
        """Generate test execution recommendations."""
        if not force_refresh and self.recommendations_cache:
            return self.recommendations_cache

        try:
            # Get current patterns
            patterns = await self.analyze_patterns()

            # Generate recommendations
            recommendations = await self.recommendation_engine.generate_recommendations(
                patterns, self.execution_history
            )

            self.recommendations_cache = recommendations
            return recommendations

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []

    async def analyze_failure(self, test_result: TestResult) -> FailureAnalysis:
        """Analyze test failure and generate detailed analysis."""
        try:
            # Get similar historical failures
            similar_failures = [
                f
                for f in self.failure_history
                if f.language == test_result.language and f.test_type == test_result.test_type
            ]

            # Analyze failure
            analysis = await self.failure_analyzer.analyze_test_failure(
                test_result, similar_failures
            )

            # Store in history
            self.failure_history.append(analysis)

            # Maintain rolling window
            max_failure_history = 500
            if len(self.failure_history) > max_failure_history:
                self.failure_history = self.failure_history[-max_failure_history:]

            return analysis

        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Failure analysis failed: {e}")
            # Return basic analysis
            return FailureAnalysis(
                analysis_id=f"error_{test_result.test_id}_{datetime.now(UTC).timestamp()}",
                test_id=test_result.test_id,
                failure_type="system_error",
                error_message="Analysis system error",
                root_cause="AI analysis failed",
                suggested_fixes=["Check system logs", "Retry analysis"],
                frequency_score=0.0,
                complexity_score=0.5,
                fix_confidence=0.0,
                language=test_result.language,
                test_type=test_result.test_type or "unknown",
            )

    async def get_execution_insights(self) -> dict[str, Any]:
        """Get comprehensive execution insights."""
        try:
            patterns = await self.analyze_patterns()
            recommendations = await self.generate_recommendations()

            # Calculate summary statistics
            total_executions = len(self.execution_history)
            success_rate = (
                sum(1 for m in self.execution_history if m.success) / total_executions
                if total_executions > 0
                else 0.0
            )

            avg_execution_time = (
                sum(m.execution_time_ms for m in self.execution_history) / total_executions
                if total_executions > 0
                else 0.0
            )

            return {
                "summary": {
                    "total_executions": total_executions,
                    "success_rate": success_rate,
                    "average_execution_time_ms": avg_execution_time,
                    "patterns_identified": len(patterns),
                    "recommendations_available": len(recommendations),
                    "failure_analyses": len(self.failure_history),
                },
                "patterns": [p.model_dump() for p in patterns],
                "recommendations": [r.model_dump() for r in recommendations],
                "recent_failures": [f.model_dump() for f in self.failure_history[-5:]],
                "generated_at": datetime.now(UTC).isoformat(),
            }

        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Failed to generate execution insights: {e}")
            return {
                "summary": {"error": "Failed to generate insights"},
                "patterns": [],
                "recommendations": [],
                "recent_failures": [],
                "generated_at": datetime.now(UTC).isoformat(),
            }


# Aliases for backward compatibility and shorter imports
TestRecommendation = PydanticAIRecommendation
