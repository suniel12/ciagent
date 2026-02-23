"""
Agent CI Core Data Models

These models define the universal trace format that all features
(diffing, cost tracking, assertions, reporting) consume.
Designed for Phase 1 (single-agent) but structured to support
Phase 2 (multi-agent) without schema changes.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid


def _new_id() -> str:
    u = uuid.uuid4()
    return u.hex[:16]


# ── Enums ──────────────────────────────────────────────

class SpanKind(str, Enum):
    """Type of span in the execution trace."""
    AGENT = "agent"          # An agent's full execution
    LLM_CALL = "llm_call"   # A single LLM API call
    TOOL_CALL = "tool_call"  # A single tool invocation
    HANDOFF = "handoff"      # Phase 2: agent-to-agent transfer


class DiffType(str, Enum):
    """Categories of detected changes between runs."""
    TOOLS_CHANGED = "tools_changed"       # Different tools called
    ARGS_CHANGED = "args_changed"         # Same tools, different arguments
    SEQUENCE_CHANGED = "sequence_changed" # Tools called in different order
    OUTPUT_CHANGED = "output_changed"     # Final output differs
    COST_SPIKE = "cost_spike"             # Cost exceeds threshold
    LATENCY_SPIKE = "latency_spike"       # Duration exceeds threshold
    STEPS_CHANGED = "steps_changed"       # Different number of LLM calls
    STOP_REASON_CHANGED = "stop_reason_changed" # LLM or Span exited for a different reason than golden baseline


class TestResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"    # Exception during execution, not a test failure


# ── Trace Components ───────────────────────────────────

class ToolCall(BaseModel):
    """A single tool/function call made by an agent."""
    tool_name: str
    arguments: dict[str, Any] = {}
    result: Any | None = None
    error: str | None = None          # If the tool call failed
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class LLMCall(BaseModel):
    """A single LLM API call within a span."""
    model: str = ""                    # e.g., "gpt-4o", "claude-sonnet-4-20250514"
    provider: str = ""                 # e.g., "openai", "anthropic"
    input_messages: list[dict[str, Any]] = []   # Stored for debugging, not diffing
    output_text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    stop_reason: str | None = None    # e.g., "end_turn", "max_tokens", "tool_use", "stop_sequence"
    cost_usd: float = 0.0             # Computed from token counts + pricing
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Span(BaseModel):
    """
    A unit of work in the execution trace.
    
    In Phase 1: One span per agent invocation.
    In Phase 2: Multiple spans form a DAG (agent A → handoff → agent B).
    """
    span_id: str = Field(default_factory=_new_id)
    parent_span_id: str | None = None  # Phase 2: enables tree structure
    kind: SpanKind = SpanKind.AGENT
    name: str = ""                     # Human-readable: "booking_agent", "search_tool"
    
    # Execution data
    input_data: Any = None             # What the agent/tool received
    output_data: Any = None            # What it returned
    stop_reason: str | None = None     # Why execution stopped ("max_tools", "complete", "error")
    graph_state: dict[str, Any] = {}   # Raw state snapshot for framework-specific parsing
    
    # Collected events
    tool_calls: list[ToolCall] = []
    llm_calls: list[LLMCall] = []
    
    # Aggregated metrics (computed after execution)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
    
    # Extensible metadata
    metadata: dict[str, Any] = {}

    def compute_metrics(self) -> None:
        """Roll up metrics from child LLM calls."""
        self.total_tokens_in = sum(c.tokens_in for c in self.llm_calls)
        self.total_tokens_out = sum(c.tokens_out for c in self.llm_calls)
        self.total_cost_usd = sum(c.cost_usd for c in self.llm_calls)


class Trace(BaseModel):
    """
    The complete execution record of a single test run.
    
    This is the universal data structure that every Agent CI feature
    consumes: diffing reads it, assertions query it, reports render it,
    and golden traces are serialized instances of it.
    """
    trace_id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # The execution data
    spans: list[Span] = []
    
    # Aggregated metrics
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    
    # Test metadata
    test_name: str = ""
    agent_name: str = ""
    framework: str = ""    # "langgraph", "crewai", "generic"
    graph_state: dict[str, Any] = {} # Final state of the graph
    metadata: dict[str, Any] = {}

    def compute_metrics(self) -> None:
        """Roll up metrics from all spans."""
        for span in self.spans:
            span.compute_metrics()
        self.total_cost_usd = sum(s.total_cost_usd for s in self.spans)
        self.total_tokens = sum(s.total_tokens_in + s.total_tokens_out for s in self.spans)
        self.total_llm_calls = sum(len(s.llm_calls) for s in self.spans)
        self.total_tool_calls = sum(len(s.tool_calls) for s in self.spans)

    @property
    def tool_call_sequence(self) -> list[str]:
        """Ordered list of tool names called across all spans."""
        calls = []
        for span in self.spans:
            calls.extend(tc.tool_name for tc in span.tool_calls)
        return calls
    
    @property
    def tool_call_details(self) -> list[ToolCall]:
        """All tool calls across all spans, in order."""
        calls = []
        for span in self.spans:
            calls.extend(span.tool_calls)
        return calls


# ── Test Definition Models ─────────────────────────────

class Assertion(BaseModel):
    """A single assertion to check against a trace."""
    type: str                          # "tool_called", "tool_not_called", 
                                       # "arg_equals", "arg_contains",
                                       # "cost_under", "steps_under",
                                       # "output_contains", "output_not_contains"
    tool: str | None = None            # Which tool (for tool-related assertions)
    field: str | None = None           # Which argument/field
    value: Any = None                  # Expected value
    threshold: float | None = None     # For numeric comparisons


class TestCase(BaseModel):
    """
    A single test scenario defined by the developer.
    
    Can be defined in Python (via decorators) or YAML (for non-code config).
    """
    name: str
    description: str = ""
    
    # Input to the agent
    input_data: Any = Field(default=None, alias="input")             # String prompt, dict, or structured input
    
    # Expected behavior
    assertions: list[Assertion] = []
    
    # Cost/performance budgets
    max_cost_usd: float | None = None
    max_duration_ms: float | None = None
    max_steps: int | None = None       # Max LLM calls
    max_tool_calls: int | None = None
    
    # Golden Trace reference
    golden_trace: str | None = None  # Path to saved golden trace JSON
    
    # Tags for filtering
    tags: list[str] = []


class TestSuite(BaseModel):
    """A collection of test cases, typically loaded from a YAML file."""
    name: str = "default"
    agent: str = ""                    # Import path: "myapp.agent:run_agent"
    framework: str = "generic"         # "langgraph", "crewai", "generic"
    mocks: str | None = None           # Path to mocks.yaml
    tests: list[TestCase] = []
    
    # Suite-level defaults
    default_max_cost_usd: float | None = None
    default_max_steps: int | None = None


# ── Diff Models ────────────────────────────────────────

class DiffResult(BaseModel):
    """A single detected difference between current and golden trace."""
    diff_type: DiffType
    severity: str = "warning"          # "error", "warning", "info"
    message: str = ""
    details: dict[str, Any] = {}       # e.g., {"expected": "NYC", "actual": "New York"}


class RunResult(BaseModel):
    """The complete result of running a single test case."""
    test_name: str
    result: TestResult
    trace: Trace
    diffs: list[DiffResult] = []
    assertion_results: list[dict[str, Any]] = []
    error_message: str | None = None
    duration_ms: float = 0.0


class SuiteResult(BaseModel):
    """The complete result of running an entire test suite."""
    suite_name: str
    results: list[RunResult] = []
    total_passed: int = 0
    total_failed: int = 0
    total_errors: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
