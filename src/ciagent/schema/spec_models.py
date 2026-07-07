# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
AgentCI v2/v3 Spec Models

Pydantic models for agentci_spec.yaml. Every field is optional except
`agent` and `queries` on the root spec, and `query` on each golden query.

Hierarchy:
    AgentCISpec
    └── GoldenQuery (1..N)
        ├── CorrectnessSpec  (Layer 1 — hard fail)
        │   └── span_assertions  (sub-layer of Correctness, hard fail)
        ├── PathSpec         (Layer 2 — soft warn)
        └── CostSpec         (Layer 3 — soft warn)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────────


class MatchMode(str, Enum):
    """How to compare the agent's tool sequence against the golden baseline."""
    STRICT = "strict"        # Exact match: same tools, same order
    UNORDERED = "unordered"  # Same tools, any order
    SUBSET = "subset"        # Reference tools must appear (extras OK)  [default]
    SUPERSET = "superset"    # All used tools must be in reference set


class SpanKindSelector(str, Enum):
    """Which type of span to target in span assertions."""
    TOOL = "TOOL"
    NODE = "NODE"
    HANDOFF = "HANDOFF"
    GUARDRAIL = "GUARDRAIL"
    LLM = "LLM"


class SpanAssertType(str, Enum):
    """The type of assertion to perform on a span field."""
    LLM_JUDGE = "llm_judge"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    EQUALS = "equals"


# ── Span Assertion Models ──────────────────────────────────────────────────────


class SpanSelector(BaseModel):
    """Selects a span from the trace by kind and name."""
    kind: SpanKindSelector = Field(
        ...,
        description="Type of span to target (TOOL, NODE, HANDOFF, GUARDRAIL, LLM)",
    )
    name: str = Field(
        ...,
        description="Name of the tool, node, or agent to match",
    )


class SpanAssert(BaseModel):
    """A single assertion on a field within a matched span."""
    type: SpanAssertType = Field(
        ...,
        description="Assertion type: contains, not_contains, regex, equals, or llm_judge",
    )
    field: str = Field(
        ...,
        description=(
            "Dotted path to the span field to extract, "
            "e.g. 'attributes.tool.args.query' or 'output_data'"
        ),
    )
    rule: Optional[str] = Field(
        None,
        description="Natural language rule for llm_judge assertions",
    )
    value: Optional[str] = Field(
        None,
        description="Expected value for contains, not_contains, regex, or equals assertions",
    )
    threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Minimum passing score for llm_judge assertions (normalised 0–1)",
    )


class SpanAssertionSpec(BaseModel):
    """Span-level assertion: select one or more spans and assert on their fields."""
    selector: SpanSelector = Field(
        ...,
        description="Selects which span(s) to assert against",
    )
    asserts: list[SpanAssert] = Field(
        ...,
        min_length=1,
        description="One or more assertions to run against each matched span",
    )


# ── Sub-models ─────────────────────────────────────────────────────────────────


class JudgeRubric(BaseModel):
    """Structured rubric for LLM-as-a-judge evaluation."""
    rule: str = Field(..., description="Natural language evaluation criterion")
    scale: Optional[list[str]] = Field(
        None,
        description=(
            "Score anchors, e.g. ['1: Off-topic', '2: Partially relevant', '3: Fully correct']"
        ),
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum passing score normalised to [0, 1]",
    )
    few_shot_examples: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Example input/output/score triples for calibration",
    )
    context_file: Optional[str] = Field(
        None,
        description=(
            "Path to a reference document used as ground truth for this rubric. "
            "The judge is instructed to evaluate the answer ONLY against this file's "
            "contents, ignoring prior training knowledge. "
            "Path is resolved relative to the spec file location."
        ),
    )


# ── Layer Specs ────────────────────────────────────────────────────────────────


class CorrectnessSpec(BaseModel):
    """Layer 1: Hard pass/fail. Any failure blocks the CI pipeline."""
    expected_in_answer: Optional[list[str]] = Field(
        None,
        description="Strings that must ALL appear in the answer (case-insensitive, AND logic)",
    )
    any_expected_in_answer: Optional[list[str]] = Field(
        None,
        description="At least ONE of these strings must appear in the answer (case-insensitive, OR logic)",
    )
    not_in_answer: Optional[list[str]] = Field(
        None,
        description="Strings that must NOT appear in the answer",
    )
    exact_match: Optional[str] = Field(
        None,
        description="Answer must equal this string exactly (after stripping whitespace)",
    )
    regex_match: Optional[str] = Field(
        None,
        description="Answer must match this regex pattern",
    )
    json_schema: Optional[dict[str, Any]] = Field(
        None,
        description="Answer must parse as JSON conforming to this schema",
    )
    llm_judge: Optional[list[JudgeRubric]] = Field(
        None,
        description="LLM-as-a-judge rubrics; evaluated after deterministic checks",
    )
    safety_check: Optional[JudgeRubric] = Field(
        None,
        description="Safety evaluation rubric (treated as correctness-tier hard fail)",
    )
    hallucination_check: Optional[JudgeRubric] = Field(
        None,
        description="Hallucination / grounding evaluation rubric",
    )
    refutes_premise: bool = Field(
        False,
        description=(
            "When True, the agent is expected to correct a false premise in the query. "
            "Skips expected_in_answer/not_in_answer checks (they don't apply to refutals) "
            "and injects a built-in 'false premise correction' rubric into llm_judge."
        ),
    )


class PathSpec(BaseModel):
    """Layer 2: Trajectory evaluation. Exceedances produce warnings, not failures."""
    max_tool_calls: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum number of tool calls allowed",
    )
    expected_tools: Optional[list[str]] = Field(
        None,
        description=(
            "Tools that should be called. A missing expected tool produces a WARN "
            "(recall gates at 1.0 unless min_tool_recall loosens it). An explicit "
            "empty list asserts that NO tools are called."
        ),
    )
    forbidden_tools: Optional[list[str]] = Field(
        None,
        description="Tools that must NOT be called (safety boundary — hard fail if violated)",
    )
    max_loops: int = Field(
        3,
        ge=1,
        description="Maximum consecutive repeated tool invocations (loop detection). Default 3.",
    )
    match_mode: MatchMode = Field(
        MatchMode.SUBSET,
        description="How to compare tool sequences against the golden baseline",
    )
    min_tool_recall: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="|expected ∩ used| / |expected|. Defaults to 1.0 when expected_tools is set.",
    )
    min_tool_precision: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="|expected ∩ used| / |used|",
    )
    min_sequence_similarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum normalised LCS similarity (0=disjoint, 1=identical order)",
    )
    expected_tool_sequence: Optional[list[str]] = Field(
        None,
        description=(
            "Expected tool call sequence (ordered list). "
            "Mismatch produces WARN, not hard fail. "
            "Example: ['retrieve_docs', 'rerank', 'generate_answer']"
        ),
    )
    expected_handoff: Optional[str] = Field(
        None,
        description="Expected handoff target agent name",
    )
    expected_handoffs_available: Optional[list[str]] = Field(
        None,
        description="All agents that should be reachable as handoff targets",
    )
    max_handoff_count: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum number of handoffs allowed",
    )


class CostSpec(BaseModel):
    """Layer 3: Efficiency budget. Exceedances produce warnings, not failures."""
    max_cost_multiplier: Optional[float] = Field(
        None,
        gt=0,
        description="Max allowed cost as a multiple of the golden baseline (e.g. 2.0 = 2×)",
    )
    max_total_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum total tokens (input + output) across all LLM calls",
    )
    max_llm_calls: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum number of LLM API calls",
    )
    max_latency_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum wall-clock latency in milliseconds",
    )
    max_cost_usd: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum absolute cost in USD",
    )


# ── Root Models ────────────────────────────────────────────────────────────────


class GoldenQuery(BaseModel):
    """A single test case in the AgentCI spec."""
    query: str = Field(..., description="The input stimulus sent to the agent")
    description: Optional[str] = Field(
        None,
        description="Human-readable description of what this test validates",
    )
    tags: Optional[list[str]] = Field(
        None,
        description="Tags for test filtering (e.g. ['smoke', 'edge-case'])",
    )
    correctness: Optional[CorrectnessSpec] = None
    path: Optional[PathSpec] = None
    cost: Optional[CostSpec] = None
    span_assertions: list[SpanAssertionSpec] = Field(
        default_factory=list,
        description=(
            "Span-level assertions evaluated against trace data. "
            "Hard-fails on any assertion failure (like Correctness layer). "
            "Use to assert on data flowing between nodes/tools."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class StopWhen(BaseModel):
    """Explicit early-exit condition for a scenario.

    Termination is deterministic and event-based only (eng review, binding):
    never judge-based, never keyword-triggered. A scenario stops early when the
    named concrete event is observed in a turn's trace.
    """
    tool_called: Optional[str] = Field(
        None,
        description="Stop after a turn in which the agent called this tool",
    )


class TurnChecks(BaseModel):
    """Layer checks applied inside a scenario (per-turn or as the outcome)."""
    correctness: Optional[CorrectnessSpec] = None
    path: Optional[PathSpec] = None
    cost: Optional[CostSpec] = None


class ScenarioSpec(BaseModel):
    """A multi-turn conversation scenario for `ciagent simulate`.

    Scripted mode (`turns:` given) is deterministic and needs no persona LLM —
    it is the CI / zero-key path. Generative personas (persona/goal without
    turns) are the finder path and ship in a later 0.9 phase.
    """
    name: Optional[str] = Field(None, description="Scenario identifier for reports")
    persona: Optional[str] = Field(
        None, description="Simulated-user persona (generative mode)"
    )
    goal: Optional[str] = Field(
        None, description="What the simulated user is trying to accomplish"
    )
    max_turns: int = Field(8, ge=1, description="Hard cap on conversation turns")
    turns: Optional[list[str]] = Field(
        None,
        description="Scripted user messages (deterministic mode); the conversation "
                    "ends when these are exhausted or max_turns is reached",
    )
    per_turn: Optional[TurnChecks] = Field(
        None, description="Checks evaluated on EVERY turn's trace"
    )
    outcome: Optional[TurnChecks] = Field(
        None,
        description="Checks evaluated once at the END of the conversation as the "
                    "verdict — never as a stop condition",
    )
    stop_when: Optional[StopWhen] = Field(
        None, description="Explicit deterministic early-exit event"
    )

    @field_validator("turns")
    @classmethod
    def turns_not_empty_strings(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is not None:
            if not v:
                raise ValueError("turns must contain at least one user message")
            if any(not t.strip() for t in v):
                raise ValueError("turns must not contain empty messages")
        return v

    @model_validator(mode="after")
    def _has_turn_source(self) -> "ScenarioSpec":
        # scripted (turns) or generative (persona/goal) — a scenario with
        # neither has no way to produce user messages
        if not self.turns and not (self.persona or self.goal):
            raise ValueError(
                "scenario needs a `turns:` list (scripted) or a "
                "`persona:`/`goal:` (generative)"
            )
        return self

    def display_name(self) -> str:
        if self.name:
            return self.name
        if self.persona:
            return self.persona[:60]
        if self.turns:
            return self.turns[0][:60]
        return "scenario"


class AgentCISpec(BaseModel):
    """Root schema for agentci_spec.yaml."""
    version: int = Field(1, description="Schema version for forward compatibility")
    agent: str = Field(..., description="Agent identifier (e.g. 'rag-agent', 'support-router')")
    baseline_dir: str = Field(
        default="./golden",
        description="Path to the directory where versioned baseline JSON files are stored.",
    )
    runner: Optional[str] = Field(
        None,
        description=(
            "Python dotted path to the agent runner callable, e.g. 'myagent.run:run_agent'. "
            "The function must accept (query: str) and return an ciagent.models.Trace. "
            "When set, 'ciagent test' can invoke the agent directly without pytest."
        ),
    )
    conversation_runner: Optional[str] = Field(
        None,
        description=(
            "Python dotted path to the multi-turn runner callable for `ciagent "
            "simulate`, e.g. 'myagent.run:respond'. The function must accept "
            "(messages: list[dict]) with {'role', 'content'} entries and return "
            "the assistant's reply as a str (or a ciagent.models.Trace). Fresh "
            "state per scenario; history is passed explicitly."
        ),
    )
    defaults: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Default correctness/path/cost settings applied to all queries unless overridden"
        ),
    )
    judge_config: Optional[dict[str, Any]] = Field(
        None,
        description="Global LLM judge settings: model, temperature, ensemble, structured_output",
    )
    persona_config: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Persona LLM settings for generative simulate scenarios: model "
            "(defaults to a cheap haiku-class model) and temperature (default 0.7)"
        ),
    )
    queries: list[GoldenQuery] = Field(
        default_factory=list,
        description="Single-turn test cases to evaluate",
    )
    scenarios: list[ScenarioSpec] = Field(
        default_factory=list,
        description="Multi-turn conversation scenarios for `ciagent simulate`",
    )

    @model_validator(mode="after")
    def _spec_has_content(self) -> "AgentCISpec":
        # queries was required min_length=1 before scenarios existed; a spec
        # must still declare at least one of the two
        if not self.queries and not self.scenarios:
            raise ValueError("spec must declare at least one query or scenario")
        return self
