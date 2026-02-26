"""
AgentCI v2 Spec Models

Pydantic models for agentci_spec.yaml. Every field is optional except
`agent` and `queries` on the root spec, and `query` on each golden query.

Hierarchy:
    AgentCISpec
    └── GoldenQuery (1..N)
        ├── CorrectnessSpec  (Layer 1 — hard fail)
        ├── PathSpec         (Layer 2 — soft warn)
        └── CostSpec         (Layer 3 — soft warn)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────


class MatchMode(str, Enum):
    """How to compare the agent's tool sequence against the golden baseline."""
    STRICT = "strict"        # Exact match: same tools, same order
    UNORDERED = "unordered"  # Same tools, any order
    SUBSET = "subset"        # Reference tools must appear (extras OK)  [default]
    SUPERSET = "superset"    # All used tools must be in reference set


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


# ── Layer Specs ────────────────────────────────────────────────────────────────


class CorrectnessSpec(BaseModel):
    """Layer 1: Hard pass/fail. Any failure blocks the CI pipeline."""
    expected_in_answer: Optional[list[str]] = Field(
        None,
        description="Strings that must appear in the answer (case-insensitive)",
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


class PathSpec(BaseModel):
    """Layer 2: Trajectory evaluation. Exceedances produce warnings, not failures."""
    max_tool_calls: Optional[int] = Field(
        None,
        ge=0,
        description="Maximum number of tool calls allowed",
    )
    expected_tools: Optional[list[str]] = Field(
        None,
        description="Tools that should be called (used for recall calculation)",
    )
    forbidden_tools: Optional[list[str]] = Field(
        None,
        description="Tools that must NOT be called (safety boundary — hard fail if violated)",
    )
    max_loops: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum consecutive repeated tool invocations (loop detection)",
    )
    match_mode: MatchMode = Field(
        MatchMode.SUBSET,
        description="How to compare tool sequences against the golden baseline",
    )
    min_tool_recall: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="|expected ∩ used| / |expected|",
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

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


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
            "The function must accept (query: str) and return an agentci.models.Trace. "
            "When set, 'agentci test' can invoke the agent directly without pytest."
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
    queries: list[GoldenQuery] = Field(
        ...,
        min_length=1,
        description="Test cases to evaluate",
    )
