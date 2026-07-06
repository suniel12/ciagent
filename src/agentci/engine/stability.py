# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Stability Engine — multi-run verdict aggregation and flip attribution.

A suite score that holds steady across runs can hide per-query verdict flips:
the aggregate looks stable because the errors move around. This module runs
the evaluation N times and answers two questions per query:

    1. Did the verdict flip across runs?
    2. If so, WHY — did the agent produce different output (agent-variance),
       or did the judge grade the same output differently (judge-flake)?

Attribution rests on one structural fact: deterministic layers cannot flip on
identical output by construction. If the normalized answer and tool sequence
are identical across runs but the verdict flipped, the flip came from the LLM
judge. If the output differs, the agent varied. Near-identical paraphrases
with a judge configured are labelled `mixed` — never guessed.

Public API:
    build_stability_report(spec, run_results) → StabilityReport
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from agentci.engine.results import QueryResult

if TYPE_CHECKING:
    from agentci.schema.spec_models import AgentCISpec, GoldenQuery

# Normalized answers with similarity at or above this ratio (but not identical)
# are ambiguous: a paraphrase could flip a judge OR a deterministic keyword check.
SIMILARITY_AMBIGUITY_THRESHOLD = 0.9


class FlipSource(str, Enum):
    """Why a query's verdict flipped across runs."""
    AGENT_VARIANCE = "agent-variance"  # agent output changed → fix the agent
    JUDGE_FLAKE = "judge-flake"        # same output, judge verdict changed → fix the eval
    MIXED = "mixed"                    # near-identical paraphrase + judge → ambiguous


@dataclass
class QueryStability:
    """Multi-run stability record for a single query."""
    query: str
    verdicts: list[bool]                    # per-run: True = passed (no hard fail)
    flip_source: Optional[FlipSource] = None
    flip_reason: str = ""                   # human-readable one-liner
    answer_similarity: float = 1.0          # min pairwise similarity of normalized answers
    cost_usd: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.verdicts)

    @property
    def flipped(self) -> bool:
        return len(set(self.verdicts)) > 1

    @property
    def pass_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(self.verdicts) / len(self.verdicts)

    @property
    def pass_at_k(self) -> float:
        """Estimated P(≥1 pass in k trials), k = observed runs, p = observed rate."""
        return round(1.0 - (1.0 - self.pass_rate) ** self.runs, 3) if self.runs else 0.0

    @property
    def pass_pow_k(self) -> float:
        """Estimated P(all k trials pass), k = observed runs, p = observed rate."""
        return round(self.pass_rate ** self.runs, 3) if self.runs else 0.0

    @property
    def always_failed(self) -> bool:
        """Failed in every run — a consistent failure, not flakiness."""
        return bool(self.verdicts) and not any(self.verdicts)

    @property
    def verdict_string(self) -> str:
        return "".join("✅" if v else "❌" for v in self.verdicts)


@dataclass
class StabilityReport:
    """Suite-level stability across N runs."""
    runs: int
    per_run_passed: list[int]               # passed count per run
    total_queries: int
    queries: list[QueryStability]

    @property
    def flipped_queries(self) -> list[QueryStability]:
        return [q for q in self.queries if q.flipped]

    @property
    def consistent_failures(self) -> list[QueryStability]:
        return [q for q in self.queries if q.always_failed]

    @property
    def is_stable(self) -> bool:
        return not self.flipped_queries

    @property
    def verdict(self) -> str:
        return "STABLE" if self.is_stable else "FLAKY"

    @property
    def per_run_scores(self) -> list[float]:
        if not self.total_queries:
            return [0.0 for _ in self.per_run_passed]
        return [round(p / self.total_queries, 3) for p in self.per_run_passed]


def build_stability_report(
    spec: "AgentCISpec",
    run_results: list[list[QueryResult]],
) -> StabilityReport:
    """Aggregate N runs of evaluation results into a stability report.

    Args:
        spec:        The evaluated spec (used to detect judge-backed queries).
        run_results: Run-major matrix: run_results[i] is run i's QueryResults.

    Returns:
        StabilityReport. Queries missing from any run (runner failure) are
        aggregated over the runs where they appear.
    """
    runs = len(run_results)
    query_specs = {q.query: q for q in spec.queries}

    # Group results by query text, preserving spec order
    grouped: dict[str, list[QueryResult]] = {q.query: [] for q in spec.queries}
    for results in run_results:
        for r in results:
            grouped.setdefault(r.query, []).append(r)

    stabilities: list[QueryStability] = []
    for query_text, results in grouped.items():
        if not results:
            continue
        qs = _build_query_stability(query_text, results, query_specs.get(query_text))
        stabilities.append(qs)

    per_run_passed = [
        sum(1 for r in results if not r.hard_fail) for results in run_results
    ]

    return StabilityReport(
        runs=runs,
        per_run_passed=per_run_passed,
        total_queries=len(query_specs),
        queries=stabilities,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────


def _build_query_stability(
    query_text: str,
    results: list[QueryResult],
    query_spec: Optional["GoldenQuery"],
) -> QueryStability:
    verdicts = [not r.hard_fail for r in results]
    answers = [_normalize_answer(_answer_of(r)) for r in results]
    tool_seqs = [_tool_sequence_of(r) for r in results]

    qs = QueryStability(
        query=query_text,
        verdicts=verdicts,
        cost_usd=[_trace_attr(r, "total_cost_usd") for r in results],
        latency_ms=[_trace_attr(r, "total_duration_ms") for r in results],
    )
    qs.answer_similarity = _min_pairwise_similarity(answers)

    if not qs.flipped:
        return qs

    qs.flip_source, qs.flip_reason = _attribute_flip(
        answers=answers,
        tool_seqs=tool_seqs,
        similarity=qs.answer_similarity,
        has_judge=_query_has_judge(query_spec),
    )
    return qs


def _attribute_flip(
    answers: list[str],
    tool_seqs: list[tuple[str, ...]],
    similarity: float,
    has_judge: bool,
) -> tuple[FlipSource, str]:
    """Attribute a verdict flip to the agent or the evaluation.

    Deterministic layers evaluate the answer string and tool sequence, so on
    identical output they return identical verdicts by construction — a flip
    on identical output can only come from the LLM judge.
    """
    identical_answers = len(set(answers)) == 1
    identical_tools = len(set(tool_seqs)) == 1

    if identical_answers and identical_tools:
        return FlipSource.JUDGE_FLAKE, "same answer, verdict flipped"

    if not identical_tools:
        return FlipSource.AGENT_VARIANCE, "tool sequence changed"

    if similarity >= SIMILARITY_AMBIGUITY_THRESHOLD and has_judge:
        # A near-identical paraphrase could flip either a judge or a keyword
        # check — do not guess.
        return (
            FlipSource.MIXED,
            f"near-identical answers (similarity {similarity:.2f}) with judge configured",
        )

    return FlipSource.AGENT_VARIANCE, "answer changed"


def _query_has_judge(query_spec: Optional["GoldenQuery"]) -> bool:
    """True if any LLM-judged rubric participates in this query's verdict."""
    if query_spec is None or query_spec.correctness is None:
        return False
    c = query_spec.correctness
    return bool(c.llm_judge or c.safety_check or c.hallucination_check)


def _answer_of(r: QueryResult) -> str:
    from agentci.engine.runner import _extract_answer

    trace = getattr(r, "trace", None)
    return _extract_answer(trace) if trace is not None else ""


def _tool_sequence_of(r: QueryResult) -> tuple[str, ...]:
    trace = getattr(r, "trace", None)
    if trace is None:
        return ()
    return tuple(getattr(trace, "tool_call_sequence", ()) or ())


def _trace_attr(r: QueryResult, attr: str) -> float:
    trace = getattr(r, "trace", None)
    return float(getattr(trace, attr, 0.0) or 0.0) if trace is not None else 0.0


def _normalize_answer(answer: str) -> str:
    """Collapse whitespace and casing so formatting noise doesn't read as variance."""
    return " ".join(answer.split()).lower()


def _min_pairwise_similarity(answers: list[str]) -> float:
    """Minimum pairwise similarity ratio across all runs' normalized answers."""
    if len(answers) < 2:
        return 1.0
    lowest = 1.0
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            ratio = difflib.SequenceMatcher(None, answers[i], answers[j]).ratio()
            lowest = min(lowest, ratio)
    return round(lowest, 3)
