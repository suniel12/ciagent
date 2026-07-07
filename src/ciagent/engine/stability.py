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

from ciagent.engine.results import QueryResult

if TYPE_CHECKING:
    from ciagent.schema.spec_models import AgentCISpec, GoldenQuery

# Normalized answers with similarity at or above this ratio (but not identical)
# are ambiguous: a paraphrase could flip a judge OR a deterministic keyword check.
SIMILARITY_AMBIGUITY_THRESHOLD = 0.9


class FlipSource(str, Enum):
    """Why a query's (or scenario's) verdict flipped across runs."""
    AGENT_VARIANCE = "agent-variance"  # agent output changed → fix the agent
    JUDGE_FLAKE = "judge-flake"        # same output/checks, judge verdict changed → fix the eval
    INFRA_ERROR = "infra-error"        # a judge call errored → fix nothing, retry
    MIXED = "mixed"                    # near-identical paraphrase + judge → ambiguous
    SIMULATION_VARIANCE = "simulation-variance"  # simulated user said different things → persona, not agent


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
    expected_runs: int = 0                  # session run count; < means partial aggregation

    @property
    def runs(self) -> int:
        return len(self.verdicts)

    @property
    def partial(self) -> bool:
        """The query is missing from at least one run (runner failure) —
        its verdicts aggregate over fewer runs than the session ran."""
        return 0 < self.runs < self.expected_runs

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
    duplicate_queries: list[str] = field(default_factory=list)  # texts appearing >1× in spec

    @property
    def partial_queries(self) -> list[QueryStability]:
        return [q for q in self.queries if q.partial]

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

    # Duplicate query texts collapse into one record when grouping by text —
    # surface that instead of hiding it.
    seen: set[str] = set()
    duplicates: list[str] = []
    for q in spec.queries:
        if q.query in seen and q.query not in duplicates:
            duplicates.append(q.query)
        seen.add(q.query)

    # Group results by query text, preserving spec order
    grouped: dict[str, list[QueryResult]] = {q.query: [] for q in spec.queries}
    for results in run_results:
        for r in results:
            grouped.setdefault(r.query, []).append(r)

    stabilities: list[QueryStability] = []
    for query_text, results in grouped.items():
        if not results:
            continue
        qs = _build_query_stability(
            query_text, results, query_specs.get(query_text), expected_runs=runs,
        )
        stabilities.append(qs)

    per_run_passed = [
        sum(1 for r in results if not r.hard_fail) for results in run_results
    ]

    return StabilityReport(
        runs=runs,
        per_run_passed=per_run_passed,
        total_queries=len(query_specs),
        queries=stabilities,
        duplicate_queries=duplicates,
    )


# ── Scenario stability (F6 Phase 3) ────────────────────────────────────────────


@dataclass
class ScenarioStability:
    """Multi-run stability record for one simulate scenario."""
    scenario: str
    verdicts: list[bool]                    # per-run: True = passed cleanly
    flip_source: Optional[FlipSource] = None
    flip_reason: str = ""
    cost_usd: list[float] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.verdicts)

    @property
    def flipped(self) -> bool:
        return len(set(self.verdicts)) > 1

    @property
    def pass_rate(self) -> float:
        return sum(self.verdicts) / len(self.verdicts) if self.verdicts else 0.0

    @property
    def always_failed(self) -> bool:
        return bool(self.verdicts) and not any(self.verdicts)

    @property
    def verdict_string(self) -> str:
        return "".join("✅" if v else "❌" for v in self.verdicts)


def build_scenario_stability(run_results: list[list]) -> list[ScenarioStability]:
    """Aggregate N simulate runs into per-scenario stability records.

    Args:
        run_results: Run-major matrix of ScenarioResults; every run evaluates
                     the same scenarios in spec order, so alignment is by index.

    Flip attribution adds a fourth source over the single-turn version:
    if the SIMULATED USER's turns differ across runs, the flip is
    `simulation-variance` — the persona said different things, and blaming
    the agent for that would be exactly the untrustworthy-eval failure mode
    this tool exists to kill. Scripted/replayed turns are identical by
    construction, so they can never attribute there.
    """
    if not run_results or not run_results[0]:
        return []

    n = min(len(run) for run in run_results)
    records: list[ScenarioStability] = []
    for i in range(n):
        per_run = [run[i] for run in run_results]
        rec = ScenarioStability(
            scenario=per_run[0].scenario.display_name(),
            verdicts=[not r.hard_fail and not r.is_partial for r in per_run],
            cost_usd=[r.cost_usd for r in per_run],
        )
        if rec.flipped:
            rec.flip_source, rec.flip_reason = _attribute_scenario_flip(per_run)
        records.append(rec)
    return records


def _attribute_scenario_flip(per_run: list) -> tuple[FlipSource, str]:
    """Attribute a scenario verdict flip. Discriminator order mirrors the
    single-turn version: infra first, then the user side, then the agent."""
    if any(r.is_infra_error or r.is_cost_aborted for r in per_run):
        return (
            FlipSource.INFRA_ERROR,
            "at least one run aborted (infra error or cost budget) — "
            "retry before trusting this flip",
        )

    user_transcripts = {tuple(r.user_turns()) for r in per_run}
    if len(user_transcripts) > 1:
        return (
            FlipSource.SIMULATION_VARIANCE,
            "the simulated user said different things across runs — "
            "record one conversation and gate on replay",
        )

    tool_seqs = {
        tuple(tuple(t.trace.tool_call_sequence or []) for t in r.turns) for r in per_run
    }
    if len(tool_seqs) > 1:
        return FlipSource.AGENT_VARIANCE, "same user turns, tool sequence changed"

    answers = {
        tuple(_normalize_answer(_extract_turn_answer(t)) for t in r.turns)
        for r in per_run
    }
    if len(answers) > 1:
        return FlipSource.AGENT_VARIANCE, "same user turns, agent answers changed"

    return FlipSource.JUDGE_FLAKE, "identical conversations, verdict flipped"


def _extract_turn_answer(turn) -> str:
    from ciagent.engine.runner import _extract_answer

    return _extract_answer(turn.trace)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _build_query_stability(
    query_text: str,
    results: list[QueryResult],
    query_spec: Optional["GoldenQuery"],
    expected_runs: int = 0,
) -> QueryStability:
    verdicts = [not r.hard_fail for r in results]
    answers = [_normalize_answer(_answer_of(r)) for r in results]
    tool_seqs = [_tool_sequence_of(r) for r in results]

    qs = QueryStability(
        query=query_text,
        verdicts=verdicts,
        cost_usd=[_trace_attr(r, "total_cost_usd") for r in results],
        latency_ms=[_trace_attr(r, "total_duration_ms") for r in results],
        expected_runs=expected_runs or len(results),
    )
    qs.answer_similarity = _min_pairwise_similarity(answers)

    if not qs.flipped:
        return qs

    qs.flip_source, qs.flip_reason = _attribute_flip(
        answers=answers,
        tool_seqs=tool_seqs,
        similarity=qs.answer_similarity,
        has_judge=_query_has_judge(query_spec),
        det_signatures=[_det_signature(r) for r in results],
        judge_verdicts=[_judge_verdict_of(r) for r in results],
        judge_errored=[_judge_errored(r) for r in results],
    )
    return qs


def _attribute_flip(
    answers: list[str],
    tool_seqs: list[tuple[str, ...]],
    similarity: float,
    has_judge: bool,
    det_signatures: list[tuple],
    judge_verdicts: list[Optional[bool]],
    judge_errored: list[bool],
) -> tuple[FlipSource, str]:
    """Attribute a verdict flip to the agent, the evaluation, or infrastructure.

    Discriminator order (most reliable signal first):
      1. A judge call errored in any run → infra-error. A transient API failure
         counts as a fail in the verdict, and must never read as "fix your rubric".
      2. Per-layer sub-verdicts: if every deterministic check returned the same
         outcome across runs but the judge's verdict changed, the flip is the
         judge's — regardless of how much the answer text was paraphrased.
      3. Identical output (answer + tool sequence) → judge-flake by construction:
         deterministic layers cannot flip on identical input.
      4. Tool sequence changed → agent-variance.
      5. Deterministic check outcomes changed → the output change caused the flip:
         agent-variance.
      6. Near-identical paraphrase with a judge configured → mixed (never guess).
    """
    if any(judge_errored):
        return (
            FlipSource.INFRA_ERROR,
            "a judge call errored during at least one run — retry before trusting this flip",
        )

    det_flipped = len(set(det_signatures)) > 1
    judge_present = any(v is not None for v in judge_verdicts)
    judge_flipped = judge_present and len({v for v in judge_verdicts if v is not None}) > 1

    if not det_flipped and judge_flipped:
        return (
            FlipSource.JUDGE_FLAKE,
            "deterministic checks agreed across runs; the judge changed its verdict",
        )

    identical_answers = len(set(answers)) == 1
    identical_tools = len(set(tool_seqs)) == 1

    if identical_answers and identical_tools:
        return FlipSource.JUDGE_FLAKE, "same answer, verdict flipped"

    if not identical_tools:
        return FlipSource.AGENT_VARIANCE, "tool sequence changed"

    if det_flipped:
        return FlipSource.AGENT_VARIANCE, "answer changed, deterministic check outcome changed"

    if similarity >= SIMILARITY_AMBIGUITY_THRESHOLD and has_judge:
        # A near-identical paraphrase could flip either a judge or a keyword
        # check — do not guess.
        return (
            FlipSource.MIXED,
            f"near-identical answers (similarity {similarity:.2f}) with judge configured",
        )

    return FlipSource.AGENT_VARIANCE, "answer changed"


# Deterministic correctness checks and where their outcome lives in
# LayerResult.details — the per-layer signature compared across runs.
_DET_DETAIL_KEYS = (
    ("expected_in_answer", "all_found"),
    ("any_expected_in_answer", "any_found"),
    ("not_in_answer", "none_found"),
    ("exact_match", "matched"),
    ("regex_match", "matched"),
    ("json_schema", "valid"),
)


def _det_signature(r: QueryResult) -> tuple:
    """Outcomes of every deterministic correctness check for one run."""
    d = r.correctness.details or {}
    sig = []
    for key, outcome_field in _DET_DETAIL_KEYS:
        entry = d.get(key)
        sig.append(entry.get(outcome_field) if isinstance(entry, dict) else None)
    return tuple(sig)


def _judge_entries(r: QueryResult) -> list[dict]:
    d = r.correctness.details or {}
    entries = [v for k, v in d.items() if k.startswith("judge_") and isinstance(v, dict)]
    for k in ("safety", "hallucination"):
        if isinstance(d.get(k), dict):
            entries.append(d[k])
    return entries


def _judge_verdict_of(r: QueryResult) -> Optional[bool]:
    """Aggregate judge verdict for one run (None = no judge ran)."""
    entries = _judge_entries(r)
    if not entries:
        return None
    return all(e.get("passed", False) for e in entries)


def _judge_errored(r: QueryResult) -> bool:
    return any(e.get("error") for e in _judge_entries(r))


def _query_has_judge(query_spec: Optional["GoldenQuery"]) -> bool:
    """True if any LLM-judged rubric participates in this query's verdict."""
    if query_spec is None or query_spec.correctness is None:
        return False
    c = query_spec.correctness
    return bool(c.llm_judge or c.safety_check or c.hallucination_check)


def _answer_of(r: QueryResult) -> str:
    from ciagent.engine.runner import _extract_answer

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
