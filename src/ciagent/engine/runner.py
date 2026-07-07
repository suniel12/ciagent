# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
AgentCI v2 Engine Runner — Orchestrator.

Wires all three evaluation layers together per query and provides a
spec-level batch evaluation function.

Public API:
    evaluate_query(query, trace, baseline_trace, judge_config) → QueryResult
    evaluate_spec(spec, traces, baselines)                     → list[QueryResult]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ciagent.engine.correctness import evaluate_correctness
from ciagent.engine.cost import evaluate_cost
from ciagent.engine.path import evaluate_path
from ciagent.engine.results import LayerResult, LayerStatus, QueryResult
from ciagent.engine.span_assertions import evaluate_span_assertions
from ciagent.schema.spec_models import AgentCISpec, GoldenQuery

if TYPE_CHECKING:
    from ciagent.models import Trace


def evaluate_query(
    query: GoldenQuery,
    trace: "Trace",
    baseline_trace: Optional["Trace"] = None,
    judge_config: Optional[dict[str, Any]] = None,
    spec_dir: Optional[str] = None,
) -> QueryResult:
    """Evaluate a single golden query across all evaluation layers.

    Args:
        query:          The GoldenQuery spec (correctness/path/retrieval/cost
                        assertions).
        trace:          The agent's execution trace for this query.
        baseline_trace: Optional golden baseline trace for comparison.
        judge_config:   Global LLM judge settings (model, temperature, ensemble).
        spec_dir:       Directory of the spec file (for context_file resolution).

    Returns:
        QueryResult containing one LayerResult per layer.
    """
    answer = _extract_answer(trace)

    # ── Layer 1: Correctness (hard fail) ──────────────────────────────────────
    if query.correctness:
        correctness = evaluate_correctness(
            answer=answer,
            spec=query.correctness,
            trace=trace,
            judge_config=judge_config,
            query=query.query,
            spec_dir=spec_dir,
        )
    else:
        correctness = LayerResult(
            status=LayerStatus.SKIP,
            details={},
            messages=["No assertions configured"],
        )

    # ── Span Assertions (sub-layer of Correctness, hard fail) ─────────────────
    if query.span_assertions:
        span_result = evaluate_span_assertions(
            spec=query.span_assertions,
            trace=trace,
            judge_config=judge_config,
            spec_dir=spec_dir,
        )
        # Merge span assertion results into correctness details
        correctness.details["span_assertions"] = span_result.details.get(
            "span_assertions", []
        )
        # If span assertions failed but correctness passed, escalate to FAIL
        if span_result.status == LayerStatus.FAIL and correctness.status != LayerStatus.FAIL:
            correctness = LayerResult(
                status=LayerStatus.FAIL,
                details=correctness.details,
                messages=correctness.messages + span_result.messages,
            )
        elif span_result.status == LayerStatus.FAIL:
            # Both failed — append span assertion failures to existing messages
            correctness = LayerResult(
                status=LayerStatus.FAIL,
                details=correctness.details,
                messages=correctness.messages + span_result.messages,
            )

    # ── Layer 2: Path (soft warn) ─────────────────────────────────────────────
    if query.path:
        path = evaluate_path(
            trace=trace,
            spec=query.path,
            baseline_trace=baseline_trace,
        )
    else:
        path = LayerResult(
            status=LayerStatus.SKIP,
            details={},
            messages=["No assertions configured"],
        )

    # ── Layer 2.5: Retrieval (soft warn; SKIPs when it cannot read) ──────────
    if query.retrieval:
        from ciagent.engine.retrieval import evaluate_retrieval

        retrieval = evaluate_retrieval(
            trace=trace,
            spec=query.retrieval,
            correctness=query.correctness,
            answer=answer,
        )
    else:
        retrieval = LayerResult(
            status=LayerStatus.SKIP,
            details={},
            messages=["No assertions configured"],
        )

    # ── Layer 3: Cost (soft warn) ─────────────────────────────────────────────
    if query.cost:
        cost = evaluate_cost(
            trace=trace,
            spec=query.cost,
            baseline_trace=baseline_trace,
        )
    else:
        cost = LayerResult(
            status=LayerStatus.SKIP,
            details={},
            messages=["No assertions configured"],
        )

    return QueryResult(
        query=query.query,
        correctness=correctness,
        path=path,
        retrieval=retrieval,
        cost=cost,
        trace=trace,
    )


def evaluate_spec(
    spec: AgentCISpec,
    traces: dict[str, "Trace"],
    baselines: Optional[dict[str, "Trace"]] = None,
    spec_dir: Optional[str] = None,
) -> list[QueryResult]:
    """Evaluate all queries in a spec against their captured traces.

    Args:
        spec:      The loaded AgentCISpec.
        traces:    Mapping of query_text → Trace (captured from agent runs).
        baselines: Optional mapping of query_text → golden baseline Trace.

    Returns:
        List of QueryResult in the same order as spec.queries.
        Queries with no matching trace are skipped (not included in output).
    """
    baselines = baselines or {}
    results: list[QueryResult] = []

    for query in spec.queries:
        trace = traces.get(query.query)
        if trace is None:
            continue
        baseline = baselines.get(query.query)
        result = evaluate_query(
            query=query,
            trace=trace,
            baseline_trace=baseline,
            judge_config=spec.judge_config,
            spec_dir=spec_dir,
        )
        results.append(result)

    return results


# ── Internal helpers ───────────────────────────────────────────────────────────


def _extract_answer(trace: "Trace") -> str:
    """Extract the agent's final text answer from the trace.

    Strategy:
        1. trace.metadata["final_output"] (explicitly set by runner).
        2. Last span's output_data (fallback for runners that don't set metadata).
        3. Empty string if nothing found.
    """
    meta_output = trace.metadata.get("final_output")
    if meta_output is not None:
        return str(meta_output)

    if trace.spans:
        last_span = trace.spans[-1]
        output = last_span.output_data
        if output is not None:
            return output if isinstance(output, str) else str(output)

    return ""
