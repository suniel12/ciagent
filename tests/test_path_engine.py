"""
Unit tests for the Path Engine (Layer 2 — Soft Warning).

Builds Trace objects using the existing agentci.models classes.
No mocks needed — all tests are deterministic.
"""

from __future__ import annotations

import pytest

from agentci.engine.path import _evaluate_match_mode, evaluate_path
from agentci.engine.results import LayerStatus
from agentci.models import Span, SpanKind, Trace, ToolCall
from agentci.schema.spec_models import MatchMode, PathSpec


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_trace(*tool_names: str, handoffs: list[tuple[str, str]] | None = None) -> Trace:
    """Build a minimal Trace with the given tool call sequence.

    Args:
        tool_names: Ordered tool names in the agent span.
        handoffs:   Optional list of (from_agent, to_agent) pairs for HANDOFF spans.
    """
    tool_calls = [ToolCall(tool_name=t) for t in tool_names]
    spans = [Span(kind=SpanKind.AGENT, name="test_agent", tool_calls=tool_calls)]
    if handoffs:
        for from_a, to_a in handoffs:
            spans.append(
                Span(kind=SpanKind.HANDOFF, from_agent=from_a, to_agent=to_a)
            )
    trace = Trace(spans=spans)
    trace.compute_metrics()
    return trace


def path(**kwargs) -> PathSpec:
    return PathSpec(**kwargs)


# ── max_tool_calls ────────────────────────────────────────────────────────────


class TestMaxToolCalls:
    def test_under_limit_passes(self):
        trace = make_trace("a", "b", "c")
        result = evaluate_path(trace, path(max_tool_calls=5))
        assert result.status == LayerStatus.PASS

    def test_at_limit_passes(self):
        trace = make_trace("a", "b", "c")
        result = evaluate_path(trace, path(max_tool_calls=3))
        assert result.status == LayerStatus.PASS

    def test_over_limit_warns(self):
        trace = make_trace("a", "b", "c", "d")
        result = evaluate_path(trace, path(max_tool_calls=2))
        assert result.status == LayerStatus.WARN
        assert any("Tool calls" in m for m in result.messages)

    def test_zero_calls_with_zero_max(self):
        trace = make_trace()  # no tools
        result = evaluate_path(trace, path(max_tool_calls=0))
        assert result.status == LayerStatus.PASS

    def test_one_call_with_zero_max_warns(self):
        trace = make_trace("a")
        result = evaluate_path(trace, path(max_tool_calls=0))
        assert result.status == LayerStatus.WARN


# ── forbidden_tools (hard FAIL) ───────────────────────────────────────────────


class TestForbiddenTools:
    def test_no_forbidden_tools_used_passes(self):
        trace = make_trace("retriever")
        result = evaluate_path(trace, path(forbidden_tools=["web_search", "tavily"]))
        assert result.status == LayerStatus.PASS

    def test_forbidden_tool_used_is_hard_fail(self):
        trace = make_trace("retriever", "web_search")
        result = evaluate_path(trace, path(forbidden_tools=["web_search"]))
        assert result.status == LayerStatus.FAIL
        assert "web_search" in str(result.messages)

    def test_multiple_forbidden_tools_all_reported(self):
        trace = make_trace("web_search", "tavily")
        result = evaluate_path(trace, path(forbidden_tools=["web_search", "tavily"]))
        assert result.status == LayerStatus.FAIL

    def test_no_tools_called_with_forbidden_list_passes(self):
        trace = make_trace()
        result = evaluate_path(trace, path(forbidden_tools=["web_search"]))
        assert result.status == LayerStatus.PASS


# ── tool recall / precision ───────────────────────────────────────────────────


class TestToolRecallAndPrecision:
    def test_full_recall_passes(self):
        trace = make_trace("a", "b")
        result = evaluate_path(trace, path(expected_tools=["a", "b"], min_tool_recall=1.0))
        assert result.status == LayerStatus.PASS

    def test_partial_recall_below_min_warns(self):
        trace = make_trace("a")
        result = evaluate_path(trace, path(expected_tools=["a", "b"], min_tool_recall=1.0))
        assert result.status == LayerStatus.WARN
        assert any("recall" in m.lower() for m in result.messages)

    def test_full_precision_passes(self):
        trace = make_trace("a", "b")
        result = evaluate_path(trace, path(expected_tools=["a", "b"], min_tool_precision=1.0))
        assert result.status == LayerStatus.PASS

    def test_extras_lower_precision_warns(self):
        # expected={a}, used={a,b,c} → precision = 1/3
        trace = make_trace("a", "b", "c")
        result = evaluate_path(trace, path(expected_tools=["a"], min_tool_precision=0.9))
        assert result.status == LayerStatus.WARN

    def test_details_contain_recall(self):
        trace = make_trace("a")
        result = evaluate_path(trace, path(expected_tools=["a", "b"]))
        assert "tool_recall" in result.details


# ── sequence similarity ────────────────────────────────────────────────────────


class TestSequenceSimilarity:
    def test_identical_sequence_passes(self):
        trace = make_trace("a", "b", "c")
        baseline = make_trace("a", "b", "c")
        result = evaluate_path(trace, path(min_sequence_similarity=1.0), baseline)
        assert result.status == LayerStatus.PASS

    def test_disjoint_sequence_warns(self):
        trace = make_trace("x", "y")
        baseline = make_trace("a", "b")
        result = evaluate_path(trace, path(min_sequence_similarity=0.5), baseline)
        assert result.status == LayerStatus.WARN

    def test_no_baseline_skips_similarity_check(self):
        trace = make_trace("a", "b")
        result = evaluate_path(trace, path(min_sequence_similarity=0.9), baseline_trace=None)
        # No baseline → check skipped → PASS (no other spec fields set)
        assert result.status == LayerStatus.PASS


# ── loop detection ─────────────────────────────────────────────────────────────


class TestLoopDetection:
    def test_no_loops_passes(self):
        # [a,b,c] has 0 loops; max_loops=1 → passes
        trace = make_trace("a", "b", "c")
        result = evaluate_path(trace, path(max_loops=1))
        assert result.status == LayerStatus.PASS

    def test_loops_exceeding_max_warns(self):
        trace = make_trace("a", "a", "a")  # 2 consecutive loops
        result = evaluate_path(trace, path(max_loops=1))
        assert result.status == LayerStatus.WARN
        assert any("Loop" in m for m in result.messages)

    def test_loops_at_max_passes(self):
        trace = make_trace("a", "a")  # 1 loop, max_loops=1 → passes
        result = evaluate_path(trace, path(max_loops=1))
        assert result.status == LayerStatus.PASS


# ── match modes ────────────────────────────────────────────────────────────────


class TestMatchModes:
    def test_strict_identical(self):
        result = _evaluate_match_mode(["a", "b", "c"], ["a", "b", "c"], MatchMode.STRICT)
        assert result["matched"] is True

    def test_strict_different_order_fails(self):
        result = _evaluate_match_mode(["b", "a"], ["a", "b"], MatchMode.STRICT)
        assert result["matched"] is False

    def test_strict_extras_fail(self):
        result = _evaluate_match_mode(["a", "b", "c"], ["a", "b"], MatchMode.STRICT)
        assert result["matched"] is False

    def test_unordered_same_set(self):
        result = _evaluate_match_mode(["b", "a"], ["a", "b"], MatchMode.UNORDERED)
        assert result["matched"] is True

    def test_unordered_different_set_fails(self):
        result = _evaluate_match_mode(["a", "c"], ["a", "b"], MatchMode.UNORDERED)
        assert result["matched"] is False

    def test_subset_reference_in_used(self):
        # reference=[a,b], used=[a,b,c] → subset passes (extras OK)
        result = _evaluate_match_mode(["a", "b", "c"], ["a", "b"], MatchMode.SUBSET)
        assert result["matched"] is True

    def test_subset_reference_missing_fails(self):
        result = _evaluate_match_mode(["a", "c"], ["a", "b"], MatchMode.SUBSET)
        assert result["matched"] is False

    def test_superset_all_used_in_reference(self):
        # reference=[a,b,c], used=[a,b] → superset passes
        result = _evaluate_match_mode(["a", "b"], ["a", "b", "c"], MatchMode.SUPERSET)
        assert result["matched"] is True

    def test_superset_unexpected_tool_fails(self):
        # used has 'x' which is not in reference
        result = _evaluate_match_mode(["a", "x"], ["a", "b"], MatchMode.SUPERSET)
        assert result["matched"] is False

    def test_match_mode_in_evaluate_path_warns_on_mismatch(self):
        trace = make_trace("a", "c")
        baseline = make_trace("a", "b")
        result = evaluate_path(
            trace,
            path(match_mode="strict"),
            baseline_trace=baseline,
        )
        assert result.status == LayerStatus.WARN


# ── handoff assertions ────────────────────────────────────────────────────────


class TestHandoffAssertions:
    def test_expected_handoff_found_passes(self):
        trace = make_trace(handoffs=[("triage", "billing")])
        result = evaluate_path(trace, path(expected_handoff="billing"))
        assert result.status == LayerStatus.PASS

    def test_expected_handoff_missing_warns(self):
        trace = make_trace(handoffs=[("triage", "technical")])
        result = evaluate_path(trace, path(expected_handoff="billing"))
        assert result.status == LayerStatus.WARN
        assert any("billing" in m for m in result.messages)

    def test_max_handoff_count_at_limit_passes(self):
        trace = make_trace(handoffs=[("a", "b")])
        result = evaluate_path(trace, path(max_handoff_count=1))
        assert result.status == LayerStatus.PASS

    def test_max_handoff_count_exceeded_warns(self):
        trace = make_trace(handoffs=[("a", "b"), ("b", "c")])
        result = evaluate_path(trace, path(max_handoff_count=1))
        assert result.status == LayerStatus.WARN


# ── empty spec ────────────────────────────────────────────────────────────────


class TestEmptySpec:
    def test_empty_spec_passes(self):
        trace = make_trace("a", "b")
        result = evaluate_path(trace, path())
        assert result.status == LayerStatus.PASS
        assert "No loops detected" in result.messages


# ── max_loops default (1.1) ───────────────────────────────────────────────────


class TestMaxLoopsDefault:
    def test_default_max_loops_is_three(self):
        """PathSpec with no max_loops argument should default to 3."""
        assert PathSpec().max_loops == 3

    def test_normal_trace_passes_with_default(self):
        """A trace with no loops passes with the default max_loops=3."""
        trace = make_trace("a", "b", "c")
        result = evaluate_path(trace, path())
        assert result.status == LayerStatus.PASS

    def test_four_loops_warns_with_default(self):
        """5 consecutive identical calls → 4 loops > default 3 → WARN."""
        trace = make_trace("a", "a", "a", "a", "a")  # loops=4
        result = evaluate_path(trace, path())  # max_loops defaults to 3
        assert result.status == LayerStatus.WARN
        assert any("Loop" in m for m in result.messages)

    def test_three_loops_passes_with_default(self):
        """4 consecutive calls → 3 loops, not > 3 → PASS with default."""
        trace = make_trace("a", "a", "a", "a")  # loops=3
        result = evaluate_path(trace, path())
        assert result.status == LayerStatus.PASS

    def test_override_max_loops_disables_default_warning(self):
        """Per-query override max_loops=10 should not warn for 4 loops."""
        trace = make_trace("a", "a", "a", "a", "a")  # loops=4
        result = evaluate_path(trace, path(max_loops=10))
        assert result.status == LayerStatus.PASS

    def test_details_always_contain_loops_detected(self):
        """loops_detected is always in details now that max_loops has a default."""
        trace = make_trace("a", "b", "c")
        result = evaluate_path(trace, path())
        assert "loops_detected" in result.details
        assert result.details["loops_detected"] == 0


# ── descriptive PASS messages ────────────────────────────────────────────────


class TestDescriptivePassMessages:
    def test_max_tool_calls_describes_count(self):
        trace = make_trace("a", "b")
        result = evaluate_path(trace, path(max_tool_calls=5))
        assert result.status == LayerStatus.PASS
        assert any("Tool calls: 2 ≤ max 5" in m for m in result.messages)

    def test_forbidden_tools_describes_clean(self):
        trace = make_trace("a", "b")
        result = evaluate_path(trace, path(forbidden_tools=["evil_tool"]))
        assert result.status == LayerStatus.PASS
        assert any("No forbidden tools used" in m for m in result.messages)

    def test_tool_recall_describes_expected(self):
        trace = make_trace("retrieve_docs", "grade_artifacts")
        result = evaluate_path(trace, path(expected_tools=["retrieve_docs"]))
        assert result.status == LayerStatus.PASS
        assert any("Tool recall" in m and "retrieve_docs" in m for m in result.messages)

    def test_no_loops_described(self):
        trace = make_trace("a", "b", "c")
        result = evaluate_path(trace, path())
        assert result.status == LayerStatus.PASS
        assert any("No loops detected" in m for m in result.messages)

    def test_loops_within_limit_described(self):
        trace = make_trace("a", "a", "a")  # 2 loops
        result = evaluate_path(trace, path(max_loops=5))
        assert result.status == LayerStatus.PASS
        assert any("Loops: 2 ≤ max 5" in m for m in result.messages)
