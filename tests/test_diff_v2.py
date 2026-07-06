"""
Tests for the AgentCI v2 Diff Engine (engine/diff.py).

Tests the three-tier DiffReport: Correctness, Path, and Cost layers.
Also tests backward compatibility with the v1 DiffResult list.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from ciagent.engine.diff import (
    DiffReport,
    MetricDelta,
    diff_baselines,
    _compute_path_deltas,
    _compute_cost_deltas,
    _extract_answer,
    _format_value,
)
from ciagent.models import Trace, Span, ToolCall, LLMCall


# ── Fixtures ───────────────────────────────────────────────────────────────────


def make_trace(
    tool_names: list[str] | None = None,
    cost_usd: float = 0.001,
    tokens_in: int = 100,
    tokens_out: int = 50,
    duration_ms: float = 500.0,
    llm_calls: int = 1,
    output: str = "Test answer",
) -> Trace:
    """Build a synthetic Trace for testing."""
    tool_calls_objs = [ToolCall(tool_name=name) for name in (tool_names or [])]
    n = max(llm_calls, 1)
    llm_call_objs = [
        LLMCall(
            model="gpt-4o-mini",
            tokens_in=tokens_in // n,
            tokens_out=tokens_out // n,
            cost_usd=cost_usd / n,
            duration_ms=duration_ms / n,
        )
        for _ in range(n)
    ]
    span = Span(
        name="agent",
        tool_calls=tool_calls_objs,
        llm_calls=llm_call_objs,
        output_data=output,
    )
    span.compute_metrics()
    trace = Trace(spans=[span])
    trace.compute_metrics()
    return trace


def trace_to_baseline(
    trace: Trace,
    version: str = "v1",
    agent: str = "test-agent",
    query: str = "test query",
) -> dict:
    """Wrap a Trace into the baseline file format."""
    return {
        "version": version,
        "agent": agent,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "metadata": {"model": "gpt-4o-mini", "precheck_passed": True},
        "trace": json.loads(trace.model_dump_json()),
    }


# ── MetricDelta tests ──────────────────────────────────────────────────────────


class TestMetricDelta:
    def test_pct_change_decrease(self):
        d = MetricDelta("tool_calls", before=10, after=2)
        assert pytest.approx(d.pct_change, abs=0.1) == -80.0

    def test_pct_change_increase(self):
        d = MetricDelta("cost_usd", before=0.001, after=0.003)
        assert pytest.approx(d.pct_change, abs=0.1) == 200.0

    def test_pct_change_zero_before(self):
        d = MetricDelta("tool_calls", before=0, after=5)
        assert d.pct_change is None

    def test_pct_change_none_values(self):
        d = MetricDelta("latency", before=None, after=500)
        assert d.pct_change is None

    def test_direction_arrow_decrease(self):
        d = MetricDelta("tool_calls", before=10, after=2)
        assert d.direction_arrow == "▼"

    def test_direction_arrow_increase(self):
        d = MetricDelta("cost", before=1, after=3)
        assert d.direction_arrow == "▲"

    def test_direction_arrow_unchanged(self):
        d = MetricDelta("latency", before=100, after=100)
        assert d.direction_arrow == "—"

    def test_pct_str_formatting(self):
        d = MetricDelta("tool_calls", before=10, after=2)
        assert "▼" in d.pct_str
        assert "80.0" in d.pct_str


# ── diff_baselines core tests ──────────────────────────────────────────────────


class TestDiffBaselines:
    def test_basic_diff_no_spec(self):
        """diff_baselines works without a spec (heuristic correctness only)."""
        t1 = make_trace(tool_names=["retrieve_docs", "rewrite_question", "grade_artifacts"] * 3,
                        cost_usd=0.008, tokens_in=3000, tokens_out=1200, llm_calls=11)
        t2 = make_trace(tool_names=[], cost_usd=0.0001, tokens_in=120, tokens_out=60, llm_calls=1)

        b1 = trace_to_baseline(t1, "v1-broken", query="What's the weather?")
        b2 = trace_to_baseline(t2, "v2-fixed", query="What's the weather?")

        report = diff_baselines(b1, b2)

        assert report.agent == "test-agent"
        assert report.from_version == "v1-broken"
        assert report.to_version == "v2-fixed"
        assert report.query == "What's the weather?"

    def test_path_deltas_detects_tool_count_change(self):
        """Path deltas: 11 tool calls → 0 detected as a large drop."""
        t1 = make_trace(tool_names=["t"] * 11, cost_usd=0.008)
        t2 = make_trace(tool_names=[], cost_usd=0.0001)
        b1 = trace_to_baseline(t1, "v1")
        b2 = trace_to_baseline(t2, "v2")

        report = diff_baselines(b1, b2)

        tool_delta = next((d for d in report.path_deltas if d.label == "tool_calls"), None)
        assert tool_delta is not None
        assert tool_delta.before == 11
        assert tool_delta.after == 0
        assert tool_delta.direction_arrow == "▼"

    def test_cost_deltas_detected(self):
        """Cost deltas: 0.008 USD → 0.0001 USD captured."""
        t1 = make_trace(cost_usd=0.008)
        t2 = make_trace(cost_usd=0.0001)
        b1 = trace_to_baseline(t1, "v1")
        b2 = trace_to_baseline(t2, "v2")

        report = diff_baselines(b1, b2)

        cost_delta = next((d for d in report.cost_deltas if d.label == "cost_usd"), None)
        assert cost_delta is not None
        assert pytest.approx(cost_delta.before, abs=0.0001) == 0.008
        assert pytest.approx(cost_delta.after, abs=0.00001) == 0.0001

    def test_no_change_produces_minimal_deltas(self):
        """Identical traces → minimal or empty path/cost deltas."""
        t = make_trace(tool_names=["tool_a", "tool_b"], cost_usd=0.002, llm_calls=2)
        b1 = trace_to_baseline(t, "v1")
        b2 = trace_to_baseline(t, "v2")

        report = diff_baselines(b1, b2)

        # Tool count should not appear as a delta (same value)
        tool_deltas = [d for d in report.path_deltas if d.label == "tool_calls"]
        assert len(tool_deltas) == 0

        # Cost delta should not appear
        cost_deltas = [d for d in report.cost_deltas if d.label == "cost_usd"]
        assert len(cost_deltas) == 0

    def test_legacy_diffs_generated(self):
        """Legacy DiffResult list is generated for backward compatibility."""
        t1 = make_trace(tool_names=["retrieve_docs", "grade"], cost_usd=0.005)
        t2 = make_trace(tool_names=["direct_answer"], cost_usd=0.001)
        b1 = trace_to_baseline(t1, "v1")
        b2 = trace_to_baseline(t2, "v2")

        report = diff_baselines(b1, b2)

        assert isinstance(report.legacy_diffs, list)
        # TOOLS_CHANGED should be in the legacy diffs since tool set changed
        diff_types = [d.diff_type for d in report.legacy_diffs]
        from ciagent.models import DiffType
        assert DiffType.TOOLS_CHANGED in diff_types

    def test_missing_trace_key_handled(self):
        """Baseline files without 'trace' key should not crash."""
        b1 = {"version": "v1", "agent": "test", "query": "q"}
        b2 = {"version": "v2", "agent": "test", "query": "q"}
        # Should not raise
        report = diff_baselines(b1, b2)
        assert report.path_deltas == []
        assert report.cost_deltas == []

    def test_version_and_agent_extracted(self):
        """Agent and version are correctly extracted from baseline metadata."""
        t = make_trace()
        b1 = trace_to_baseline(t, "v1-broken", agent="rag-agent")
        b2 = trace_to_baseline(t, "v2-fixed", agent="rag-agent")

        report = diff_baselines(b1, b2)

        assert report.agent == "rag-agent"
        assert report.from_version == "v1-broken"
        assert report.to_version == "v2-fixed"


# ── DiffReport properties ──────────────────────────────────────────────────────


class TestDiffReportProperties:
    def test_has_regression_on_correctness_fail(self):
        """has_regression is True when correctness degrades pass→fail."""
        report = DiffReport(
            agent="a", from_version="v1", to_version="v2",
            correctness_delta={"before": "pass", "after": "fail", "changed": True},
        )
        assert report.has_regression is True

    def test_no_regression_pass_to_pass(self):
        """has_regression is False when correctness stays pass→pass."""
        report = DiffReport(
            agent="a", from_version="v1", to_version="v2",
            correctness_delta={"before": "pass", "after": "pass", "changed": False},
        )
        assert report.has_regression is False

    def test_no_regression_tool_count_small_increase(self):
        """A small tool count increase (3→4) is not flagged as regression."""
        report = DiffReport(
            agent="a", from_version="v1", to_version="v2",
            correctness_delta={"before": "pass", "after": "pass", "changed": False},
            path_deltas=[MetricDelta("tool_calls", before=3, after=4)],
        )
        assert report.has_regression is False

    def test_has_improvement_large_cost_drop(self):
        """has_improvement is True when cost drops significantly."""
        report = DiffReport(
            agent="a", from_version="v1", to_version="v2",
            correctness_delta={"before": "pass", "after": "pass", "changed": False},
            cost_deltas=[MetricDelta("cost_usd", before=0.008, after=0.0001)],
        )
        assert report.has_improvement is True

    def test_not_improvement_if_regression(self):
        """has_improvement is False if has_regression is True."""
        report = DiffReport(
            agent="a", from_version="v1", to_version="v2",
            correctness_delta={"before": "pass", "after": "fail", "changed": True},
            cost_deltas=[MetricDelta("cost_usd", before=0.008, after=0.0001)],
        )
        assert report.has_improvement is False


# ── Console output ─────────────────────────────────────────────────────────────


class TestDiffReportConsole:
    def test_summary_console_contains_versions(self):
        report = DiffReport(agent="rag-agent", from_version="v1-broken", to_version="v2-fixed")
        console = report.summary_console()
        assert "v1-broken" in console
        assert "v2-fixed" in console
        assert "rag-agent" in console

    def test_summary_console_shows_correctness_pass(self):
        report = DiffReport(
            agent="a", from_version="v1", to_version="v2",
            correctness_delta={"before": "pass", "after": "pass", "changed": False},
        )
        console = report.summary_console()
        assert "✅" in console
        assert "PASS" in console

    def test_summary_console_shows_correctness_fail(self):
        report = DiffReport(
            agent="a", from_version="v1", to_version="v2",
            correctness_delta={"before": "pass", "after": "fail", "changed": True},
        )
        console = report.summary_console()
        assert "❌" in console
        assert "FAIL" in console

    def test_summary_json_shape(self):
        report = DiffReport(
            agent="a", from_version="v1", to_version="v2",
            correctness_delta={"before": "pass", "after": "pass", "changed": False},
            path_deltas=[MetricDelta("tool_calls", 11, 0)],
            cost_deltas=[MetricDelta("cost_usd", 0.008, 0.0001)],
        )
        j = report.summary_json()
        assert j["agent"] == "a"
        assert j["has_regression"] is False
        assert len(j["path"]) == 1
        assert len(j["cost"]) == 1
        assert j["path"][0]["metric"] == "tool_calls"
        assert j["path"][0]["before"] == 11
        assert j["path"][0]["after"] == 0


# ── _compute_path_deltas ───────────────────────────────────────────────────────


class TestComputePathDeltas:
    def test_tool_count_delta_zero_to_many(self):
        t1 = make_trace(tool_names=["a", "b", "c"] * 3)
        t2 = make_trace(tool_names=[])
        deltas = _compute_path_deltas(t2, t1)
        # After=0, Before=9
        tool_d = next(d for d in deltas if d.label == "tool_calls")
        assert tool_d.before == 0
        assert tool_d.after == 9

    def test_no_deltas_for_identical_traces(self):
        t = make_trace(tool_names=["a", "b"])
        deltas = _compute_path_deltas(t, t)
        tool_deltas = [d for d in deltas if d.label == "tool_calls"]
        assert tool_deltas == []  # Same count → no delta

    def test_llm_calls_included(self):
        t1 = make_trace(llm_calls=1)
        t2 = make_trace(llm_calls=5)
        deltas = _compute_path_deltas(t1, t2)
        llm_d = next((d for d in deltas if d.label == "llm_calls"), None)
        assert llm_d is not None
        assert llm_d.before == 1
        assert llm_d.after == 5


# ── _compute_cost_deltas ───────────────────────────────────────────────────────


class TestComputeCostDeltas:
    def test_cost_delta_captured(self):
        t1 = make_trace(cost_usd=0.008, tokens_in=3000, tokens_out=1200)
        t2 = make_trace(cost_usd=0.0001, tokens_in=80, tokens_out=40)
        deltas = _compute_cost_deltas(t1, t2)

        labels = {d.label for d in deltas}
        assert "cost_usd" in labels      # cost changed significantly
        assert "total_tokens" in labels  # token counts differ

    def test_no_delta_when_costs_equal(self):
        t = make_trace(cost_usd=0.002)
        deltas = _compute_cost_deltas(t, t)
        cost_d = [d for d in deltas if d.label == "cost_usd"]
        assert cost_d == []


# ── _extract_answer ────────────────────────────────────────────────────────────


class TestExtractAnswer:
    def test_extracts_string_output(self):
        t = make_trace(output="Hello AgentCI!")
        assert _extract_answer(t) == "Hello AgentCI!"

    def test_empty_trace_returns_empty_string(self):
        t = Trace(spans=[])
        assert _extract_answer(t) == ""

    def test_none_output_returns_empty_string(self):
        span = Span(name="agent", output_data=None)
        t = Trace(spans=[span])
        assert _extract_answer(t) == ""

    def test_non_string_output_converted(self):
        span = Span(name="agent", output_data={"key": "value"})
        t = Trace(spans=[span])
        result = _extract_answer(t)
        assert isinstance(result, str)
        assert "key" in result

    def test_metadata_fallback_when_output_none(self):
        """When span.output_data is None, fall back to trace.metadata['final_output']."""
        span = Span(name="agent", output_data=None)
        t = Trace(spans=[span], metadata={"final_output": "The answer from metadata"})
        assert _extract_answer(t) == "The answer from metadata"

    def test_metadata_takes_priority_over_span(self):
        """trace.metadata['final_output'] should be preferred over span output."""
        span = Span(name="agent", output_data="From span")
        t = Trace(spans=[span], metadata={"final_output": "From metadata"})
        assert _extract_answer(t) == "From metadata"

    def test_span_fallback_when_no_metadata(self):
        """span.output_data should be used when metadata has no final_output."""
        span = Span(name="agent", output_data="From span")
        t = Trace(spans=[span], metadata={})
        assert _extract_answer(t) == "From span"

    def test_no_spans_with_metadata_fallback(self):
        """Empty spans list should still check metadata."""
        t = Trace(spans=[], metadata={"final_output": "Fallback answer"})
        assert _extract_answer(t) == "Fallback answer"
