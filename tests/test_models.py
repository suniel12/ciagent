"""
Tests for data models.
"""
import json

import pytest

from ciagent.models import Span, SpanKind, Trace, ToolCall


def test_trace_initialization():
    trace = Trace(test_name="example")
    assert trace.test_name == "example"
    assert len(trace.spans) == 0


# ── Span.attributes (Milestone 3.1) ─────────────────────────────────────────


class TestSpanAttributes:
    def test_attributes_default_empty_dict(self):
        """Span.attributes defaults to {} (backward-compatible)."""
        span = Span(name="tool_span", kind=SpanKind.TOOL_CALL)
        assert span.attributes == {}

    def test_attributes_can_be_set(self):
        """Span.attributes can hold arbitrary key/value pairs."""
        span = Span(
            name="retrieve_docs",
            kind=SpanKind.TOOL_CALL,
            attributes={
                "tool.args": {"query": "How do I install AgentCI?"},
                "tool.result": "pip install agentci",
            },
        )
        assert span.attributes["tool.args"]["query"] == "How do I install AgentCI?"
        assert span.attributes["tool.result"] == "pip install agentci"

    def test_attributes_round_trip_json(self):
        """Span.attributes serializes and deserializes through JSON correctly."""
        span = Span(
            name="search_tool",
            kind=SpanKind.TOOL_CALL,
            attributes={
                "tool.args": {"query": "pricing"},
                "tool.result": {"results": ["plan A", "plan B"]},
            },
        )
        serialized = span.model_dump_json()
        restored = Span.model_validate_json(serialized)
        assert restored.attributes["tool.args"]["query"] == "pricing"
        assert restored.attributes["tool.result"]["results"] == ["plan A", "plan B"]

    def test_attributes_in_trace_round_trip(self):
        """Span.attributes survives Trace-level serialization."""
        span = Span(
            name="my_tool",
            kind=SpanKind.TOOL_CALL,
            attributes={"tool.args.search_query": "weather"},
        )
        trace = Trace(spans=[span])
        json_str = trace.model_dump_json()
        restored_trace = Trace.model_validate_json(json_str)
        assert restored_trace.spans[0].attributes["tool.args.search_query"] == "weather"

    def test_attributes_accepts_nested_structures(self):
        """Span.attributes can store nested dicts and lists."""
        span = Span(
            name="complex_tool",
            kind=SpanKind.TOOL_CALL,
            attributes={
                "tool.args": {
                    "filters": ["active", "premium"],
                    "pagination": {"page": 1, "size": 10},
                }
            },
        )
        assert span.attributes["tool.args"]["pagination"]["page"] == 1
        assert "active" in span.attributes["tool.args"]["filters"]


# ── Trace assertion helpers ──────────────────────────────────────────────────


def _make_trace_with_tools(*tool_names: str, cost_usd: float = 0.0, llm_calls: int = 0) -> Trace:
    """Helper: build a Trace with given tool calls in a single span."""
    span = Span(kind=SpanKind.AGENT)
    for name in tool_names:
        span.tool_calls.append(ToolCall(tool_name=name))
    trace = Trace(spans=[span])
    trace.total_cost_usd = cost_usd
    trace.total_llm_calls = llm_calls
    return trace


class TestTraceAssertionHelpers:
    def test_called_returns_true_when_tool_in_sequence(self):
        trace = _make_trace_with_tools("retrieve_docs", "grade_artifacts")
        assert trace.called("retrieve_docs") is True

    def test_called_returns_false_when_tool_absent(self):
        trace = _make_trace_with_tools("grade_artifacts")
        assert trace.called("retrieve_docs") is False

    def test_called_empty_trace(self):
        trace = Trace()
        assert trace.called("any_tool") is False

    def test_never_called_is_inverse_of_called_when_present(self):
        trace = _make_trace_with_tools("retrieve_docs")
        assert trace.never_called("retrieve_docs") is False

    def test_never_called_is_inverse_of_called_when_absent(self):
        trace = _make_trace_with_tools("grade_artifacts")
        assert trace.never_called("retrieve_docs") is True

    def test_loop_count_counts_single_occurrence(self):
        trace = _make_trace_with_tools("retrieve_docs")
        assert trace.loop_count("retrieve_docs") == 1

    def test_loop_count_counts_multiple_occurrences(self):
        trace = _make_trace_with_tools("rewrite_question", "retrieve_docs", "rewrite_question", "retrieve_docs")
        assert trace.loop_count("rewrite_question") == 2
        assert trace.loop_count("retrieve_docs") == 2

    def test_loop_count_zero_for_absent_tool(self):
        trace = _make_trace_with_tools("retrieve_docs")
        assert trace.loop_count("rewrite_question") == 0

    def test_cost_under_returns_true_when_below_threshold(self):
        trace = _make_trace_with_tools(cost_usd=0.005)
        assert trace.cost_under(0.01) is True

    def test_cost_under_returns_false_when_at_threshold(self):
        trace = _make_trace_with_tools(cost_usd=0.01)
        assert trace.cost_under(0.01) is False

    def test_cost_under_returns_false_when_above_threshold(self):
        trace = _make_trace_with_tools(cost_usd=0.02)
        assert trace.cost_under(0.01) is False

    def test_llm_calls_under_returns_true_when_below_count(self):
        trace = _make_trace_with_tools(llm_calls=3)
        assert trace.llm_calls_under(5) is True

    def test_llm_calls_under_returns_false_when_at_count(self):
        trace = _make_trace_with_tools(llm_calls=5)
        assert trace.llm_calls_under(5) is False

    def test_llm_calls_under_returns_false_when_above_count(self):
        trace = _make_trace_with_tools(llm_calls=7)
        assert trace.llm_calls_under(5) is False
