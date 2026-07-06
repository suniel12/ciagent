"""
Unit tests for the Span Assertions Engine.

All LLM judge calls are mocked — no real API calls in this file.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ciagent.engine.span_assertions import (
    _resolve_field,
    _select_spans,
    evaluate_span_assertions,
)
from ciagent.engine.results import LayerStatus
from ciagent.models import Span, SpanKind, Trace, ToolCall
from ciagent.schema.spec_models import (
    SpanAssert,
    SpanAssertionSpec,
    SpanAssertType,
    SpanKindSelector,
    SpanSelector,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_tool_span(
    name: str = "retrieve_docs",
    output_data: str = "retrieved content",
    attributes: dict | None = None,
    tool_calls: list | None = None,
) -> Span:
    return Span(
        kind=SpanKind.TOOL_CALL,
        name=name,
        output_data=output_data,
        attributes=attributes or {},
        tool_calls=tool_calls or [],
    )


def make_trace(*spans: Span) -> Trace:
    trace = Trace(spans=list(spans))
    trace.compute_metrics()
    return trace


def make_selector(kind: SpanKindSelector = SpanKindSelector.TOOL, name: str = "retrieve_docs") -> SpanSelector:
    return SpanSelector(kind=kind, name=name)


def make_assert(
    type: SpanAssertType = SpanAssertType.CONTAINS,
    field: str = "output_data",
    value: str | None = "AgentCI",
    rule: str | None = None,
    threshold: float = 0.8,
) -> SpanAssert:
    return SpanAssert(type=type, field=field, value=value, rule=rule, threshold=threshold)


def make_spec(
    selector: SpanSelector | None = None,
    asserts: list[SpanAssert] | None = None,
) -> SpanAssertionSpec:
    return SpanAssertionSpec(
        selector=selector or make_selector(),
        asserts=asserts or [make_assert()],
    )


def judge_pass() -> dict:
    return {"passed": True, "score": 4, "label": "pass", "rationale": "Good"}


def judge_fail() -> dict:
    return {"passed": False, "score": 2, "label": "fail", "rationale": "Poor"}


# ── evaluate_span_assertions: empty spec ──────────────────────────────────────


class TestEmptySpec:
    def test_empty_spec_list_returns_pass(self):
        trace = make_trace(make_tool_span())
        result = evaluate_span_assertions([], trace)
        assert result.status == LayerStatus.PASS


# ── No matching span → FAIL ───────────────────────────────────────────────────


class TestNoMatchingSpan:
    def test_no_span_of_correct_name_fails(self):
        span = make_tool_span(name="other_tool")
        trace = make_trace(span)
        spec = [make_spec(selector=make_selector(name="retrieve_docs"))]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.FAIL
        assert any("No span matched" in m for m in result.messages)

    def test_no_span_of_correct_kind_fails(self):
        # Span is TOOL_CALL but we're looking for NODE
        span = make_tool_span(name="my_node")
        trace = make_trace(span)
        spec = [make_spec(selector=make_selector(kind=SpanKindSelector.NODE, name="my_node"))]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.FAIL
        assert any("No span matched" in m for m in result.messages)

    def test_error_message_includes_selector_info(self):
        trace = make_trace()
        spec = [make_spec(selector=make_selector(kind=SpanKindSelector.TOOL, name="missing_tool"))]
        result = evaluate_span_assertions(spec, trace)
        assert any("TOOL:missing_tool" in m for m in result.messages)


# ── CONTAINS assertion ─────────────────────────────────────────────────────────


class TestContainsAssertion:
    def test_value_found_in_output_data_passes(self):
        span = make_tool_span(output_data="AgentCI is great")
        trace = make_trace(span)
        spec = [make_spec(asserts=[make_assert(type=SpanAssertType.CONTAINS, field="output_data", value="AgentCI")])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.PASS

    def test_value_not_found_fails(self):
        span = make_tool_span(output_data="something else")
        trace = make_trace(span)
        spec = [make_spec(asserts=[make_assert(type=SpanAssertType.CONTAINS, field="output_data", value="AgentCI")])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.FAIL
        assert any("AgentCI" in m for m in result.messages)

    def test_contains_in_attributes_field(self):
        span = make_tool_span(
            attributes={"tool.args": {"query": "How to install AgentCI"}}
        )
        trace = make_trace(span)
        sa = make_assert(
            type=SpanAssertType.CONTAINS,
            field="attributes.tool.args",
            value="install",
        )
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.PASS


# ── NOT_CONTAINS assertion ────────────────────────────────────────────────────


class TestNotContainsAssertion:
    def test_value_absent_passes(self):
        span = make_tool_span(output_data="clean output")
        trace = make_trace(span)
        sa = make_assert(type=SpanAssertType.NOT_CONTAINS, field="output_data", value="error")
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.PASS

    def test_value_present_fails(self):
        span = make_tool_span(output_data="an error occurred")
        trace = make_trace(span)
        sa = make_assert(type=SpanAssertType.NOT_CONTAINS, field="output_data", value="error")
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.FAIL


# ── EQUALS assertion ───────────────────────────────────────────────────────────


class TestEqualsAssertion:
    def test_exact_match_passes(self):
        span = make_tool_span(output_data="exact value")
        trace = make_trace(span)
        sa = make_assert(type=SpanAssertType.EQUALS, field="output_data", value="exact value")
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.PASS

    def test_partial_match_fails(self):
        span = make_tool_span(output_data="exact value with extras")
        trace = make_trace(span)
        sa = make_assert(type=SpanAssertType.EQUALS, field="output_data", value="exact value")
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.FAIL


# ── REGEX assertion ────────────────────────────────────────────────────────────


class TestRegexAssertion:
    def test_regex_matches_passes(self):
        span = make_tool_span(output_data="version 2.3.1")
        trace = make_trace(span)
        sa = make_assert(
            type=SpanAssertType.REGEX,
            field="output_data",
            value=r"\d+\.\d+\.\d+",
        )
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.PASS

    def test_regex_no_match_fails(self):
        span = make_tool_span(output_data="no version here")
        trace = make_trace(span)
        sa = make_assert(
            type=SpanAssertType.REGEX,
            field="output_data",
            value=r"\d+\.\d+\.\d+",
        )
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.FAIL


# ── LLM_JUDGE assertion ────────────────────────────────────────────────────────


class TestLLMJudgeAssertion:
    def test_judge_pass_returns_pass(self):
        span = make_tool_span(output_data="relevant content about installation")
        trace = make_trace(span)
        sa = make_assert(
            type=SpanAssertType.LLM_JUDGE,
            field="output_data",
            rule="Content is relevant to the user's query",
            threshold=0.7,
        )
        spec = [make_spec(asserts=[sa])]
        with patch("ciagent.engine.judge.run_judge", return_value=judge_pass()):
            result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.PASS

    def test_judge_fail_returns_fail(self):
        span = make_tool_span(output_data="irrelevant content")
        trace = make_trace(span)
        sa = make_assert(
            type=SpanAssertType.LLM_JUDGE,
            field="output_data",
            rule="Content is relevant to the user's query",
            threshold=0.7,
        )
        spec = [make_spec(asserts=[sa])]
        with patch("ciagent.engine.judge.run_judge", return_value=judge_fail()):
            result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.FAIL

    def test_judge_called_with_field_value(self):
        """Judge receives the actual field value as the answer."""
        span = make_tool_span(output_data="field content to judge")
        trace = make_trace(span)
        sa = make_assert(
            type=SpanAssertType.LLM_JUDGE,
            field="output_data",
            rule="Content rule",
            threshold=0.7,
        )
        spec = [make_spec(asserts=[sa])]
        with patch("ciagent.engine.judge.run_judge", return_value=judge_pass()) as mock_judge:
            evaluate_span_assertions(spec, trace)
        call_kwargs = mock_judge.call_args
        assert call_kwargs[1]["answer"] == "field content to judge"

    def test_llm_judge_missing_rule_fails(self):
        """LLM_JUDGE without a rule immediately fails."""
        span = make_tool_span(output_data="some content")
        trace = make_trace(span)
        sa = SpanAssert(type=SpanAssertType.LLM_JUDGE, field="output_data", threshold=0.7)
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.FAIL
        assert any("rule" in m.lower() for m in result.messages)


# ── Dotted path resolution ─────────────────────────────────────────────────────


class TestFieldResolution:
    def test_output_data_direct(self):
        span = make_tool_span(output_data="direct output")
        value = _resolve_field(span, "output_data")
        assert value == "direct output"

    def test_attributes_flat_key(self):
        span = make_tool_span(attributes={"tool.args": {"query": "install"}})
        value = _resolve_field(span, "attributes.tool.args")
        assert value == {"query": "install"}

    def test_attributes_nested_dotted(self):
        span = make_tool_span(attributes={"tool": {"args": {"query": "install"}}})
        value = _resolve_field(span, "attributes.tool.args.query")
        assert value == "install"

    def test_metadata_key(self):
        span = Span(
            kind=SpanKind.TOOL_CALL,
            name="test",
            metadata={"handoffs": ["billing", "technical"]},
        )
        value = _resolve_field(span, "metadata.handoffs")
        assert value == ["billing", "technical"]

    def test_missing_field_returns_none(self):
        span = make_tool_span()
        value = _resolve_field(span, "attributes.nonexistent.path")
        assert value is None

    def test_missing_field_in_assertion_fails(self):
        span = make_tool_span(attributes={})
        trace = make_trace(span)
        sa = make_assert(field="attributes.nonexistent.path", value="anything")
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.FAIL
        assert any("not found" in m.lower() for m in result.messages)


# ── Multiple spans matched ─────────────────────────────────────────────────────


class TestMultipleSpansMatched:
    def test_all_spans_must_pass(self):
        """When 2 matching spans exist, BOTH must pass the assertion."""
        span1 = make_tool_span(name="retrieve_docs", output_data="AgentCI content")
        span2 = make_tool_span(name="retrieve_docs", output_data="unrelated content")
        trace = make_trace(span1, span2)
        sa = make_assert(type=SpanAssertType.CONTAINS, field="output_data", value="AgentCI")
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        # span2 fails → overall FAIL
        assert result.status == LayerStatus.FAIL

    def test_all_spans_passing_returns_pass(self):
        """When all matched spans pass, result is PASS."""
        span1 = make_tool_span(name="retrieve_docs", output_data="AgentCI content 1")
        span2 = make_tool_span(name="retrieve_docs", output_data="AgentCI content 2")
        trace = make_trace(span1, span2)
        sa = make_assert(type=SpanAssertType.CONTAINS, field="output_data", value="AgentCI")
        spec = [make_spec(asserts=[sa])]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.PASS


# ── Multiple assertions on same span ──────────────────────────────────────────


class TestMultipleAssertionsOnSpan:
    def test_all_assertions_must_pass(self):
        """Multiple asserts on same selector: ALL must pass."""
        span = make_tool_span(output_data="AgentCI install error")
        trace = make_trace(span)
        spec = [
            make_spec(
                asserts=[
                    make_assert(type=SpanAssertType.CONTAINS, field="output_data", value="AgentCI"),
                    make_assert(type=SpanAssertType.NOT_CONTAINS, field="output_data", value="error"),
                ]
            )
        ]
        result = evaluate_span_assertions(spec, trace)
        # Second assertion fails (output contains "error")
        assert result.status == LayerStatus.FAIL

    def test_all_assertions_passing(self):
        span = make_tool_span(output_data="AgentCI install guide")
        trace = make_trace(span)
        spec = [
            make_spec(
                asserts=[
                    make_assert(type=SpanAssertType.CONTAINS, field="output_data", value="AgentCI"),
                    make_assert(type=SpanAssertType.NOT_CONTAINS, field="output_data", value="error"),
                    make_assert(type=SpanAssertType.CONTAINS, field="output_data", value="install"),
                ]
            )
        ]
        result = evaluate_span_assertions(spec, trace)
        assert result.status == LayerStatus.PASS


# ── Details populated ──────────────────────────────────────────────────────────


class TestDetailsPopulated:
    def test_details_contain_span_assertions_key(self):
        span = make_tool_span(output_data="AgentCI")
        trace = make_trace(span)
        spec = [make_spec()]
        result = evaluate_span_assertions(spec, trace)
        assert "span_assertions" in result.details

    def test_details_contain_matched_spans_count(self):
        span = make_tool_span(output_data="AgentCI")
        trace = make_trace(span)
        spec = [make_spec()]
        result = evaluate_span_assertions(spec, trace)
        assertion_detail = result.details["span_assertions"][0]
        assert "matched_spans" in assertion_detail
        assert assertion_detail["matched_spans"] == 1


# ── Integration with evaluate_query ───────────────────────────────────────────


class TestRunnerIntegration:
    def test_span_assertion_failure_escalates_correctness(self):
        """A span assertion FAIL escalates to a correctness FAIL in evaluate_query."""
        from ciagent.engine.runner import evaluate_query
        from ciagent.schema.spec_models import GoldenQuery

        span = make_tool_span(name="retrieve_docs", output_data="no relevant content")
        trace = make_trace(span)

        query = GoldenQuery(
            query="How do I install?",
            span_assertions=[
                SpanAssertionSpec(
                    selector=SpanSelector(kind=SpanKindSelector.TOOL, name="retrieve_docs"),
                    asserts=[
                        SpanAssert(
                            type=SpanAssertType.CONTAINS,
                            field="output_data",
                            value="AgentCI",
                        )
                    ],
                )
            ],
        )
        result = evaluate_query(query, trace)
        assert result.hard_fail is True
        assert any("span" in m.lower() or "AgentCI" in m for m in result.correctness.messages)

    def test_span_assertion_pass_does_not_affect_existing_pass(self):
        """A passing span assertion keeps correctness PASS."""
        from ciagent.engine.runner import evaluate_query
        from ciagent.schema.spec_models import GoldenQuery

        span = make_tool_span(name="retrieve_docs", output_data="AgentCI documentation")
        trace = make_trace(span)

        query = GoldenQuery(
            query="How do I install?",
            span_assertions=[
                SpanAssertionSpec(
                    selector=SpanSelector(kind=SpanKindSelector.TOOL, name="retrieve_docs"),
                    asserts=[
                        SpanAssert(
                            type=SpanAssertType.CONTAINS,
                            field="output_data",
                            value="AgentCI",
                        )
                    ],
                )
            ],
        )
        result = evaluate_query(query, trace)
        assert result.hard_fail is False
