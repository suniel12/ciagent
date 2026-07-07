# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the F4 retrieval layer (layer 2.5) and its cross-feature wiring.

Contract under test (eng review 2026-07-05, binding): the layer reads
``ToolCall.result`` with explicit rules and SKIPs whenever it cannot read —
uncaptured, unparseable — never guesses; empty means None/[]/""/whitespace
(plus spec markers); ``facts_in_context`` is informational-only in v1;
retrieval never hard-fails.

ADR required tests live here:
  - empty-retrieval pass-on-refusal
  - facts_in_context SKIP when no correctness terms
  - min_results boundaries
"""

from __future__ import annotations

import json

from ciagent.engine.results import LayerResult, LayerStatus, QueryResult
from ciagent.engine.retrieval import (
    evaluate_retrieval,
    extract_source_set,
    is_empty_retrieval,
    retrieval_signature,
)
from ciagent.models import DiffType, Span, SpanKind, ToolCall, Trace
from ciagent.schema.spec_models import (
    AgentCISpec,
    CorrectnessSpec,
    GoldenQuery,
    RetrievalSpec,
    ScenarioSpec,
    TurnChecks,
)

RETRIEVER = "retrieve_docs"

DOCS = [
    {"source": "returns.md", "content": "Returns accepted within 30 days."},
    {"source": "pricing.md", "content": "The APR rate is 4.5% for new accounts."},
]


def rag_trace(
    answer: str = "You have 30 days to return items.",
    result=None,
    tool: str = RETRIEVER,
    extra_calls: list[ToolCall] | None = None,
    include_retriever: bool = True,
) -> Trace:
    tool_calls = list(extra_calls or [])
    if include_retriever:
        tool_calls.insert(
            0, ToolCall(tool_name=tool, arguments={"query": "q"}, result=result)
        )
    trace = Trace(
        agent_name="rag-agent",
        spans=[Span(kind=SpanKind.AGENT, name="rag-agent", tool_calls=tool_calls)],
    )
    trace.metadata["final_output"] = answer
    trace.compute_metrics()
    return trace


# ── Reading rules: empty / uncaptured / unparseable ────────────────────────────


class TestReadingRules:
    def test_empty_list_is_empty(self):
        assert is_empty_retrieval(rag_trace(result=[]), RetrievalSpec(tool=RETRIEVER)) is True

    def test_empty_string_and_whitespace_are_empty(self):
        assert is_empty_retrieval(rag_trace(result=""), RetrievalSpec(tool=RETRIEVER)) is True
        assert is_empty_retrieval(rag_trace(result="   \n "), RetrievalSpec(tool=RETRIEVER)) is True

    def test_empty_marker_string_is_empty(self):
        spec = RetrievalSpec(tool=RETRIEVER, empty_markers=["No results found"])
        assert is_empty_retrieval(rag_trace(result="no results found "), spec) is True

    def test_non_marker_text_is_not_empty(self):
        spec = RetrievalSpec(tool=RETRIEVER, empty_markers=["No results found"])
        assert is_empty_retrieval(rag_trace(result="one hit"), spec) is False

    def test_none_result_is_empty_when_adapter_captures_results(self):
        # Another tool call carries a result → the adapter captures; the
        # retriever's None is a real empty return.
        trace = rag_trace(
            result=None,
            extra_calls=[ToolCall(tool_name="other", result="something")],
        )
        assert is_empty_retrieval(trace, RetrievalSpec(tool=RETRIEVER)) is True

    def test_none_result_skips_when_nothing_in_trace_captured(self):
        # No tool call anywhere has a result → indistinguishable from an
        # adapter that doesn't capture → SKIP, never guess.
        trace = rag_trace(result=None)
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, forbid_empty=True))
        assert result.status == LayerStatus.SKIP
        assert "not captured" in result.messages[0]
        assert is_empty_retrieval(trace, RetrievalSpec(tool=RETRIEVER)) is None

    def test_retriever_not_called_skips(self):
        trace = rag_trace(include_retriever=False)
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, forbid_empty=True))
        assert result.status == LayerStatus.SKIP
        assert "was not called" in result.messages[0]

    def test_unparseable_for_list_hint_skips(self):
        trace = rag_trace(result="plain prose, not a list")
        spec = RetrievalSpec(tool=RETRIEVER, result_format="list", min_results=1)
        result = evaluate_retrieval(trace, spec)
        assert result.status == LayerStatus.SKIP
        assert "did not parse" in result.messages[0]

    def test_json_string_parses_under_list_hint(self):
        trace = rag_trace(result=json.dumps(DOCS))
        spec = RetrievalSpec(tool=RETRIEVER, result_format="list", min_results=2)
        assert evaluate_retrieval(trace, spec).status == LayerStatus.PASS

    def test_multiple_calls_any_nonempty_is_not_empty(self):
        trace = Trace(
            agent_name="a",
            spans=[Span(kind=SpanKind.AGENT, name="a", tool_calls=[
                ToolCall(tool_name=RETRIEVER, result=[]),
                ToolCall(tool_name=RETRIEVER, result=DOCS),
            ])],
        )
        assert is_empty_retrieval(trace, RetrievalSpec(tool=RETRIEVER)) is False


# ── forbid_empty (ADR: empty-retrieval pass-on-refusal) ────────────────────────


class TestForbidEmpty:
    def test_empty_retrieval_with_substantive_answer_warns(self):
        trace = rag_trace(answer="We do not sell smart thermostats.", result=[])
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, forbid_empty=True))
        assert result.status == LayerStatus.WARN
        assert "Ungrounded answer" in result.messages[0]

    def test_empty_retrieval_pass_on_refusal(self):
        # ADR required test: the agent refusing on empty retrieval is the
        # CORRECT behavior and must pass.
        trace = rag_trace(
            answer="I couldn't find any information about that in our docs.",
            result=[],
        )
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, forbid_empty=True))
        assert result.status == LayerStatus.PASS
        assert result.details["forbid_empty"]["refusal_detected"] is True

    def test_custom_refusal_markers_override_defaults(self):
        spec = RetrievalSpec(
            tool=RETRIEVER, forbid_empty=True, refusal_markers=["cannot help with that"],
        )
        trace = rag_trace(answer="Sorry, I cannot help with that today.", result=[])
        assert evaluate_retrieval(trace, spec).status == LayerStatus.PASS
        # default marker no longer applies once overridden
        trace2 = rag_trace(answer="I couldn't find anything.", result=[])
        assert evaluate_retrieval(trace2, spec).status == LayerStatus.WARN

    def test_empty_retrieval_with_empty_answer_passes(self):
        trace = rag_trace(answer="", result=[])
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, forbid_empty=True))
        assert result.status == LayerStatus.PASS

    def test_nonempty_retrieval_passes(self):
        trace = rag_trace(result=DOCS)
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, forbid_empty=True))
        assert result.status == LayerStatus.PASS


# ── min_results (ADR: boundaries) ──────────────────────────────────────────────


class TestMinResults:
    def test_exactly_at_minimum_passes(self):
        trace = rag_trace(result=DOCS)  # 2 items
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, min_results=2))
        assert result.status == LayerStatus.PASS

    def test_one_below_minimum_warns(self):
        trace = rag_trace(result=DOCS[:1])
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, min_results=2))
        assert result.status == LayerStatus.WARN
        assert "1 < min 2" in result.messages[0]

    def test_zero_minimum_passes_on_empty_list(self):
        trace = rag_trace(result=[])
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, min_results=0))
        assert result.status == LayerStatus.PASS

    def test_counts_sum_across_multiple_calls(self):
        trace = Trace(
            agent_name="a",
            spans=[Span(kind=SpanKind.AGENT, name="a", tool_calls=[
                ToolCall(tool_name=RETRIEVER, result=DOCS[:1]),
                ToolCall(tool_name=RETRIEVER, result=DOCS[1:]),
            ])],
        )
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, min_results=2))
        assert result.status == LayerStatus.PASS

    def test_non_list_result_skips_count_never_guesses(self):
        trace = rag_trace(result="a text blob of retrieved context")
        result = evaluate_retrieval(trace, RetrievalSpec(tool=RETRIEVER, min_results=1))
        # the count is unknowable — the check reports SKIP and the layer
        # (with nothing else evaluated) is SKIP, not PASS or WARN
        assert result.status == LayerStatus.SKIP
        assert result.details["min_results"] == {"skipped": "result is not a list"}


# ── expected_sources ───────────────────────────────────────────────────────────


class TestExpectedSources:
    def test_all_sources_found_passes(self):
        trace = rag_trace(result=DOCS)
        spec = RetrievalSpec(tool=RETRIEVER, expected_sources=["returns.md", "pricing.md"])
        assert evaluate_retrieval(trace, spec).status == LayerStatus.PASS

    def test_missing_source_warns(self):
        trace = rag_trace(result=DOCS[:1])
        spec = RetrievalSpec(tool=RETRIEVER, expected_sources=["pricing.md"])
        result = evaluate_retrieval(trace, spec)
        assert result.status == LayerStatus.WARN
        assert result.details["expected_sources"]["missing"] == ["pricing.md"]

    def test_match_is_case_insensitive_substring(self):
        trace = rag_trace(result=[{"source": "KB/Returns.MD", "content": "x"}])
        spec = RetrievalSpec(tool=RETRIEVER, expected_sources=["returns.md"])
        assert evaluate_retrieval(trace, spec).status == LayerStatus.PASS


# ── facts_in_context (ADR: SKIP when no correctness terms) ─────────────────────


class TestFactsInContext:
    def test_skip_when_no_correctness_terms(self):
        # ADR required test: nothing to cross-check → SKIP, never a guess.
        trace = rag_trace(result=DOCS)
        spec = RetrievalSpec(tool=RETRIEVER, facts_in_context=True)
        result = evaluate_retrieval(trace, spec, correctness=CorrectnessSpec())
        assert result.status == LayerStatus.SKIP
        assert "no expected_in_answer" in result.messages[0]

    def test_skip_when_correctness_absent(self):
        trace = rag_trace(result=DOCS)
        spec = RetrievalSpec(tool=RETRIEVER, facts_in_context=True)
        assert evaluate_retrieval(trace, spec, correctness=None).status == LayerStatus.SKIP

    def test_ungrounded_fact_is_informational_never_warns(self):
        # "informational-only in v1" is binding: reported, status unaffected.
        trace = rag_trace(result=DOCS)
        spec = RetrievalSpec(tool=RETRIEVER, facts_in_context=True)
        correctness = CorrectnessSpec(expected_in_answer=["99.9%"])  # not in DOCS
        result = evaluate_retrieval(trace, spec, correctness=correctness)
        assert result.status == LayerStatus.PASS
        assert result.details["facts_in_context"]["ungrounded"] == ["99.9%"]
        assert any("wrong reason" in m for m in result.messages)

    def test_grounded_facts_reported(self):
        trace = rag_trace(result=DOCS)
        spec = RetrievalSpec(tool=RETRIEVER, facts_in_context=True)
        correctness = CorrectnessSpec(
            expected_in_answer=["30 days"], any_expected_in_answer=["4.5%"],
        )
        result = evaluate_retrieval(trace, spec, correctness=correctness)
        assert result.status == LayerStatus.PASS
        assert result.details["facts_in_context"]["ungrounded"] == []


# ── Pipeline wiring: evaluate_query, QueryResult, scenario verdict ─────────────


class TestPipelineWiring:
    def test_evaluate_query_populates_retrieval_layer(self):
        from ciagent.engine.runner import evaluate_query

        gq = GoldenQuery(
            query="what is the return window?",
            correctness=CorrectnessSpec(expected_in_answer=["30 days"]),
            retrieval=RetrievalSpec(tool=RETRIEVER, forbid_empty=True, min_results=1),
        )
        result = evaluate_query(gq, rag_trace(result=DOCS))
        assert result.retrieval.status == LayerStatus.PASS
        assert not result.hard_fail

    def test_retrieval_warn_is_soft_never_hard_fail(self):
        from ciagent.engine.runner import evaluate_query

        gq = GoldenQuery(
            query="q",
            retrieval=RetrievalSpec(tool=RETRIEVER, forbid_empty=True),
        )
        result = evaluate_query(gq, rag_trace(answer="Confident answer.", result=[]))
        assert result.retrieval.status == LayerStatus.WARN
        assert not result.hard_fail
        assert result.has_warnings

    def test_query_result_default_retrieval_is_skip(self):
        # Pre-F4 construction sites omit retrieval — must stay valid.
        qr = QueryResult(
            query="q",
            correctness=LayerResult(status=LayerStatus.PASS),
            path=LayerResult(status=LayerStatus.SKIP),
            cost=LayerResult(status=LayerStatus.SKIP),
        )
        assert qr.retrieval.status == LayerStatus.SKIP
        assert not qr.has_warnings

    def test_scenario_verdict_includes_retrieval(self):
        from ciagent.engine.simulate import run_scenario, scenario_verdict

        scenario = ScenarioSpec(
            name="rag-check",
            turns=["what is the return window?"],
            outcome=TurnChecks(
                retrieval=RetrievalSpec(tool=RETRIEVER, forbid_empty=True),
            ),
        )
        result = run_scenario(scenario, lambda messages: rag_trace(result=DOCS))
        verdict = scenario_verdict(result)
        assert verdict["outcome"]["retrieval"]["status"] == "pass"
        # byte-determinism: same run twice serializes identically
        result2 = run_scenario(scenario, lambda messages: rag_trace(result=DOCS))
        assert json.dumps(scenario_verdict(result2)) == json.dumps(verdict)

    def test_serialize_result_includes_retrieval(self):
        from ciagent.engine.reporter import _serialize_result

        qr = QueryResult(
            query="q",
            correctness=LayerResult(status=LayerStatus.PASS),
            path=LayerResult(status=LayerStatus.SKIP),
            cost=LayerResult(status=LayerStatus.SKIP),
            retrieval=LayerResult(status=LayerStatus.WARN, messages=["m"]),
        )
        data = _serialize_result(qr)
        assert data["retrieval"]["status"] == "warn"
        assert data["has_warnings"] is True


# ── Mock runner synthesis (zero-key path) ──────────────────────────────────────


class TestMockSynthesis:
    def test_mock_run_satisfies_retrieval_spec(self):
        from ciagent.engine.mock_runner import mock_run

        gq = GoldenQuery(
            query="return window?",
            correctness=CorrectnessSpec(expected_in_answer=["30 days"]),
            retrieval=RetrievalSpec(
                tool=RETRIEVER,
                forbid_empty=True,
                min_results=3,
                expected_sources=["returns.md"],
                facts_in_context=True,
            ),
        )
        trace = mock_run(gq.query, gq.model_dump())
        result = evaluate_retrieval(
            trace, gq.retrieval, correctness=gq.correctness,
            answer=trace.metadata["final_output"],
        )
        assert result.status == LayerStatus.PASS
        assert result.details["min_results"]["actual"] >= 3
        assert result.details["facts_in_context"]["ungrounded"] == []

    def test_mock_reuses_expected_tools_retriever_call(self):
        from ciagent.engine.mock_runner import mock_run

        gq = GoldenQuery(
            query="q",
            path={"expected_tools": [RETRIEVER]},
            retrieval=RetrievalSpec(tool=RETRIEVER, forbid_empty=True),
        )
        trace = mock_run(gq.query, gq.model_dump())
        calls = [tc for s in trace.spans for tc in s.tool_calls if tc.tool_name == RETRIEVER]
        assert len(calls) == 1  # no duplicate retriever call added
        assert calls[0].result  # and it now carries a synthesized result

    def test_mock_text_format_synthesizes_text(self):
        from ciagent.engine.mock_runner import mock_run

        gq = GoldenQuery(
            query="q",
            retrieval=RetrievalSpec(
                tool=RETRIEVER, forbid_empty=True, result_format="text",
            ),
        )
        trace = mock_run(gq.query, gq.model_dump())
        spec = gq.retrieval
        assert evaluate_retrieval(trace, spec).status == LayerStatus.PASS

    def test_mock_conversation_outcome_retrieval_satisfied(self):
        from ciagent.engine.mock_runner import mock_conversation_runner
        from ciagent.engine.simulate import run_scenario

        scenario = ScenarioSpec(
            name="rag-scenario",
            turns=["hi", "what is the return window?"],
            outcome=TurnChecks(
                retrieval=RetrievalSpec(
                    tool=RETRIEVER, forbid_empty=True, min_results=2,
                ),
            ),
        )
        result = run_scenario(scenario, mock_conversation_runner(scenario))
        assert result.outcome is not None
        assert result.outcome.retrieval.status == LayerStatus.PASS


# ── Stability: retrieval-variance flip source ──────────────────────────────────


def _stability_result(passed: bool, retrieval_result, answer: str) -> QueryResult:
    trace = rag_trace(answer=answer, result=retrieval_result)
    status = LayerStatus.PASS if passed else LayerStatus.FAIL
    return QueryResult(
        query="q",
        correctness=LayerResult(
            status=status,
            details={"expected_in_answer": {"all_found": passed}},
        ),
        path=LayerResult(status=LayerStatus.SKIP),
        cost=LayerResult(status=LayerStatus.SKIP),
        trace=trace,
    )


class TestRetrievalVariance:
    def _spec(self) -> AgentCISpec:
        return AgentCISpec(agent="a", queries=[GoldenQuery(
            query="q",
            correctness=CorrectnessSpec(expected_in_answer=["30 days"]),
            retrieval=RetrievalSpec(tool=RETRIEVER, forbid_empty=True),
        )])

    def test_same_tools_different_retrieval_is_retrieval_variance(self):
        from ciagent.engine.stability import FlipSource, build_stability_report

        runs = [
            [_stability_result(True, DOCS, "You have 30 days.")],
            [_stability_result(False, [], "We don't accept returns.")],
        ]
        report = build_stability_report(self._spec(), runs)
        q = report.queries[0]
        assert q.flipped
        assert q.flip_source == FlipSource.RETRIEVAL_VARIANCE

    def test_uncaptured_retrieval_never_attributes_to_retriever(self):
        from ciagent.engine.stability import FlipSource, build_stability_report

        # result=None with nothing captured anywhere → signature None →
        # attribution falls through to agent-variance (fail closed)
        runs = [
            [_stability_result(True, None, "You have 30 days.")],
            [_stability_result(False, None, "We don't accept returns.")],
        ]
        report = build_stability_report(self._spec(), runs)
        q = report.queries[0]
        assert q.flip_source == FlipSource.AGENT_VARIANCE

    def test_no_retrieval_spec_keeps_agent_variance(self):
        from ciagent.engine.stability import FlipSource, build_stability_report

        spec = AgentCISpec(agent="a", queries=[GoldenQuery(
            query="q", correctness=CorrectnessSpec(expected_in_answer=["30 days"]),
        )])
        runs = [
            [_stability_result(True, DOCS, "You have 30 days.")],
            [_stability_result(False, [], "We don't accept returns.")],
        ]
        report = build_stability_report(spec, runs)
        assert report.queries[0].flip_source == FlipSource.AGENT_VARIANCE

    def test_scenario_flip_attributes_retrieval_variance(self):
        from ciagent.engine.simulate import ScenarioResult, TurnResult
        from ciagent.engine.stability import FlipSource, build_scenario_stability

        scenario = ScenarioSpec(
            name="s",
            turns=["what is the return window?"],
            outcome=TurnChecks(
                retrieval=RetrievalSpec(tool=RETRIEVER, forbid_empty=True),
            ),
        )

        def scenario_run(passed: bool, retrieval_result, answer: str) -> ScenarioResult:
            qr = _stability_result(passed, retrieval_result, answer)
            return ScenarioResult(
                scenario=scenario,
                turns=[TurnResult(turn_index=0, user_message=scenario.turns[0], trace=qr.trace)],
                outcome=qr,
                termination="scripted-turns-exhausted",
            )

        runs = [
            [scenario_run(True, DOCS, "30 days.")],
            [scenario_run(False, [], "No returns.")],
        ]
        records = build_scenario_stability(runs)
        assert records[0].flipped
        assert records[0].flip_source == FlipSource.RETRIEVAL_VARIANCE


# ── Judge audit: judged against empty retrieval ────────────────────────────────


class TestJudgeAuditEmptyRetrieval:
    def _spec(self) -> AgentCISpec:
        from ciagent.schema.spec_models import JudgeRubric

        return AgentCISpec(agent="a", queries=[GoldenQuery(
            query="do you sell thermostats?",
            correctness=CorrectnessSpec(llm_judge=[JudgeRubric(rule="is it right?")]),
            retrieval=RetrievalSpec(tool=RETRIEVER),
        )])

    def test_empty_retrieval_row_reported(self):
        from ciagent.engine.judge_audit import run_judge_audit

        judge_fn = lambda **kw: {"passed": True, "rationale": "looks fine"}  # noqa: E731
        report = run_judge_audit(
            self._spec(),
            answers={"do you sell thermostats?": "We do not sell those."},
            repeats=1,
            judge_fn=judge_fn,
            retrieval_flags={"do you sell thermostats?": True},
        )
        assert len(report.empty_retrieval_judged) == 1
        assert report.queries[0].judged_against_empty_retrieval

    def test_unknown_flag_is_never_reported_as_empty(self):
        from ciagent.engine.judge_audit import run_judge_audit

        judge_fn = lambda **kw: {"passed": True, "rationale": ""}  # noqa: E731
        report = run_judge_audit(
            self._spec(),
            answers={"do you sell thermostats?": "We do not sell those."},
            repeats=1,
            judge_fn=judge_fn,
            retrieval_flags={"do you sell thermostats?": None},
        )
        assert report.empty_retrieval_judged == []

    def test_load_retrieval_flags_from_baselines(self, tmp_path):
        from ciagent.engine.judge_audit import load_retrieval_flags_from_baselines

        spec = self._spec()
        query = spec.queries[0].query
        trace = rag_trace(answer="We do not sell those.", result=[])
        trace.test_name = query
        baseline = {"agent": "a", "query": query, "trace": json.loads(trace.model_dump_json())}
        (tmp_path / "v1.json").write_text(json.dumps(baseline), encoding="utf-8")

        flags = load_retrieval_flags_from_baselines(str(tmp_path), spec)
        assert flags[query] is True

    def test_serialization_includes_empty_retrieval_count(self):
        from ciagent.engine.judge_audit import run_judge_audit
        from ciagent.engine.reporter import _serialize_judge_audit

        judge_fn = lambda **kw: {"passed": True, "rationale": ""}  # noqa: E731
        report = run_judge_audit(
            self._spec(),
            answers={"do you sell thermostats?": "We do not sell those."},
            repeats=1,
            judge_fn=judge_fn,
            retrieval_flags={"do you sell thermostats?": True},
        )
        data = _serialize_judge_audit(report)
        assert data["judged_against_empty_retrieval"] == 1
        assert data["queries"][0]["retrieval_empty"] is True


# ── Diff engine: RETRIEVAL_CHANGED ─────────────────────────────────────────────


class TestRetrievalChangedDiff:
    def test_changed_source_set_emits_retrieval_changed(self):
        from ciagent.diff_engine import diff_traces

        golden = rag_trace(result=DOCS)
        current = rag_trace(result=[{"source": "shipping.md", "content": "x"}])
        diffs = diff_traces(current, golden, retriever_tool=RETRIEVER)
        kinds = [d.diff_type for d in diffs]
        assert DiffType.RETRIEVAL_CHANGED in kinds
        d = next(d for d in diffs if d.diff_type == DiffType.RETRIEVAL_CHANGED)
        assert d.severity == "warning"
        assert "shipping.md" in d.details["added"]

    def test_same_source_set_emits_nothing(self):
        from ciagent.diff_engine import diff_traces

        golden = rag_trace(result=DOCS)
        current = rag_trace(result=list(DOCS))
        diffs = diff_traces(current, golden, retriever_tool=RETRIEVER)
        assert DiffType.RETRIEVAL_CHANGED not in [d.diff_type for d in diffs]

    def test_without_retriever_tool_emits_nothing(self):
        from ciagent.diff_engine import diff_traces

        golden = rag_trace(result=DOCS)
        current = rag_trace(result=[{"source": "shipping.md", "content": "x"}])
        diffs = diff_traces(current, golden)
        assert DiffType.RETRIEVAL_CHANGED not in [d.diff_type for d in diffs]

    def test_unextractable_sources_emit_nothing_fail_closed(self):
        from ciagent.diff_engine import diff_traces

        golden = rag_trace(result="text blob one")
        current = rag_trace(result="different text blob")
        diffs = diff_traces(current, golden, retriever_tool=RETRIEVER)
        assert DiffType.RETRIEVAL_CHANGED not in [d.diff_type for d in diffs]

    def test_extract_source_set(self):
        assert extract_source_set(rag_trace(result=DOCS), RETRIEVER) == {
            "returns.md", "pricing.md",
        }
        assert extract_source_set(rag_trace(result="prose"), RETRIEVER) is None
        assert extract_source_set(rag_trace(include_retriever=False), RETRIEVER) is None


# ── Signatures (stability input) ───────────────────────────────────────────────


class TestRetrievalSignature:
    def test_signature_stable_and_discriminating(self):
        a1 = retrieval_signature(rag_trace(result=DOCS), RETRIEVER)
        a2 = retrieval_signature(rag_trace(result=list(DOCS)), RETRIEVER)
        b = retrieval_signature(rag_trace(result=[]), RETRIEVER)
        assert a1 == a2
        assert a1 != b

    def test_signature_none_when_not_called_or_uncaptured(self):
        assert retrieval_signature(rag_trace(include_retriever=False), RETRIEVER) is None
        assert retrieval_signature(rag_trace(result=None), RETRIEVER) is None


# ── Adapter capture ────────────────────────────────────────────────────────────


class _FakeAIMessage:
    type = "ai"

    def __init__(self, tool_calls, content=""):
        self.tool_calls = tool_calls
        self.content = content


class _FakeToolMessage:
    type = "tool"
    tool_calls = None

    def __init__(self, tool_call_id, content):
        self.tool_call_id = tool_call_id
        self.content = content


class TestAdapterCapture:
    def test_langgraph_parse_state_pairs_tool_results(self):
        from ciagent.adapters.langgraph import LangGraphAdapter

        state = {"messages": [
            _FakeAIMessage([{"name": RETRIEVER, "args": {"q": "x"}, "id": "call_1"}]),
            _FakeToolMessage("call_1", json.dumps(DOCS)),
            _FakeAIMessage([], content="You have 30 days."),
        ]}
        trace = LangGraphAdapter().parse_state(state)
        calls = [tc for s in trace.spans for tc in s.tool_calls if tc.tool_name == RETRIEVER]
        assert calls[0].result == json.dumps(DOCS)

    def test_langgraph_unpaired_call_keeps_none(self):
        from ciagent.adapters.langgraph import LangGraphAdapter

        state = {"messages": [
            _FakeAIMessage([{"name": RETRIEVER, "args": {}, "id": "call_1"}]),
        ]}
        trace = LangGraphAdapter().parse_state(state)
        calls = [tc for s in trace.spans for tc in s.tool_calls]
        assert calls[0].result is None

    def test_attach_langgraph_state_pairs_tool_results(self):
        from ciagent.capture import TraceContext

        state = {"messages": [
            _FakeAIMessage([{"name": RETRIEVER, "args": {"q": "x"}, "id": "call_1"}]),
            _FakeToolMessage("call_1", "retrieved chunk text"),
        ]}
        with TraceContext(agent_name="a") as ctx:
            ctx.attach_langgraph_state(state)
        calls = [
            tc for s in ctx.trace.spans for tc in s.tool_calls
            if tc.tool_name == RETRIEVER
        ]
        assert calls[0].result == "retrieved chunk text"

    def test_openai_backfill_from_role_tool_message(self):
        from ciagent.capture import TraceContext

        ctx = TraceContext(agent_name="a")
        tc = ToolCall(tool_name=RETRIEVER, arguments={})
        ctx._pending_tool_results["call_1"] = tc
        ctx._backfill_openai_tool_results([
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(DOCS)},
        ])
        assert tc.result == json.dumps(DOCS)
        assert "call_1" not in ctx._pending_tool_results

    def test_anthropic_backfill_from_tool_result_block(self):
        from ciagent.capture import TraceContext

        ctx = TraceContext(agent_name="a")
        tc = ToolCall(tool_name=RETRIEVER, arguments={})
        ctx._pending_tool_results["tu_1"] = tc
        ctx._backfill_anthropic_tool_results([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": [
                    {"type": "text", "text": "retrieved chunk"},
                ]},
            ]},
        ])
        assert tc.result == "retrieved chunk"

    def test_tool_result_content_keeps_structured_payloads_raw(self):
        from ciagent.capture import _tool_result_content

        assert _tool_result_content("plain") == "plain"
        assert _tool_result_content(DOCS) == DOCS  # not text blocks → raw
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _tool_result_content(blocks) == "a\nb"
