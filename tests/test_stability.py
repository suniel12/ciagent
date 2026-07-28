# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit + integration tests for the Stability Engine (multi-run flip attribution).

Attribution invariant under test: deterministic layers cannot flip on
identical output by construction, so identical answers + flipped verdict must
attribute to the judge; different answers attribute to the agent; ambiguous
paraphrases with a judge configured attribute to neither (mixed).
"""

from __future__ import annotations

import pytest

from ciagent.engine.results import LayerResult, LayerStatus, QueryResult
from ciagent.engine.stability import (
    FlipSource,
    build_stability_report,
    _min_pairwise_similarity,
)
from ciagent.models import Span, SpanKind, ToolCall, Trace
from ciagent.schema.spec_models import (
    CIAgentSpec,
    CorrectnessSpec,
    GoldenQuery,
    JudgeRubric,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_result(
    query: str,
    passed: bool,
    answer: str = "same answer",
    tools: tuple[str, ...] = (),
    det_details: dict | None = None,
    judge_passed: bool | None = None,
    judge_error: bool = False,
) -> QueryResult:
    tool_calls = [ToolCall(tool_name=t) for t in tools]
    trace = Trace(spans=[Span(kind=SpanKind.AGENT, name="agent", tool_calls=tool_calls)])
    trace.metadata["final_output"] = answer
    trace.compute_metrics()
    status = LayerStatus.PASS if passed else LayerStatus.FAIL
    details = dict(det_details or {})
    if judge_passed is not None or judge_error:
        entry: dict = {"passed": bool(judge_passed)}
        if judge_error:
            entry["error"] = "api down"
        details["judge_is it good?"] = entry
    return QueryResult(
        query=query,
        correctness=LayerResult(status=status, details=details),
        path=LayerResult(status=LayerStatus.SKIP),
        cost=LayerResult(status=LayerStatus.SKIP),
        trace=trace,
    )


def make_spec(*queries: GoldenQuery) -> CIAgentSpec:
    return CIAgentSpec(agent="stability-test", queries=list(queries))


def plain_query(text: str) -> GoldenQuery:
    return GoldenQuery(
        query=text,
        correctness=CorrectnessSpec(expected_in_answer=["answer"]),
    )


def judged_query(text: str) -> GoldenQuery:
    return GoldenQuery(
        query=text,
        correctness=CorrectnessSpec(llm_judge=[JudgeRubric(rule="is it good?")]),
    )


# ── Aggregation ────────────────────────────────────────────────────────────────


class TestAggregation:
    def test_stable_suite_no_flips(self):
        spec = make_spec(plain_query("q1"), plain_query("q2"))
        runs = [
            [make_result("q1", True), make_result("q2", True)],
            [make_result("q1", True), make_result("q2", True)],
        ]
        report = build_stability_report(spec, runs)
        assert report.verdict == "STABLE"
        assert report.is_stable
        assert report.flipped_queries == []
        assert report.per_run_passed == [2, 2]
        assert report.per_run_scores == [1.0, 1.0]

    def test_flip_detected(self):
        spec = make_spec(plain_query("q1"))
        runs = [
            [make_result("q1", True, answer="A")],
            [make_result("q1", False, answer="B")],
            [make_result("q1", True, answer="A")],
        ]
        report = build_stability_report(spec, runs)
        assert report.verdict == "FLAKY"
        assert len(report.flipped_queries) == 1
        q = report.flipped_queries[0]
        assert q.verdicts == [True, False, True]
        assert q.verdict_string == "✅❌✅"

    def test_consistent_failure_is_not_flaky(self):
        spec = make_spec(plain_query("q1"), plain_query("q2"))
        runs = [
            [make_result("q1", False), make_result("q2", True)],
            [make_result("q1", False), make_result("q2", True)],
        ]
        report = build_stability_report(spec, runs)
        assert report.verdict == "STABLE"
        assert len(report.consistent_failures) == 1
        assert report.consistent_failures[0].query == "q1"

    def test_query_missing_from_one_run_aggregates_over_present_runs(self):
        spec = make_spec(plain_query("q1"), plain_query("q2"))
        runs = [
            [make_result("q1", True), make_result("q2", True)],
            [make_result("q1", True)],  # q2's runner failed this run
        ]
        report = build_stability_report(spec, runs)
        q2 = next(q for q in report.queries if q.query == "q2")
        assert q2.runs == 1
        assert not q2.flipped


# ── pass@k / pass^k ────────────────────────────────────────────────────────────


class TestPassMetrics:
    def test_pass_rate_and_estimates(self):
        spec = make_spec(plain_query("q1"))
        runs = [
            [make_result("q1", True, answer="A")],
            [make_result("q1", False, answer="B")],
            [make_result("q1", True, answer="A")],
        ]
        report = build_stability_report(spec, runs)
        q = report.queries[0]
        assert q.pass_rate == pytest.approx(2 / 3)
        # p=2/3, k=3: pass@k = 1-(1/3)^3 ≈ 0.963, pass^k = (2/3)^3 ≈ 0.296
        assert q.pass_at_k == pytest.approx(0.963, abs=0.001)
        assert q.pass_pow_k == pytest.approx(0.296, abs=0.001)

    def test_all_pass_metrics_are_one(self):
        spec = make_spec(plain_query("q1"))
        runs = [[make_result("q1", True)], [make_result("q1", True)]]
        report = build_stability_report(spec, runs)
        q = report.queries[0]
        assert q.pass_rate == 1.0
        assert q.pass_at_k == 1.0
        assert q.pass_pow_k == 1.0


# ── Flip attribution ───────────────────────────────────────────────────────────


class TestFlipAttribution:
    def test_identical_answers_flipped_verdict_is_judge_flake(self):
        spec = make_spec(judged_query("q1"))
        runs = [
            [make_result("q1", True, answer="The rate is 4.5%", tools=("kb",))],
            [make_result("q1", False, answer="The rate is 4.5%", tools=("kb",))],
        ]
        report = build_stability_report(spec, runs)
        q = report.flipped_queries[0]
        assert q.flip_source == FlipSource.JUDGE_FLAKE

    def test_identical_answers_normalization_ignores_whitespace_and_case(self):
        spec = make_spec(judged_query("q1"))
        runs = [
            [make_result("q1", True, answer="The Rate is  4.5%")],
            [make_result("q1", False, answer="the rate is 4.5%")],
        ]
        report = build_stability_report(spec, runs)
        assert report.flipped_queries[0].flip_source == FlipSource.JUDGE_FLAKE

    def test_different_answers_is_agent_variance(self):
        spec = make_spec(plain_query("q1"))
        runs = [
            [make_result("q1", True, answer="The correct answer is 42.")],
            [make_result("q1", False, answer="I could not find that information anywhere.")],
        ]
        report = build_stability_report(spec, runs)
        q = report.flipped_queries[0]
        assert q.flip_source == FlipSource.AGENT_VARIANCE

    def test_near_identical_with_judge_is_mixed(self):
        spec = make_spec(judged_query("q1"))
        base = "our return window is 30 days from the date of delivery for all items"
        variant = "our return window is 30 days from the day of delivery for all items"
        runs = [
            [make_result("q1", True, answer=base)],
            [make_result("q1", False, answer=variant)],
        ]
        report = build_stability_report(spec, runs)
        q = report.flipped_queries[0]
        assert q.answer_similarity >= 0.9
        assert q.flip_source == FlipSource.MIXED

    def test_near_identical_without_judge_is_agent_variance(self):
        # No judge configured → a deterministic verdict flipped, so the output
        # difference caused it, however small. That IS agent variance.
        spec = make_spec(plain_query("q1"))
        base = "our return window is 30 days from the date of delivery for all items"
        variant = "our return window is 30 days from the day of delivery for all items"
        runs = [
            [make_result("q1", True, answer=base)],
            [make_result("q1", False, answer=variant)],
        ]
        report = build_stability_report(spec, runs)
        assert report.flipped_queries[0].flip_source == FlipSource.AGENT_VARIANCE

    def test_same_answer_different_tools_is_agent_variance(self):
        spec = make_spec(judged_query("q1"))
        runs = [
            [make_result("q1", True, answer="same", tools=("kb", "search"))],
            [make_result("q1", False, answer="same", tools=("search",))],
        ]
        report = build_stability_report(spec, runs)
        assert report.flipped_queries[0].flip_source == FlipSource.AGENT_VARIANCE

    def test_no_flip_no_attribution(self):
        spec = make_spec(plain_query("q1"))
        runs = [
            [make_result("q1", True, answer="A")],
            [make_result("q1", True, answer="B")],  # answer varies, verdict doesn't
        ]
        report = build_stability_report(spec, runs)
        q = report.queries[0]
        assert not q.flipped
        assert q.flip_source is None

    def test_judge_error_flip_is_infra_error(self):
        # A judge API failure counted as a fail must never read as judge-flake
        spec = make_spec(judged_query("q1"))
        runs = [
            [make_result("q1", True, answer="same", judge_passed=True)],
            [make_result("q1", False, answer="same", judge_passed=False, judge_error=True)],
        ]
        report = build_stability_report(spec, runs)
        q = report.flipped_queries[0]
        assert q.flip_source == FlipSource.INFRA_ERROR

    def test_paraphrased_answer_with_stable_checks_flipped_judge_is_judge_flake(self):
        # Answers differ a lot, but every deterministic check agreed across runs
        # and only the judge changed its mind — layer sub-verdicts must win over
        # answer-text comparison.
        det = {"any_expected_in_answer": {"any_found": True}}
        spec = make_spec(judged_query("q1"))
        runs = [
            [make_result("q1", True, answer="The rate is 4.5% APR on all standard cards.",
                         det_details=det, judge_passed=True)],
            [make_result("q1", False, answer="Standard cards carry a 4.5% annual rate.",
                         det_details=det, judge_passed=False)],
        ]
        report = build_stability_report(spec, runs)
        q = report.flipped_queries[0]
        assert q.answer_similarity < 0.9  # would previously mislabel agent-variance
        assert q.flip_source == FlipSource.JUDGE_FLAKE

    def test_det_outcome_change_beats_mixed(self):
        # Near-identical paraphrase + judge configured would be `mixed` — but the
        # deterministic check outcome itself changed, so the output caused it.
        base = "our return window is 30 days from the date of delivery for all items"
        variant = "our return window is 3O days from the date of delivery for all items"
        spec = make_spec(judged_query("q1"))
        runs = [
            [make_result("q1", True, answer=base,
                         det_details={"expected_in_answer": {"all_found": True}})],
            [make_result("q1", False, answer=variant,
                         det_details={"expected_in_answer": {"all_found": False}})],
        ]
        report = build_stability_report(spec, runs)
        assert report.flipped_queries[0].flip_source == FlipSource.AGENT_VARIANCE


# ── Duplicate + partial aggregation flags ──────────────────────────────────────


class TestRobustnessFlags:
    def test_duplicate_query_texts_flagged(self):
        spec = make_spec(plain_query("same text"), plain_query("same text"))
        runs = [[make_result("same text", True)]]
        report = build_stability_report(spec, runs)
        assert report.duplicate_queries == ["same text"]

    def test_partial_aggregation_flagged(self):
        spec = make_spec(plain_query("q1"), plain_query("q2"))
        runs = [
            [make_result("q1", True), make_result("q2", True)],
            [make_result("q1", True)],  # q2's runner failed in run 2
        ]
        report = build_stability_report(spec, runs)
        q2 = next(q for q in report.queries if q.query == "q2")
        assert q2.partial
        assert q2.runs == 1 and q2.expected_runs == 2
        assert report.partial_queries == [q2]

    def test_full_aggregation_not_partial(self):
        spec = make_spec(plain_query("q1"))
        runs = [[make_result("q1", True)], [make_result("q1", True)]]
        report = build_stability_report(spec, runs)
        assert not report.queries[0].partial


# ── Similarity helper ──────────────────────────────────────────────────────────


class TestSimilarity:
    def test_single_answer_is_fully_similar(self):
        assert _min_pairwise_similarity(["only one"]) == 1.0

    def test_identical_answers(self):
        assert _min_pairwise_similarity(["a b c", "a b c"]) == 1.0

    def test_disjoint_answers_low_similarity(self):
        sim = _min_pairwise_similarity(["totally different text", "qqqq zzzz 12345"])
        assert sim < 0.5


# ── CLI integration (mock mode, zero API keys) ─────────────────────────────────


class TestCLIStability:
    @pytest.fixture()
    def spec_file(self, tmp_path):
        spec = tmp_path / "ciagent_spec.yaml"
        spec.write_text(
            """
agent: stability-cli-test
queries:
  - query: "flaky one"
    correctness:
      expected_in_answer: ["documentation"]
  - query: "stable one"
    correctness:
      expected_in_answer: ["documentation"]
"""
        )
        return spec

    def _invoke(self, spec_file, args, env=None):
        from click.testing import CliRunner

        from ciagent.cli import cli

        runner = CliRunner()
        return runner.invoke(
            cli,
            ["test", "--config", str(spec_file), "--mock", "--yes", *args],
            env=env or {},
        )

    def test_stable_mock_runs(self, spec_file):
        result = self._invoke(spec_file, ["--runs", "3"])
        assert result.exit_code == 0, result.output
        assert "Stability Report" in result.output
        assert "STABLE" in result.output

    def test_flaky_mock_runs_flag_attribution_and_exit_zero(self, spec_file):
        # CIAGENT_MOCK_FLAKY breaks even-indexed queries on odd runs
        result = self._invoke(
            spec_file, ["--runs", "3"], env={"CIAGENT_MOCK_FLAKY": "1"},
        )
        assert result.exit_code == 0, result.output
        assert "FLAKY" in result.output
        assert "agent-variance" in result.output

    def test_fail_on_flaky_exits_one(self, spec_file):
        result = self._invoke(
            spec_file, ["--runs", "3", "--fail-on-flaky"],
            env={"CIAGENT_MOCK_FLAKY": "1"},
        )
        assert result.exit_code == 1, result.output

    def test_json_format_includes_stability_block(self, spec_file):
        import json

        result = self._invoke(
            spec_file, ["--runs", "2", "--format", "json"],
            env={"CIAGENT_MOCK_FLAKY": "1"},
        )
        payload = json.loads(result.output[result.output.index("{"):])
        assert payload["stability"]["runs"] == 2
        assert payload["stability"]["verdict"] == "FLAKY"
        assert "estimate" in payload["stability"]["estimate_note"]
        flipped = [q for q in payload["stability"]["queries"] if q["flipped"]]
        assert flipped and flipped[0]["flip_source"] == "agent-variance"
        # estimates are labeled as such in JSON; console shows observed facts only
        assert "pass_at_k_estimate" in flipped[0]
        assert "pass_at_k" not in {k for k in flipped[0] if k == "pass_at_k"}

    def test_single_run_has_no_stability_section(self, spec_file):
        result = self._invoke(spec_file, [])
        assert result.exit_code == 0, result.output
        assert "Stability Report" not in result.output


# ── Per-run answers (--runs N is only worth its cost if the answers survive) ───


class TestPerRunAnswers:
    """`--runs N` costs N times the API spend. Before this, the JSON output
    kept one answer per query and discarded the other N-1, so anyone grading
    externally (LLM-as-judge, oracle grading, human review, majority vote) paid
    for repeats they could not read. These tests pin the answers to the output.
    """

    def test_answers_align_with_verdicts(self):
        spec = make_spec(plain_query("q1"))
        runs = [
            [make_result("q1", True, answer="first answer")],
            [make_result("q1", False, answer="second answer")],
            [make_result("q1", True, answer="third answer")],
        ]
        q = build_stability_report(spec, runs).queries[0]
        assert q.answers == ["first answer", "second answer", "third answer"]
        assert q.verdicts == [True, False, True]
        # the failing run's answer is recoverable by index
        assert q.answers[q.verdicts.index(False)] == "second answer"

    def test_per_run_cost_attribution_is_index_aligned(self):
        spec = make_spec(plain_query("q1"))
        runs = [
            [make_result("q1", True, answer="a")],
            [make_result("q1", True, answer="b")],
        ]
        q = build_stability_report(spec, runs).queries[0]
        for series in (q.trace_ids, q.total_tokens, q.cost_usd, q.latency_ms):
            assert len(series) == len(q.verdicts)
        assert len(set(q.trace_ids)) == 2  # one trace per run, not reused

    def test_partial_query_answers_track_present_runs(self):
        spec = make_spec(plain_query("q1"), plain_query("q2"))
        runs = [
            [make_result("q1", True, answer="a"), make_result("q2", True, answer="b")],
            [make_result("q1", True, answer="c")],  # q2's adapter failed
        ]
        q2 = next(q for q in build_stability_report(spec, runs).queries
                  if q.query == "q2")
        assert q2.answers == ["b"]
        assert len(q2.answers) == len(q2.verdicts)

    def test_normalization_does_not_leak_into_stored_answers(self):
        # Attribution compares normalized text; the stored answers must stay
        # verbatim or an external grader sees a lowercased, reflowed answer.
        spec = make_spec(plain_query("q1"))
        runs = [
            [make_result("q1", True, answer="  Mixed   CASE answer\n")],
            [make_result("q1", True, answer="Mixed CASE answer")],
        ]
        q = build_stability_report(spec, runs).queries[0]
        assert q.answers[0] == "  Mixed   CASE answer\n"
        assert q.answer_similarity == 1.0  # still normalized for comparison


class TestPerRunAnswersJSON:
    """End-to-end through the CLI with a stub adapter that returns a
    distinguishable answer per call, so ordering is observable."""

    ANSWERS = {
        "alpha query": [
            "alpha refund answer one",
            "alpha refund answer two",
            "alpha refund answer three",
        ],
        "beta query": [
            "beta refund answer one",
            "beta declined answer two",  # missing the keyword: run 2 fails
            "beta refund answer three",
        ],
    }

    STUB = '''
import threading

from ciagent.models import LLMCall, Span, SpanKind, Trace

_LOCK = threading.Lock()
_CALLS: dict[str, int] = {}

ANSWERS = %r


def run(query: str) -> Trace:
    with _LOCK:
        index = _CALLS.get(query, 0)
        _CALLS[query] = index + 1
    span = Span(
        kind=SpanKind.AGENT,
        name="agent",
        llm_calls=[LLMCall(model="stub", tokens_in=5, tokens_out=7)],
    )
    trace = Trace(spans=[span])
    trace.metadata["final_output"] = ANSWERS[query][index]
    trace.compute_metrics()
    return trace
'''

    SPEC = """
agent: per-run-answers
adapter: "{module}:run"
queries:
  - query: "alpha query"
    correctness:
      expected_in_answer: ["refund"]
  - query: "beta query"
    correctness:
      expected_in_answer: ["refund"]
"""

    def _run(self, tmp_path, monkeypatch, module, args):
        import json
        import sys

        from click.testing import CliRunner

        from ciagent.cli import cli

        (tmp_path / f"{module}.py").write_text(self.STUB % self.ANSWERS)
        spec = tmp_path / "ciagent_spec.yaml"
        spec.write_text(self.SPEC.format(module=module))

        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        try:
            result = CliRunner().invoke(
                cli,
                ["test", "--config", str(spec), "--yes", "--format", "json", *args],
            )
        finally:
            sys.modules.pop(module, None)
        assert result.exception is None or isinstance(
            result.exception, SystemExit
        ), result.exception
        raw = result.stdout
        return json.loads(raw[raw.index("{"):]), result

    def test_runs_three_preserves_every_answer_in_order(self, tmp_path, monkeypatch):
        payload, _ = self._run(
            tmp_path, monkeypatch, "stub_runs_three", ["--runs", "3"],
        )
        by_query = {r["query"]: r for r in payload["results"]}
        for query, expected in self.ANSWERS.items():
            assert by_query[query]["answers"] == expected, query
            # `answer` keeps its old meaning: the representative (last) run
            assert by_query[query]["answer"] == expected[-1]

    def test_answers_index_align_with_stability_verdicts(self, tmp_path, monkeypatch):
        payload, _ = self._run(
            tmp_path, monkeypatch, "stub_align", ["--runs", "3"],
        )
        answers = {r["query"]: r["answers"] for r in payload["results"]}
        stability = {q["query"]: q for q in payload["stability"]["queries"]}

        beta = stability["beta query"]
        assert beta["verdicts"] == [True, False, True]
        failing_run = beta["verdicts"].index(False)
        assert answers["beta query"][failing_run] == "beta declined answer two"
        assert "refund" not in answers["beta query"][failing_run]

        alpha = stability["alpha query"]
        assert alpha["verdicts"] == [True, True, True]
        assert len(answers["alpha query"]) == len(alpha["verdicts"]) == 3

    def test_per_run_cost_attribution_present(self, tmp_path, monkeypatch):
        payload, _ = self._run(
            tmp_path, monkeypatch, "stub_cost", ["--runs", "3"],
        )
        beta = next(q for q in payload["stability"]["queries"]
                    if q["query"] == "beta query")
        assert len(beta["trace_ids"]) == 3
        assert len(set(beta["trace_ids"])) == 3
        assert beta["total_tokens"] == [12, 12, 12]
        assert len(beta["cost_usd"]) == len(beta["latency_ms"]) == 3

    def test_single_run_shape_unchanged(self, tmp_path, monkeypatch):
        payload, _ = self._run(tmp_path, monkeypatch, "stub_single", [])
        assert "stability" not in payload
        for result in payload["results"]:
            assert "answers" not in result
            assert set(result) == {
                "query", "answer", "hard_fail", "has_warnings",
                "correctness", "path", "retrieval", "cost",
            }
        by_query = {r["query"]: r for r in payload["results"]}
        assert by_query["alpha query"]["answer"] == "alpha refund answer one"
