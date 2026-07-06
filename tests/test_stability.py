# Copyright 2025-2026 The AgentCI Authors
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

from agentci.engine.results import LayerResult, LayerStatus, QueryResult
from agentci.engine.stability import (
    FlipSource,
    build_stability_report,
    _min_pairwise_similarity,
)
from agentci.models import Span, SpanKind, ToolCall, Trace
from agentci.schema.spec_models import (
    AgentCISpec,
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
) -> QueryResult:
    tool_calls = [ToolCall(tool_name=t) for t in tools]
    trace = Trace(spans=[Span(kind=SpanKind.AGENT, name="agent", tool_calls=tool_calls)])
    trace.metadata["final_output"] = answer
    trace.compute_metrics()
    status = LayerStatus.PASS if passed else LayerStatus.FAIL
    return QueryResult(
        query=query,
        correctness=LayerResult(status=status),
        path=LayerResult(status=LayerStatus.SKIP),
        cost=LayerResult(status=LayerStatus.SKIP),
        trace=trace,
    )


def make_spec(*queries: GoldenQuery) -> AgentCISpec:
    return AgentCISpec(agent="stability-test", queries=list(queries))


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
        spec = tmp_path / "agentci_spec.yaml"
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

        from agentci.cli import cli

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
        # AGENTCI_MOCK_FLAKY breaks even-indexed queries on odd runs
        result = self._invoke(
            spec_file, ["--runs", "3"], env={"AGENTCI_MOCK_FLAKY": "1"},
        )
        assert result.exit_code == 0, result.output
        assert "FLAKY" in result.output
        assert "agent-variance" in result.output

    def test_fail_on_flaky_exits_one(self, spec_file):
        result = self._invoke(
            spec_file, ["--runs", "3", "--fail-on-flaky"],
            env={"AGENTCI_MOCK_FLAKY": "1"},
        )
        assert result.exit_code == 1, result.output

    def test_json_format_includes_stability_block(self, spec_file):
        import json

        result = self._invoke(
            spec_file, ["--runs", "2", "--format", "json"],
            env={"AGENTCI_MOCK_FLAKY": "1"},
        )
        payload = json.loads(result.output[result.output.index("{"):])
        assert payload["stability"]["runs"] == 2
        assert payload["stability"]["verdict"] == "FLAKY"
        flipped = [q for q in payload["stability"]["queries"] if q["flipped"]]
        assert flipped and flipped[0]["flip_source"] == "agent-variance"

    def test_single_run_has_no_stability_section(self, spec_file):
        result = self._invoke(spec_file, [])
        assert result.exit_code == 0, result.output
        assert "Stability Report" not in result.output
