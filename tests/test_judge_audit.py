# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit + integration tests for the Judge Audit engine.

The judge is injected as a fake (judge_fn) so every mode is testable with
zero API calls: Mode 1 (judge vs deterministic checks), Mode 2 (retest flip
rate), Mode 3 (hand labels, agreement + Cohen's kappa).
"""

from __future__ import annotations

import pytest

from ciagent.engine.judge_audit import (
    load_answers_from_baselines,
    load_labels_file,
    run_judge_audit,
)
from ciagent.schema.spec_models import (
    AgentCISpec,
    CorrectnessSpec,
    GoldenQuery,
    JudgeRubric,
)


# ── Fake judges ────────────────────────────────────────────────────────────────


def always_pass(**_kwargs):
    return {"passed": True, "score": 5, "rationale": "looks great"}


def always_fail(**_kwargs):
    return {"passed": False, "score": 1, "rationale": "not good"}


class FlakyJudge:
    """Alternates verdicts per call — deterministic retest instability."""

    def __init__(self):
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        return {"passed": self.calls % 2 == 1, "score": 3, "rationale": "hmm"}


def make_spec(*queries: GoldenQuery) -> AgentCISpec:
    return AgentCISpec(agent="audit-test", queries=list(queries))


def checkable_query(text: str, expect: str) -> GoldenQuery:
    """Query with BOTH a deterministic check and a judge rubric."""
    return GoldenQuery(
        query=text,
        correctness=CorrectnessSpec(
            expected_in_answer=[expect],
            llm_judge=[JudgeRubric(rule="is the answer helpful?")],
        ),
    )


def judged_only_query(text: str) -> GoldenQuery:
    return GoldenQuery(
        query=text,
        correctness=CorrectnessSpec(llm_judge=[JudgeRubric(rule="is it good?")]),
    )


# ── Mode 1: judge vs deterministic checks ──────────────────────────────────────


class TestJudgeVsChecks:
    def test_false_pass_detected(self):
        # Deterministic check fails (fact missing) but the judge passes it —
        # the post's 1-in-7 failure mode.
        spec = make_spec(checkable_query("q1", expect="4.5%"))
        answers = {"q1": "Our rate is competitive and customer-friendly."}
        report = run_judge_audit(spec, answers, repeats=1, judge_fn=always_pass)
        assert len(report.false_passes) == 1
        assert report.false_passes[0].query == "q1"
        assert report.false_pass_rate == 1.0
        assert report.verdict == "UNRELIABLE"

    def test_agreement_when_both_pass(self):
        spec = make_spec(checkable_query("q1", expect="4.5%"))
        answers = {"q1": "The rate is 4.5% APR."}
        report = run_judge_audit(spec, answers, repeats=1, judge_fn=always_pass)
        assert report.agreement_rate == 1.0
        assert report.false_passes == []
        assert report.verdict == "TRUSTWORTHY"

    def test_judge_only_fail_is_not_a_false_pass(self):
        # Judge stricter than checks — listed as false_alarm, not counted
        # against the judge's trust verdict.
        spec = make_spec(checkable_query("q1", expect="4.5%"))
        answers = {"q1": "The rate is 4.5% APR."}
        report = run_judge_audit(spec, answers, repeats=1, judge_fn=always_fail)
        assert report.false_passes == []
        assert len(report.false_alarms) == 1

    def test_judgment_only_queries_counted(self):
        spec = make_spec(judged_only_query("q1"), checkable_query("q2", expect="x"))
        answers = {"q1": "anything", "q2": "x marks the spot"}
        report = run_judge_audit(spec, answers, repeats=1, judge_fn=always_pass)
        assert report.judgment_only_count == 1
        assert len(report.checkable_queries) == 1

    def test_queries_without_rubrics_are_skipped(self):
        spec = make_spec(
            GoldenQuery(query="q1", correctness=CorrectnessSpec(expected_in_answer=["a"])),
        )
        report = run_judge_audit(spec, {"q1": "a"}, repeats=1, judge_fn=always_pass)
        assert report.judged == []

    def test_queries_without_recorded_answer_are_skipped(self):
        spec = make_spec(checkable_query("q1", expect="a"))
        report = run_judge_audit(spec, {}, repeats=1, judge_fn=always_pass)
        assert report.judged == []


# ── Mode 2: retest stability ───────────────────────────────────────────────────


class TestRetestStability:
    def test_flaky_judge_flip_rate(self):
        spec = make_spec(judged_only_query("q1"))
        report = run_judge_audit(
            spec, {"q1": "answer"}, repeats=3, judge_fn=FlakyJudge(),
        )
        q = report.judged[0]
        assert q.judge_flipped
        assert q.judge_verdicts == [True, False, True]
        assert report.flip_rate == 1.0
        assert report.verdict == "UNRELIABLE"

    def test_stable_judge_no_flips(self):
        spec = make_spec(judged_only_query("q1"))
        report = run_judge_audit(
            spec, {"q1": "answer"}, repeats=3, judge_fn=always_pass,
        )
        assert report.flip_rate == 0.0
        assert not report.judged[0].judge_flipped

    def test_single_repeat_has_no_flip_rate(self):
        spec = make_spec(judged_only_query("q1"))
        report = run_judge_audit(spec, {"q1": "answer"}, repeats=1, judge_fn=always_pass)
        assert report.flip_rate is None

    def test_majority_verdict_ties_fail(self):
        # 1 pass, 1 fail over 2 repeats → tie → conservative fail
        spec = make_spec(judged_only_query("q1"))
        report = run_judge_audit(spec, {"q1": "answer"}, repeats=2, judge_fn=FlakyJudge())
        assert report.judged[0].judge_verdict is False


# ── Mode 3: hand labels ────────────────────────────────────────────────────────


class TestHandLabels:
    def test_perfect_agreement_kappa(self):
        spec = make_spec(judged_only_query("q1"), judged_only_query("q2"))
        answers = {"q1": "a", "q2": "b"}

        def judge(answer="", **_kwargs):
            return {"passed": answer == "a", "score": 3, "rationale": ""}

        labels = {"q1": True, "q2": False}
        report = run_judge_audit(spec, answers, repeats=1, labels=labels, judge_fn=judge)
        assert report.label_agreement == 1.0
        assert report.cohens_kappa == 1.0

    def test_disagreement_lowers_kappa_and_verdict(self):
        spec = make_spec(*(judged_only_query(f"q{i}") for i in range(4)))
        answers = {f"q{i}": "x" for i in range(4)}
        labels = {"q0": False, "q1": False, "q2": True, "q3": False}
        # Judge passes everything; labels mostly fail → low agreement
        report = run_judge_audit(spec, answers, repeats=1, labels=labels, judge_fn=always_pass)
        assert report.label_agreement == 0.25
        assert report.verdict == "UNRELIABLE"

    def test_unlabeled_queries_excluded(self):
        spec = make_spec(judged_only_query("q1"), judged_only_query("q2"))
        answers = {"q1": "a", "q2": "b"}
        report = run_judge_audit(
            spec, answers, repeats=1, labels={"q1": True}, judge_fn=always_pass,
        )
        assert len(report.labeled_queries) == 1


# ── Sampling, erroring judges, scope note ──────────────────────────────────────


class TestMisc:
    def test_sample_caps_judged_queries(self):
        spec = make_spec(*(judged_only_query(f"q{i}") for i in range(5)))
        answers = {f"q{i}": "x" for i in range(5)}
        report = run_judge_audit(spec, answers, repeats=1, sample=2, judge_fn=always_pass)
        assert len(report.judged) == 2

    def test_erroring_judge_counts_as_fail(self):
        def broken(**_kwargs):
            raise RuntimeError("api down")

        spec = make_spec(judged_only_query("q1"))
        report = run_judge_audit(spec, {"q1": "x"}, repeats=1, judge_fn=broken)
        assert report.judged[0].judge_verdict is False
        assert "judge error" in report.judged[0].judge_rationales[0]

    def test_all_calls_errored_yields_error_verdict(self):
        # A judge that never ran must not be scored TRUSTWORTHY
        def broken(**_kwargs):
            raise RuntimeError("no api key")

        spec = make_spec(checkable_query("q1", expect="4.5%"))
        report = run_judge_audit(
            spec, {"q1": "no rate mentioned here"}, repeats=2, judge_fn=broken,
        )
        assert report.all_judge_calls_errored
        assert report.verdict == "ERROR"

    def test_partial_errors_do_not_mask_verdict(self):
        calls = {"n": 0}

        def sometimes_broken(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("blip")
            return {"passed": True, "score": 5, "rationale": "ok"}

        spec = make_spec(judged_only_query("q1"))
        report = run_judge_audit(spec, {"q1": "x"}, repeats=3, judge_fn=sometimes_broken)
        assert report.total_judge_errors == 1
        assert not report.all_judge_calls_errored
        assert report.verdict != "ERROR"

    def test_scope_note_states_the_inferential_leap(self):
        spec = make_spec(checkable_query("q1", expect="a"), judged_only_query("q2"))
        answers = {"q1": "a", "q2": "x"}
        report = run_judge_audit(spec, answers, repeats=1, judge_fn=always_pass)
        note = report.scope_note
        assert "1 fact-checkable" in note
        assert "smoke test, not a guarantee" in note
        assert "judgment-only" in note

    def test_low_sample_flagged(self):
        spec = make_spec(checkable_query("q1", expect="a"))
        report = run_judge_audit(spec, {"q1": "a"}, repeats=1, judge_fn=always_pass)
        assert report.low_sample
        assert "anecdotes" in report.scope_note


# ── Loaders ────────────────────────────────────────────────────────────────────


class TestLoaders:
    def test_load_answers_bare_trace_shape(self, tmp_path):
        import json

        (tmp_path / "a.golden.json").write_text(json.dumps({
            "spans": [{"kind": "agent", "name": "x", "output_data": "span answer"}],
            "test_name": "what is x?",
            "metadata": {"final_output": "the answer is x"},
        }))
        answers = load_answers_from_baselines(str(tmp_path))
        assert answers == {"what is x?": "the answer is x"}

    def test_load_answers_wrapped_trace_shape(self, tmp_path):
        import json

        (tmp_path / "b.json").write_text(json.dumps({
            "trace": {
                "spans": [{"kind": "agent", "name": "x", "output_data": "fallback"}],
                "query": "wrapped query",
                "metadata": {},
            }
        }))
        answers = load_answers_from_baselines(str(tmp_path))
        assert answers == {"wrapped query": "fallback"}

    def test_load_answers_skips_malformed(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not json")
        assert load_answers_from_baselines(str(tmp_path)) == {}

    def test_load_labels_yaml_variants(self, tmp_path):
        f = tmp_path / "labels.yaml"
        f.write_text('"q1": pass\n"q2": fail\n"q3": true\n"q4": 0\n')
        labels = load_labels_file(str(f))
        assert labels == {"q1": True, "q2": False, "q3": True, "q4": False}

    def test_load_labels_rejects_garbage(self, tmp_path):
        f = tmp_path / "labels.yaml"
        f.write_text('"q1": maybe\n')
        with pytest.raises(ValueError):
            load_labels_file(str(f))


# ── CLI integration (fake judge via monkeypatch) ──────────────────────────────


class TestCLIJudgeAudit:
    @pytest.fixture()
    def project(self, tmp_path):
        import json

        spec = tmp_path / "agentci_spec.yaml"
        spec.write_text(
            """
agent: audit-cli-test
baseline_dir: ./golden
queries:
  - query: "what rate do you charge?"
    correctness:
      expected_in_answer: ["4.5%"]
      llm_judge:
        - rule: "is the answer helpful?"
"""
        )
        golden = tmp_path / "golden"
        golden.mkdir()
        (golden / "rate.golden.json").write_text(json.dumps({
            "spans": [{"kind": "agent", "name": "a", "output_data": ""}],
            "test_name": "what rate do you charge?",
            "metadata": {"final_output": "Our rates are very competitive!"},
        }))
        return tmp_path

    def test_cli_detects_false_pass(self, project, monkeypatch):
        from click.testing import CliRunner

        import ciagent.engine.judge_audit as ja
        from ciagent.cli import cli

        monkeypatch.setattr(
            "ciagent.engine.judge.run_judge",
            lambda **kw: {"passed": True, "score": 5, "rationale": "sounds fine"},
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["judge-audit", "--config", str(project / "agentci_spec.yaml"),
             "--baseline-dir", str(project / "golden"), "--repeats", "2", "--yes"],
        )
        assert "Judge Audit" in result.output
        assert "judge PASS / check FAIL: 1" in result.output
        assert "UNRELIABLE" in result.output
        assert result.exit_code == 1, result.output

    def test_cli_json_format(self, project, monkeypatch):
        import json

        from click.testing import CliRunner

        from ciagent.cli import cli

        monkeypatch.setattr(
            "ciagent.engine.judge.run_judge",
            lambda **kw: {"passed": True, "score": 5, "rationale": ""},
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["judge-audit", "--config", str(project / "agentci_spec.yaml"),
             "--baseline-dir", str(project / "golden"), "--repeats", "1",
             "--format", "json", "--yes"],
        )
        payload = json.loads(result.output[result.output.index("{"):])
        assert payload["verdict"] == "UNRELIABLE"
        assert payload["queries"][0]["false_pass"] is True

    def test_cli_no_baselines_exits_2(self, tmp_path):
        from click.testing import CliRunner

        from ciagent.cli import cli

        spec = tmp_path / "agentci_spec.yaml"
        spec.write_text(
            "agent: empty\nqueries:\n  - query: q\n    correctness:\n"
            "      llm_judge:\n        - rule: r\n"
        )
        runner = CliRunner()
        result = runner.invoke(
            cli, ["judge-audit", "--config", str(spec), "--baseline-dir",
                  str(tmp_path / "nope"), "--yes"],
        )
        assert result.exit_code == 2
