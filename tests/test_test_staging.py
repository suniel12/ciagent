# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Promotion v2 slice 3: single-turn `test` staging.

Failing live queries stage as one-turn envelopes (mode="single") via the
query_result_to_envelope adapter; classification comes from QueryStability
(incl. the `mixed` state that only exists in single-turn stability); mock
failures are synthetic and never stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

from ciagent.cli import cli
from ciagent.conversation import load_envelope

QA_SPEC = """
agent: qa-test
baseline_dir: ./golden
runner: "toy_qa:run"
queries:
  - query: "what is the refund policy?"
    correctness:
      expected_in_answer: ["refund"]
"""

TOY_RUNNER = '''
from ciagent.models import Span, SpanKind, Trace

ANSWER = "i cannot help"

def run(query):
    span = Span(kind=SpanKind.AGENT, name="agent")
    span.output_data = ANSWER
    t = Trace(agent_name="qa-test", test_name=query, spans=[span])
    t.metadata["final_output"] = ANSWER
    t.compute_metrics()
    return t
'''


def _write(spec_extra: str = "", answer: str | None = None):
    Path("agentci_spec.yaml").write_text(QA_SPEC + spec_extra)
    body = TOY_RUNNER if answer is None else TOY_RUNNER.replace(
        '"i cannot help"', repr(answer)
    )
    Path("toy_qa.py").write_text(body)


def _invoke(args):
    sys.path.insert(0, ".")
    try:
        return CliRunner().invoke(cli, args)
    finally:
        sys.path.remove(".")
        sys.modules.pop("toy_qa", None)


class TestSingleTurnStaging:
    def test_failing_live_query_stages_by_default(self):
        r = CliRunner()
        with r.isolated_filesystem():
            _write()
            res = _invoke(["test", "--yes"])
            assert res.exit_code == 1, res.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            assert len(staged) == 1, res.output
            env = load_envelope(staged[0])
            assert env.mode == "single"
            assert env.staging["source"] == "test"
            assert env.staging["classification"] == "unverified"
            assert env.turns[0].user_message == "what is the refund policy?"
            # the adapter embeds a replayable one-turn scenario spec
            spec = env.scenario["spec"]
            assert spec["turns"] == ["what is the refund policy?"]
            assert spec["outcome"]["correctness"]["expected_in_answer"] == ["refund"]

    def test_multi_run_consistent_classification(self):
        r = CliRunner()
        with r.isolated_filesystem():
            _write()
            res = _invoke(["test", "--yes", "--runs", "3"])
            assert res.exit_code == 1, res.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            assert len(staged) == 1
            env = load_envelope(staged[0])
            assert env.staging["classification"] == "consistent"
            assert env.staging["runs_observed"] == 3

    def test_no_stage_flag_disables(self):
        r = CliRunner()
        with r.isolated_filesystem():
            _write()
            res = _invoke(["test", "--yes", "--no-stage"])
            assert res.exit_code == 1, res.output
            assert not Path(".ciagent/staged").exists()

    def test_staging_disabled_in_spec_prints_notice(self):
        r = CliRunner()
        with r.isolated_filesystem():
            _write("staging: false\n")
            res = _invoke(["test", "--yes"])
            assert res.exit_code == 1, res.output
            assert "enable staging" in res.output
            assert not Path(".ciagent/staged").exists()

    def test_mock_failures_never_stage(self, monkeypatch):
        # AGENTCI_MOCK_FLAKY forces verdict flips across mock runs — synthetic
        # failures, marked as such by never entering the staging area.
        monkeypatch.setenv("AGENTCI_MOCK_FLAKY", "1")
        r = CliRunner()
        with r.isolated_filesystem():
            _write()
            res = CliRunner().invoke(cli, ["test", "--mock", "--runs", "3"])
            assert not Path(".ciagent/staged").exists(), res.output

    def test_staged_query_is_redacted(self):
        r = CliRunner()
        with r.isolated_filesystem():
            _write(answer="ask alice@corp.example.org, key sk-abc123DEF456ghi789jkl")
            res = _invoke(["test", "--yes"])
            assert res.exit_code == 1, res.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            raw = staged[0].read_text()
            assert "sk-abc123DEF456ghi789jkl" not in raw
            assert "alice@corp.example.org" not in raw

    def test_promote_test_staged_entry(self):
        r = CliRunner()
        with r.isolated_filesystem():
            _write()
            res = _invoke(["test", "--yes", "--runs", "3"])
            assert res.exit_code == 1, res.output
            from ciagent.promotion import StageStore

            sid = StageStore(Path(".ciagent/staged")).list()[0].stage_id
            res = _invoke(["promote", sid, "--yes"])
            assert res.exit_code == 0, res.output
            goldens = list(Path("golden/qa-test/scenarios").glob("*.json"))
            assert len(goldens) == 1
            g = load_envelope(goldens[0])
            assert g.provenance["lifecycle"] == "gate"
            assert g.mode == "single"

    def test_json_stdout_stays_pure_with_staging(self):
        import json as _json

        r = CliRunner()
        with r.isolated_filesystem():
            _write()
            res = _invoke(["test", "--yes", "--format", "json"])
            _json.loads(res.stdout)
            assert "staged" in res.stderr
