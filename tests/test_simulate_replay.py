# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
F6 Phase 2: record + replay + conversation-aware diff.

ADR required-test checklist items covered here:
- replay determinism (replay twice ⇒ byte-identical verdicts)
- recorded envelope is a self-contained regression test (found-bug →
  regression conversion: record failing run, fix agent, replay gates)
- replay feeds recorded user turns verbatim — the scripted source embedded
  in the envelope's spec is never consulted
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ciagent.cli import cli
from ciagent.conversation import ConversationEnvelope, load_envelope
from ciagent.engine.diff import diff_envelopes
from ciagent.engine.simulate import (
    TERM_SCRIPT_EXHAUSTED,
    envelope_to_scenario,
    record_scenario_result,
    replay_envelope,
    run_scenario,
    scenario_slug,
    scenario_verdict,
)
from ciagent.schema.spec_models import ScenarioSpec

FIXTURES = Path(__file__).parent / "fixtures"


def echo_runner(messages):
    return f"reply to: {messages[-1]['content']}"


def make_scenario(**kw) -> ScenarioSpec:
    kw.setdefault("name", "refund flow")
    kw.setdefault("turns", ["hello", "i want a refund"])
    kw.setdefault("outcome", {"correctness": {"expected_in_answer": ["refund"]}})
    return ScenarioSpec(**kw)


# ── Recording ───────────────────────────────────────────────────────────────────


class TestRecord:
    def test_slug(self):
        assert scenario_slug("Refund flow #2!") == "refund-flow-2"
        assert scenario_slug("---") == "scenario"

    def test_recorded_envelope_round_trips(self, tmp_path):
        r = run_scenario(make_scenario(), echo_runner)
        path = record_scenario_result(r, tmp_path, agent="toy", mode="scripted")
        assert path == tmp_path / "toy" / "scenarios" / "refund-flow.json"

        env = load_envelope(path)
        assert env.schema_version == 2
        assert env.mode == "scripted"
        assert env.agent == "toy"
        assert env.captured_at
        assert [t.user_message for t in env.turns] == ["hello", "i want a refund"]
        # self-contained: the full scenario spec (checks included) is embedded
        assert env.scenario["spec"]["outcome"]["correctness"]["expected_in_answer"] == ["refund"]

    def test_failed_scenario_records_with_checks_passed_false(self, tmp_path):
        r = run_scenario(
            make_scenario(outcome={"correctness": {"expected_in_answer": ["NEVER-SAID"]}}),
            echo_runner,
        )
        assert r.hard_fail
        path = record_scenario_result(r, tmp_path, agent="toy")
        env = load_envelope(path)
        assert env.metadata["checks_passed"] is False


# ── Replay reconstruction ──────────────────────────────────────────────────────


class TestEnvelopeToScenario:
    def test_recorded_turns_override_embedded_script(self, tmp_path):
        # The envelope's embedded spec says turns A/B, but the RECORDED turns
        # (what actually happened) differ — recorded turns must win verbatim:
        # the scripted source is never consulted on replay (binding rule).
        r = run_scenario(make_scenario(), echo_runner)
        path = record_scenario_result(r, tmp_path, agent="toy")
        data = json.loads(path.read_text())
        data["scenario"]["spec"]["turns"] = ["SOMETHING", "ELSE", "ENTIRELY"]
        path.write_text(json.dumps(data))

        scenario = envelope_to_scenario(load_envelope(path))
        assert scenario.turns == ["hello", "i want a refund"]
        assert scenario.max_turns == 2
        # checks carry over from the embedded spec
        assert scenario.outcome.correctness.expected_in_answer == ["refund"]

    def test_empty_envelope_is_rejected(self):
        with pytest.raises(ValueError, match="nothing to replay"):
            envelope_to_scenario(ConversationEnvelope(agent="toy"))

    def test_legacy_single_turn_baseline_is_replayable(self):
        # Envelope is the only format: a pre-0.9 wrapper normalizes to a
        # 1-turn envelope, so it replays through the same driver.
        env = load_envelope(FIXTURES / "legacy" / "v1-legacy-wrapper.json")
        scenario = envelope_to_scenario(env)
        assert scenario.turns == ["what is your return policy?"]
        r = replay_envelope(env, echo_runner)
        assert len(r.turns) == 1
        assert r.termination == TERM_SCRIPT_EXHAUSTED


# ── Replay determinism (ADR required test) ─────────────────────────────────────


class TestReplayDeterminism:
    def test_replay_twice_yields_byte_identical_verdicts(self, tmp_path):
        r = run_scenario(
            make_scenario(per_turn={"correctness": {"expected_in_answer": ["reply"]}}),
            echo_runner,
        )
        env = load_envelope(record_scenario_result(r, tmp_path, agent="toy"))

        first = replay_envelope(env, echo_runner)
        second = replay_envelope(env, echo_runner)
        b1 = json.dumps(scenario_verdict(first), sort_keys=True).encode()
        b2 = json.dumps(scenario_verdict(second), sort_keys=True).encode()
        assert b1 == b2

    def test_verdict_has_no_trace_ids_or_timestamps(self):
        r = run_scenario(make_scenario(), echo_runner)
        text = json.dumps(scenario_verdict(r))
        assert "trace_id" not in text
        assert "captured_at" not in text
        assert "duration" not in text


# ── Found-bug → regression-test conversion ─────────────────────────────────────


class TestBugToRegression:
    def test_record_failing_run_then_replay_fixed_agent_passes(self, tmp_path):
        def broken_agent(messages):
            return "i cannot help with that"

        def fixed_agent(messages):
            return f"your refund is on the way ({messages[-1]['content']})"

        # 1. simulate finds the bug; --record saves it, failure and all
        found = run_scenario(make_scenario(), broken_agent)
        assert found.hard_fail
        path = record_scenario_result(found, tmp_path, agent="toy")

        # 2. the golden gates the suite: replay against the fixed agent passes
        env = load_envelope(path)
        replayed = replay_envelope(env, fixed_agent)
        assert not replayed.hard_fail
        assert replayed.outcome is not None and not replayed.outcome.hard_fail

        # 3. the conversation-aware diff shows what changed
        diff = diff_envelopes(env, replayed.to_envelope(agent="toy", mode="replay"))
        assert diff.has_changes
        assert not diff.turn_count_changed
        assert all(t.answer_changed for t in diff.turn_diffs)


# ── Conversation-aware diff ────────────────────────────────────────────────────


class TestDiffEnvelopes:
    def _envelope(self, runner, **scenario_kw) -> ConversationEnvelope:
        return run_scenario(make_scenario(**scenario_kw), runner).to_envelope(agent="toy")

    def test_identical_conversations_have_no_changes(self):
        a, b = self._envelope(echo_runner), self._envelope(echo_runner)
        d = diff_envelopes(a, b)
        assert not d.has_changes
        assert d.turns_before == d.turns_after == 2
        assert d.summary_json()["turns"] == []  # only changed turns serialize

    def test_turn_count_change_detected(self):
        a = self._envelope(echo_runner)
        b = self._envelope(echo_runner, turns=["hello"])
        d = diff_envelopes(a, b)
        assert d.turn_count_changed and d.has_changes
        assert d.turns_before == 2 and d.turns_after == 1
        # the unmatched golden turn reports its side of the pair
        last = d.turn_diffs[1]
        assert last.answer_before and not last.answer_after

    def test_per_turn_tool_and_answer_changes(self):
        from ciagent.models import Span, SpanKind, ToolCall, Trace

        def tool_agent(tool_name):
            def run(messages):
                span = Span(kind=SpanKind.AGENT, name="a")
                span.tool_calls.append(ToolCall(tool_name=tool_name, arguments={}))
                span.output_data = f"answer via {tool_name}"
                t = Trace(agent_name="a", spans=[span])
                t.metadata["final_output"] = f"answer via {tool_name}"
                t.compute_metrics()
                return t
            return run

        a = self._envelope(tool_agent("search_kb"))
        b = self._envelope(tool_agent("escalate"))
        d = diff_envelopes(a, b)
        assert d.tools_changed
        t0 = d.turn_diffs[0]
        assert t0.tools_before == ["search_kb"] and t0.tools_after == ["escalate"]
        assert t0.answer_changed
        assert d.summary_json()["turns"][0]["tools_changed"] is True


# ── CLI ─────────────────────────────────────────────────────────────────────────

SPEC_YAML = """
agent: sim-test
baseline_dir: ./golden
scenarios:
  - name: happy path
    turns: ["hello", "i want a refund"]
    outcome:
      correctness:
        expected_in_answer: ["refund"]
"""


class TestSimulateRecordReplayCLI:
    def _write_spec(self, content=SPEC_YAML):
        with open("agentci_spec.yaml", "w") as f:
            f.write(content)

    def test_record_writes_golden_into_baseline_dir(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec()
            result = runner.invoke(cli, ["simulate", "--mock", "--record"])
            assert result.exit_code == 0, result.output
            golden = Path("golden/sim-test/scenarios/happy-path.json")
            assert golden.exists()
            assert "happy-path.json" in result.output
            env = load_envelope(golden)
            assert env.metadata["mock"] is True

    def test_record_dir_overrides_baseline_dir(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec()
            result = runner.invoke(cli, ["simulate", "--mock", "--record-dir", "captured"])
            assert result.exit_code == 0, result.output
            assert Path("captured/sim-test/scenarios/happy-path.json").exists()

    def test_failed_scenario_still_records_and_exits_1(self):
        # Mock always satisfies checks, so drive a live toy agent that fails.
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec(
                SPEC_YAML + 'conversation_runner: "toy_failing_agent:respond"\n'
            )
            with open("toy_failing_agent.py", "w") as f:
                f.write("def respond(messages):\n    return 'i cannot help'\n")
            import sys

            sys.path.insert(0, ".")
            try:
                result = runner.invoke(cli, ["simulate", "--yes", "--record"])
            finally:
                sys.path.remove(".")
                sys.modules.pop("toy_failing_agent", None)
            assert result.exit_code == 1, result.output
            env = load_envelope(Path("golden/sim-test/scenarios/happy-path.json"))
            assert env.metadata["checks_passed"] is False

    def test_replay_from_directory_and_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec()
            runner.invoke(cli, ["simulate", "--mock", "--record"])
            # directory form: baseline_dir root (finds <agent>/scenarios/*.json)
            r_dir = runner.invoke(cli, ["simulate", "--mock", "--replay", "./golden"])
            assert r_dir.exit_code == 0, r_dir.output
            assert "no changes" in r_dir.output
            # file form
            r_file = runner.invoke(
                cli,
                ["simulate", "--mock", "--replay", "golden/sim-test/scenarios/happy-path.json"],
            )
            assert r_file.exit_code == 0, r_file.output

    def test_replay_missing_path_exits_2(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec()
            result = runner.invoke(cli, ["simulate", "--mock", "--replay", "./nowhere"])
            assert result.exit_code == 2
            assert "Nothing to replay" in result.output

    def test_replay_gates_on_recorded_checks(self):
        # golden records a failing conversation; live replay against an agent
        # that still fails must exit 1 — the suite gates on the golden.
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec(
                SPEC_YAML.replace('"refund"', '"NEVER-SAID"')
                + 'conversation_runner: "toy_replay_agent:respond"\n'
            )
            with open("toy_replay_agent.py", "w") as f:
                f.write("def respond(messages):\n    return 'hello there'\n")
            import sys

            sys.path.insert(0, ".")
            try:
                rec = runner.invoke(cli, ["simulate", "--yes", "--record"])
                assert rec.exit_code == 1, rec.output
                rep = runner.invoke(cli, ["simulate", "--yes", "--replay", "./golden"])
            finally:
                sys.path.remove(".")
                sys.modules.pop("toy_replay_agent", None)
            assert rep.exit_code == 1, rep.output
            assert "FAIL" in rep.output

    def test_json_format_includes_diff_and_recorded(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec()
            runner.invoke(cli, ["simulate", "--mock", "--record"])
            result = runner.invoke(
                cli, ["simulate", "--mock", "--replay", "./golden", "--format", "json", "--record"]
            )
            payload = json.loads(result.output[result.output.index("{"):])
            assert payload["mode"] == "replay"
            sc = payload["scenarios"][0]
            assert sc["diff"]["has_changes"] is False
            assert payload["recorded"]  # re-recorded (golden-update flow)
            env = load_envelope(Path(payload["recorded"][0]))
            assert env.mode == "replay"
