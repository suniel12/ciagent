# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
F6 Phase 1: conversation driver + `ciagent simulate` (scripted personas).

ADR required-test checklist items covered here:
- driver termination (scripted-turns-exhausted AND max-turns-reached)
- agent-raises-mid-conversation (infra-error, partial turns kept)
- per_turn + outcome evaluation
- outcome is the END verdict, never a stop condition (turn-1 keyword match
  must not end the scenario)
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from ciagent.cli import cli
from ciagent.engine.mock_runner import mock_conversation_runner
from ciagent.engine.simulate import (
    TERM_INFRA_ERROR,
    TERM_MAX_TURNS,
    TERM_SCRIPT_EXHAUSTED,
    TERM_STOP_WHEN,
    run_scenario,
)
from ciagent.models import Span, SpanKind, ToolCall, Trace
from ciagent.schema.spec_models import ScenarioSpec


def echo_runner(messages):
    """Deterministic toy agent: replies mentioning the last user message."""
    return f"reply to: {messages[-1]['content']}"


def make_scenario(**kw) -> ScenarioSpec:
    kw.setdefault("turns", ["one", "two", "three"])
    return ScenarioSpec(**kw)


# ── Termination (both deterministic causes + stop_when) ─────────────────────────


class TestTermination:
    def test_scripted_turns_exhausted(self):
        r = run_scenario(make_scenario(turns=["a", "b"], max_turns=8), echo_runner)
        assert r.termination == TERM_SCRIPT_EXHAUSTED
        assert len(r.turns) == 2

    def test_max_turns_caps_script(self):
        r = run_scenario(make_scenario(turns=["a", "b", "c", "d"], max_turns=2), echo_runner)
        assert r.termination == TERM_MAX_TURNS
        assert len(r.turns) == 2

    def test_stop_when_tool_called_exits_early(self):
        def tool_on_turn_two(messages):
            n = sum(1 for m in messages if m["role"] == "user")
            span = Span(kind=SpanKind.AGENT, name="a")
            if n == 2:
                span.tool_calls.append(ToolCall(tool_name="escalate", arguments={}))
            span.output_data = "ok"
            t = Trace(agent_name="a", spans=[span])
            t.metadata["final_output"] = "ok"
            t.compute_metrics()
            return t

        r = run_scenario(
            make_scenario(turns=["a", "b", "c"], stop_when={"tool_called": "escalate"}),
            tool_on_turn_two,
        )
        assert r.termination == TERM_STOP_WHEN
        assert len(r.turns) == 2

    def test_scenario_without_turns_is_rejected(self):
        with pytest.raises(ValueError, match="no scripted turns"):
            run_scenario(ScenarioSpec(persona="angry customer"), echo_runner)


# ── Agent raises mid-conversation ───────────────────────────────────────────────


class TestAgentRaises:
    def test_infra_error_keeps_partial_turns(self):
        def explodes_on_second(messages):
            if sum(1 for m in messages if m["role"] == "user") == 2:
                raise RuntimeError("agent crashed")
            return "fine"

        r = run_scenario(make_scenario(turns=["a", "b", "c"]), explodes_on_second)
        assert r.termination == TERM_INFRA_ERROR
        assert r.is_infra_error
        assert "RuntimeError" in r.error and "turn 2" in r.error
        assert len(r.turns) == 1  # completed turn kept
        assert not r.hard_fail  # infra is not a correctness failure


# ── per_turn + outcome evaluation ───────────────────────────────────────────────


class TestEvaluation:
    def test_per_turn_checks_run_on_every_turn(self):
        scenario = make_scenario(
            turns=["a", "b"],
            per_turn={"correctness": {"expected_in_answer": ["reply"]}},
        )
        r = run_scenario(scenario, echo_runner)
        assert all(t.checks is not None for t in r.turns)
        assert not r.hard_fail

    def test_per_turn_failure_is_hard_fail(self):
        scenario = make_scenario(
            turns=["a", "b"],
            per_turn={"correctness": {"expected_in_answer": ["MISSING-TOKEN"]}},
        )
        r = run_scenario(scenario, echo_runner)
        assert r.hard_fail

    def test_outcome_evaluated_on_final_trace(self):
        scenario = make_scenario(
            turns=["a", "final question"],
            outcome={"correctness": {"expected_in_answer": ["final question"]}},
        )
        r = run_scenario(scenario, echo_runner)
        assert r.outcome is not None
        assert not r.outcome.hard_fail

    def test_outcome_is_never_a_stop_condition(self):
        # Turn 1's answer already contains the outcome keyword; the scenario
        # must still run to the end of the script (ADR binding rule).
        scenario = make_scenario(
            turns=["magic", "b", "c"],
            outcome={"correctness": {"expected_in_answer": ["reply"]}},
        )
        r = run_scenario(scenario, echo_runner)
        assert len(r.turns) == 3
        assert r.termination == TERM_SCRIPT_EXHAUSTED

    def test_history_grows_with_assistant_replies(self):
        seen: list[int] = []

        def recorder(messages):
            seen.append(len(messages))
            return "ok"

        run_scenario(make_scenario(turns=["a", "b", "c"]), recorder)
        # turn i sees 2i-1 messages (users + prior assistant replies)
        assert seen == [1, 3, 5]

    def test_envelope_conversion(self):
        r = run_scenario(make_scenario(turns=["a", "b"]), echo_runner)
        env = r.to_envelope(agent="toy", mode="scripted")
        assert env.mode == "scripted"
        assert len(env.turns) == 2
        assert env.metadata["termination"] == TERM_SCRIPT_EXHAUSTED
        assert env.turns[1].trace.metadata["final_output"] == "reply to: b"


# ── Mock conversation runner ────────────────────────────────────────────────────


class TestMockConversationRunner:
    def test_mock_satisfies_per_turn_and_outcome(self):
        scenario = make_scenario(
            turns=["a", "b"],
            per_turn={"path": {"expected_tools": ["search_kb"]}},
            outcome={"correctness": {"expected_in_answer": ["refund"]}},
        )
        r = run_scenario(scenario, mock_conversation_runner(scenario))
        assert not r.hard_fail
        assert r.outcome is not None and not r.outcome.hard_fail


# ── CLI ─────────────────────────────────────────────────────────────────────────

SPEC_YAML = """
agent: sim-test
scenarios:
  - name: happy
    turns: ["hello", "i want a refund"]
    outcome:
      correctness:
        expected_in_answer: ["refund"]
"""


class TestSimulateCLI:
    def _write_spec(self, content=SPEC_YAML):
        with open("agentci_spec.yaml", "w") as f:
            f.write(content)

    def test_mock_run_passes(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec()
            result = runner.invoke(cli, ["simulate", "--mock"])
        assert result.exit_code == 0, result.output
        assert "2/2 passed" in result.output or "1/1 passed" in result.output

    def test_json_format(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec()
            result = runner.invoke(cli, ["simulate", "--mock", "--format", "json"])
        payload = json.loads(result.output[result.output.index("{"):])
        assert payload["summary"]["passed"] == payload["summary"]["total"]
        sc = payload["scenarios"][0]
        assert sc["termination"] == "scripted-turns-exhausted"
        assert sc["turns"][0]["answer"]
        assert sc["outcome"]["correctness"]["status"] == "pass"

    def test_no_scenarios_exits_2(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec("agent: a\nqueries:\n  - query: q\n")
            result = runner.invoke(cli, ["simulate", "--mock"])
        assert result.exit_code == 2
        assert "scenarios" in result.output.lower()

    def test_unscripted_scenario_exits_2_with_hint(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec(
                "agent: a\nscenarios:\n  - persona: angry customer\n    goal: refund\n"
            )
            result = runner.invoke(cli, ["simulate", "--mock"])
        assert result.exit_code == 2
        assert "turns" in result.output

    def test_live_without_conversation_runner_exits_2(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec()
            result = runner.invoke(cli, ["simulate", "--yes"])
        assert result.exit_code == 2
        assert "conversation_runner" in result.output

    def test_failing_outcome_exits_1(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_spec(
                "agent: a\n"
                "conversation_runner: \"toy_sim_agent:respond\"\n"
                "scenarios:\n"
                "  - turns: [\"hi\"]\n"
                "    outcome:\n"
                "      correctness: {expected_in_answer: [\"NEVER-SAID\"]}\n"
            )
            with open("toy_sim_agent.py", "w") as f:
                f.write("def respond(messages):\n    return 'hello there'\n")
            import sys

            sys.path.insert(0, ".")
            try:
                result = runner.invoke(cli, ["simulate", "--yes"])
            finally:
                sys.path.remove(".")
                sys.modules.pop("toy_sim_agent", None)
        assert result.exit_code == 1, result.output
        assert "FAIL" in result.output
