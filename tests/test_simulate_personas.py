# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
F6 Phase 3: generative personas + cost guardrails + stability + parallel.

ADR required-test checklist items covered here:
- persona-derail (unusable persona output → infra-error, partial turns kept,
  never a silent grade of the agent)
- --max-cost mid-session abort with partial report (session-level budget,
  stops mid-conversation, outcome NOT evaluated on partial conversations)
Plus: simulation-variance flip attribution, parallel scenarios, and the
simulate cost estimate.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from ciagent.cli import cli, _simulation_turn_plan
from ciagent.engine.cost_estimator import estimate_simulation_cost
from ciagent.engine.mock_runner import mock_persona_turn_source
from ciagent.engine.persona import (
    PersonaError,
    build_persona_system_prompt,
    generate_user_turn,
    persona_turn_source,
)
from ciagent.engine.simulate import (
    TERM_COST_ABORT,
    TERM_INFRA_ERROR,
    TERM_MAX_TURNS,
    CostBudget,
    run_scenario,
    run_scenarios_parallel,
)
from ciagent.engine.stability import FlipSource, build_scenario_stability
from ciagent.models import LLMCall, Span, SpanKind, Trace
from ciagent.schema.spec_models import ScenarioSpec


def echo_runner(messages):
    return f"reply to: {messages[-1]['content']}"


def costly_runner(cost_per_turn: float):
    """Toy agent whose every turn costs a fixed amount."""
    def run(messages):
        span = Span(kind=SpanKind.AGENT, name="a")
        span.llm_calls.append(
            LLMCall(model="m", tokens_in=1, tokens_out=1, cost_usd=cost_per_turn)
        )
        span.output_data = "ok"
        t = Trace(agent_name="a", spans=[span])
        t.metadata["final_output"] = "ok"
        t.compute_metrics()
        return t

    return run


def generative_scenario(**kw) -> ScenarioSpec:
    kw.setdefault("name", "angry refund")
    kw.setdefault("persona", "angry customer")
    kw.setdefault("goal", "get a refund")
    kw.setdefault("max_turns", 3)
    return ScenarioSpec(**kw)


# ── Persona turn generation ────────────────────────────────────────────────────


class TestPersona:
    def test_generates_turn_with_persona_and_goal_in_prompt(self):
        seen = {}

        def fake_complete(system, user, model, temperature):
            seen.update(system=system, user=user, model=model, temperature=temperature)
            return '  "I want my money back!"  '

        turn = generate_user_turn(
            "angry customer", "get a refund", [], model="test-model",
            complete_fn=fake_complete,
        )
        assert turn == "I want my money back!"  # stripped, unquoted
        assert "angry customer" in seen["system"]
        assert "get a refund" in seen["system"]
        assert seen["model"] == "test-model"
        assert "not started" in seen["user"]  # opening-message prompt

    def test_transcript_rendered_from_user_perspective(self):
        seen = {}

        def fake_complete(system, user, model, temperature):
            seen["user"] = user
            return "next"

        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello, how can I help?"},
        ]
        generate_user_turn("p", "g", history, complete_fn=fake_complete)
        assert "You: hi" in seen["user"]
        assert "Agent: hello, how can I help?" in seen["user"]

    def test_empty_output_is_derail(self):
        with pytest.raises(PersonaError, match="empty"):
            generate_user_turn("p", "g", [], complete_fn=lambda *a: "   ")

    def test_oversized_output_is_derail(self):
        with pytest.raises(PersonaError, match="derailed"):
            generate_user_turn("p", "g", [], complete_fn=lambda *a: "x" * 5000)

    def test_llm_failure_is_derail(self):
        def boom(*a):
            raise ConnectionError("api down")

        with pytest.raises(PersonaError, match="persona LLM call failed"):
            generate_user_turn("p", "g", [], complete_fn=boom)

    def test_system_prompt_forbids_meta(self):
        prompt = build_persona_system_prompt("p", "g")
        assert "ONLY the user's next message" in prompt
        assert "Never break character" in prompt


# ── Generative scenarios in the driver ─────────────────────────────────────────


class TestGenerativeDriver:
    def test_persona_never_terminates_runs_to_max_turns(self):
        # The persona keeps talking; only max_turns ends the conversation.
        scenario = generative_scenario(max_turns=4)
        source = persona_turn_source(
            scenario, complete_fn=lambda s, u, m, t: "still waiting on my refund"
        )
        r = run_scenario(scenario, echo_runner, turn_source=source)
        assert r.termination == TERM_MAX_TURNS
        assert len(r.turns) == 4
        assert r.mode == "simulated"

    def test_persona_derail_mid_conversation_keeps_partial_turns(self):
        # ADR required test: a derailed persona marks the scenario
        # infra-error with completed turns kept — it must never silently
        # grade the agent on a broken simulated user.
        calls = {"n": 0}

        def flaky_persona(system, user, model, temperature):
            calls["n"] += 1
            if calls["n"] >= 3:
                return ""  # derails on turn 3
            return f"user turn {calls['n']}"

        scenario = generative_scenario(
            max_turns=5,
            outcome={"correctness": {"expected_in_answer": ["reply"]}},
        )
        source = persona_turn_source(scenario, complete_fn=flaky_persona)
        r = run_scenario(scenario, echo_runner, turn_source=source)
        assert r.termination == TERM_INFRA_ERROR
        assert "PersonaError" in r.error and "user turn 3" in r.error
        assert len(r.turns) == 2  # completed turns kept
        assert r.outcome is None  # no verdict on a broken conversation
        assert not r.hard_fail

    def test_stop_when_still_exits_generative_conversations(self):
        def tool_on_turn_two(messages):
            n = sum(1 for m in messages if m["role"] == "user")
            span = Span(kind=SpanKind.AGENT, name="a")
            if n == 2:
                from ciagent.models import ToolCall

                span.tool_calls.append(ToolCall(tool_name="escalate", arguments={}))
            span.output_data = "ok"
            t = Trace(agent_name="a", spans=[span])
            t.metadata["final_output"] = "ok"
            t.compute_metrics()
            return t

        scenario = generative_scenario(
            max_turns=6, stop_when={"tool_called": "escalate"}
        )
        source = persona_turn_source(scenario, complete_fn=lambda *a: "again")
        r = run_scenario(scenario, tool_on_turn_two, turn_source=source)
        assert r.termination == "stop-when-event"
        assert len(r.turns) == 2

    def test_mock_persona_is_deterministic(self):
        scenario = generative_scenario()
        s1, s2 = mock_persona_turn_source(scenario), mock_persona_turn_source(scenario)
        turns1 = [s1([], i) for i in range(3)]
        turns2 = [s2([], i) for i in range(3)]
        assert turns1 == turns2
        assert all("get a refund" in t for t in turns1)


# ── Cost budget: --max-cost mid-session abort (ADR required test) ──────────────


class TestCostBudget:
    def test_budget_aborts_mid_conversation_keeps_partial(self):
        budget = CostBudget(max_usd=0.5)
        scenario = ScenarioSpec(
            name="expensive",
            turns=["a", "b", "c"],
            outcome={"correctness": {"expected_in_answer": ["ok"]}},
        )
        r = run_scenario(scenario, costly_runner(1.0), budget=budget)
        assert r.termination == TERM_COST_ABORT
        assert r.is_cost_aborted and r.is_partial
        assert len(r.turns) == 1  # turn 1 ran ($1.00 > $0.50), turn 2 never did
        assert r.outcome is None  # partial conversation is never graded
        assert "cost budget breached" in r.error
        assert not r.hard_fail

    def test_completed_script_is_not_marked_partial_on_breach(self):
        # The budget gates the NEXT turn only: a conversation whose script
        # exhausted right as the budget breached completed cleanly — it keeps
        # its termination and its outcome verdict (found by CLI smoke test).
        # $0.4/turn against a $0.5 cap: turn 2 is allowed (spent $0.4), and
        # the script exhausts exactly as the budget breaches (spent $0.8).
        budget = CostBudget(max_usd=0.5)
        scenario = ScenarioSpec(
            name="finished",
            turns=["a", "b"],
            outcome={"correctness": {"expected_in_answer": ["ok"]}},
        )
        r = run_scenario(scenario, costly_runner(0.4), budget=budget)
        assert r.termination == "scripted-turns-exhausted"
        assert not r.is_partial
        assert len(r.turns) == 2
        assert r.outcome is not None and not r.outcome.hard_fail
        assert budget.exceeded  # the NEXT scenario still cost-aborts

    def test_breached_budget_stops_scenarios_before_they_start(self):
        budget = CostBudget(max_usd=0.5)
        budget.charge(1.0)
        r = run_scenario(
            ScenarioSpec(name="late", turns=["a"]), echo_runner, budget=budget
        )
        assert r.is_cost_aborted
        assert len(r.turns) == 0

    def test_budget_is_thread_safe_accumulator(self):
        from concurrent.futures import ThreadPoolExecutor

        budget = CostBudget(max_usd=1e9)
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda _: budget.charge(0.001), range(1000)))
        assert budget.spent_usd == pytest.approx(1.0)

    def test_run_without_budget_is_unaffected(self):
        r = run_scenario(
            ScenarioSpec(name="free", turns=["a", "b"]), costly_runner(100.0)
        )
        assert len(r.turns) == 2
        assert r.cost_usd == pytest.approx(200.0)


# ── Parallel scenarios ─────────────────────────────────────────────────────────


class TestParallelScenarios:
    def test_results_return_in_scenario_order(self):
        import time

        scenarios = [
            ScenarioSpec(name=f"s{i}", turns=[f"q{i}"]) for i in range(6)
        ]

        def slow_first(scenario):
            def run(messages):
                if scenario.name == "s0":
                    time.sleep(0.05)  # finishes last, must still report first
                return f"answer for {scenario.name}"

            return run

        results = run_scenarios_parallel(scenarios, slow_first, max_workers=4)
        assert [r.scenario.name for r in results] == [f"s{i}" for i in range(6)]
        assert all(len(r.turns) == 1 for r in results)

    def test_shared_budget_aborts_across_workers(self):
        budget = CostBudget(max_usd=0.5)
        scenarios = [
            ScenarioSpec(name=f"s{i}", turns=["a", "b", "c"]) for i in range(3)
        ]
        results = run_scenarios_parallel(
            scenarios,
            lambda s: costly_runner(1.0),
            max_workers=1,  # deterministic: sequential submission
            budget=budget,
        )
        assert results[0].is_cost_aborted and len(results[0].turns) == 1
        assert all(r.is_cost_aborted and len(r.turns) == 0 for r in results[1:])


# ── Simulation-variance flip attribution ───────────────────────────────────────


def _scripted_result(answers, verdict_fail=False, user_turns=None, name="s"):
    """Build a ScenarioResult by driving the real driver with canned answers."""
    turns = user_turns or [f"q{i}" for i in range(len(answers))]
    it = iter(answers)

    scenario = ScenarioSpec(
        name=name,
        turns=turns,
        outcome={
            "correctness": {
                "expected_in_answer": ["NEVER-SAID" if verdict_fail else "answer"]
            }
        },
    )
    return run_scenario(scenario, lambda messages: f"answer: {next(it)}")


class TestSimulationVariance:
    def test_stable_scenarios_do_not_flip(self):
        runs = [
            [_scripted_result(["x", "y"])],
            [_scripted_result(["x", "y"])],
        ]
        recs = build_scenario_stability(runs)
        assert not recs[0].flipped
        assert recs[0].flip_source is None

    def test_different_user_turns_attribute_to_simulation_variance(self):
        runs = [
            [_scripted_result(["x"], user_turns=["how do I get a refund?"])],
            [_scripted_result(["x"], verdict_fail=True, user_turns=["REFUND. NOW."])],
        ]
        recs = build_scenario_stability(runs)
        assert recs[0].flipped
        assert recs[0].flip_source == FlipSource.SIMULATION_VARIANCE
        assert "simulated user" in recs[0].flip_reason

    def test_same_user_turns_different_answers_is_agent_variance(self):
        runs = [
            [_scripted_result(["x"])],
            [_scripted_result(["TOTALLY DIFFERENT"], verdict_fail=True)],
        ]
        recs = build_scenario_stability(runs)
        assert recs[0].flip_source == FlipSource.AGENT_VARIANCE

    def test_aborted_run_attributes_to_infra(self):
        aborted = run_scenario(
            ScenarioSpec(name="s", turns=["q0"]),
            costly_runner(1.0),
            budget=CostBudget(max_usd=-1.0),  # instantly breached
        )
        ok = _scripted_result(["x"])
        recs = build_scenario_stability([[ok], [aborted]])
        assert recs[0].flip_source == FlipSource.INFRA_ERROR


# ── Cost estimate ──────────────────────────────────────────────────────────────


class TestSimulationEstimate:
    def test_turn_plan_scripted_vs_generative(self):
        scenarios = [
            ScenarioSpec(name="s", turns=["a", "b"], max_turns=8),
            ScenarioSpec(name="g", persona="p", goal="g", max_turns=5),
        ]
        agent_turns, persona_turns, judged = _simulation_turn_plan(scenarios)
        assert agent_turns == 7  # 2 scripted + 5 generative ceiling
        assert persona_turns == 5
        assert judged == 0

    def test_judged_turns_counted(self):
        scenarios = [
            ScenarioSpec(
                name="s",
                turns=["a", "b"],
                per_turn={"correctness": {"llm_judge": [{"rule": "is helpful"}]}},
                outcome={"correctness": {"llm_judge": [{"rule": "is helpful"}]}},
            ),
        ]
        _, _, judged = _simulation_turn_plan(scenarios)
        assert judged == 3  # 2 per-turn + 1 outcome

    def test_estimate_scales_with_runs_and_includes_persona(self):
        one = estimate_simulation_cost(
            agent_turns=10, persona_turns=10, runs=1, persona_model="claude-haiku-4-5"
        )
        three = estimate_simulation_cost(
            agent_turns=10, persona_turns=10, runs=3, persona_model="claude-haiku-4-5"
        )
        assert one["persona_cost"] > 0
        assert three["total_estimate"] == pytest.approx(one["total_estimate"] * 3)
        # haiku-class persona is a small fraction of the agent cost
        assert one["persona_cost"] < one["agent_cost"]


# ── CLI ─────────────────────────────────────────────────────────────────────────

COSTLY_AGENT = """
from ciagent.models import Trace, Span, SpanKind, LLMCall

def respond(messages):
    span = Span(kind=SpanKind.AGENT, name="a")
    span.llm_calls.append(LLMCall(model="m", tokens_in=1, tokens_out=1, cost_usd=1.0))
    span.output_data = "ok"
    t = Trace(agent_name="a", spans=[span])
    t.metadata["final_output"] = "ok"
    t.compute_metrics()
    return t
"""


class TestSimulateCLIPhase3:
    def _invoke_with_module(self, runner, spec, module_src, args):
        import sys

        with runner.isolated_filesystem():
            with open("agentci_spec.yaml", "w") as f:
                f.write(spec)
            with open("toy_costly_agent.py", "w") as f:
                f.write(module_src)
            sys.path.insert(0, ".")
            try:
                return runner.invoke(cli, args)
            finally:
                sys.path.remove(".")
                sys.modules.pop("toy_costly_agent", None)

    def test_max_cost_aborts_session_with_partial_report(self):
        # ADR required test: --max-cost mid-session abort, partial report
        # clearly marked, exit code 2.
        spec = (
            "agent: a\n"
            'conversation_runner: "toy_costly_agent:respond"\n'
            "scenarios:\n"
            "  - name: first\n    turns: [\"a\", \"b\", \"c\"]\n"
            "  - name: second\n    turns: [\"a\"]\n"
        )
        result = self._invoke_with_module(
            CliRunner(), spec, COSTLY_AGENT,
            ["simulate", "--yes", "--max-cost", "0.5", "-w", "1"],
        )
        assert result.exit_code == 2, result.output
        assert "SESSION ABORTED" in result.output
        assert "PARTIAL" in result.output
        assert "max-cost-aborted" in result.output

    def test_max_cost_json_marks_partial_and_spent(self):
        spec = (
            "agent: a\n"
            'conversation_runner: "toy_costly_agent:respond"\n'
            "scenarios:\n"
            "  - name: first\n    turns: [\"a\", \"b\"]\n"
        )
        result = self._invoke_with_module(
            CliRunner(), spec, COSTLY_AGENT,
            ["simulate", "--yes", "--max-cost", "0.5", "-w", "1", "--format", "json"],
        )
        payload = json.loads(result.output[result.output.index("{"):])
        assert payload["cost_aborted"] is True
        assert payload["spent_usd"] == pytest.approx(1.0)
        sc = payload["scenarios"][0]
        assert sc["partial"] is True
        assert sc["termination"] == "max-cost-aborted"
        assert sc["cost_usd"] == pytest.approx(1.0)

    def test_runs_reports_stability_block(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("agentci_spec.yaml", "w") as f:
                f.write(
                    "agent: a\nscenarios:\n"
                    "  - name: happy\n    turns: [\"hi\"]\n"
                    "    outcome:\n      correctness: {expected_in_answer: [\"documentation\"]}\n"
                )
            result = runner.invoke(cli, ["simulate", "--mock", "--runs", "2"])
        assert result.exit_code == 0, result.output
        assert "Stability across 2 runs" in result.output
        assert "1/1 stable" in result.output

    def test_generative_record_then_replay_never_calls_persona(self):
        # Recorded simulated conversation replays with zero persona calls:
        # envelope mode is "simulated", replay works fully mock/deterministic.
        from pathlib import Path

        from ciagent.conversation import load_envelope

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("agentci_spec.yaml", "w") as f:
                f.write(
                    "agent: a\nscenarios:\n"
                    "  - name: gen\n    persona: angry customer\n"
                    "    goal: refund\n    max_turns: 2\n"
                )
            rec = runner.invoke(cli, ["simulate", "--mock", "--record"])
            assert rec.exit_code == 0, rec.output
            golden = Path("golden/a/scenarios/gen.json")
            env = load_envelope(golden)
            assert env.mode == "simulated"
            assert len(env.turns) == 2

            rep = runner.invoke(cli, ["simulate", "--mock", "--replay", str(golden)])
            assert rep.exit_code == 0, rep.output
            assert "no changes" in rep.output
