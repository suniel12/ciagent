# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Conversation driver for `ciagent simulate` (F6).

Phase 1 scope: scripted personas only — fixed user turns from the scenario
spec, fully deterministic, zero API keys. Generative personas arrive in a
later phase; this driver is the seed for every F6 test either way.

Termination rules (eng review 2026-07-05, binding):
- a conversation runs until scripted turns are exhausted or `max_turns` is
  reached, whichever comes first
- `outcome` checks are evaluated once at the END as the verdict, NEVER as a
  stop condition — a turn-1 keyword match must not end the scenario before
  the multi-turn bug surfaces
- early exit only via explicit `stop_when` events on concrete trace facts
  (a named tool was called); no judge in the control loop, no keyword triggers
- an agent exception mid-conversation marks the scenario `infra-error` and
  keeps the completed turns (partial results are reported, not discarded)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from ..conversation import ConversationEnvelope, ConversationTurn
from ..models import Trace
from ..schema.spec_models import GoldenQuery, ScenarioSpec
from .results import QueryResult

# A conversation runner receives the full message history
# [{"role": "user"|"assistant", "content": str}, ...] and returns the
# assistant's reply (str) or a full Trace.
ConversationRunner = Callable[[list[dict[str, str]]], Union[str, "Trace"]]

# Termination reasons (deterministic, reported per scenario)
TERM_SCRIPT_EXHAUSTED = "scripted-turns-exhausted"
TERM_MAX_TURNS = "max-turns-reached"
TERM_STOP_WHEN = "stop-when-event"
TERM_INFRA_ERROR = "infra-error"


@dataclass
class TurnResult:
    """One executed turn: the user message, the traced reply, per-turn checks."""
    turn_index: int
    user_message: str
    trace: "Trace"
    checks: Optional[QueryResult] = None  # per_turn layer results, if configured


@dataclass
class ScenarioResult:
    """Outcome of driving one scenario to termination."""
    scenario: ScenarioSpec
    turns: list[TurnResult] = field(default_factory=list)
    outcome: Optional[QueryResult] = None  # evaluated on the final trace
    termination: str = ""
    error: Optional[str] = None  # set when termination == infra-error

    @property
    def hard_fail(self) -> bool:
        """Correctness failure in the outcome or any per-turn check."""
        if self.outcome is not None and self.outcome.hard_fail:
            return True
        return any(t.checks is not None and t.checks.hard_fail for t in self.turns)

    @property
    def is_infra_error(self) -> bool:
        return self.termination == TERM_INFRA_ERROR

    def to_envelope(self, agent: str = "", mode: str = "scripted") -> ConversationEnvelope:
        return ConversationEnvelope(
            mode=mode,
            agent=agent,
            scenario={
                "name": self.scenario.display_name(),
                "persona": self.scenario.persona,
                "goal": self.scenario.goal,
                "max_turns": self.scenario.max_turns,
            },
            metadata={"termination": self.termination, **({"error": self.error} if self.error else {})},
            turns=[
                ConversationTurn(
                    turn_index=t.turn_index,
                    user_message=t.user_message,
                    trace=t.trace,
                )
                for t in self.turns
            ],
        )


def _run_turn(
    conv_runner: ConversationRunner,
    messages: list[dict[str, str]],
    agent_name: str,
    turn_label: str,
) -> "Trace":
    """Execute one conversation turn with capture + return-type coercion.

    Same contract as the single-turn executor: wrap in TraceContext so string
    returns still get LLM/tool capture; Trace returns pass through (merging
    captured spans when the runner's Trace has none).
    """
    from ..capture import TraceContext
    from .parallel import _wrap_str_as_trace

    with TraceContext(agent_name=agent_name, test_name=turn_label) as ctx:
        result = conv_runner(messages)

    if isinstance(result, Trace):
        if not result.spans and ctx.trace.spans:
            result.spans = ctx.trace.spans
            result.compute_metrics()
        return result

    text = result if isinstance(result, str) else str(result)
    trace = ctx.trace
    trace.metadata["final_output"] = text
    if trace.spans:
        trace.spans[0].output_data = text
    if not trace.spans:
        trace = _wrap_str_as_trace(text, turn_label)
    return trace


def _checks_to_query(user_message: str, checks) -> GoldenQuery:
    return GoldenQuery(
        query=user_message,
        correctness=checks.correctness,
        path=checks.path,
        cost=checks.cost,
    )


def _stop_event_observed(scenario: ScenarioSpec, trace: "Trace") -> bool:
    sw = scenario.stop_when
    if sw is None:
        return False
    if sw.tool_called:
        return sw.tool_called in (trace.tool_call_sequence or [])
    return False


def run_scenario(
    scenario: ScenarioSpec,
    conv_runner: ConversationRunner,
    agent_name: str = "",
    judge_config: Optional[dict[str, Any]] = None,
    spec_dir: Optional[str] = None,
) -> ScenarioResult:
    """Drive one scenario against a conversation runner until termination."""
    from .runner import evaluate_query

    if not scenario.turns:
        raise ValueError(
            f"Scenario '{scenario.display_name()}' has no scripted turns. "
            "Generative personas (persona/goal without turns) are not available "
            "yet — give the scenario a `turns:` list."
        )

    result = ScenarioResult(scenario=scenario)
    messages: list[dict[str, str]] = []
    n_turns = min(len(scenario.turns), scenario.max_turns)

    for i in range(n_turns):
        user_message = scenario.turns[i]
        messages.append({"role": "user", "content": user_message})
        label = f"{scenario.display_name()} [turn {i + 1}]"

        try:
            trace = _run_turn(conv_runner, list(messages), agent_name, label)
        except Exception as exc:  # noqa: BLE001 — agent code is arbitrary
            result.termination = TERM_INFRA_ERROR
            result.error = f"turn {i + 1}: {type(exc).__name__}: {exc}"
            return result

        from .runner import _extract_answer

        messages.append({"role": "assistant", "content": _extract_answer(trace)})

        turn = TurnResult(turn_index=i, user_message=user_message, trace=trace)
        if scenario.per_turn is not None:
            turn.checks = evaluate_query(
                query=_checks_to_query(user_message, scenario.per_turn),
                trace=trace,
                judge_config=judge_config,
                spec_dir=spec_dir,
            )
        result.turns.append(turn)

        if _stop_event_observed(scenario, trace):
            result.termination = TERM_STOP_WHEN
            break
    else:
        result.termination = (
            TERM_SCRIPT_EXHAUSTED if n_turns == len(scenario.turns) else TERM_MAX_TURNS
        )

    # Outcome: the verdict, evaluated once at the end on the final trace
    if scenario.outcome is not None and result.turns:
        final = result.turns[-1]
        result.outcome = evaluate_query(
            query=_checks_to_query(final.user_message, scenario.outcome),
            trace=final.trace,
            judge_config=judge_config,
            spec_dir=spec_dir,
        )
    return result
