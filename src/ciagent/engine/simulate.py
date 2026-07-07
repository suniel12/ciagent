# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Conversation driver for `ciagent simulate` (F6).

Phase 1: scripted personas — fixed user turns from the scenario spec, fully
deterministic, zero API keys. Generative personas arrive in a later phase;
this driver is the seed for every F6 test either way.

Phase 2: record + replay. Any driven conversation can be recorded as a
schema_version-2 golden envelope (`record_scenario_result`). A recorded
envelope replays through the SAME driver (`replay_envelope`): the recorded
user turns are fed back verbatim — the persona/scripted source is never
consulted — so only the agent side can vary. Replaying a deterministic agent
twice yields byte-identical verdicts (`scenario_verdict` is the contract).
This is the one-command found-bug → regression-test conversion: a failed
scenario recorded with `--record` becomes a golden the suite gates on via
`--replay`.

Phase 3: generative personas + cost guardrails + parallel scenarios. A
scenario with `persona:`/`goal:` and no `turns:` gets its user turns from a
persona LLM (engine/persona.py) — nondeterministic by design, the finder
path. Termination rules DO NOT change: max_turns / stop_when only; the
persona never decides when a conversation ends, and a derailed persona
(empty/unusable output) marks the scenario infra-error rather than silently
grading the agent. A shared CostBudget hard-aborts the session mid-
conversation when `--max-cost` is breached; partial turns are kept and
clearly marked, and the outcome verdict is NOT evaluated on a partial
conversation. Scenarios run in parallel (they are independent, binding);
turns within a conversation stay sequential.

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
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from pathlib import Path

from ..conversation import ConversationEnvelope, ConversationTurn
from ..models import Trace
from ..schema.spec_models import GoldenQuery, ScenarioSpec
from .results import QueryResult

# A conversation runner receives the full message history
# [{"role": "user"|"assistant", "content": str}, ...] and returns the
# assistant's reply (str) or a full Trace.
ConversationRunner = Callable[[list[dict[str, str]]], Union[str, "Trace"]]

# A turn source produces the NEXT user message from (history, turn_index):
# a str to continue, or None when a scripted source is exhausted. Generative
# sources never return None — they end only via max_turns / stop_when.
TurnSource = Callable[[list[dict[str, str]], int], Optional[str]]

# Termination reasons (deterministic, reported per scenario)
TERM_SCRIPT_EXHAUSTED = "scripted-turns-exhausted"
TERM_MAX_TURNS = "max-turns-reached"
TERM_STOP_WHEN = "stop-when-event"
TERM_INFRA_ERROR = "infra-error"
TERM_COST_ABORT = "max-cost-aborted"


@dataclass
class CostBudget:
    """Session-level cost ceiling shared across scenarios (and threads).

    The driver checks `exceeded` before every turn and charges each turn's
    trace cost after it runs — so a breach stops the session mid-conversation
    at the next turn boundary, including scenarios that haven't started yet.
    """
    max_usd: float
    spent_usd: float = 0.0

    def __post_init__(self) -> None:
        import threading

        self._lock = threading.Lock()

    def charge(self, amount_usd: Optional[float]) -> None:
        with self._lock:
            self.spent_usd += amount_usd or 0.0

    @property
    def exceeded(self) -> bool:
        return self.spent_usd > self.max_usd


def scripted_turn_source(turns: list[str]) -> TurnSource:
    """Turn source over a fixed `turns:` list; None when exhausted."""
    def next_turn(messages: list[dict[str, str]], turn_index: int) -> Optional[str]:
        return turns[turn_index] if turn_index < len(turns) else None

    return next_turn


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
    error: Optional[str] = None  # set when termination == infra-error / cost-abort
    mode: str = "scripted"  # how user turns were produced: scripted | simulated

    @property
    def hard_fail(self) -> bool:
        """Correctness failure in the outcome or any per-turn check."""
        if self.outcome is not None and self.outcome.hard_fail:
            return True
        return any(t.checks is not None and t.checks.hard_fail for t in self.turns)

    @property
    def is_infra_error(self) -> bool:
        return self.termination == TERM_INFRA_ERROR

    @property
    def is_cost_aborted(self) -> bool:
        return self.termination == TERM_COST_ABORT

    @property
    def is_partial(self) -> bool:
        """The conversation did not run to a clean termination — partial
        results are reported but the outcome verdict was not evaluated."""
        return self.is_infra_error or self.is_cost_aborted

    @property
    def cost_usd(self) -> float:
        """Agent-side cost of this conversation (sum of turn trace costs)."""
        return sum(t.trace.total_cost_usd or 0.0 for t in self.turns)

    def user_turns(self) -> list[str]:
        return [t.user_message for t in self.turns]

    def to_envelope(self, agent: str = "", mode: Optional[str] = None) -> ConversationEnvelope:
        return ConversationEnvelope(
            mode=mode or self.mode,
            agent=agent,
            scenario={
                "name": self.scenario.display_name(),
                "persona": self.scenario.persona,
                "goal": self.scenario.goal,
                "max_turns": self.scenario.max_turns,
                # Full spec (checks, stop_when) so a recorded envelope is a
                # self-contained regression test — replay needs no spec file.
                "spec": self.scenario.model_dump(exclude_none=True),
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
    retry_count: int = 2,
    retry_backoff: float = 1.0,
) -> "Trace":
    """Execute one conversation turn with capture + return-type coercion.

    Same contract as the single-turn executor: wrap in TraceContext so string
    returns still get LLM/tool capture; Trace returns pass through (merging
    captured spans when the runner's Trace has none). Transient infra errors
    (rate limits, timeouts) retry with the same exponential backoff the
    single-turn worker pool uses; anything else propagates to the driver's
    infra-error handling.
    """
    import time

    from ..capture import TraceContext
    from .parallel import _RETRYABLE_EXCEPTIONS, _wrap_str_as_trace

    for attempt in range(retry_count + 1):
        try:
            with TraceContext(agent_name=agent_name, test_name=turn_label) as ctx:
                result = conv_runner(messages)
        except _RETRYABLE_EXCEPTIONS:
            if attempt >= retry_count:
                raise
            time.sleep(retry_backoff * (2 ** attempt))
            continue
        return _coerce_turn_trace(result, ctx.trace, turn_label, _wrap_str_as_trace)
    raise RuntimeError("unreachable: retry loop exits by return or raise")


def _coerce_turn_trace(result, captured: "Trace", turn_label: str, wrap_str) -> "Trace":
    """Coerce a runner's return (Trace | str) into a Trace with captured spans."""
    if isinstance(result, Trace):
        if not result.spans and captured.spans:
            result.spans = captured.spans
            result.compute_metrics()
        return result

    text = result if isinstance(result, str) else str(result)
    trace = captured
    trace.metadata["final_output"] = text
    if trace.spans:
        trace.spans[0].output_data = text
    if not trace.spans:
        trace = wrap_str(text, turn_label)
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
    turn_source: Optional[TurnSource] = None,
    persona_config: Optional[dict[str, Any]] = None,
    budget: Optional[CostBudget] = None,
) -> ScenarioResult:
    """Drive one scenario against a conversation runner until termination.

    User turns come from `turn_source` when given (tests, mock personas);
    otherwise from the scenario itself — its `turns:` list (scripted), or a
    persona LLM built from `persona:`/`goal:` (simulated, the finder path).
    """
    from .runner import evaluate_query

    if turn_source is None:
        if scenario.turns:
            turn_source = scripted_turn_source(scenario.turns)
        elif scenario.persona or scenario.goal:
            from .persona import persona_turn_source

            turn_source = persona_turn_source(scenario, persona_config)
        else:
            raise ValueError(
                f"Scenario '{scenario.display_name()}' has neither scripted turns "
                "nor a persona — give it a `turns:` list or a `persona:`/`goal:`."
            )

    result = ScenarioResult(
        scenario=scenario,
        mode="scripted" if scenario.turns else "simulated",
    )
    messages: list[dict[str, str]] = []
    script_exhausted = False

    for i in range(scenario.max_turns):
        try:
            user_message = turn_source(list(messages), i)
        except Exception as exc:  # noqa: BLE001 — persona derail / source failure
            result.termination = TERM_INFRA_ERROR
            result.error = f"user turn {i + 1}: {type(exc).__name__}: {exc}"
            return result

        if user_message is None:
            script_exhausted = True
            break

        # Budget gates the NEXT turn only — a conversation whose script just
        # exhausted completed cleanly and must not read as partial.
        if budget is not None and budget.exceeded:
            result.termination = TERM_COST_ABORT
            result.error = (
                f"cost budget breached before turn {i + 1}: "
                f"${budget.spent_usd:.4f} spent, max ${budget.max_usd:.4f}"
            )
            return result  # partial turns kept; outcome NOT evaluated

        messages.append({"role": "user", "content": user_message})
        label = f"{scenario.display_name()} [turn {i + 1}]"

        try:
            trace = _run_turn(conv_runner, list(messages), agent_name, label)
        except Exception as exc:  # noqa: BLE001 — agent code is arbitrary
            result.termination = TERM_INFRA_ERROR
            result.error = f"turn {i + 1}: {type(exc).__name__}: {exc}"
            return result

        if budget is not None:
            budget.charge(trace.total_cost_usd)

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

    if not result.termination:
        # Scripted turns fitting inside max_turns exhaust the script even when
        # the last turn IS the max — preserves Phase 1 semantics.
        if script_exhausted or (
            scenario.turns and len(scenario.turns) <= scenario.max_turns
        ):
            result.termination = TERM_SCRIPT_EXHAUSTED
        else:
            result.termination = TERM_MAX_TURNS

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


def run_scenarios_parallel(
    scenarios: list[ScenarioSpec],
    conv_runner_for: Callable[[ScenarioSpec], ConversationRunner],
    agent_name: str = "",
    judge_config: Optional[dict[str, Any]] = None,
    spec_dir: Optional[str] = None,
    max_workers: int = 4,
    turn_source_for: Optional[Callable[[ScenarioSpec], Optional[TurnSource]]] = None,
    persona_config: Optional[dict[str, Any]] = None,
    budget: Optional[CostBudget] = None,
) -> list[ScenarioResult]:
    """Run scenarios concurrently; turns within each stay sequential.

    Scenarios are independent (eng review, binding), so this is the same
    worker-pool shape the single-turn path uses. Results come back in
    scenario order. A shared budget still aborts the whole session: workers
    check it at every turn boundary.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[Optional[ScenarioResult]] = [None] * len(scenarios)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_scenario,
                s,
                conv_runner_for(s),
                agent_name=agent_name,
                judge_config=judge_config,
                spec_dir=spec_dir,
                turn_source=turn_source_for(s) if turn_source_for else None,
                persona_config=persona_config,
                budget=budget,
            ): i
            for i, s in enumerate(scenarios)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return [r for r in results if r is not None]


# ── Phase 2: record + replay ────────────────────────────────────────────────────


def scenario_slug(name: str) -> str:
    """Filesystem-safe golden filename for a scenario display name."""
    slug = "".join(c if c.isalnum() else "-" for c in name.lower())
    slug = "-".join(p for p in slug.split("-") if p)
    return slug[:80] or "scenario"


def record_scenario_result(
    result: ScenarioResult,
    directory: Union[str, "Path"],
    agent: str = "",
    mode: Optional[str] = None,
    mock: bool = False,
) -> "Path":
    """Save a driven scenario as a golden conversation envelope.

    Written to ``<directory>/<agent>/scenarios/<slug>.json`` — the
    ``scenarios/`` subdirectory keeps conversation goldens out of the
    single-turn version listing (`ciagent baselines`). Recording never
    prechecks: a FAILED scenario is exactly what gets recorded when
    converting a found bug into a regression test.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from ..conversation import save_envelope

    envelope = result.to_envelope(agent=agent, mode=mode)
    envelope.captured_at = datetime.now(timezone.utc).isoformat()
    envelope.metadata["checks_passed"] = not result.hard_fail
    if mock:
        envelope.metadata["mock"] = True

    out_dir = Path(directory) / agent / "scenarios"
    return save_envelope(envelope, out_dir / f"{scenario_slug(result.scenario.display_name())}.json")


def envelope_to_scenario(envelope: ConversationEnvelope) -> ScenarioSpec:
    """Reconstruct a replayable ScenarioSpec from a recorded envelope.

    The recorded user turns are the script, fed back verbatim — the original
    persona/scripted source is never consulted, whatever the embedded spec
    says its `turns:` were. Checks and stop_when carry over from the embedded
    spec so the golden gates on the same verdict it was recorded with.
    """
    recorded_turns = [t.user_message for t in envelope.turns]
    if not recorded_turns:
        raise ValueError(
            f"Envelope '{(envelope.scenario or {}).get('name', envelope.agent)}' has "
            "no recorded turns — nothing to replay."
        )
    spec_dict = dict((envelope.scenario or {}).get("spec") or {})
    if not spec_dict.get("name"):
        spec_dict["name"] = (envelope.scenario or {}).get("name")
    spec_dict["turns"] = recorded_turns
    spec_dict["max_turns"] = len(recorded_turns)
    return ScenarioSpec(**spec_dict)


def replay_envelope(
    envelope: ConversationEnvelope,
    conv_runner: ConversationRunner,
    agent_name: str = "",
    judge_config: Optional[dict[str, Any]] = None,
    spec_dir: Optional[str] = None,
) -> ScenarioResult:
    """Replay a recorded conversation's user turns against the agent."""
    return run_scenario(
        envelope_to_scenario(envelope),
        conv_runner,
        agent_name=agent_name or envelope.agent,
        judge_config=judge_config,
        spec_dir=spec_dir,
    )


def scenario_verdict(result: ScenarioResult) -> dict[str, Any]:
    """Deterministic verdict serialization for a driven scenario.

    This is the replay-determinism contract: statuses and messages only, no
    trace ids, timestamps, or latencies — replaying a deterministic agent
    twice must yield byte-identical ``json.dumps(scenario_verdict(r))``.
    """
    def layer(qr: Optional[QueryResult]) -> Optional[dict[str, Any]]:
        if qr is None:
            return None
        return {
            "hard_fail": qr.hard_fail,
            "correctness": {"status": qr.correctness.status.value, "messages": qr.correctness.messages},
            "path": {"status": qr.path.status.value, "messages": qr.path.messages},
            "cost": {"status": qr.cost.status.value, "messages": qr.cost.messages},
        }

    return {
        "scenario": result.scenario.display_name(),
        "termination": result.termination,
        "error": result.error,
        "hard_fail": result.hard_fail,
        "turns": [
            {"turn_index": t.turn_index, "user_message": t.user_message, "checks": layer(t.checks)}
            for t in result.turns
        ],
        "outcome": layer(result.outcome),
    }
