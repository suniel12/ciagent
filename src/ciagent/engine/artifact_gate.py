# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Artifact Gate — validate a candidate artifact against the engine BEFORE it
is written.

Three features write artifacts that the suite will later trust blindly:
F3 writes generated checks into the spec, F6 records conversation goldens,
F7 imports production traces as golden baselines. All three share one
failure mode — a bad artifact planted today is a permanent false regression
(or a check that flags correct output) in the user's CI tomorrow — so they
share one gate (eng review 2026-07-05): evaluate the candidate → accept, or
reject with NAMED reasons. Nothing is ever written silently broken.

Gate verdicts:
    accepted    — safe to write
    rejected    — would misbehave; reasons name exactly what failed
    unvalidated — cannot be checked (no ground truth available); written
                  only with explicit user consent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ciagent.conversation import ConversationEnvelope
    from ciagent.models import Trace

ACCEPTED = "accepted"
REJECTED = "rejected"
UNVALIDATED = "unvalidated"


@dataclass
class GateResult:
    """Outcome of gating one candidate artifact."""
    status: str                              # accepted | rejected | unvalidated
    reasons: list[str] = field(default_factory=list)  # named, human-readable

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    @property
    def rejected(self) -> bool:
        return self.status == REJECTED

    def summary(self) -> str:
        if self.accepted:
            return "accepted"
        return f"{self.status}: " + "; ".join(self.reasons)


def _accept() -> GateResult:
    return GateResult(status=ACCEPTED)


def _reject(*reasons: str) -> GateResult:
    return GateResult(status=REJECTED, reasons=list(reasons))


def _unvalidated(reason: str) -> GateResult:
    return GateResult(status=UNVALIDATED, reasons=[reason])


# ── F3: generated-check brittleness gate ───────────────────────────────────────


def gate_candidate_check(
    check_field: str,
    check_value: Any,
    known_good_answers: list[str],
) -> GateResult:
    """Gate one generated deterministic check against known-good answers.

    A check that fails a known-good answer would flag correct output as
    wrong — exactly the brittleness a determinism brand must not ship.
    """
    import re

    from ciagent.engine.correctness import evaluate_correctness
    from ciagent.engine.results import LayerStatus
    from ciagent.schema.spec_models import CorrectnessSpec

    if check_field == "regex_match":
        try:
            re.compile(check_value)
        except re.error as e:
            return _reject(f"invalid regex: {e}")

    if not known_good_answers:
        return _unvalidated("no recorded known-good answer to validate against")

    single = CorrectnessSpec(**{check_field: check_value})
    for answer in known_good_answers:
        result = evaluate_correctness(answer=answer, spec=single)
        if result.status != LayerStatus.PASS:
            return _reject(
                f"fails a known-good answer: \"{answer[:80]}\" — "
                "would flag correct output as wrong"
            )
    return _accept()


# ── F6: conversation-golden replay gate ────────────────────────────────────────


def gate_conversation_envelope(envelope: "ConversationEnvelope") -> GateResult:
    """Gate a conversation envelope before it is written as a golden.

    A golden that cannot replay is a permanent false regression: the reasons
    name the structural defect instead of letting `--replay` explode later.
    Check outcomes are deliberately NOT gated — recording a FAILED scenario
    is the found-bug → regression-test conversion working as intended.
    """
    reasons: list[str] = []

    if not envelope.turns:
        reasons.append("envelope has no recorded turns — nothing to replay")
    else:
        for t in envelope.turns:
            if not (t.user_message or "").strip():
                reasons.append(
                    f"turn {t.turn_index}: empty user_message — replay would "
                    "feed the agent an empty turn"
                )
                break
        if any(t.trace is None for t in envelope.turns):
            reasons.append("a recorded turn has no trace")

    # The embedded spec must reconstruct into a replayable scenario
    if not reasons:
        try:
            from ciagent.engine.simulate import envelope_to_scenario

            envelope_to_scenario(envelope)
        except Exception as e:  # noqa: BLE001 — reason is the payload
            reasons.append(f"embedded scenario spec does not reconstruct: {e}")

    return _reject(*reasons) if reasons else _accept()


# ── F7: imported-golden round-trip gate ────────────────────────────────────────


def gate_imported_golden(
    trace: "Trace",
    query: Optional[str],
) -> GateResult:
    """Gate a trace imported from production before it becomes a golden.

    Round-trip requirement (eng review, binding): every import must produce
    a golden that loads and evaluates cleanly before it is written. Partial
    traces — no user input, no final output, no spans — are rejected with
    the missing fields named, never silently imported: a golden that can
    never pass is a permanent false regression planted in the user's CI.
    """
    from ciagent.engine.runner import _extract_answer, evaluate_query
    from ciagent.models import Trace
    from ciagent.schema.spec_models import GoldenQuery

    reasons: list[str] = []
    if not query or not str(query).strip():
        reasons.append(
            "no user input found — the trace carries no query text "
            "(expected gen_ai.input.messages or equivalent)"
        )
    if not trace.spans:
        reasons.append("trace has no spans — nothing was captured")
    if not _extract_answer(trace):
        reasons.append(
            "no final output found — the golden could never be evaluated "
            "(expected gen_ai.output.messages or metadata.final_output)"
        )
    if reasons:
        return _reject(*reasons)

    # Round trip: serialize → reload → dry evaluation must not raise.
    try:
        reloaded = Trace(**__import__("json").loads(trace.model_dump_json()))
        evaluate_query(GoldenQuery(query=str(query)), reloaded)
    except Exception as e:  # noqa: BLE001 — reason is the payload
        return _reject(f"golden does not round-trip through the engine: {e}")

    return _accept()
