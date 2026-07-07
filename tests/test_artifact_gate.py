# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the shared artifact gate (engine/artifact_gate.py).

One behavior, one rejection format, one test suite (eng review): F3's
generated-check gate, F6's conversation-golden gate, and F7's import
round-trip gate all refuse to write an artifact that would misbehave —
with named reasons, never silently.
"""

from __future__ import annotations

import pytest

from ciagent.conversation import ConversationEnvelope, ConversationTurn
from ciagent.engine.artifact_gate import (
    gate_candidate_check,
    gate_conversation_envelope,
    gate_imported_golden,
)
from ciagent.models import Span, SpanKind, ToolCall, Trace


def _trace(answer: str = "the rate is 4.5%", spans: bool = True) -> Trace:
    t = Trace(agent_name="a")
    if spans:
        t.spans.append(Span(kind=SpanKind.AGENT, name="a", tool_calls=[
            ToolCall(tool_name="lookup", result="found it"),
        ]))
    if answer:
        t.metadata["final_output"] = answer
    t.compute_metrics()
    return t


# ── F3: candidate checks ───────────────────────────────────────────────────────


class TestCandidateCheckGate:
    def test_check_passing_known_good_answers_accepted(self):
        gate = gate_candidate_check(
            "any_expected_in_answer", ["4.5%"], ["The rate is 4.5% APR."],
        )
        assert gate.accepted

    def test_check_failing_a_known_good_answer_rejected_with_name(self):
        gate = gate_candidate_check(
            "any_expected_in_answer", ["9.9%"], ["The rate is 4.5% APR."],
        )
        assert gate.rejected
        assert "fails a known-good answer" in gate.reasons[0]

    def test_invalid_regex_rejected(self):
        gate = gate_candidate_check("regex_match", "([unclosed", ["anything"])
        assert gate.rejected
        assert "invalid regex" in gate.reasons[0]

    def test_no_ground_truth_is_unvalidated_not_rejected(self):
        gate = gate_candidate_check("any_expected_in_answer", ["4.5%"], [])
        assert gate.status == "unvalidated"
        assert not gate.rejected

    def test_check_generator_consumes_gate_with_same_statuses(self):
        # The F3 consumer keeps its historical status/reason contract
        from ciagent.engine.check_generator import (
            CandidateCheck,
            GenerationResult,
            validate_candidates,
        )

        result = GenerationResult(candidates=[
            CandidateCheck(query="q", field="any_expected_in_answer", value=["4.5%"]),
            CandidateCheck(query="q", field="any_expected_in_answer", value=["9.9%"]),
            CandidateCheck(query="unknown", field="any_expected_in_answer", value=["x"]),
        ])
        validate_candidates(result, {"q": ["The rate is 4.5% APR."]})
        assert [c.status for c in result.candidates] == [
            "validated", "rejected", "unvalidated",
        ]
        assert "fails a known-good answer" in result.candidates[1].reason


# ── F6: conversation goldens ───────────────────────────────────────────────────


class TestConversationEnvelopeGate:
    def _envelope(self, turns=None) -> ConversationEnvelope:
        return ConversationEnvelope(
            mode="scripted",
            agent="a",
            scenario={"name": "s", "spec": {"turns": ["hi"], "max_turns": 1}},
            turns=turns if turns is not None else [
                ConversationTurn(turn_index=0, user_message="hi", trace=_trace()),
            ],
        )

    def test_replayable_envelope_accepted(self):
        assert gate_conversation_envelope(self._envelope()).accepted

    def test_empty_envelope_rejected_with_reason(self):
        gate = gate_conversation_envelope(self._envelope(turns=[]))
        assert gate.rejected
        assert "no recorded turns" in gate.reasons[0]

    def test_empty_user_message_rejected(self):
        env = self._envelope(turns=[
            ConversationTurn(turn_index=0, user_message="   ", trace=_trace()),
        ])
        gate = gate_conversation_envelope(env)
        assert gate.rejected
        assert "empty user_message" in gate.reasons[0]

    def test_failed_checks_still_record(self):
        # The found-bug → regression-test conversion: check outcomes are NOT
        # the gate's business, only structure is.
        env = self._envelope()
        env.metadata["checks_passed"] = False
        assert gate_conversation_envelope(env).accepted

    def test_record_scenario_result_refuses_unreplayable(self, tmp_path):
        from ciagent.engine.simulate import ScenarioResult, record_scenario_result
        from ciagent.schema.spec_models import ScenarioSpec

        result = ScenarioResult(
            scenario=ScenarioSpec(name="s", turns=["hi"]),
            turns=[],  # infra-error before turn 1
            termination="infra-error",
        )
        with pytest.raises(ValueError, match="refusing to record"):
            record_scenario_result(result, tmp_path, agent="a")


# ── F7: imported goldens ───────────────────────────────────────────────────────


class TestImportedGoldenGate:
    def test_complete_trace_accepted(self):
        assert gate_imported_golden(_trace(), "what rate?").accepted

    def test_missing_query_rejected_named(self):
        gate = gate_imported_golden(_trace(), None)
        assert gate.rejected
        assert any("no user input" in r for r in gate.reasons)

    def test_missing_final_output_rejected_named(self):
        gate = gate_imported_golden(_trace(answer=""), "what rate?")
        assert gate.rejected
        assert any("no final output" in r for r in gate.reasons)

    def test_empty_trace_rejects_all_reasons_named(self):
        gate = gate_imported_golden(_trace(answer="", spans=False), "  ")
        assert gate.rejected
        joined = " ".join(gate.reasons)
        assert "no user input" in joined
        assert "no spans" in joined
        assert "no final output" in joined
