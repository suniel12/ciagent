# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
F6 Phase 0: schema_version + conversation envelope.

Backward-compat contract under test (workspace critical rule: the golden trace
format must remain backward-compatible):
- pre-0.9 files on disk (unversioned wrapper, bare trace) keep loading through
  every reader: the envelope normalizer, baselines.load_baseline, and
  judge-audit's answer collector — proven against static fixtures checked into
  tests/fixtures/legacy/, not shapes rebuilt at test time
- new single-trace baselines carry schema_version 1; envelopes carry 2
- files newer than the reader are rejected by name, never guessed at
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ciagent.conversation import (
    ENVELOPE_SCHEMA_VERSION,
    ConversationEnvelope,
    ConversationTurn,
    load_envelope,
    normalize_to_envelope,
    save_envelope,
)
from ciagent.exceptions import BaselineError
from ciagent.models import Span, SpanKind, Trace

FIXTURES = Path(__file__).parent / "fixtures" / "legacy"


def make_trace(answer: str, query: str) -> Trace:
    span = Span(kind=SpanKind.AGENT, name="agent")
    span.output_data = answer
    trace = Trace(agent_name="test-agent", test_name=query, spans=[span])
    trace.metadata["final_output"] = answer
    trace.metadata["query"] = query
    trace.compute_metrics()
    return trace


# ── Envelope round-trip ─────────────────────────────────────────────────────────


class TestEnvelopeRoundTrip:
    def test_serialize_load_round_trip(self, tmp_path):
        env = ConversationEnvelope(
            mode="scripted",
            agent="support-agent",
            version="v1-scenario-refund",
            scenario={"persona": "frustrated customer", "goal": "get a refund", "max_turns": 3},
            turns=[
                ConversationTurn(turn_index=0, user_message="hi", trace=make_trace("hello!", "hi")),
                ConversationTurn(
                    turn_index=1,
                    user_message="I want a refund",
                    trace=make_trace("Refunds take 5-7 business days.", "I want a refund"),
                ),
            ],
        )
        path = save_envelope(env, tmp_path / "conv.json")
        loaded = load_envelope(path)
        assert loaded.schema_version == ENVELOPE_SCHEMA_VERSION
        assert loaded.mode == "scripted"
        assert loaded.scenario["persona"] == "frustrated customer"
        assert [t.user_message for t in loaded.turns] == ["hi", "I want a refund"]
        assert loaded.turns[1].trace.metadata["final_output"] == "Refunds take 5-7 business days."
        # second round trip is byte-identical (replay determinism groundwork)
        path2 = save_envelope(loaded, tmp_path / "conv2.json")
        assert path.read_text() == path2.read_text()

    def test_final_trace_and_single_turn(self):
        env = ConversationEnvelope(
            turns=[ConversationTurn(user_message="q", trace=make_trace("a", "q"))]
        )
        assert env.is_single_turn
        assert env.final_trace().metadata["final_output"] == "a"


# ── Legacy fixture loading (pre-0.9 files must keep working) ────────────────────


class TestLegacyFixtures:
    def test_wrapper_fixture_normalizes_to_one_turn(self):
        data = json.loads((FIXTURES / "v1-legacy-wrapper.json").read_text())
        assert "schema_version" not in data  # the fixture IS pre-0.9
        env = normalize_to_envelope(data, source="fixture")
        assert env.is_single_turn
        assert env.mode == "single"
        assert env.agent == "legacy-agent"
        assert env.turns[0].user_message == "what is your return policy?"
        assert "30 days" in env.turns[0].trace.metadata["final_output"]

    def test_bare_trace_fixture_normalizes_to_one_turn(self):
        data = json.loads((FIXTURES / "legacy-bare-trace.golden.json").read_text())
        env = normalize_to_envelope(data, source="fixture")
        assert env.is_single_turn
        assert env.turns[0].trace.total_tool_calls >= 1

    def test_load_baseline_still_reads_wrapper_unchanged(self, tmp_path):
        # the diff engine's loader must keep returning the raw dict shape
        from ciagent.baselines import load_baseline

        target = tmp_path / "legacy-agent"
        target.mkdir()
        (target / "v1.json").write_text((FIXTURES / "v1-legacy-wrapper.json").read_text())
        data = load_baseline("legacy-agent", "v1", baseline_dir=str(tmp_path))
        assert data["version"] == "v1"
        assert "spans" in data["trace"]

    def test_judge_audit_collector_reads_both_fixtures(self, tmp_path):
        from ciagent.engine.judge_audit import load_answers_from_baselines

        import shutil

        shutil.copy(FIXTURES / "v1-legacy-wrapper.json", tmp_path / "a.json")
        shutil.copy(FIXTURES / "legacy-bare-trace.golden.json", tmp_path / "b.json")
        answers = load_answers_from_baselines(str(tmp_path))
        assert "what is your return policy?" in answers
        assert "30 days" in answers["what is your return policy?"]


# ── schema_version semantics ────────────────────────────────────────────────────


class TestSchemaVersion:
    def test_save_baseline_writes_schema_version_1(self, tmp_path):
        from ciagent.baselines import save_baseline
        from ciagent.schema.spec_models import AgentCISpec, GoldenQuery

        spec = AgentCISpec(
            agent="a", queries=[GoldenQuery(query="q")],
        )
        out = save_baseline(
            trace=make_trace("answer", "q"), agent="a", version="v1",
            spec=spec, query_text="q", baseline_dir=str(tmp_path), force=True,
        )
        assert json.loads(out.read_text())["schema_version"] == 1

    def test_versioned_wrapper_with_schema_version_1_loads(self):
        data = json.loads((FIXTURES / "v1-legacy-wrapper.json").read_text())
        data["schema_version"] = 1
        env = normalize_to_envelope(data)
        assert env.is_single_turn

    def test_newer_schema_version_rejected_by_name(self):
        data = json.loads((FIXTURES / "v1-legacy-wrapper.json").read_text())
        data["schema_version"] = 99
        with pytest.raises(BaselineError, match="schema_version 99"):
            normalize_to_envelope(data)

    def test_unrecognized_shape_rejected_with_keys_named(self):
        with pytest.raises(BaselineError, match="Unrecognized baseline shape"):
            normalize_to_envelope({"foo": 1, "bar": 2})
