# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Golden Promotion Pipeline (v1) — StageStore, TriageClassifier, PromotionService.

Unit-level coverage of the three collaborators in ciagent.promotion. CLI and
simulate-integration behavior are exercised in test_promotion_cli.py and
test_simulate_staging.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ciagent.conversation import ConversationEnvelope, ConversationTurn
from ciagent.engine.stability import FlipSource
from ciagent.models import Span, SpanKind, Trace
from ciagent.promotion import (
    Classification,
    PromotionRefused,
    PromotionService,
    StageAmbiguous,
    StageNotFound,
    StageStore,
    TriageClassifier,
)


# ── helpers ──────────────────────────────────────────────────────────────────────


def make_trace(answer: str, query: str) -> Trace:
    span = Span(kind=SpanKind.AGENT, name="agent")
    span.output_data = answer
    trace = Trace(agent_name="test-agent", test_name=query, spans=[span])
    trace.metadata["final_output"] = answer
    trace.metadata["query"] = query
    trace.compute_metrics()
    return trace


def make_envelope(agent="support-agent", name="refund-flow", answer="here") -> ConversationEnvelope:
    return ConversationEnvelope(
        mode="simulated",
        agent=agent,
        scenario={"name": name, "spec": {"name": name, "turns": ["hi"]}},
        turns=[ConversationTurn(turn_index=0, user_message="hi", trace=make_trace(answer, "hi"))],
    )


def staging_block(classification="consistent", run_id="run-1", scenario_id="refund-flow"):
    return {
        "run_id": run_id,
        "staged_at": "2026-07-09T00:00:00+00:00",
        "scenario_id": scenario_id,
        "source": "simulate",
        "classification": classification,
        "runs_observed": 3,
        "verdicts": [False, False, False],
        "flip_source": None,
        "flip_reason": "",
        "failure_summary": "outcome failed",
    }


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw) -> None:
        self.t = self.t + timedelta(**kw)


class FakeStability:
    """Duck-types StabilityLike."""

    def __init__(self, *, always_failed=False, flipped=False, flip_source=None):
        self._always_failed = always_failed
        self._flipped = flipped
        self.flip_source = flip_source

    @property
    def always_failed(self) -> bool:
        return self._always_failed

    @property
    def flipped(self) -> bool:
        return self._flipped


# ── TriageClassifier (step 3) ──────────────────────────────────────────────────


class TestTriageClassifier:
    def test_none_stability_is_unverified(self):
        assert TriageClassifier.classify(None, runs=5) == Classification.UNVERIFIED

    def test_single_run_is_unverified(self):
        s = FakeStability(always_failed=True)
        assert TriageClassifier.classify(s, runs=1) == Classification.UNVERIFIED

    def test_always_failed_no_flip_is_consistent(self):
        s = FakeStability(always_failed=True, flipped=False)
        assert TriageClassifier.classify(s, runs=3) == Classification.CONSISTENT

    def test_agent_variance_is_flaky_agent(self):
        s = FakeStability(flipped=True, flip_source=FlipSource.AGENT_VARIANCE)
        assert TriageClassifier.classify(s, runs=3) == Classification.FLAKY_AGENT

    def test_retrieval_variance_is_flaky_agent(self):
        s = FakeStability(flipped=True, flip_source=FlipSource.RETRIEVAL_VARIANCE)
        assert TriageClassifier.classify(s, runs=3) == Classification.FLAKY_AGENT

    def test_simulation_variance_is_held(self):
        s = FakeStability(flipped=True, flip_source=FlipSource.SIMULATION_VARIANCE)
        assert TriageClassifier.classify(s, runs=3) == Classification.HELD

    def test_judge_flake_is_held(self):
        s = FakeStability(flipped=True, flip_source=FlipSource.JUDGE_FLAKE)
        assert TriageClassifier.classify(s, runs=3) == Classification.HELD

    def test_mixed_is_held(self):
        s = FakeStability(flipped=True, flip_source=FlipSource.MIXED)
        assert TriageClassifier.classify(s, runs=3) == Classification.HELD

    def test_infra_error_is_held_infra(self):
        s = FakeStability(flipped=True, flip_source=FlipSource.INFRA_ERROR)
        assert TriageClassifier.classify(s, runs=3) == Classification.HELD_INFRA

    def test_flipped_but_no_source_is_held(self):
        s = FakeStability(flipped=True, flip_source=None)
        assert TriageClassifier.classify(s, runs=3) == Classification.HELD

    def test_every_flip_source_maps(self):
        for src in FlipSource:
            s = FakeStability(flipped=True, flip_source=src)
            assert isinstance(TriageClassifier.classify(s, runs=3), Classification)

    def test_consistent_meaning_is_reproducible_not_agent_bug(self):
        from ciagent.promotion import CLASSIFICATION_MEANING

        meaning = CLASSIFICATION_MEANING[Classification.CONSISTENT].lower()
        assert "reproducible" in meaning
        assert "not attributed" in meaning


# ── StageStore (step 2) ─────────────────────────────────────────────────────────


class TestStageStore:
    def _store(self, tmp_path, clock, **kw):
        return StageStore(tmp_path / "staged", now=clock, **kw)

    def test_stage_and_load_round_trip(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock)
        path = store.stage(make_envelope(), staging_block=staging_block())
        assert path.exists()
        entries = store.list()
        assert len(entries) == 1
        e = entries[0]
        assert e.agent == "support-agent"
        assert e.scenario_id == "refund-flow"
        assert e.classification == "consistent"
        _, env = store.load(e.stage_id)
        assert env.staging["run_id"] == "run-1"

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock)
        store.stage(make_envelope(), staging_block=staging_block())
        assert list((tmp_path / "staged").rglob("*.tmp")) == []

    def test_two_concurrent_stages_same_scenario_no_collision(self, tmp_path):
        # Same timestamp, same scenario — hash suffix must keep them distinct.
        clock = FakeClock(datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock)
        p1 = store.stage(make_envelope(answer="a"), staging_block=staging_block())
        p2 = store.stage(make_envelope(answer="b"), staging_block=staging_block())
        assert p1 != p2
        assert len(store.list()) == 2

    def test_cap_eviction_keeps_newest(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, 0, 0, 0, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock, cap=3)
        for i in range(6):
            clock.advance(seconds=1)
            store.stage(make_envelope(answer=f"a{i}"), staging_block=staging_block())
        entries = store.list(agent="support-agent")
        assert len(entries) == 3

    def test_age_eviction(self, tmp_path):
        clock = FakeClock(datetime(2026, 6, 1, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock, cap=100, max_age_days=30)
        store.stage(make_envelope(answer="old"), staging_block=staging_block())
        assert len(store.list()) == 1
        # jump past the age cutoff, then stage another → the old one is GC'd
        clock.advance(days=40)
        store.stage(make_envelope(answer="new"), staging_block=staging_block())
        entries = store.list()
        assert len(entries) == 1

    def test_global_file_cap_eviction(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, 0, 0, 0, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock, cap=100, global_max_files=4)
        for i in range(6):
            clock.advance(seconds=1)
            # different scenario ids so the per-scenario cap doesn't fire
            store.stage(
                make_envelope(name=f"s{i}"),
                staging_block=staging_block(scenario_id=f"s{i}"),
            )
        total = len(list((tmp_path / "staged").rglob("*.json")))
        assert total == 4

    def test_drop(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock)
        store.stage(make_envelope(), staging_block=staging_block())
        sid = store.list()[0].stage_id
        store.drop(sid)
        assert store.list() == []

    def test_update_staging_block(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock)
        store.stage(make_envelope(), staging_block=staging_block(classification="unverified"))
        sid = store.list()[0].stage_id
        store.update_staging_block(sid, staging_block(classification="consistent"))
        assert store.list()[0].classification == "consistent"

    def test_load_missing_raises(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock)
        with pytest.raises(StageNotFound):
            store.load("nope/nope/nope")

    def test_list_sorted_best_first(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, 0, 0, 0, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock)
        for klass in ["held", "consistent", "unverified", "flaky-agent", "held-infra"]:
            clock.advance(seconds=1)
            store.stage(
                make_envelope(name=klass),
                staging_block=staging_block(classification=klass, scenario_id=klass),
            )
        order = [e.classification for e in store.list()]
        assert order == ["consistent", "flaky-agent", "unverified", "held", "held-infra"]

    def test_ambiguous_prefix_raises(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock)
        store.stage(make_envelope(name="a"), staging_block=staging_block(scenario_id="a"))
        store.stage(make_envelope(name="b"), staging_block=staging_block(scenario_id="b"))
        # the agent name is a substring of both ids → ambiguous
        with pytest.raises(StageAmbiguous):
            store.load("support-agent")

    def test_prefix_id_resolution(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc))
        store = self._store(tmp_path, clock)
        store.stage(make_envelope(), staging_block=staging_block())
        stem = store.list()[0].path.stem  # run-ts+hash
        _, env = store.load(stem)
        assert env.staging["run_id"] == "run-1"


# ── PromotionService (step 6 unit — provenance stamping + refusal) ──────────────


class TestPromotionServiceUnit:
    def _seed(self, tmp_path, clock, classification="consistent"):
        store = StageStore(tmp_path / "staged", now=clock)
        store.stage(
            make_envelope(),
            staging_block=staging_block(classification=classification),
        )
        return store, store.list()[0].stage_id

    def test_promote_stamps_provenance_and_drops_staging(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store, sid = self._seed(tmp_path, clock)
        svc = PromotionService(store, now=clock)
        out = svc.promote(sid, baseline_dir=str(tmp_path / "golden"))
        from ciagent.conversation import load_envelope

        env = load_envelope(out)
        assert env.staging is None
        assert env.provenance["lifecycle"] == "gate"
        assert env.provenance["classification_at_promotion"] == "consistent"
        assert env.provenance["staged_run_id"] == "run-1"
        assert env.provenance["promoted_at"].startswith("2026-07-09")
        # lands where --record writes
        assert out.parts[-2] == "scenarios"
        # promote is a MOVE: the staged copy is consumed on success
        assert store.list() == []

    def test_refused_promote_keeps_staged_file(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store, sid = self._seed(tmp_path, clock, classification="held")
        svc = PromotionService(store, now=clock)
        with pytest.raises(PromotionRefused):
            svc.promote(sid, baseline_dir=str(tmp_path / "golden"))
        assert len(store.list()) == 1

    def test_promote_refuses_held_without_force(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store, sid = self._seed(tmp_path, clock, classification="held")
        svc = PromotionService(store, now=clock)
        with pytest.raises(PromotionRefused):
            svc.promote(sid, baseline_dir=str(tmp_path / "golden"))

    def test_promote_held_with_force_succeeds(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store, sid = self._seed(tmp_path, clock, classification="held")
        svc = PromotionService(store, now=clock)
        out = svc.promote(sid, baseline_dir=str(tmp_path / "golden"), force=True)
        assert out.exists()

    def test_promote_unverified_refused(self, tmp_path):
        clock = FakeClock(datetime(2026, 7, 9, tzinfo=timezone.utc))
        store, sid = self._seed(tmp_path, clock, classification="unverified")
        svc = PromotionService(store, now=clock)
        with pytest.raises(PromotionRefused):
            svc.promote(sid, baseline_dir=str(tmp_path / "golden"))
