# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Steps 5 & 6 — `ciagent stage` group + `ciagent promote` over a real CLI.

The stage-group tests seed a staging area with StageStore directly; the promote
integration test drives the full loop: simulate stages a failing conversation →
promote writes a golden → `simulate --replay` gates (exit 1) → fix the agent →
replay passes (exit 0).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from ciagent.cli import cli
from ciagent.conversation import ConversationEnvelope, ConversationTurn, load_envelope
from ciagent.models import Span, SpanKind, Trace
from ciagent.promotion import StageStore

SPEC = """
agent: sim-test
baseline_dir: ./golden
scenarios:
  - name: placeholder
    turns: ["hi"]
"""


def make_trace(answer):
    span = Span(kind=SpanKind.AGENT, name="agent")
    span.output_data = answer
    t = Trace(agent_name="sim-test", test_name="q", spans=[span])
    t.metadata["final_output"] = answer
    t.compute_metrics()
    return t


def make_env(name="refund-flow"):
    return ConversationEnvelope(
        mode="simulated", agent="sim-test",
        scenario={"name": name, "spec": {"name": name, "turns": ["hi"]}},
        turns=[ConversationTurn(turn_index=0, user_message="hi", trace=make_trace("no"))],
    )


def block(klass="consistent", scenario_id="refund-flow"):
    return {
        "run_id": "run-1", "staged_at": "2026-07-09T00:00:00+00:00",
        "scenario_id": scenario_id, "source": "simulate", "classification": klass,
        "runs_observed": 3, "verdicts": [False, False, False],
        "flip_source": None, "flip_reason": "", "failure_summary": "outcome failed",
    }


def _seed(root, *pairs):
    store = StageStore(Path(root), now=lambda: datetime(2026, 7, 9, tzinfo=timezone.utc))
    for klass, sid in pairs:
        store.stage(make_env(sid), staging_block=block(klass, sid))
    return store


class TestStageGroup:
    def test_list_empty_exits_0(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SPEC)
            res = r.invoke(cli, ["stage", "list"])
            assert res.exit_code == 0, res.output
            assert "No staged" in res.output

    def test_list_sorted_and_filtered(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SPEC)
            _seed(".ciagent/staged", ("held", "h"), ("consistent", "c"), ("unverified", "u"))
            res = r.invoke(cli, ["stage", "list", "--format", "json"])
            assert res.exit_code == 0, res.output
            import json
            data = json.loads(res.output)
            assert [d["classification"] for d in data] == ["consistent", "unverified", "held"]
            # filter
            res2 = r.invoke(cli, ["stage", "list", "--classification", "held", "--format", "json"])
            assert len(json.loads(res2.output)) == 1

    def test_show_and_export(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SPEC)
            store = _seed(".ciagent/staged", ("consistent", "c"))
            sid = store.list()[0].stage_id
            res = r.invoke(cli, ["stage", "show", sid, "--export", "out.json"])
            assert res.exit_code == 0, res.output
            assert Path("out.json").exists()
            assert "exported" in res.output

    def test_show_missing_exits_1(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SPEC)
            res = r.invoke(cli, ["stage", "show", "nope"])
            assert res.exit_code == 1

    def test_drop_single(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SPEC)
            store = _seed(".ciagent/staged", ("consistent", "c"))
            sid = store.list()[0].stage_id
            res = r.invoke(cli, ["stage", "drop", sid])
            assert res.exit_code == 0, res.output
            assert store.list() == []

    def test_drop_held(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SPEC)
            store = _seed(".ciagent/staged", ("held", "h"), ("consistent", "c"))
            res = r.invoke(cli, ["stage", "drop", "--held", "--yes"])
            assert res.exit_code == 0, res.output
            assert [e.classification for e in store.list()] == ["consistent"]

    def test_drop_bad_combo_exits_2(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SPEC)
            res = r.invoke(cli, ["stage", "drop", "someid", "--all"])
            assert res.exit_code == 2

    def test_gc(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SPEC)
            _seed(".ciagent/staged", ("consistent", "c"))
            res = r.invoke(cli, ["stage", "gc"])
            assert res.exit_code == 0, res.output
            assert "gc complete" in res.output

    def test_verify_mock_reclassifies(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SPEC)
            store = _seed(".ciagent/staged", ("unverified", "refund-flow"))
            sid = store.list()[0].stage_id
            # mock runner satisfies checks; the seeded scenario spec has no
            # checks, so all runs "pass" → not always_failed → reclassified.
            res = r.invoke(cli, ["stage", "verify", sid, "--mock", "--runs", "3"])
            assert res.exit_code == 0, res.output
            assert "re-classified" in res.output


# ── Full promote → replay → fix → replay loop ───────────────────────────────────

PROMOTE_SPEC = """
agent: sim-test
baseline_dir: ./golden
conversation_runner: "toy_agent:respond"
staging:
  enabled: true
scenarios:
  - name: refund path
    turns: ["hello", "i want a refund"]
    outcome:
      correctness:
        expected_in_answer: ["refund"]
"""

FAILING = "def respond(messages):\n    return 'i cannot help'\n"
FIXED = "def respond(messages):\n    return 'sure, your refund is on the way'\n"


def _invoke(args):
    sys.path.insert(0, ".")
    try:
        return CliRunner().invoke(cli, args)
    finally:
        sys.path.remove(".")
        sys.modules.pop("toy_agent", None)


class TestPromoteReplayLoop:
    def test_stage_promote_replay_gate_then_fix(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(PROMOTE_SPEC)
            Path("toy_agent.py").write_text(FAILING)

            # 1. simulate 3× → stages a consistent failing conversation
            res = _invoke(["simulate", "--yes", "--runs", "3"])
            assert res.exit_code == 1, res.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            assert len(staged) == 1
            env = load_envelope(staged[0])
            assert env.staging["classification"] == "consistent"

            from ciagent.promotion import StageStore
            store = StageStore(Path(".ciagent/staged"))
            sid = store.list()[0].stage_id

            # 2. promote → writes a golden with provenance
            res = _invoke(["promote", sid, "--yes"])
            assert res.exit_code == 0, res.output
            golden = Path("golden/sim-test/scenarios/refund-path.json")
            assert golden.exists()
            g = load_envelope(golden)
            assert g.staging is None
            assert g.provenance["lifecycle"] == "gate"

            # 3. replay while the bug reproduces → gate red (exit 1)
            res = _invoke(["simulate", "--yes", "--replay", "./golden"])
            assert res.exit_code == 1, res.output

            # 4. fix the agent → replay goes green on its own (exit 0)
            Path("toy_agent.py").write_text(FIXED)
            res = _invoke(["simulate", "--yes", "--replay", "./golden"])
            assert res.exit_code == 0, res.output

class TestInitGitignoreScaffold:
    def test_writes_ignore_line(self):
        from ciagent.cli import _scaffold_staging_gitignore

        r = CliRunner()
        with r.isolated_filesystem():
            assert _scaffold_staging_gitignore() is True
            content = Path(".gitignore").read_text()
            assert ".ciagent/staged/" in content

    def test_appends_without_clobbering(self):
        from ciagent.cli import _scaffold_staging_gitignore

        r = CliRunner()
        with r.isolated_filesystem():
            Path(".gitignore").write_text("*.pyc\n")
            _scaffold_staging_gitignore()
            content = Path(".gitignore").read_text()
            assert "*.pyc" in content
            assert ".ciagent/staged/" in content

    def test_idempotent(self):
        from ciagent.cli import _scaffold_staging_gitignore

        r = CliRunner()
        with r.isolated_filesystem():
            assert _scaffold_staging_gitignore() is True
            assert _scaffold_staging_gitignore() is False
            # only one occurrence
            assert Path(".gitignore").read_text().count(".ciagent/staged/") == 1


class TestPromoteRefusalDetail:
    def test_promote_refuses_unverified_without_force(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(PROMOTE_SPEC)
            Path("toy_agent.py").write_text(FAILING)
            # single run → unverified
            _invoke(["simulate", "--yes"])
            from ciagent.promotion import StageStore
            store = StageStore(Path(".ciagent/staged"))
            sid = store.list()[0].stage_id
            res = _invoke(["promote", sid, "--yes"])
            assert res.exit_code == 1, res.output
            assert "Refused" in res.output
            # --force promotes anyway
            res = _invoke(["promote", sid, "--yes", "--force"])
            assert res.exit_code == 0, res.output


class TestXfailReplayLoop:
    """Full xfail loop: promote --xfail → replay green while failing (XFAIL)
    → fix → replay green with XPASS flag → promote --flip → gate again."""

    def test_xfail_loop(self):
        import json as _json

        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(PROMOTE_SPEC)
            Path("toy_agent.py").write_text(FAILING)

            res = _invoke(["simulate", "--yes", "--runs", "3"])
            assert res.exit_code == 1, res.output

            from ciagent.promotion import StageStore
            sid = StageStore(Path(".ciagent/staged")).list()[0].stage_id

            # promote as xfail
            res = _invoke(["promote", sid, "--yes", "--xfail"])
            assert res.exit_code == 0, res.output
            golden = Path("golden/sim-test/scenarios/refund-path.json")
            g = load_envelope(golden)
            assert g.provenance["lifecycle"] == "xfail"

            # replay while the bug reproduces: XFAIL → CI green (exit 0)
            res = _invoke(["simulate", "--yes", "--replay", "./golden"])
            assert res.exit_code == 0, res.output
            assert "XFAIL" in res.output

            # json surface carries the lifecycle fold
            res = _invoke(["simulate", "--yes", "--replay", "./golden",
                           "--format", "json"])
            assert res.exit_code == 0, res.output
            payload = _json.loads(res.stdout)
            assert payload["scenarios"][0]["lifecycle"] == "xfail"
            assert payload["summary"]["xfail_expected"] == 1

            # fix the agent → XPASS, still green, flip suggested
            Path("toy_agent.py").write_text(FIXED)
            res = _invoke(["simulate", "--yes", "--replay", "./golden"])
            assert res.exit_code == 0, res.output
            assert "XPASS" in res.output
            assert "promote --flip" in res.output

            # flip → gate lifecycle with flipped_at
            res = _invoke(["promote", "--flip", str(golden), "--yes"])
            assert res.exit_code == 0, res.output
            g = load_envelope(golden)
            assert g.provenance["lifecycle"] == "gate"
            assert g.provenance["flipped_at"]

            # regression returns → gate golden goes red again
            Path("toy_agent.py").write_text(FAILING)
            res = _invoke(["simulate", "--yes", "--replay", "./golden"])
            assert res.exit_code == 1, res.output

    def test_flip_bad_combo_exits_2(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(PROMOTE_SPEC)
            res = r.invoke(cli, ["promote", "--flip", "x", "--xfail"])
            assert res.exit_code == 2, res.output

    def test_flip_without_target_exits_2(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(PROMOTE_SPEC)
            res = r.invoke(cli, ["promote", "--flip"])
            assert res.exit_code == 2, res.output

    def test_flip_gate_golden_refused(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(PROMOTE_SPEC)
            Path("toy_agent.py").write_text(FAILING)
            _invoke(["simulate", "--yes", "--runs", "3"])
            from ciagent.promotion import StageStore
            sid = StageStore(Path(".ciagent/staged")).list()[0].stage_id
            res = _invoke(["promote", sid, "--yes"])  # gate lifecycle
            assert res.exit_code == 0, res.output
            golden = "golden/sim-test/scenarios/refund-path.json"
            res = _invoke(["promote", "--flip", golden, "--yes"])
            assert res.exit_code == 1, res.output
            assert "nothing to flip" in res.output
