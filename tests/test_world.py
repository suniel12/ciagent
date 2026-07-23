# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Simulated World MVP slice 1: matching, sequence, freeze, decorator seam, CLI.
Plan: Plan_docs/world_sim_mvp.md (A1-A14 binding).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ciagent.cli import cli
from ciagent.conversation import ConversationEnvelope, ConversationTurn
from ciagent.models import Span, SpanKind, ToolCall, Trace
from ciagent.world import (
    Fixture,
    ToolWorld,
    World,
    WorldError,
    WorldMiss,
    activate,
    freeze_envelope,
    world_tool,
)


def make_world(**tools):
    return World({name: tw for name, tw in tools.items()}, name="w", agent="a")


def fx(match, response, **kw):
    return Fixture(match=match, response=response, **kw)


# ── Matching (A1) ───────────────────────────────────────────────────────────────


class TestMatching:
    def test_exact_match_serves(self):
        w = make_world(t=ToolWorld(fixtures=[fx({"a": 1}, "ok")]))
        assert w.serve("t", {"a": 1}) == "ok"

    def test_reusable_fixture_serves_repeatedly(self):
        w = make_world(t=ToolWorld(fixtures=[fx({"a": 1}, "ok")]))
        assert w.serve("t", {"a": 1}) == "ok"
        assert w.serve("t", {"a": 1}) == "ok"

    def test_coerced_scalar_matches(self):
        # pydantic coerces "5" -> 5 between frozen JSON and the runtime call
        w = make_world(t=ToolWorld(fixtures=[fx({"n": "5"}, "ok")]))
        assert w.serve("t", {"n": 5}) == "ok"

    def test_ignored_field_matches_any_value(self):
        w = make_world(t=ToolWorld(
            fixtures=[fx({"id": "INV-1", "reason": "too slow"}, "ok",
                         ignore=["reason"])]))
        assert w.serve("t", {"id": "INV-1", "reason": "totally different"}) == "ok"

    def test_extra_offered_key_is_miss(self):
        w = make_world(t=ToolWorld(fixtures=[fx({"a": 1}, "ok")]))
        with pytest.raises(WorldMiss):
            w.serve("t", {"a": 1, "b": 2})

    def test_extra_key_equal_to_signature_default_matches(self):
        # A1: the framework fills defaults the LLM omitted
        w = make_world(t=ToolWorld(fixtures=[fx({"a": 1}, "ok")]))
        assert w.serve("t", {"a": 1, "limit": 10}, defaults={"limit": 10}) == "ok"

    def test_extra_key_differing_from_default_is_miss(self):
        w = make_world(t=ToolWorld(fixtures=[fx({"a": 1}, "ok")]))
        with pytest.raises(WorldMiss):
            w.serve("t", {"a": 1, "limit": 99}, defaults={"limit": 10})

    def test_unknown_tool_is_miss(self):
        w = make_world(t=ToolWorld(fixtures=[fx({"a": 1}, "ok")]))
        with pytest.raises(WorldMiss):
            w.serve("nope", {"a": 1})

    def test_miss_records_are_authoritative(self):
        # A3: the recorded miss list is the signal, not the exception
        w = make_world(t=ToolWorld(fixtures=[fx({"a": 1}, "ok")]))
        for offered in ({"a": 2}, {"a": 3}):
            with pytest.raises(WorldMiss):
                w.serve("t", offered)
        assert w.report().miss_count == 2

    def test_miss_diff_suggests_ignore(self):
        # A5a: value-only mismatches produce a ready-to-paste suggestion
        w = make_world(t=ToolWorld(
            fixtures=[fx({"id": "INV-1", "reason": "slow"}, "ok")]))
        with pytest.raises(WorldMiss) as e:
            w.serve("t", {"id": "INV-1", "reason": "very slow"})
        assert '"ignore": ["reason"]' in str(e.value)


class TestSequence:
    def _seq_world(self):
        return make_world(t=ToolWorld(sequence=True, fixtures=[
            fx({"id": "INV-1"}, "initiated"),
            fx({"id": "INV-1"}, "already in progress"),
        ]))

    def test_fifo_consumption(self):
        w = self._seq_world()
        assert w.serve("t", {"id": "INV-1"}) == "initiated"
        assert w.serve("t", {"id": "INV-1"}) == "already in progress"

    def test_exhaustion_is_miss(self):
        w = self._seq_world()
        w.serve("t", {"id": "INV-1"})
        w.serve("t", {"id": "INV-1"})
        with pytest.raises(WorldMiss) as e:
            w.serve("t", {"id": "INV-1"})
        assert "consumed" in str(e.value)

    def test_clone_isolates_consumption(self):
        # A4: per-scenario clones have independent sequence state
        w = self._seq_world()
        a, b = w.clone(), w.clone()
        assert a.serve("t", {"id": "INV-1"}) == "initiated"
        assert b.serve("t", {"id": "INV-1"}) == "initiated"
        assert a.report().miss_count == 0
        assert w.report().served == {}

    def test_unconsumed_reported(self):
        w = self._seq_world()
        w.serve("t", {"id": "INV-1"})
        assert w.report().unconsumed == {"t": 1}


class TestAmbiguityInvariant:
    def test_reusable_conflicting_fixtures_rejected(self):
        # A10b: same effective match, different responses, sequence: false
        with pytest.raises(WorldError, match="ambiguous"):
            make_world(t=ToolWorld(fixtures=[
                fx({"a": 1}, "x"), fx({"a": 1}, "y"),
            ]))

    def test_ignore_edit_can_introduce_ambiguity(self):
        with pytest.raises(WorldError, match="ambiguous"):
            make_world(t=ToolWorld(fixtures=[
                fx({"a": 1, "ts": "t1"}, "x", ignore=["ts"]),
                fx({"a": 1, "ts": "t2"}, "y", ignore=["ts"]),
            ]))


# ── Freeze (D3, A9, A10, A5b) ───────────────────────────────────────────────────


def env_with_calls(calls, mode="simulated"):
    """calls: list of (turn_index, tool, args, result)."""
    turns = {}
    for turn_index, tool, args, result in calls:
        turns.setdefault(turn_index, []).append(
            ToolCall(tool_name=tool, arguments=args, result=result))
    convo = []
    for turn_index in sorted(turns):
        span = Span(kind=SpanKind.AGENT, name="agent")
        span.tool_calls = turns[turn_index]
        trace = Trace(agent_name="a", test_name="q", spans=[span])
        trace.metadata["final_output"] = "answer"
        convo.append(ConversationTurn(turn_index=turn_index, user_message="hi",
                                      trace=trace))
    return ConversationEnvelope(
        mode=mode, agent="a",
        scenario={"name": "s", "spec": {"name": "s", "turns": ["hi"]}},
        turns=convo,
    )


class TestFreeze:
    def test_freeze_groups_and_serves(self):
        env = env_with_calls([(0, "lookup", {"email": "alice@corp.example.org"}, "found")])
        w = freeze_envelope(env)
        assert w.serve("lookup", {"email": "alice@corp.example.org"}) == "found"

    def test_zero_tool_calls_refused(self):
        env = env_with_calls([])
        env.turns = [ConversationTurn(
            turn_index=0, user_message="hi",
            trace=Trace(agent_name="a", test_name="q", spans=[]))]
        with pytest.raises(WorldError, match="no tool calls"):
            freeze_envelope(env)

    def test_duplicate_identical_calls_dedupe_reusable(self):
        env = env_with_calls([
            (0, "lookup", {"email": "x"}, "found"),
            (1, "lookup", {"email": "x"}, "found"),
        ])
        w = freeze_envelope(env)
        assert not w.tools["lookup"].sequence
        assert len(w.tools["lookup"].fixtures) == 1

    def test_same_args_new_result_becomes_sequence(self):
        env = env_with_calls([
            (0, "refund", {"id": "INV-1"}, "initiated"),
            (1, "refund", {"id": "INV-1"}, "already in progress"),
        ])
        w = freeze_envelope(env)
        assert w.tools["refund"].sequence
        assert len(w.tools["refund"].fixtures) == 2
        assert w.serve("refund", {"id": "INV-1"}) == "initiated"
        assert w.serve("refund", {"id": "INV-1"}) == "already in progress"

    def test_gaps_refused_without_allow(self):
        env = env_with_calls([
            (0, "lookup", {"email": "x"}, "found"),
            (0, "verify", {"email": "x"}, None),
        ])
        with pytest.raises(WorldError, match="WILL miss"):
            freeze_envelope(env)

    def test_allow_gaps_records_them(self):
        env = env_with_calls([
            (0, "lookup", {"email": "x"}, "found"),
            (1, "verify", {"email": "x"}, None),
        ])
        w = freeze_envelope(env, allow_gaps=True)
        assert w.gaps == [{"tool": "verify", "args": {"email": "x"}, "turn": 1}]
        with pytest.raises(WorldMiss) as e:
            w.serve("verify", {"email": "x"})
        assert "without" in str(e.value).lower() or "gap" in str(e.value).lower()

    def test_tools_filter(self):
        env = env_with_calls([
            (0, "lookup", {"email": "x"}, "found"),
            (0, "node_emit", {"content": "..."}, "..."),
        ])
        w = freeze_envelope(env, tools_filter=["lookup"])
        assert list(w.tools) == ["lookup"]

    def test_suggested_ignore_from_varying_field(self):
        env = env_with_calls([
            (0, "refund", {"id": "INV-1", "reason": "too slow"}, "ok-1"),
            (1, "refund", {"id": "INV-2", "reason": "too slow"}, "ok-2"),
        ])
        w = freeze_envelope(env)
        assert "id" in w.tools["refund"].suggested_ignore

    def test_suggested_ignore_long_free_text(self):
        long_text = "customer says " + "very " * 20 + "unhappy"
        env = env_with_calls([(0, "log", {"note": long_text}, "ok")])
        w = freeze_envelope(env)
        assert "note" in w.tools["log"].suggested_ignore

    def test_roundtrip(self, tmp_path):
        env = env_with_calls([
            (0, "refund", {"id": "INV-1"}, "initiated"),
            (1, "refund", {"id": "INV-1"}, "again"),
            (0, "lookup", {"email": "x"}, {"structured": ["r1", "r2"]}),
        ])
        w = freeze_envelope(env)
        p = w.save(tmp_path / "w.world.json")
        w2 = World.load(p)
        assert w2.serve("lookup", {"email": "x"}) == {"structured": ["r1", "r2"]}
        assert w2.serve("refund", {"id": "INV-1"}) == "initiated"

    def test_load_rejects_unknown_schema(self, tmp_path):
        p = tmp_path / "bad.world.json"
        p.write_text(json.dumps({"world_schema": 99, "tools": {}}))
        with pytest.raises(WorldError, match="world_schema"):
            World.load(p)


# ── The decorator seam (D1, A2) ─────────────────────────────────────────────────


class TestWorldTool:
    def test_passthrough_without_world(self):
        @world_tool
        def lookup(email: str) -> str:
            return f"live:{email}"
        assert lookup("x@y.z") == "live:x@y.z"

    def test_serves_when_active(self):
        @world_tool
        def lookup(email: str) -> str:
            return "live"
        w = make_world(lookup=ToolWorld(fixtures=[fx({"email": "x"}, "frozen")]))
        with activate(w):
            assert lookup("x") == "frozen"
        assert lookup("x") == "live"

    def test_default_filled_by_framework_matches(self):
        @world_tool
        def search(q: str, limit: int = 10) -> str:
            return "live"
        w = make_world(search=ToolWorld(fixtures=[fx({"q": "hi"}, "frozen")]))
        with activate(w):
            # framework passes the default positionally even when LLM omitted it
            assert search("hi", 10) == "frozen"

    def test_async_tool(self):
        @world_tool
        async def lookup(email: str) -> str:
            return "live"
        w = make_world(lookup=ToolWorld(fixtures=[fx({"email": "x"}, "frozen")]))
        with activate(w):
            assert asyncio.run(lookup("x")) == "frozen"
        assert asyncio.run(lookup("x")) == "live"

    def test_context_param_stripped(self):
        class RunContextWrapper:  # stand-in matching by annotation name
            pass

        @world_tool
        def lookup(ctx: RunContextWrapper, email: str) -> str:
            return "live"
        w = make_world(lookup=ToolWorld(fixtures=[fx({"email": "x"}, "frozen")]))
        with activate(w):
            assert lookup(RunContextWrapper(), "x") == "frozen"

    def test_rejects_non_function(self):
        class FunctionTool:
            pass
        with pytest.raises(TypeError, match="INNERMOST"):
            world_tool(FunctionTool())

    def test_fail_closed_never_calls_real_fn(self):
        calls = []

        @world_tool
        def lookup(email: str) -> str:
            calls.append(email)
            return "live"
        w = make_world(lookup=ToolWorld(fixtures=[fx({"email": "x"}, "frozen")]))
        with activate(w):
            with pytest.raises(WorldMiss):
                lookup("other@y.z")
        assert calls == []
        assert w.report().miss_count == 1


# ── CLI: world freeze / show ────────────────────────────────────────────────────


class TestWorldCLI:
    SPEC = "agent: a\nbaseline_dir: ./golden\nscenarios:\n  - name: s\n    turns: [hi]\n"

    def _write_golden(self, path="golden.json", leaky=False):
        from ciagent.conversation import save_envelope
        result = "found sk-abc123DEF456ghi789jkl" if leaky else "found"
        env = env_with_calls([(0, "lookup", {"email": "alice@corp.example.org"}, result)])
        save_envelope(env, Path(path))
        return path

    def test_freeze_from_golden_and_show(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(self.SPEC)
            # golden with raw email: default redactor rewrites it → refusal
            # without --force-redact (A8)
            p = self._write_golden()
            res = r.invoke(cli, ["world", "freeze", p])
            assert res.exit_code == 2, res.output
            assert "force-redact" in res.output

            res = r.invoke(cli, ["world", "freeze", p, "--force-redact"])
            assert res.exit_code == 0, res.output
            out = Path("worlds/s.world.json")
            assert out.exists()
            w = World.load(out)
            # fixture inherited envelope-level redaction (A8)
            [f] = w.tools["lookup"].fixtures
            assert f.match["email"] == "redacted-1@example.com"

            res = r.invoke(cli, ["world", "show", str(out), "--format", "json"])
            assert res.exit_code == 0, res.output
            payload = json.loads(res.stdout)
            assert payload["world_schema"] == 1

    def test_freeze_redaction_is_envelope_level_not_per_fixture(self):
        # A8: two different emails must get DIFFERENT placeholders
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(self.SPEC)
            from ciagent.conversation import save_envelope
            env = env_with_calls([
                (0, "lookup", {"email": "a@x.co"}, "found a"),
                (1, "lookup", {"email": "b@y.io"}, "found b"),
            ])
            save_envelope(env, Path("g.json"))
            res = r.invoke(cli, ["world", "freeze", "g.json", "--force-redact"])
            assert res.exit_code == 0, res.output
            w = World.load(Path("worlds/s.world.json"))
            emails = sorted(f.match["email"] for f in w.tools["lookup"].fixtures)
            assert emails == ["redacted-1@example.com", "redacted-2@example.com"]

    def test_freeze_missing_source_exits_1(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(self.SPEC)
            res = r.invoke(cli, ["world", "freeze", "no-such-stage-id"])
            assert res.exit_code == 1, res.output

    def test_freeze_from_staged_entry(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(self.SPEC)
            from ciagent.promotion import StageStore
            from ciagent.redaction import Redactor
            store = StageStore(Path(".ciagent/staged"), redactor=Redactor())
            env = env_with_calls([(0, "lookup", {"email": "alice@corp.example.org"}, "found")])
            store.stage(env, staging_block={
                "run_id": "r1", "scenario_id": "s", "source": "simulate",
                "classification": "consistent", "runs_observed": 3,
                "verdicts": [False] * 3, "flip_source": None,
                "flip_reason": "", "failure_summary": "f",
            })
            sid = store.list()[0].stage_id
            res = r.invoke(cli, ["world", "freeze", sid])
            assert res.exit_code == 0, res.output
            w = World.load(Path("worlds/s.world.json"))
            # staged file was already redacted at capture; freeze re-redaction
            # is a no-op and requires no flag
            [f] = w.tools["lookup"].fixtures
            assert f.match["email"] == "redacted-1@example.com"


# ── Slice 2: replay integration ─────────────────────────────────────────────────

WORLD_SPEC = """
agent: sim-test
baseline_dir: ./golden
conversation_runner: "toy_world_agent:respond"
scenarios:
  - name: refund path
    turns: ["i want a refund for INV-1"]
    outcome:
      correctness:
        expected_in_answer: ["refund initiated"]
"""

# A deterministic toy agent: answers from its (wrapped) tool. When BACKEND_UP
# is deleted, a live call raises — proving replay hit the frozen world.
TOY_AGENT = '''
import os
from ciagent.world import world_tool
from ciagent.models import Span, SpanKind, Trace, ToolCall

@world_tool
def process_refund(invoice_id: str) -> str:
    if not os.path.exists("BACKEND_UP"):
        raise RuntimeError("backend is gone")
    return "refund is stuck"          # the bug: never says "initiated"

def respond(messages):
    result = process_refund("INV-1")
    span = Span(kind=SpanKind.AGENT, name="agent")
    span.tool_calls = [ToolCall(tool_name="process_refund",
                                arguments={"invoice_id": "INV-1"},
                                result=result)]
    span.output_data = result
    t = Trace(agent_name="sim-test", test_name="q", spans=[span])
    t.metadata["final_output"] = result
    t.compute_metrics()
    return t
'''

# Divergent agent: calls the tool with DIFFERENT args than the frozen run.
TOY_AGENT_DIVERGED = TOY_AGENT.replace(
    'process_refund("INV-1")', 'process_refund("INV-999")'
).replace('if not os.path.exists("BACKEND_UP"):\n        raise', 'if False:\n        raise')


def _winvoke(args):
    import sys as _sys
    _sys.path.insert(0, ".")
    try:
        return CliRunner().invoke(cli, args)
    finally:
        _sys.path.remove(".")
        _sys.modules.pop("toy_world_agent", None)


class TestWorldReplayLoop:
    def _setup(self):
        Path("agentci_spec.yaml").write_text(WORLD_SPEC)
        Path("toy_world_agent.py").write_text(TOY_AGENT)
        Path("BACKEND_UP").write_text("1")

    def test_full_loop_freeze_then_replay_without_backend(self):
        r = CliRunner()
        with r.isolated_filesystem():
            self._setup()
            # 1. live failing run stages (default-ON staging)
            res = _winvoke(["simulate", "--yes"])
            assert res.exit_code == 1, res.output
            from ciagent.promotion import StageStore
            store = StageStore(Path(".ciagent/staged"))
            sid = store.list()[0].stage_id

            # 2. freeze the failing run's tool traffic
            res = _winvoke(["world", "freeze", sid])
            assert res.exit_code == 0, res.output
            world_file = next(Path("worlds").glob("*.world.json"))

            # 3. promote the repro to a golden
            res = _winvoke(["promote", sid, "--yes", "--force"])
            assert res.exit_code == 0, res.output

            # 4. the backend dies — live replay would crash; world replay
            #    serves the frozen response and the gate still fires (exit 1,
            #    the bug still reproduces on the frozen world)
            Path("BACKEND_UP").unlink()
            res = _winvoke(["simulate", "--yes", "--replay", "./golden",
                            "--world", str(world_file)])
            assert res.exit_code == 1, res.output
            assert "World:" in res.output
            assert "0 miss(es)" in res.output

    def test_divergence_is_world_miss_exit_1(self):
        import json as _json

        r = CliRunner()
        with r.isolated_filesystem():
            self._setup()
            _winvoke(["simulate", "--yes"])
            from ciagent.promotion import StageStore
            sid = StageStore(Path(".ciagent/staged")).list()[0].stage_id
            _winvoke(["world", "freeze", sid])
            world_file = next(Path("worlds").glob("*.world.json"))
            _winvoke(["promote", sid, "--yes", "--force"])

            # agent now calls the tool with different args → fail-closed miss
            Path("toy_world_agent.py").write_text(TOY_AGENT_DIVERGED)
            res = _winvoke(["simulate", "--yes", "--replay", "./golden",
                            "--world", str(world_file), "--format", "json"])
            assert res.exit_code == 1, res.output
            payload = _json.loads(res.stdout)
            assert payload["summary"]["world_misses"] >= 1
            assert payload["scenarios"][0]["world_misses"] >= 1

    def test_world_requires_replay(self):
        r = CliRunner()
        with r.isolated_filesystem():
            self._setup()
            Path("w.world.json").write_text('{"world_schema": 1, "tools": {}}')
            res = _winvoke(["simulate", "--yes", "--world", "w.world.json"])
            assert res.exit_code == 2, res.output
            assert "requires --replay" in res.output

    def test_world_rejects_mock(self):
        r = CliRunner()
        with r.isolated_filesystem():
            self._setup()
            Path("w.world.json").write_text('{"world_schema": 1, "tools": {}}')
            res = _winvoke(["simulate", "--mock", "--replay", ".",
                            "--world", "w.world.json"])
            assert res.exit_code == 2, res.output

    def test_verify_with_world(self):
        r = CliRunner()
        with r.isolated_filesystem():
            self._setup()
            _winvoke(["simulate", "--yes", "--runs", "2"])
            from ciagent.promotion import StageStore
            store = StageStore(Path(".ciagent/staged"))
            sid = store.list()[0].stage_id
            res = _winvoke(["world", "freeze", sid])
            assert res.exit_code == 0, res.output
            world_file = next(Path("worlds").glob("*.world.json"))

            Path("BACKEND_UP").unlink()   # backend gone; world serves
            res = _winvoke(["stage", "verify", sid, "--yes", "--runs", "2",
                            "--world", str(world_file)])
            assert res.exit_code == 0, res.output
            _path, env = store.load(sid)
            assert env.staging["verified_via"] == "replay+world"
            assert env.staging["classification"] == "consistent"

    def test_verify_all_runs_missed_leaves_block_untouched(self):
        r = CliRunner()
        with r.isolated_filesystem():
            self._setup()
            _winvoke(["simulate", "--yes"])
            from ciagent.promotion import StageStore
            store = StageStore(Path(".ciagent/staged"))
            sid = store.list()[0].stage_id
            _winvoke(["world", "freeze", sid])
            world_file = next(Path("worlds").glob("*.world.json"))
            before = store.load(sid)[1].staging["classification"]

            Path("toy_world_agent.py").write_text(TOY_AGENT_DIVERGED)
            res = _winvoke(["stage", "verify", sid, "--yes", "--runs", "2",
                            "--world", str(world_file)])
            assert res.exit_code == 1, res.output
            assert "diverged from the frozen world" in res.output
            assert store.load(sid)[1].staging["classification"] == before
