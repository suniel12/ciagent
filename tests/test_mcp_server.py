# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
MCP server (Plan_docs/mcp_server.md, A1-A13 binding).

Tool logic is plain async functions; these tests exercise them directly —
no `mcp` package needed except for the one build_server test.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ciagent.mcp_server import (
    SUPPORTS_JSON,
    SUPPORTS_YES,
    GuardrailRefused,
    ServerConfig,
    jail,
    make_envelope,
    require_live_ack,
    tool_promote,
    tool_simulate,
    tool_stage_list,
    tool_stage_show,
    tool_test,
    tool_world_freeze,
    tool_world_show,
)

QA_SPEC = """
agent: qa-test
baseline_dir: ./golden
runner: "toy_qa:run"
queries:
  - query: "what is the refund policy?"
    correctness:
      expected_in_answer: ["refund"]
"""

TOY_RUNNER = '''
from ciagent.models import Span, SpanKind, Trace, ToolCall

def run(query):
    span = Span(kind=SpanKind.AGENT, name="agent")
    span.output_data = "i cannot help"
    span.tool_calls = [ToolCall(tool_name="lookup", arguments={"q": query},
                                result="nothing found")]
    t = Trace(agent_name="qa-test", test_name=query, spans=[span])
    t.metadata["final_output"] = "i cannot help"
    t.compute_metrics()
    return t
'''


def cfg_for(tmp_path) -> ServerConfig:
    return ServerConfig(project_root=Path(tmp_path))


def run(coro):
    return asyncio.run(coro)


# ── A1: appended flags actually exist on every command ─────────────────────────


class TestFlagCapabilities:
    @pytest.mark.parametrize("subcmd,flag", (
        [(sc, "--format") for sc in sorted(SUPPORTS_JSON)]
        + [(sc, "--yes") for sc in sorted(SUPPORTS_YES)]
    ))
    def test_flag_exists(self, subcmd, flag):
        out = subprocess.run(
            [sys.executable, "-m", "ciagent.cli", *subcmd, "--help"],
            capture_output=True, text=True,
        )
        assert flag in out.stdout, f"{subcmd} lacks {flag}"

    def test_python_dash_m_entrypoints(self):
        for mod in ("ciagent", "ciagent.cli"):
            out = subprocess.run([sys.executable, "-m", mod, "--version"],
                                 capture_output=True, text=True)
            assert out.returncode == 0 and "ciagent" in out.stdout


# ── Guardrails ──────────────────────────────────────────────────────────────────


class TestGuardrails:
    def test_live_simulate_requires_max_cost(self, tmp_path):
        env = run(tool_simulate(cfg_for(tmp_path), mock=False))
        assert env["ok"] is False and "max_cost" in env["error"]

    def test_live_world_replay_requires_max_cost(self, tmp_path):
        env = run(tool_simulate(cfg_for(tmp_path), mock=False,
                                replay="golden", world="w.json"))
        assert env["ok"] is False and "max_cost" in env["error"]

    def test_live_test_requires_allow_live(self, tmp_path):
        env = run(tool_test(cfg_for(tmp_path), mock=False))
        assert env["ok"] is False and "allow_live" in env["error"]

    def test_jail_blocks_escape(self, tmp_path):
        with pytest.raises(GuardrailRefused):
            jail(cfg_for(tmp_path), "../outside.json")

    def test_jail_blocks_absolute_escape(self, tmp_path):
        with pytest.raises(GuardrailRefused):
            jail(cfg_for(tmp_path), "/etc/passwd")

    def test_jail_allows_inside(self, tmp_path):
        (tmp_path / "w.json").write_text("{}")
        assert jail(cfg_for(tmp_path), "w.json", must_exist=True)

    def test_promote_requires_stage_id(self, tmp_path):
        env = run(tool_promote(cfg_for(tmp_path), stage_id=""))
        assert env["ok"] is False and "stage_id" in env["error"]

    def test_mock_never_refused(self):
        require_live_ack(mock=True, tool="x")  # no raise


# ── Envelope shaping (A3/A7/A12) ────────────────────────────────────────────────


class TestEnvelope:
    def test_json_parsed(self, tmp_path):
        env = make_envelope(cfg_for(tmp_path), ("test",), ["test"], 1,
                            '{"a": 1}', "chrome")
        assert env["ok"] and env["data"] == {"a": 1} and env["exit_code"] == 1

    def test_empty_stdout_nonzero_exit_is_not_a_parse_failure(self, tmp_path):
        env = make_envelope(cfg_for(tmp_path), ("promote",), ["promote"], 1,
                            "", "Refused: held")
        assert env["ok"] is True and env["data"] is None
        assert "Refused" in env["stderr_tail"]

    def test_no_json_command_carries_stdout_text(self, tmp_path):
        env = make_envelope(cfg_for(tmp_path), ("stage", "verify"),
                            ["stage", "verify"], 0, "re-classified x", "")
        assert env["data"] is None and "re-classified" in env["stdout_text"]

    def test_oversized_data_truncated_to_file(self, tmp_path):
        cfg = cfg_for(tmp_path)
        cfg.data_cap_bytes = 100
        big = {"summary": {"total": 1}, "stability": None, "recorded": [],
               "scenarios": [{"name": "s", "hard_fail": True,
                              "turns": ["x" * 500]}]}
        env = make_envelope(cfg, ("simulate",), ["simulate"], 1,
                            json.dumps(big), "")
        assert env["data_truncated"] is True
        assert Path(env["data_file"]).exists()
        assert env["data"]["summary"] == {"total": 1}
        assert "turns" not in env["data"]["scenarios"][0]


# ── Integration: real subprocesses against the real CLI ─────────────────────────


class TestIntegration:
    def _project(self, tmp_path) -> ServerConfig:
        (tmp_path / "agentci_spec.yaml").write_text(QA_SPEC)
        (tmp_path / "toy_qa.py").write_text(TOY_RUNNER)
        return cfg_for(tmp_path)

    def test_full_loop_test_stage_show_promote_freeze(self, tmp_path):
        cfg = self._project(tmp_path)

        # live failing test run (toy runner, no LLM — allow_live acknowledges)
        env = run(tool_test(cfg, mock=False, allow_live=True, runs=2))
        assert env["ok"], env
        assert env["exit_code"] == 1  # the gate detected the failure

        env = run(tool_stage_list(cfg))
        assert env["ok"] and len(env["data"]) == 1
        sid = env["data"][0]["id"]
        assert env["data"][0]["classification"] == "consistent"

        env = run(tool_stage_show(cfg, stage_id=sid))
        assert env["ok"] and env["data"]["agent"] == "qa-test"

        env = run(tool_world_freeze(cfg, source=sid))
        assert env["exit_code"] == 0, env
        assert Path(env["world_file"]).exists()

        env = run(tool_world_show(cfg, path=env["world_file"]))
        assert env["ok"] and "lookup" in env["data"]["tools"]

        env = run(tool_promote(cfg, stage_id=sid))
        assert env["ok"] and env["exit_code"] == 0, env
        assert env["data"]["lifecycle"] == "gate"

    def test_mock_test_run(self, tmp_path):
        cfg = self._project(tmp_path)
        env = run(tool_test(cfg, mock=True))
        assert env["ok"], env

    def test_timeout_kills_process_group(self, tmp_path):
        cfg = self._project(tmp_path)
        cfg.timeout_s = 1
        (tmp_path / "toy_qa.py").write_text(
            "import time\n" + TOY_RUNNER.replace(
                "def run(query):", "def run(query):\n    time.sleep(30)")
        )
        env = run(tool_test(cfg, mock=False, allow_live=True))
        assert env["ok"] is False and "timed out" in env["error"]


class TestBuildServer:
    def test_registers_tools(self, tmp_path):
        pytest.importorskip("mcp")
        from ciagent.mcp_server import build_server

        s = build_server(cfg_for(tmp_path))
        names = {t.name for t in run(s.list_tools())}
        assert {"ciagent_test", "ciagent_simulate", "ciagent_stage_list",
                "ciagent_promote", "ciagent_world_freeze",
                "ciagent_import"} <= names
