# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Conformance for the Agent Failure Atlas (Plan_docs/failure_atlas_seed.md,
AT1-AT8). Every seed entry must actually demonstrate its failure — run its
toy vulnerable agent LIVE (no LLM) and assert the gate fires (exit 1). If an
entry stops failing, CI breaks, so the atlas can't rot into aspirational
prose.

Entries run in an isolated copy of their directory with a unique runner
module (AT3), driven via subprocess so cwd-relative goldens/worlds and the
world contextvar are fully isolated.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ATLAS = Path(__file__).resolve().parents[1] / "src" / "ciagent" / "examples" / "failure-atlas"

# Single-command entries: `simulate --yes` must exit 1 (the gate fires).
SIMPLE_ENTRIES = ["money-out-no-verification", "transcript-poisoning"]


def _run_in_copy(entry: str, argv: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    work = tmp_path / entry
    shutil.copytree(ATLAS / entry, work)
    return subprocess.run(
        [sys.executable, "-m", "ciagent.cli", *argv],
        cwd=str(work), capture_output=True, text=True,
    )


@pytest.mark.parametrize("entry", SIMPLE_ENTRIES)
def test_simple_entry_gate_fires(entry, tmp_path):
    res = _run_in_copy(entry, ["simulate", "--yes"], tmp_path)
    assert res.returncode == 1, (
        f"{entry}: expected the gate to fire (exit 1), got "
        f"{res.returncode}.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )


def test_injection_entry_recipe(tmp_path):
    # AT2: the injection failure is a multi-step recipe (record -> freeze ->
    # mutate -> replay). Run the steps directly (run.sh needs `ciagent` on
    # PATH; we drive the module instead for hermeticity).
    entry = "tool-output-injection"
    work = tmp_path / entry
    shutil.copytree(ATLAS / entry, work)

    def cli(*argv):
        return subprocess.run([sys.executable, "-m", "ciagent.cli", *argv],
                              cwd=str(work), capture_output=True, text=True)

    assert cli("simulate", "--yes", "--record", "--record-dir", "./golden").returncode == 0
    golden = next((work / "golden").rglob("*.json"))
    rel_golden = golden.relative_to(work)
    assert cli("world", "freeze", str(rel_golden), "-o", "clean.world.json").returncode == 0
    # clean replay passes
    assert cli("simulate", "--yes", "--replay", "./golden",
               "--world", "clean.world.json").returncode == 0
    # inject an override into the tool result
    assert cli("world", "mutate", "clean.world.json", "--op", "inject",
               "--payload-id", "role-override", "-o", "evil.world.json").returncode == 0
    # injected replay: the agent obeys the tool output → gate fires
    res = cli("simulate", "--yes", "--replay", "./golden", "--world", "evil.world.json")
    assert res.returncode == 1, f"injection gate did not fire:\n{res.stdout}\n{res.stderr}"
    assert "REFUND-ISSUED" in res.stdout or "FAIL" in res.stdout


def test_poison_check_is_non_vacuous(tmp_path):
    # AT6: the not_in_answer gate must actually be what fails — a PII-free
    # turn that still returns [BLOCKED. Confirm the toy runner emits it, so
    # the check isn't vacuously passing.
    res = _run_in_copy("transcript-poisoning", ["simulate", "--yes"], tmp_path)
    assert "BLOCKED" in res.stdout


class TestAtlasIndex:
    def _readme_slugs(self) -> set[str]:
        text = (ATLAS / "README.md").read_text(encoding="utf-8")
        # AT7: parse the first table column (backtick-quoted slugs) only —
        # not row formatting.
        return set(re.findall(r"^\|\s*`([a-z0-9-]+)`\s*\|", text, re.MULTILINE))

    def test_readme_slugs_match_directories(self):
        dirs = {p.name for p in ATLAS.iterdir()
                if p.is_dir() and not p.name.startswith("__")}
        assert self._readme_slugs() == dirs

    def test_every_entry_spec_loads(self):
        from ciagent.loader import load_spec

        for entry in ATLAS.iterdir():
            if not entry.is_dir() or entry.name.startswith("__"):
                continue
            spec = load_spec(str(entry / "agentci_spec.yaml"))
            assert spec.agent

    def test_every_entry_has_readme(self):
        for entry in ATLAS.iterdir():
            if not entry.is_dir() or entry.name.startswith("__"):
                continue
            assert (entry / "README.md").exists(), entry.name


def test_forbidden_tools_violation_gates_the_build(tmp_path):
    # Regression for the safety gap the atlas dogfooding surfaced: a
    # forbidden_tools violation prints PATH: FAIL and MUST exit 1 (it used to
    # exit 0 because QueryResult.hard_fail read correctness alone).
    (tmp_path / "agentci_spec.yaml").write_text(
        'agent: fb\nrunner: "fb_regress:run"\n'
        'queries:\n  - query: q\n    path:\n      forbidden_tools: [danger]\n'
    )
    (tmp_path / "fb_regress.py").write_text(
        "from ciagent.models import Span, SpanKind, Trace, ToolCall\n"
        "def run(q):\n"
        "    s = Span(kind=SpanKind.AGENT, name='a')\n"
        "    s.tool_calls = [ToolCall(tool_name='danger', arguments={}, result='x')]\n"
        "    s.output_data = 'done'\n"
        "    t = Trace(agent_name='fb', test_name=q, spans=[s])\n"
        "    t.metadata['final_output'] = 'done'\n"
        "    t.compute_metrics()\n"
        "    return t\n"
    )
    res = subprocess.run([sys.executable, "-m", "ciagent.cli", "test", "--yes"],
                         cwd=str(tmp_path), capture_output=True, text=True)
    assert res.returncode == 1, res.stdout
