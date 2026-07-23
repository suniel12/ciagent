# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Step 4 — auto-stage in simulate_cmd (opt-in v1).

A failing simulate conversation stages + classifies when staging is enabled,
prints a one-line notice when it is off, and a staging error never changes the
run's exit code. Uses a live toy failing agent because the mock runner always
satisfies checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

from ciagent.cli import cli
from ciagent.conversation import load_envelope

BASE_SPEC = """
agent: sim-test
baseline_dir: ./golden
conversation_runner: "toy_failing_agent:respond"
scenarios:
  - name: refund path
    turns: ["hello", "i want a refund"]
    outcome:
      correctness:
        expected_in_answer: ["refund"]
"""

FAILING_AGENT = "def respond(messages):\n    return 'i cannot help'\n"


def _write(spec_extra: str = ""):
    Path("agentci_spec.yaml").write_text(BASE_SPEC + spec_extra)
    Path("toy_failing_agent.py").write_text(FAILING_AGENT)


def _invoke(args):
    sys.path.insert(0, ".")
    try:
        return CliRunner().invoke(cli, args)
    finally:
        sys.path.remove(".")
        sys.modules.pop("toy_failing_agent", None)


class TestSimulateAutoStage:
    def test_failure_stages_when_enabled_via_flag(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write()
            result = _invoke(["simulate", "--yes", "--stage"])
            assert result.exit_code == 1, result.output  # gate lifecycle: red
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            assert len(staged) == 1, result.output
            env = load_envelope(staged[0])
            assert env.staging is not None
            # single run (default --runs 1) → unverified
            assert env.staging["classification"] == "unverified"
            assert env.staging["source"] == "simulate"
            assert env.staging["failure_summary"]

    def test_multi_run_failure_classifies_consistent(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write()
            result = _invoke(["simulate", "--yes", "--stage", "--runs", "3"])
            assert result.exit_code == 1, result.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            assert len(staged) == 1
            env = load_envelope(staged[0])
            # deterministic toy agent fails every run, no flip → consistent
            assert env.staging["classification"] == "consistent"
            assert env.staging["runs_observed"] == 3

    def test_staging_off_by_default_prints_notice(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write()
            result = _invoke(["simulate", "--yes"])
            assert result.exit_code == 1, result.output
            assert "enable staging" in result.output
            assert not Path(".ciagent/staged").exists()

    def test_no_stage_flag_overrides_spec_enabled(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write("staging:\n  enabled: true\n")
            result = _invoke(["simulate", "--yes", "--no-stage"])
            assert result.exit_code == 1, result.output
            assert not Path(".ciagent/staged").exists()

    def test_spec_enabled_stages_without_flag(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write("staging:\n  enabled: true\n")
            result = _invoke(["simulate", "--yes"])
            assert result.exit_code == 1
            assert list(Path(".ciagent/staged").rglob("*.json"))

    def test_staging_error_does_not_change_exit_code(self, monkeypatch):
        # Force StageStore.stage to blow up; the run must still exit 1 (the
        # failure), staging is best-effort and never load-bearing.
        import ciagent.promotion as promotion

        def boom(self, *a, **k):
            raise RuntimeError("disk full")

        runner = CliRunner()
        with runner.isolated_filesystem():
            _write()
            monkeypatch.setattr(promotion.StageStore, "stage", boom)
            result = _invoke(["simulate", "--yes", "--stage"])
            assert result.exit_code == 1, result.output
            assert "staging warning" in result.output

    def test_passing_scenario_stages_nothing(self):
        # Mock runner satisfies checks → no failure → nothing staged, exit 0.
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write("staging:\n  enabled: true\n")
            result = CliRunner().invoke(cli, ["simulate", "--mock", "--stage"])
            assert result.exit_code == 0, result.output
            assert not Path(".ciagent/staged").exists()
