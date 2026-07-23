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

    def test_staging_on_by_default_stages_and_gitignores(self):
        # 0.12 default flip (redaction ADR D5): a spec with no `staging:` key
        # stages on failure and scaffolds the gitignore entry.
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write()
            result = _invoke(["simulate", "--yes"])
            assert result.exit_code == 1, result.output
            assert list(Path(".ciagent/staged").rglob("*.json"))
            assert ".ciagent/staged/" in Path(".gitignore").read_text()

    def test_staging_disabled_in_spec_prints_notice(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write("staging: false\n")
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


class TestCaptureTimeRedaction:
    """Integration for the redaction slice (Plan_docs/redaction_capture.md)."""

    LEAKY_AGENT = (
        "def respond(messages):\n"
        "    return 'contact alice@corp.example.org, key sk-abc123DEF456ghi789jkl'\n"
    )

    def _write_leaky(self, spec_extra=""):
        Path("agentci_spec.yaml").write_text(BASE_SPEC + spec_extra)
        Path("toy_failing_agent.py").write_text(self.LEAKY_AGENT)

    def test_staged_file_is_scrubbed_with_counts(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_leaky()
            result = _invoke(["simulate", "--yes"])
            assert result.exit_code == 1, result.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            assert len(staged) == 1
            raw = staged[0].read_text()
            assert "sk-abc123DEF456ghi789jkl" not in raw
            assert "alice@corp.example.org" not in raw
            env = load_envelope(staged[0])
            red = env.staging["redaction"]
            assert red["applied"] is True
            assert red["counts"]["secret"] == 1
            assert red["counts"]["email"] == 1

    def test_failure_summary_scrubbed(self):
        # ADR A1: the correctness message embeds a raw answer preview; the
        # block is attached before the walk so it cannot leak.
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_leaky()
            result = _invoke(["simulate", "--yes"])
            assert result.exit_code == 1, result.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            env = load_envelope(staged[0])
            assert "sk-abc123DEF456ghi789jkl" not in env.staging["failure_summary"]

    def test_redact_false_warns_and_writes_raw(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_leaky("staging:\n  enabled: true\n  redact: false\n")
            result = _invoke(["simulate", "--yes"])
            assert result.exit_code == 1, result.output
            assert "redact is disabled" in result.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            env = load_envelope(staged[0])
            assert env.staging["redaction"]["applied"] is False

    def test_custom_pattern_from_spec(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(
                BASE_SPEC + 'staging:\n  redact_patterns: ["cannot help"]\n'
            )
            Path("toy_failing_agent.py").write_text(FAILING_AGENT)
            result = _invoke(["simulate", "--yes"])
            assert result.exit_code == 1, result.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            raw = staged[0].read_text()
            assert "cannot help" not in raw
            assert "[SECRET:custom#1]" in raw

    def test_show_and_export_scrub_pre_redaction_file(self):
        # A staged file written raw (redact: false) is scrubbed on every
        # show/export path with the current (default) config.
        import json as _json

        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_leaky("staging:\n  enabled: true\n  redact: false\n")
            result = _invoke(["simulate", "--yes"])
            assert result.exit_code == 1, result.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            assert "sk-abc123DEF456ghi789jkl" in staged[0].read_text()

            # flip the spec back to default redact-on for the read paths
            Path("agentci_spec.yaml").write_text(BASE_SPEC)
            sid = staged[0].stem
            res = CliRunner().invoke(cli, ["stage", "show", sid, "--format", "json"])
            assert res.exit_code == 0, res.output
            assert "sk-abc123DEF456ghi789jkl" not in res.stdout
            _json.loads(res.stdout)

            res = CliRunner().invoke(
                cli, ["stage", "show", sid, "--export", "shared.json"]
            )
            assert res.exit_code == 0, res.output
            assert "exported redacted copy" in res.output
            assert "sk-abc123DEF456ghi789jkl" not in Path("shared.json").read_text()

    def test_verify_preserves_redaction_block(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._write_leaky()
            result = _invoke(["simulate", "--yes"])
            assert result.exit_code == 1, result.output
            from ciagent.promotion import StageStore

            sid = StageStore(Path(".ciagent/staged")).list()[0].stage_id
            res = CliRunner().invoke(
                cli, ["stage", "verify", sid, "--mock", "--runs", "2"]
            )
            assert res.exit_code == 0, res.output
            staged = list(Path(".ciagent/staged").rglob("*.json"))
            env = load_envelope(staged[0])
            assert env.staging["redaction"]["applied"] is True
