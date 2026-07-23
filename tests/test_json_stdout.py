# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Issue #39 regression: in `--format json`, stdout must carry exactly one JSON
document. All rich chrome (banner, mode line, notices) goes to stderr instead.

`json.loads(result.stdout)` with no slicing IS the assertion — the historical
workaround was `raw[raw.index('{'):]`, which these tests exist to delete.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from ciagent.cli import cli

SIM_SPEC = """
agent: sim-test
scenarios:
  - name: refund path
    turns: ["hello", "i want a refund"]
    outcome:
      correctness:
        expected_in_answer: ["refund"]
"""

TEST_SPEC = """
agent: q-test
queries:
  - query: "what is the refund policy?"
    expect:
      answer_contains: ["refund"]
"""


class TestSimulateJsonStdout:
    def test_stdout_is_pure_json(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SIM_SPEC)
            res = r.invoke(cli, ["simulate", "--mock", "--format", "json"])
        payload = json.loads(res.stdout)  # no banner-slicing workaround
        assert "scenarios" in payload

    def test_banner_moved_to_stderr(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SIM_SPEC)
            res = r.invoke(cli, ["simulate", "--mock", "--format", "json"])
        assert "CIAgent" not in res.stdout
        assert "CIAgent" in res.stderr

    def test_console_format_keeps_banner_on_stdout(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(SIM_SPEC)
            res = r.invoke(cli, ["simulate", "--mock"])
        assert "CIAgent" in res.stdout

    def test_staging_notices_not_on_stdout(self):
        # Staging chrome (staged notice, gitignore scaffold line) must be
        # stderr — a FAILING run's JSON must still parse. Mock always
        # satisfies checks, so use a toy agent that never says "refund".
        import sys

        spec = SIM_SPEC + 'conversation_runner: "toy_agent:respond"\n'
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(spec)
            Path("toy_agent.py").write_text(
                "def respond(messages):\n    return 'i cannot help'\n"
            )
            sys.path.insert(0, ".")
            try:
                res = r.invoke(
                    cli, ["simulate", "--yes", "--format", "json"]
                )
            finally:
                sys.path.remove(".")
                sys.modules.pop("toy_agent", None)
        payload = json.loads(res.stdout)
        assert payload["summary"]["passed"] < payload["summary"]["total"]
        # staging is default-ON: the staged notice is chrome, on stderr
        assert "staged 1 failing conversation" in res.stderr
        assert "staged" not in res.stdout or json.loads(res.stdout)

    def test_disabled_staging_repro_notice_not_on_stdout(self):
        import sys

        spec = (SIM_SPEC + 'conversation_runner: "toy_agent:respond"\n'
                + 'staging: false\n')
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(spec)
            Path("toy_agent.py").write_text(
                "def respond(messages):\n    return 'i cannot help'\n"
            )
            sys.path.insert(0, ".")
            try:
                res = r.invoke(
                    cli, ["simulate", "--yes", "--format", "json"]
                )
            finally:
                sys.path.remove(".")
                sys.modules.pop("toy_agent", None)
        json.loads(res.stdout)
        assert "repro was found" in res.stderr


class TestTestCmdJsonStdout:
    def test_stdout_is_pure_json(self):
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(TEST_SPEC)
            res = r.invoke(cli, ["test", "--mock", "--format", "json"])
        payload = json.loads(res.stdout)
        assert isinstance(payload, dict)
        assert "CIAgent" not in res.stdout

    def test_chrome_route_resets_between_invocations(self):
        # `console` is module-global: a json run must not leak stderr routing
        # into a later console-mode run in the same process.
        r = CliRunner()
        with r.isolated_filesystem():
            Path("agentci_spec.yaml").write_text(TEST_SPEC)
            r.invoke(cli, ["test", "--mock", "--format", "json"])
            res = r.invoke(cli, ["test", "--mock"])
        assert "CIAgent" in res.stdout
