# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the F5 coding-agent plugin and its CLI enablers.

Three guarantees:
- `agentci bootstrap --yes` records goldens fully non-interactively from a
  runner that returns a plain string (the coding-agent onboarding path).
- The `--format json` output carries the answer text (JSON consumers must see
  what the agent said, not just verdicts).
- The plugin artifacts stay truthful: manifests parse with required fields,
  and every `agentci` command line the skills instruct is one the CLI accepts
  (subcommand exists, flags exist on it).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from agentci.cli import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / "plugins" / "agentci"

TOY_RUNNER = '''
def run_for_agentci(query: str) -> str:
    if "return" in query.lower():
        return "You can return items within 30 days."
    return "Please contact support@example.com."
'''


@pytest.fixture()
def toy_repo(tmp_path, monkeypatch):
    (tmp_path / "toy_runner.py").write_text(TOY_RUNNER)
    (tmp_path / "queries.txt").write_text(
        "what is your return policy?\ndo you sell rocket fuel?\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ── bootstrap --yes (non-interactive golden recording) ─────────────────────────


class TestBootstrapYes:
    def test_string_runner_records_goldens_noninteractive(self, toy_repo):
        result = CliRunner().invoke(
            cli,
            ["bootstrap", "--runner", "toy_runner:run_for_agentci",
             "--queries", "queries.txt", "--agent", "toy", "--yes"],
        )
        assert result.exit_code == 0, result.output
        baselines = list((toy_repo / "baselines" / "toy").glob("*.json"))
        assert len(baselines) == 2
        # baseline carries the captured answer
        data = json.loads(baselines[0].read_text())
        assert data["trace"]["metadata"]["final_output"]
        # spec written and loadable
        spec_file = toy_repo / "agentci_spec.yaml"
        assert spec_file.exists()
        from agentci.loader import load_spec

        spec = load_spec(spec_file)
        assert len(spec.queries) == 2

    def test_bootstrapped_spec_runs_multi_run_test(self, toy_repo):
        r1 = CliRunner().invoke(
            cli,
            ["bootstrap", "--runner", "toy_runner:run_for_agentci",
             "--queries", "queries.txt", "--agent", "toy", "--yes"],
        )
        assert r1.exit_code == 0, r1.output
        r2 = CliRunner().invoke(cli, ["test", "--yes", "--runs", "3"])
        assert r2.exit_code == 0, r2.output
        assert "STABLE" in r2.output

    def test_yes_requires_queries(self, toy_repo):
        result = CliRunner().invoke(
            cli, ["bootstrap", "--runner", "toy_runner:run_for_agentci", "--yes"]
        )
        assert result.exit_code == 2
        assert "--queries" in result.output

    def test_declining_confirm_skips_baseline(self, toy_repo):
        (toy_repo / "one.txt").write_text("what is your return policy?\n")
        result = CliRunner().invoke(
            cli,
            ["bootstrap", "--runner", "toy_runner:run_for_agentci",
             "--queries", "one.txt", "--agent", "toy"],
            input="n\n",
        )
        assert result.exit_code == 0, result.output
        assert not list((toy_repo / "baselines" / "toy").glob("*.json"))


# ── JSON output carries the answer ──────────────────────────────────────────────


class TestJsonAnswerField:
    def test_mock_json_includes_answer(self, toy_repo):
        (toy_repo / "agentci_spec.yaml").write_text(
            """
agent: json-test
queries:
  - query: "what is your return policy?"
    correctness:
      expected_in_answer: ["documentation"]
"""
        )
        result = CliRunner().invoke(cli, ["test", "--mock", "--format", "json"])
        payload = json.loads(result.output[result.output.index("{"):])
        assert payload["results"][0]["answer"]
        assert "documentation" in payload["results"][0]["answer"]

    def test_answer_falls_back_to_span_output(self):
        # Codex review finding: traces whose answer lives only in the last
        # span's output_data (no metadata final_output) must still serialize
        # an answer — same fallback the correctness evaluator uses.
        from agentci.engine.reporter import _serialize_result
        from agentci.engine.results import LayerResult, LayerStatus, QueryResult
        from agentci.models import Span, SpanKind, Trace

        span = Span(kind=SpanKind.AGENT, name="a")
        span.output_data = "answer from span"
        trace = Trace(spans=[span])
        ok = LayerResult(status=LayerStatus.PASS)
        r = QueryResult(query="q", correctness=ok, path=ok, cost=ok, trace=trace)
        assert _serialize_result(r)["answer"] == "answer from span"


# ── Plugin artifact integrity ───────────────────────────────────────────────────


class TestPluginArtifacts:
    def test_marketplace_manifest(self):
        m = json.loads((REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text())
        assert m["name"] == "agentci"
        assert m["owner"]["name"]
        entry = m["plugins"][0]
        assert entry["name"] == "agentci"
        # relative source must resolve inside the repo
        assert (REPO_ROOT / entry["source"]).is_dir()

    def test_plugin_manifest(self):
        p = json.loads((PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text())
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", p["name"])
        assert p["description"]

    @pytest.mark.parametrize("skill", ["onboard", "check"])
    def test_skill_frontmatter(self, skill):
        text = (PLUGIN_DIR / "skills" / skill / "SKILL.md").read_text()
        assert text.startswith("---\n")
        frontmatter = yaml.safe_load(text.split("---\n")[1])
        assert frontmatter["name"] == skill
        # description drives auto-invocation; combined cap is 1536 chars
        assert 50 < len(frontmatter["description"]) <= 1536
        # skills must stay loadable-small
        assert len(text.splitlines()) < 500

    @pytest.mark.parametrize("skill", ["onboard", "check"])
    def test_skill_commands_are_real(self, skill):
        """Every `agentci <cmd> --flag` the skill teaches must exist in the CLI."""
        text = (PLUGIN_DIR / "skills" / skill / "SKILL.md").read_text()
        group_flags = {o for p in cli.params for o in (*p.opts, *p.secondary_opts)}
        for cmd_line in re.findall(r"agentci\s+([a-z-]+)((?:\s+--?[a-zA-Z-]+(?:\s+<[^>]+>|\s+[\w./-]+)?)*)", text):
            sub, flag_blob = cmd_line
            if sub.startswith("-"):
                assert sub in group_flags, (
                    f"skill '{skill}': top-level flag '{sub}' not accepted by the CLI group"
                )
                continue
            assert sub in cli.commands, f"skill '{skill}' references unknown command 'agentci {sub}'"
            param_names = set()
            for param in cli.commands[sub].params:
                param_names.update(param.opts + param.secondary_opts)
            for flag in re.findall(r"--?[a-zA-Z][a-zA-Z-]*", flag_blob):
                assert flag in param_names, (
                    f"skill '{skill}': 'agentci {sub}' does not accept '{flag}'"
                )
