# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the converged `ciagent record` command and baseline discovery.

Covers the v2 port (record via spec.runner into versioned baselines that
`ciagent test` discovers) and the deprecated v1 path (single golden trace
from ciagent.yaml).
"""

import json

import pytest
from click.testing import CliRunner

from ciagent.baselines import discover_baselines
from ciagent.cli import cli

FAKE_RUNNER = '''
from ciagent.models import Trace, Span, ToolCall

def run_agent(query):
    span = Span(
        name="agent",
        tool_calls=[ToolCall(tool_name="search_kb", arguments={"q": query})],
        output_data=f"answer to {query}",
    )
    return Trace(
        spans=[span],
        total_cost_usd=0.01,
        metadata={"final_output": f"answer to {query}"},
    )
'''

V2_SPEC = """\
agent: demo-agent
runner: "fake_runner:run_agent"
baseline_dir: ./baselines
queries:
  - query: "What is the refund policy?"
  - query: "How do I reset my password?"
"""

V1_AGENT = '''
def run_agent(input_text):
    return f"handled: {input_text}"
'''

V1_SUITE = """\
name: legacy-suite
agent: "fake_v1_agent:run_agent"
tests:
  - name: test_hello
    input: "hello there"
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Isolated project dir on sys.path with cwd set to it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _write(project, name, content):
    (project / name).write_text(content, encoding="utf-8")


class TestRecordV2:
    def test_records_all_queries_as_versioned_baselines(self, project):
        _write(project, "fake_runner.py", FAKE_RUNNER)
        _write(project, "ciagent_spec.yaml", V2_SPEC)

        result = CliRunner().invoke(cli, ["record", "--json"])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        assert len(payload["saved"]) == 2
        assert payload["failed"] == []

        agent_dir = project / "baselines" / "demo-agent"
        files = sorted(agent_dir.glob("v1-*.json"))
        assert len(files) == 2

        data = json.loads(files[0].read_text())
        assert data["agent"] == "demo-agent"
        assert data["query"] in (
            "What is the refund policy?",
            "How do I reset my password?",
        )
        assert data["trace"]["total_cost_usd"] == 0.01

    def test_records_single_query_by_substring(self, project):
        _write(project, "fake_runner.py", FAKE_RUNNER)
        _write(project, "ciagent_spec.yaml", V2_SPEC)

        result = CliRunner().invoke(cli, ["record", "refund", "--json"])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        assert len(payload["saved"]) == 1
        assert payload["saved"][0]["query"] == "What is the refund policy?"

    def test_ambiguous_target_exits_2(self, project):
        _write(project, "fake_runner.py", FAKE_RUNNER)
        _write(project, "ciagent_spec.yaml", V2_SPEC)

        # "policy" and "password" both contain "p"; use a clearly shared token
        result = CliRunner().invoke(cli, ["record", "?", "--json"])
        assert result.exit_code == 2
        assert "matches" in json.loads(result.output)["error"]

    def test_unknown_target_exits_2_and_lists_queries(self, project):
        _write(project, "fake_runner.py", FAKE_RUNNER)
        _write(project, "ciagent_spec.yaml", V2_SPEC)

        result = CliRunner().invoke(cli, ["record", "nonexistent", "--json"])
        assert result.exit_code == 2
        payload = json.loads(result.output)
        assert "No query matches" in payload["error"]
        assert len(payload["queries"]) == 2

    def test_missing_runner_exits_2(self, project):
        _write(project, "ciagent_spec.yaml",
               V2_SPEC.replace('runner: "fake_runner:run_agent"\n', ""))

        result = CliRunner().invoke(cli, ["record", "--json"])
        assert result.exit_code == 2
        assert "runner" in json.loads(result.output)["error"].lower()

    def test_recorded_baselines_are_discovered_for_test_cmd(self, project):
        """The record → test bridge: what record writes, discovery finds."""
        _write(project, "fake_runner.py", FAKE_RUNNER)
        _write(project, "ciagent_spec.yaml", V2_SPEC)

        result = CliRunner().invoke(cli, ["record", "--json"])
        assert result.exit_code == 0, result.output

        baselines = discover_baselines("./baselines", agent="demo-agent")
        assert set(baselines) == {
            "What is the refund policy?",
            "How do I reset my password?",
        }
        trace = baselines["What is the refund policy?"]
        assert trace.tool_call_sequence == ["search_kb"]


class TestRecordV1Legacy:
    def test_v1_record_still_works_with_deprecation_notice(self, project):
        _write(project, "fake_v1_agent.py", V1_AGENT)
        _write(project, "ciagent.yaml", V1_SUITE)

        result = CliRunner().invoke(cli, ["record", "test_hello", "--json"])
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        assert payload["saved"] is True
        assert (project / "golden" / "test_hello.golden.json").exists()

    def test_v1_record_without_target_exits_2(self, project):
        _write(project, "ciagent.yaml", V1_SUITE)

        result = CliRunner().invoke(cli, ["record", "--json"])
        assert result.exit_code == 2
        assert "required" in json.loads(result.output)["error"].lower()

    def test_v1_record_human_mode_prints_deprecation(self, project):
        _write(project, "fake_v1_agent.py", V1_AGENT)
        _write(project, "ciagent.yaml", V1_SUITE)

        result = CliRunner().invoke(cli, ["record", "test_hello"], input="n\n")
        assert "DEPRECATED" in result.output


class TestDiscoverBaselines:
    def _baseline(self, query, schema_version=1):
        return {
            "schema_version": schema_version,
            "version": "v1",
            "agent": "demo-agent",
            "query": query,
            "metadata": {},
            "trace": {"spans": [], "total_cost_usd": 0.0},
        }

    def test_flat_layout(self, tmp_path):
        (tmp_path / "b.json").write_text(json.dumps(self._baseline("q1")))
        found = discover_baselines(str(tmp_path))
        assert list(found) == ["q1"]

    def test_agent_subdir_layout(self, tmp_path):
        agent_dir = tmp_path / "demo-agent"
        agent_dir.mkdir()
        (agent_dir / "v1-x.json").write_text(json.dumps(self._baseline("q2")))
        found = discover_baselines(str(tmp_path), agent="demo-agent")
        assert list(found) == ["q2"]

    def test_agent_subdir_wins_over_flat_for_same_query(self, tmp_path):
        flat = self._baseline("q")
        flat["trace"]["total_cost_usd"] = 1.0
        (tmp_path / "flat.json").write_text(json.dumps(flat))

        agent_dir = tmp_path / "demo-agent"
        agent_dir.mkdir()
        versioned = self._baseline("q")
        versioned["trace"]["total_cost_usd"] = 2.0
        (agent_dir / "v.json").write_text(json.dumps(versioned))

        found = discover_baselines(str(tmp_path), agent="demo-agent")
        assert found["q"].total_cost_usd == 2.0

    def test_skips_malformed_and_envelope_files(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not json")
        (tmp_path / "envelope.json").write_text(
            json.dumps(self._baseline("conv", schema_version=2)))
        (tmp_path / "noquery.json").write_text(
            json.dumps({"schema_version": 1, "trace": {"spans": []}}))
        (tmp_path / "good.json").write_text(json.dumps(self._baseline("ok")))

        found = discover_baselines(str(tmp_path))
        assert list(found) == ["ok"]

    def test_missing_dir_returns_empty(self, tmp_path):
        assert discover_baselines(str(tmp_path / "nope")) == {}
