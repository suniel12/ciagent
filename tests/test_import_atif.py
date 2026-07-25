# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ATIF trajectory importer.

Fixture provenance (the no-docs-guessing rule): a REAL export — a Harbor
run of the hello-world task with the claude-code agent, the trajectory.json
Harbor writes for every trial (ATIF-v1.7) — checked in as
tests/fixtures/atif_real.json (3 steps: user / agent+tool / agent).
"""

from __future__ import annotations

import json

import pytest

from ciagent.importers import import_trace_file
from ciagent.importers.atif import (
    AtifImportError,
    load_trajectory,
    looks_like_atif,
    trace_from_atif,
)

FIXTURE = "tests/fixtures/atif_real.json"


class TestRealAtifExport:
    def test_maps_query_answer_and_agent(self):
        trace, query = trace_from_atif(load_trajectory(FIXTURE))
        assert "hello.txt" in query
        assert trace.metadata["final_output"].startswith("Done!")
        assert trace.agent_name == "claude-code"
        assert trace.framework == "atif"
        assert trace.metadata["atif_schema_version"] == "ATIF-v1.7"

    def test_maps_agent_steps_to_llm_calls_with_tokens(self):
        trace, _ = trace_from_atif(load_trajectory(FIXTURE))
        calls = trace.spans[0].llm_calls
        assert len(calls) == 2
        assert calls[0].model == "claude-haiku-4-5-20251001"
        assert (calls[0].tokens_in, calls[0].tokens_out) == (22651, 177)

    def test_maps_tool_call_with_observation_result(self):
        trace, _ = trace_from_atif(load_trajectory(FIXTURE))
        tc = trace.spans[0].tool_calls[0]
        assert tc.tool_name == "Write"
        assert tc.arguments["file_path"] == "/app/hello.txt"
        assert isinstance(tc.result, str) and "File created successfully" in tc.result

    def test_total_cost_from_final_metrics(self):
        trace, _ = trace_from_atif(load_trajectory(FIXTURE))
        assert trace.total_cost_usd == pytest.approx(0.01328)

    def test_duration_from_step_window(self):
        trace, _ = trace_from_atif(load_trajectory(FIXTURE))
        assert trace.total_duration_ms > 0

    def test_gate_accepts_the_real_import(self):
        from ciagent.engine.artifact_gate import gate_imported_golden

        trace, query = trace_from_atif(load_trajectory(FIXTURE))
        assert gate_imported_golden(trace, query).accepted

    def test_dispatcher_detects_atif(self):
        trace, query, fmt = import_trace_file(FIXTURE)
        assert fmt == "atif"
        assert query and trace.tool_call_sequence == ["Write"]

    def test_cli_import_atif_end_to_end(self, tmp_path):
        from pathlib import Path

        from click.testing import CliRunner

        from ciagent.cli import cli

        spec_path = tmp_path / "ciagent_spec.yaml"
        spec_path.write_text(
            "agent: atif-import\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n  - query: \"existing\"\n"
        )
        result = CliRunner().invoke(
            cli, ["import", str(Path(FIXTURE).resolve()), "-c", str(spec_path)],
        )
        assert result.exit_code == 0, result.output
        assert "atif" in result.output
        assert list((tmp_path / "golden" / "atif-import").glob("imported-*.json"))


class TestTrajectoryShapes:
    def test_content_block_array_message(self):
        # RFC 0001: message may be a content-block array, not just a string
        data = load_trajectory(FIXTURE)
        data["steps"][0]["message"] = [
            {"type": "text", "text": "Create a file called"},
            {"type": "text", "text": "hello.txt"},
        ]
        _, query = trace_from_atif(data)
        assert query == "Create a file called\nhello.txt"

    def test_deterministic_step_adds_no_llm_call(self):
        # llm_call_count == 0 marks a step no LLM produced (RFC 0001)
        data = load_trajectory(FIXTURE)
        data["steps"][2]["llm_call_count"] = 0
        trace, _ = trace_from_atif(data)
        assert len(trace.spans[0].llm_calls) == 1

    def test_unmatched_observation_result_stays_none(self):
        data = load_trajectory(FIXTURE)
        step = data["steps"][1]
        step["observation"]["results"][0]["source_call_id"] = "some-other-call"
        step["observation"]["results"].append({"content": "second, unattributable"})
        trace, _ = trace_from_atif(data)
        assert trace.spans[0].tool_calls[0].result is None

    def test_single_result_without_call_id_pairs_up(self):
        data = load_trajectory(FIXTURE)
        del data["steps"][1]["observation"]["results"][0]["source_call_id"]
        trace, _ = trace_from_atif(data)
        assert "File created successfully" in trace.spans[0].tool_calls[0].result

    def test_garbage_raises_import_error(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text('{"not": "a trajectory"}')
        with pytest.raises(AtifImportError):
            load_trajectory(f)

    def test_looks_like_atif_discriminates(self):
        assert looks_like_atif(json.loads(open(FIXTURE).read()))
        assert looks_like_atif({"schema_version": "ATIF-v1.7", "steps": []})
        assert not looks_like_atif({"schema_version": 1.7, "steps": []})
        assert not looks_like_atif({"schema_version": "ATIF-v1.7"})
        assert not looks_like_atif([{"run_type": "llm"}])
        assert not looks_like_atif({"resourceSpans": []})

    def test_non_atif_fixtures_still_route_elsewhere(self):
        _, _, fmt = import_trace_file("tests/fixtures/langsmith_runs_real.json")
        assert fmt == "langsmith-runs"
        _, _, fmt = import_trace_file("tests/fixtures/langfuse_spans_real.json")
        assert fmt != "atif"
