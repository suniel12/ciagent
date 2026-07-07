# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the LangSmith run-format importer.

Fixture provenance (the no-docs-guessing rule): a REAL export — a live
LangChain tool-calling agent traced by langchain-core's BaseTracer, the
exact machinery LangSmith's LangChainTracer uses — checked in as
tests/fixtures/langsmith_runs_real.json (7 runs: llm/tool/chain).
"""

from __future__ import annotations

import json

import pytest

from ciagent.importers import import_trace_file
from ciagent.importers.langsmith import (
    LangsmithImportError,
    load_runs,
    looks_like_runs,
    trace_from_langsmith,
)

FIXTURE = "tests/fixtures/langsmith_runs_real.json"


class TestRealLangsmithExport:
    def test_maps_query_answer_and_agent(self):
        trace, query = trace_from_langsmith(load_runs(FIXTURE))
        assert "charged twice for CloudSync Pro" in query
        assert trace.metadata["final_output"].startswith("I found two invoices")
        assert trace.agent_name == "LangGraph"
        assert trace.framework == "langsmith"

    def test_maps_llm_runs_with_tokens_and_provider(self):
        trace, _ = trace_from_langsmith(load_runs(FIXTURE))
        calls = trace.spans[0].llm_calls
        assert len(calls) == 2
        assert calls[0].model == "gpt-4o-mini-2024-07-18"
        assert calls[0].provider == "openai"
        assert (calls[0].tokens_in, calls[0].tokens_out) == (79, 17)

    def test_maps_tool_run_with_unwrapped_result(self):
        trace, _ = trace_from_langsmith(load_runs(FIXTURE))
        tc = trace.spans[0].tool_calls[0]
        assert tc.tool_name == "lookup_invoice"
        assert tc.arguments == {"customer_email": "alice@example.com"}
        # ToolMessage dict unwraps to its content string
        assert isinstance(tc.result, str) and "INV-2024-001" in tc.result

    def test_duration_from_run_window(self):
        trace, _ = trace_from_langsmith(load_runs(FIXTURE))
        assert trace.total_duration_ms > 0

    def test_gate_accepts_the_real_import(self):
        from ciagent.engine.artifact_gate import gate_imported_golden

        trace, query = trace_from_langsmith(load_runs(FIXTURE))
        assert gate_imported_golden(trace, query).accepted

    def test_dispatcher_detects_langsmith(self):
        trace, query, fmt = import_trace_file(FIXTURE)
        assert fmt == "langsmith-runs"
        assert query and trace.tool_call_sequence == ["lookup_invoice"]

    def test_cli_import_langsmith_end_to_end(self, tmp_path):
        from pathlib import Path

        from click.testing import CliRunner

        from ciagent.cli import cli

        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(
            "agent: ls-import\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n  - query: \"existing\"\n"
        )
        result = CliRunner().invoke(
            cli, ["import", str(Path(FIXTURE).resolve()), "-c", str(spec_path)],
        )
        assert result.exit_code == 0, result.output
        assert "langsmith-runs" in result.output
        assert list((tmp_path / "golden" / "ls-import").glob("imported-*.json"))


class TestRunShapes:
    def test_jsonl_input(self, tmp_path):
        runs = load_runs(FIXTURE)
        f = tmp_path / "runs.jsonl"
        f.write_text("\n".join(json.dumps(r, default=str) for r in runs))
        trace, query = trace_from_langsmith(load_runs(f))
        assert query and trace.tool_call_sequence == ["lookup_invoice"]

    def test_nested_runtree_input(self, tmp_path):
        runs = load_runs(FIXTURE)
        root = next(r for r in runs if not r.get("parent_run_id"))
        nested = dict(root)
        nested["child_runs"] = [r for r in runs if r is not root]
        f = tmp_path / "tree.json"
        f.write_text(json.dumps(nested, default=str))
        trace, query = trace_from_langsmith(load_runs(f))
        assert query and len(trace.spans[0].llm_calls) == 2

    def test_runs_wrapper_input(self, tmp_path):
        runs = load_runs(FIXTURE)
        f = tmp_path / "wrapped.json"
        f.write_text(json.dumps({"runs": runs}, default=str))
        assert len(load_runs(f)) == len(runs)

    def test_plain_input_output_chain_shape(self):
        # Classic AgentExecutor-style root: {"input": ...} / {"output": ...}
        runs = [{
            "run_type": "chain", "name": "AgentExecutor",
            "parent_run_id": None,
            "inputs": {"input": "what rate do you charge?"},
            "outputs": {"output": "The rate is 4.5%."},
        }]
        trace, query = trace_from_langsmith(runs)
        assert query == "what rate do you charge?"
        assert trace.metadata["final_output"] == "The rate is 4.5%."

    def test_tool_run_without_output_stays_none(self):
        runs = [{
            "run_type": "tool", "name": "search",
            "parent_run_id": "x", "inputs": {"q": "a"}, "outputs": None,
        }]
        trace, _ = trace_from_langsmith(runs)
        assert trace.spans[0].tool_calls[0].result is None

    def test_garbage_raises_import_error(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text('{"not": "runs"}')
        with pytest.raises(LangsmithImportError):
            load_runs(f)

    def test_looks_like_runs_discriminates(self):
        assert looks_like_runs([{"run_type": "llm"}])
        assert looks_like_runs({"runs": [{"run_type": "tool"}]})
        assert not looks_like_runs([{"attributes": {}, "name": "span"}])
        assert not looks_like_runs({"resourceSpans": []})
