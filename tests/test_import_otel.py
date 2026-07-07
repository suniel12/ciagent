# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Tests for `ciagent import` — OTel GenAI semconv → spec query + golden.

The binding contract under test: partial traces are rejected by the
round-trip gate with the missing fields NAMED, never silently imported;
an accepted import produces a golden that loads and evaluates cleanly.
"""

from __future__ import annotations

import json

import pytest

from ciagent.importers.otel import OtelImportError, load_spans, trace_from_otel

QUERY = "Do you sell smart thermostats?"
ANSWER = "We don't sell smart thermostats — CloudSync is our only product line."


def _flat(output: str) -> str:
    """Collapse whitespace — rich wraps console output at the CI terminal
    width, splitting asserted phrases across lines."""
    return " ".join(output.split())



def _chat_span(with_content: bool = True) -> dict:
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.response.model": "gpt-4o-2024-08-06",
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 40,
    }
    if with_content:
        attrs["gen_ai.input.messages"] = json.dumps([
            {"role": "system", "content": "you are support"},
            {"role": "user", "content": QUERY},
        ])
        attrs["gen_ai.output.messages"] = json.dumps([
            {"role": "assistant", "parts": [{"type": "text", "content": ANSWER}]},
        ])
    return {
        "name": "chat gpt-4o",
        "startTimeUnixNano": 1_000_000_000,
        "endTimeUnixNano": 1_450_000_000,
        "attributes": attrs,
    }


def _tool_span(with_result: bool = True) -> dict:
    attrs = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "search_products",
        "gen_ai.tool.call.arguments": json.dumps({"query": "smart thermostat"}),
    }
    if with_result:
        attrs["gen_ai.tool.call.result"] = json.dumps([])
    return {
        "name": "execute_tool search_products",
        "startTimeUnixNano": 1_050_000_000,
        "endTimeUnixNano": 1_150_000_000,
        "attributes": attrs,
    }


def _agent_span() -> dict:
    return {
        "name": "invoke_agent shop-assistant",
        "attributes": {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "shop-assistant",
        },
    }


class TestLoadSpans:
    def test_flat_list(self, tmp_path):
        f = tmp_path / "t.json"
        f.write_text(json.dumps([_chat_span()]))
        assert len(load_spans(f)) == 1

    def test_spans_wrapper(self, tmp_path):
        f = tmp_path / "t.json"
        f.write_text(json.dumps({"spans": [_chat_span(), _tool_span()]}))
        assert len(load_spans(f)) == 2

    def test_otlp_envelope_with_attribute_lists(self, tmp_path):
        span = {
            "name": "chat gpt-4o",
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4o"}},
                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "120"}},
            ],
        }
        f = tmp_path / "t.json"
        f.write_text(json.dumps({
            "resourceSpans": [{"scopeSpans": [{"spans": [span]}]}],
        }))
        spans = load_spans(f)
        assert spans[0]["attributes"]["gen_ai.request.model"] == "gpt-4o"
        assert spans[0]["attributes"]["gen_ai.usage.input_tokens"] == 120

    def test_non_span_json_raises_import_error(self, tmp_path):
        f = tmp_path / "t.json"
        f.write_text('{"hello": "world"}')
        with pytest.raises(OtelImportError, match="not an OTel span export"):
            load_spans(f)

    def test_invalid_json_raises_import_error(self, tmp_path):
        f = tmp_path / "t.json"
        f.write_text("not json {{{")
        with pytest.raises(OtelImportError, match="cannot read"):
            load_spans(f)


class TestTraceMapping:
    def test_full_mapping(self):
        trace, query = trace_from_otel([_agent_span(), _tool_span(), _chat_span()])
        assert query == QUERY
        assert trace.metadata["final_output"] == ANSWER
        assert trace.agent_name == "shop-assistant"
        assert trace.tool_call_sequence == ["search_products"]
        tc = trace.spans[0].tool_calls[0]
        assert tc.arguments == {"query": "smart thermostat"}
        assert tc.result == []  # captured empty retrieval — F4 can see it
        llm = trace.spans[0].llm_calls[0]
        assert llm.model == "gpt-4o-2024-08-06"
        assert llm.tokens_in == 120 and llm.tokens_out == 40
        assert trace.total_llm_calls == 1

    def test_tool_without_result_stays_none(self):
        # Absent opt-in content attribute → result None; the retrieval layer
        # SKIPs on uncaptured results rather than guess.
        trace, _ = trace_from_otel([_tool_span(with_result=False), _chat_span()])
        assert trace.spans[0].tool_calls[0].result is None

    def test_contentless_export_has_no_query_or_answer(self):
        trace, query = trace_from_otel([_chat_span(with_content=False)])
        assert query is None
        assert "final_output" not in trace.metadata

    def test_plain_content_message_shape_accepted(self):
        span = _chat_span()
        span["attributes"]["gen_ai.output.messages"] = json.dumps([
            {"role": "assistant", "content": "plain-shape answer"},
        ])
        trace, _ = trace_from_otel([span])
        assert trace.metadata["final_output"] == "plain-shape answer"


class TestImportCLI:
    def _project(self, tmp_path):
        spec = tmp_path / "agentci_spec.yaml"
        spec.write_text(
            "agent: import-test\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n"
            "  - query: \"existing query\"\n"
        )
        return spec

    def test_import_writes_spec_query_and_golden(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        import yaml

        from ciagent.cli import cli

        spec_path = self._project(tmp_path)
        trace_file = tmp_path / "prod.json"
        trace_file.write_text(json.dumps([_agent_span(), _tool_span(), _chat_span()]))

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            cli, ["import", str(trace_file), "-c", str(spec_path)],
        )
        assert result.exit_code == 0, result.output

        spec_data = yaml.safe_load(spec_path.read_text())
        imported = [q for q in spec_data["queries"] if q["query"] == QUERY]
        assert imported and imported[0]["tags"] == ["imported"]
        assert (tmp_path / "agentci_spec.yaml.bak").exists()

        goldens = list((tmp_path / "golden" / "import-test").glob("imported-*.json"))
        assert len(goldens) == 1
        golden = json.loads(goldens[0].read_text())
        assert golden["query"] == QUERY
        assert golden["schema_version"] == 1

        # Round trip: the written golden loads through the standard loaders
        from ciagent.engine.judge_audit import load_answers_from_baselines

        answers = load_answers_from_baselines(str(tmp_path / "golden"))
        assert answers[QUERY] == ANSWER

    def test_partial_trace_rejected_exit_1_nothing_written(self, tmp_path):
        from click.testing import CliRunner

        from ciagent.cli import cli

        spec_path = self._project(tmp_path)
        trace_file = tmp_path / "partial.json"
        trace_file.write_text(json.dumps([_chat_span(with_content=False)]))

        result = CliRunner().invoke(
            cli, ["import", str(trace_file), "-c", str(spec_path)],
        )
        assert result.exit_code == 1
        assert "no user input" in _flat(result.output)
        assert "no final output" in _flat(result.output)
        assert not (tmp_path / "golden").exists()
        assert "existing query" in spec_path.read_text()

    def test_dry_run_writes_nothing(self, tmp_path):
        from click.testing import CliRunner

        from ciagent.cli import cli

        spec_path = self._project(tmp_path)
        trace_file = tmp_path / "prod.json"
        trace_file.write_text(json.dumps([_chat_span()]))

        result = CliRunner().invoke(
            cli, ["import", str(trace_file), "-c", str(spec_path), "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert "nothing written" in _flat(result.output)
        assert not (tmp_path / "golden").exists()

    def test_reimport_same_query_leaves_spec_alone(self, tmp_path):
        from click.testing import CliRunner

        from ciagent.cli import cli

        spec_path = self._project(tmp_path)
        trace_file = tmp_path / "prod.json"
        trace_file.write_text(json.dumps([_chat_span()]))

        runner = CliRunner()
        first = runner.invoke(cli, ["import", str(trace_file), "-c", str(spec_path)])
        assert first.exit_code == 0, first.output
        spec_after_first = spec_path.read_text()

        second = runner.invoke(cli, ["import", str(trace_file), "-c", str(spec_path)])
        assert second.exit_code == 0, second.output
        assert "spec unchanged" in _flat(second.output)
        assert spec_path.read_text() == spec_after_first
        # but a second golden version was written
        goldens = list((tmp_path / "golden" / "import-test").glob("imported-*.json"))
        assert len(goldens) == 2

    def test_garbage_file_exits_2(self, tmp_path):
        from click.testing import CliRunner

        from ciagent.cli import cli

        self._project(tmp_path)
        bad = tmp_path / "bad.json"
        bad.write_text('{"hello": 1}')
        result = CliRunner().invoke(
            cli, ["import", str(bad), "-c", str(tmp_path / "agentci_spec.yaml")],
        )
        assert result.exit_code == 2
        assert "not an OTel span export" in _flat(result.output)
