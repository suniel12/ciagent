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

    def test_import_into_spec_with_path_assertion_serializes_enums(self, tmp_path):
        # Regression (dogfood 2026-07-07): the spec rewrite used
        # model_dump(exclude_none=True), leaving path.match_mode as a MatchMode
        # enum that yaml.safe_dump cannot represent — crashing import for ANY
        # spec that carries a path assertion.
        from click.testing import CliRunner

        import yaml

        from ciagent.cli import cli
        from ciagent.loader import load_spec

        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(
            "agent: import-test\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n"
            '  - query: "route me"\n'
            "    path:\n"
            '      expected_handoff: "Billing Agent"\n'
            "      match_mode: subset\n"
        )
        trace_file = tmp_path / "prod.json"
        trace_file.write_text(json.dumps([_chat_span()]))

        result = CliRunner().invoke(cli, ["import", str(trace_file), "-c", str(spec_path)])
        assert result.exit_code == 0, result.output

        reloaded = yaml.safe_load(spec_path.read_text())
        routed = [q for q in reloaded["queries"] if q["query"] == "route me"][0]
        assert routed["path"]["match_mode"] == "subset"  # plain string, not an enum
        load_spec(str(spec_path))  # the rewritten spec still loads

    def test_failing_trace_blocked_then_forced(self, tmp_path):
        # F7's purpose is importing FAILURES. When the matching query already
        # gates on correctness the trace fails, the save prechecks and stops
        # (exit 1) unless --force-save keeps the failure as the golden.
        from click.testing import CliRunner

        from ciagent.cli import cli

        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(
            "agent: import-test\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n"
            f"  - query: {json.dumps(QUERY)}\n"
            "    correctness:\n"
            '      expected_in_answer: ["thermostats are in stock"]\n'
        )
        trace_file = tmp_path / "prod.json"
        trace_file.write_text(json.dumps([_chat_span()]))

        runner = CliRunner()
        blocked = runner.invoke(cli, ["import", str(trace_file), "-c", str(spec_path)])
        assert blocked.exit_code == 1, blocked.output
        assert "force-save" in _flat(blocked.output)
        assert not (tmp_path / "golden").exists()

        forced = runner.invoke(
            cli, ["import", str(trace_file), "-c", str(spec_path), "--force-save"],
        )
        assert forced.exit_code == 0, forced.output
        goldens = list((tmp_path / "golden" / "import-test").glob("*.json"))
        assert len(goldens) == 1


class TestRealOpenllmetryExport:
    """Against a REAL export: live OpenAI call traced by openllmetry 0.62
    (tests/fixtures/otel_openllmetry_real.json). Two chat spans, no
    execute_tool span — the tool call lives in message content."""

    FIXTURE = "tests/fixtures/otel_openllmetry_real.json"

    def test_maps_llm_calls_query_and_answer(self):
        spans = load_spans(self.FIXTURE)
        trace, query = trace_from_otel(spans)
        assert "charged twice for CloudSync Pro" in query
        assert trace.metadata["final_output"].startswith("I've checked your invoices")
        span = trace.spans[0]
        assert len(span.llm_calls) == 2
        assert span.llm_calls[0].model == "gpt-4o-mini-2024-07-18"
        assert span.llm_calls[1].tokens_in == 178

    def test_recovers_tool_call_with_result_from_messages(self):
        # No execute_tool span exists; the call comes from a `tool_call`
        # output part and its result from the paired `tool_call_response`.
        spans = load_spans(self.FIXTURE)
        trace, _ = trace_from_otel(spans)
        assert trace.tool_call_sequence == ["lookup_invoice"]
        tc = trace.spans[0].tool_calls[0]
        assert tc.arguments == {"customer_email": "alice@example.com"}
        assert isinstance(tc.result, list) and tc.result[0]["id"] == "INV-2024-001"

    def test_echoed_tool_call_not_duplicated(self):
        # Span 2's input messages echo span 1's tool_call — id-dedupe keeps one.
        spans = load_spans(self.FIXTURE)
        trace, _ = trace_from_otel(spans)
        assert len(trace.spans[0].tool_calls) == 1

    def test_imported_result_is_visible_to_retrieval_layer(self):
        # The F7→F4 hand-off: a captured tool result in an imported golden
        # is exactly what the retrieval layer evaluates.
        from ciagent.engine.results import LayerStatus
        from ciagent.engine.retrieval import evaluate_retrieval
        from ciagent.schema.spec_models import RetrievalSpec

        spans = load_spans(self.FIXTURE)
        trace, _ = trace_from_otel(spans)
        result = evaluate_retrieval(
            trace,
            RetrievalSpec(tool="lookup_invoice", forbid_empty=True, min_results=2),
        )
        assert result.status == LayerStatus.PASS

    def test_cli_import_real_export(self, tmp_path):
        from pathlib import Path

        from click.testing import CliRunner

        from ciagent.cli import cli

        fixture = Path(self.FIXTURE).resolve()
        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(
            "agent: real-import\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n  - query: \"existing\"\n"
        )
        result = CliRunner().invoke(
            cli, ["import", str(fixture), "-c", str(spec_path)],
        )
        assert result.exit_code == 0, result.output
        goldens = list((tmp_path / "golden" / "real-import").glob("imported-*.json"))
        assert len(goldens) == 1


class TestRealAnthropicExport:
    """Against a REAL export: live Anthropic tool_use conversation traced by
    openllmetry's AnthropicInstrumentor (tests/fixtures/anthropic_otel_real.json).
    Two `anthropic.chat` spans, provider=anthropic, tool call + result carried
    in message content. Verifies the F7 import path works on the Anthropic
    dialect with zero importer changes."""

    FIXTURE = "tests/fixtures/anthropic_otel_real.json"

    def test_maps_query_answer_and_provider(self):
        spans = load_spans(self.FIXTURE)
        trace, query = trace_from_otel(spans)
        assert query == "What's the weather in Paris right now?"
        assert "Paris" in trace.metadata["final_output"]
        assert trace.total_llm_calls == 2
        llm = trace.spans[0].llm_calls[0]
        assert llm.model == "claude-sonnet-4-5-20250929"
        assert llm.tokens_in > 0 and llm.tokens_out > 0

    def test_recovers_tool_call_with_result(self):
        # The F7→F4 hand-off on Anthropic: tool_use call + tool_result survive
        # import from message content (no execute_tool span).
        spans = load_spans(self.FIXTURE)
        trace, _ = trace_from_otel(spans)
        assert trace.tool_call_sequence == ["get_weather"]
        tc = trace.spans[0].tool_calls[0]
        assert tc.arguments == {"city": "Paris"}
        assert tc.result == {"temp_c": 18, "condition": "light rain", "city": "Paris"}

    def test_cli_import_real_anthropic_export(self, tmp_path):
        from pathlib import Path

        from click.testing import CliRunner

        from ciagent.cli import cli

        fixture = Path(self.FIXTURE).resolve()
        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(
            "agent: anthropic-import\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n  - query: \"existing\"\n"
        )
        result = CliRunner().invoke(cli, ["import", str(fixture), "-c", str(spec_path)])
        assert result.exit_code == 0, result.output
        assert "otel-genai" in _flat(result.output)
        goldens = list((tmp_path / "golden" / "anthropic-import").glob("imported-*.json"))
        assert len(goldens) == 1


class TestRealCrewAIExport:
    """Against a REAL export: a live CrewAI crew (agent + tool + task, gpt-4o-mini)
    traced by openllmetry's CrewAI + OpenAI instrumentors
    (tests/fixtures/crewai_otel_real.json). CrewAI runs LLM calls through
    litellm -> the OpenAI client, so the gen_ai content lives on `openai.chat`
    spans alongside CrewAI's `invoke_agent` workflow spans. Verifies the F7
    import path on the CrewAI dialect with zero importer changes.

    Note: CrewAI's imported query is its full constructed task prompt (it is the
    last user message CrewAI sent) — faithful to the emitter, just verbose.
    (The bloated crewai.agent.* object-dump attributes — which the instrumentor
    fills with the whole Agent incl. the API key — were stripped from the
    fixture; the importer never reads them.)"""

    FIXTURE = "tests/fixtures/crewai_otel_real.json"

    def test_maps_answer_and_llm_calls(self):
        spans = load_spans(self.FIXTURE)
        trace, query = trace_from_otel(spans)
        assert "get_weather" in query  # the constructed task prompt
        assert "Paris" in trace.metadata["final_output"]
        assert trace.total_llm_calls >= 2

    def test_recovers_tool_call_with_result(self):
        spans = load_spans(self.FIXTURE)
        trace, _ = trace_from_otel(spans)
        assert trace.tool_call_sequence == ["get_weather"]
        tc = trace.spans[0].tool_calls[0]
        assert tc.arguments == {"city": "Paris"}
        assert tc.result == {"temp_c": 18, "condition": "light rain", "city": "Paris"}

    def test_fixture_carries_no_secret(self):
        # Regression guard: the CrewAI instrumentor serializes the Agent object
        # (incl. api_key) into span attributes — the fixture must stay scrubbed.
        import re

        from pathlib import Path

        text = Path(self.FIXTURE).read_text()
        assert "api_key=" not in text
        assert not re.search(r"sk-[A-Za-z0-9_\-]{20,}", text)

    def test_cli_import_real_crewai_export(self, tmp_path):
        from pathlib import Path

        from click.testing import CliRunner

        from ciagent.cli import cli

        fixture = Path(self.FIXTURE).resolve()
        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(
            "agent: crewai-import\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n  - query: \"existing\"\n"
        )
        result = CliRunner().invoke(cli, ["import", str(fixture), "-c", str(spec_path)])
        assert result.exit_code == 0, result.output
        goldens = list((tmp_path / "golden" / "crewai-import").glob("imported-*.json"))
        assert len(goldens) == 1


class TestRealADKExport:
    """Against a REAL export: a live Google ADK agent (google-adk 2.3 +
    gemini-2.5-flash, agent + tool) using ADK's NATIVE OTel self-instrumentation
    (tests/fixtures/adk_otel_real.json). ADK emits a `gcp.vertex.agent.*`
    dialect — query/answer live in llm_request/llm_response (google-genai
    Content shape), tool args/result in tool_call_args/tool_response — NOT the
    gen_ai.input/output.messages shape. This is the dialect gap the importer's
    otel-adk branch closes (like Langfuse before it)."""

    FIXTURE = "tests/fixtures/adk_otel_real.json"

    def test_detected_as_adk_dialect(self):
        from ciagent.importers import import_trace_file

        trace, query, fmt = import_trace_file(self.FIXTURE)
        assert fmt == "otel-adk"
        assert trace.framework == "otel-adk"
        assert query == "What's the weather in Paris right now?"
        assert "Paris" in trace.metadata["final_output"]
        assert trace.agent_name == "weather_agent"

    def test_maps_llm_calls_without_double_counting_generate_content(self):
        # 2 call_llm spans -> 2 LLM calls; the gemini generate_content child
        # spans must NOT become extra empty LLM calls.
        trace, _ = trace_from_otel(load_spans(self.FIXTURE))
        assert trace.total_llm_calls == 2
        llm = trace.spans[0].llm_calls[0]
        assert llm.model == "gemini-2.5-flash"
        assert llm.tokens_in > 0 and llm.tokens_out > 0

    def test_recovers_tool_call_with_result(self):
        trace, _ = trace_from_otel(load_spans(self.FIXTURE))
        assert trace.tool_call_sequence == ["get_weather"]
        tc = trace.spans[0].tool_calls[0]
        assert tc.arguments == {"city": "Paris"}
        assert tc.result == {"temp_c": 18, "condition": "light rain", "city": "Paris"}

    def test_cli_import_real_adk_export(self, tmp_path):
        from pathlib import Path

        from click.testing import CliRunner

        from ciagent.cli import cli

        fixture = Path(self.FIXTURE).resolve()
        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(
            "agent: adk-import\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n  - query: \"existing\"\n"
        )
        result = CliRunner().invoke(cli, ["import", str(fixture), "-c", str(spec_path)])
        assert result.exit_code == 0, result.output
        assert "otel-adk" in _flat(result.output)
        goldens = list((tmp_path / "golden" / "adk-import").glob("imported-*.json"))
        assert len(goldens) == 1


class TestRealClaudeAgentSdkExport:
    """Against a REAL export: a live Claude Agent SDK query() (claude-agent-sdk
    0.2, SDK MCP weather tool) instrumented by otel-instrumentation-claude-agent-sdk
    0.0.6 (GenAI semconv 1.42), tests/fixtures/claude_agent_sdk_otel_real.json.

    The Agent SDK dialect: ONE invoke_agent span carries provider, model,
    session-aggregate usage, and (opt-in) the full input/output messages;
    execute_tool child spans carry gen_ai.tool.call.id but NOT
    arguments/result — that content only exists in the invoke_agent span's
    messages. The importer pairs the two views by call id; without that,
    every tool call double-counts (once bare from its span, once with
    content from the messages).

    Real-harness quirk kept faithful: the CLI subprocess loads deferred
    tools, so the model calls ToolSearch before the MCP weather tool."""

    FIXTURE = "tests/fixtures/claude_agent_sdk_otel_real.json"

    def test_maps_query_answer_model_and_session(self):
        spans = load_spans(self.FIXTURE)
        trace, query = trace_from_otel(spans)
        assert query == "What's the weather in Paris right now?"
        assert "Paris" in trace.metadata["final_output"]
        # The invoke_agent span is the only model/usage carrier -> one
        # session-aggregate LLM call.
        assert trace.total_llm_calls == 1
        llm = trace.spans[0].llm_calls[0]
        assert llm.provider == "anthropic"
        assert llm.model.startswith("claude-")
        assert llm.tokens_in > 0 and llm.tokens_out > 0

    def test_execute_tool_spans_merge_with_message_content_by_call_id(self):
        # The dialect gap this class exists for: execute_tool spans have ids
        # but no content; messages have content keyed by the same ids. The
        # merged result must be exactly one ToolCall per real call, each
        # with arguments AND result.
        spans = load_spans(self.FIXTURE)
        trace, _ = trace_from_otel(spans)
        assert trace.tool_call_sequence == ["ToolSearch", "mcp__weather__get_weather"]
        weather = trace.spans[0].tool_calls[1]
        assert weather.arguments == {"city": "Paris"}
        assert weather.result == [
            {"type": "text", "text": '{"temp_c": 18, "condition": "light rain", "city": "Paris"}'}
        ]
        # The span-derived ToolSearch call got its arguments backfilled too.
        assert trace.spans[0].tool_calls[0].arguments != {}

    def test_fixture_carries_no_secret(self):
        import re
        from pathlib import Path

        text = Path(self.FIXTURE).read_text()
        assert "api_key=" not in text
        assert not re.search(r"sk-[A-Za-z0-9_\-]{20,}", text)

    def test_cli_import_real_claude_agent_sdk_export(self, tmp_path):
        from pathlib import Path

        from click.testing import CliRunner

        from ciagent.cli import cli

        fixture = Path(self.FIXTURE).resolve()
        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(
            "agent: claude-agent-sdk-import\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n  - query: \"existing\"\n"
        )
        result = CliRunner().invoke(cli, ["import", str(fixture), "-c", str(spec_path)])
        assert result.exit_code == 0, result.output
        assert "otel-genai" in _flat(result.output)
        goldens = list(
            (tmp_path / "golden" / "claude-agent-sdk-import").glob("imported-*.json")
        )
        assert len(goldens) == 1


class TestLangsmithDetection:
    def test_langsmith_runs_export_gets_targeted_error(self, tmp_path):
        f = tmp_path / "runs.json"
        f.write_text(json.dumps([
            {"run_type": "llm", "name": "openai", "inputs": {}, "outputs": {}},
        ]))
        with pytest.raises(OtelImportError, match="LangSmith runs export"):
            load_spans(f)


class TestRealLangfuseExport:
    """Against a REAL export: live OpenAI tool-call conversation traced by
    the langfuse 4.13 SDK (tests/fixtures/langfuse_spans_real.json). The
    dialect is pure langfuse.* attributes — zero gen_ai.* — which the plain
    semconv mapping would reject; this is why we verify against emitters."""

    FIXTURE = "tests/fixtures/langfuse_spans_real.json"

    def test_maps_generations_with_model_and_tokens(self):
        spans = load_spans(self.FIXTURE)
        trace, _ = trace_from_otel(spans)
        calls = trace.spans[0].llm_calls
        assert len(calls) == 2
        assert calls[0].model == "gpt-4o-mini-2024-07-18"
        assert (calls[0].tokens_in, calls[0].tokens_out) == (79, 17)
        assert trace.framework == "otel-langfuse"

    def test_maps_tool_observation_with_structured_result(self):
        spans = load_spans(self.FIXTURE)
        trace, _ = trace_from_otel(spans)
        tc = trace.spans[0].tool_calls[0]
        assert tc.tool_name == "lookup_invoice"
        assert tc.arguments == {"customer_email": "alice@example.com"}
        # JSON-string output parses to a structured list — countable by F4
        assert isinstance(tc.result, list) and tc.result[0]["id"] == "INV-2024-001"

    def test_root_observation_carries_query_answer_and_agent(self):
        spans = load_spans(self.FIXTURE)
        trace, query = trace_from_otel(spans)
        assert "charged twice for CloudSync Pro" in query
        assert trace.metadata["final_output"].startswith("I found two invoices")
        assert trace.agent_name == "billing-support-agent"

    def test_generation_only_export_still_yields_query_and_answer(self):
        # A user exporting just the generation spans (no root observation)
        # still gets a gateable golden from the message content.
        spans = [s for s in load_spans(self.FIXTURE)
                 if s["attributes"].get("langfuse.observation.type") == "generation"]
        trace, query = trace_from_otel(spans)
        assert query and "charged twice" in query
        assert trace.metadata["final_output"].startswith("I found two invoices")

    def test_gate_and_retrieval_layer_accept_the_import(self):
        from ciagent.engine.artifact_gate import gate_imported_golden
        from ciagent.engine.results import LayerStatus
        from ciagent.engine.retrieval import evaluate_retrieval
        from ciagent.schema.spec_models import RetrievalSpec

        spans = load_spans(self.FIXTURE)
        trace, query = trace_from_otel(spans)
        assert gate_imported_golden(trace, query).accepted
        result = evaluate_retrieval(
            trace,
            RetrievalSpec(tool="lookup_invoice", forbid_empty=True, min_results=2),
        )
        assert result.status == LayerStatus.PASS

    def test_dispatcher_reports_langfuse_dialect(self, tmp_path):
        from pathlib import Path

        from click.testing import CliRunner

        from ciagent.cli import cli
        from ciagent.importers import import_trace_file

        _, _, fmt = import_trace_file(self.FIXTURE)
        assert fmt == "otel-langfuse"

        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(
            "agent: lf-import\n"
            f"baseline_dir: {tmp_path / 'golden'}\n"
            "queries:\n  - query: \"existing\"\n"
        )
        result = CliRunner().invoke(
            cli, ["import", str(Path(self.FIXTURE).resolve()), "-c", str(spec_path)],
        )
        assert result.exit_code == 0, result.output
        assert "otel-langfuse" in _flat(result.output)
        assert list((tmp_path / "golden" / "lf-import").glob("imported-*.json"))
