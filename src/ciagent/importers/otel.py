# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
OTel GenAI → CIAgent Trace importer (F7).

Converts spans exported under the OpenTelemetry **GenAI semantic
conventions** into a `ciagent.models.Trace`. That target is a binding
decision (eng review 2026-07-05): Langfuse/LangSmith/instrumentation-library
exporters already emit the GenAI semconv, so one mapping covers the
ecosystem instead of per-vendor adapters.

Accepted file shapes (JSON):
  - OTLP/JSON envelope: {"resourceSpans": [{"scopeSpans": [{"spans": [...]}]}]}
  - a flat list of span dicts: [{...}, ...]
  - a {"spans": [...]} wrapper

Attribute mapping (GenAI semconv; older aliases read-tolerated):
  gen_ai.operation.name       chat / text_completion / generate_content → LLM call
                              execute_tool → tool call
                              invoke_agent / create_agent → agent identity
  gen_ai.request.model /
  gen_ai.response.model       LLM model
  gen_ai.provider.name        provider (alias: gen_ai.system)
  gen_ai.usage.input_tokens / output_tokens   token counts
  gen_ai.tool.name            tool name
  gen_ai.tool.call.arguments  tool arguments (JSON string; opt-in content)
  gen_ai.tool.call.result     tool result   (JSON string; opt-in content —
                              absent results stay None and the retrieval
                              layer SKIPs, never guesses)
  gen_ai.input.messages /
  gen_ai.output.messages      conversation content (opt-in). The query is the
                              last user message; the final output is the last
                              assistant message.

This module only MAPS. Whether the result is fit to become a golden is the
artifact gate's call (`engine.artifact_gate.gate_imported_golden`) — partial
traces are rejected there with the missing fields named, never silently
imported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from ciagent.cost import compute_cost
from ciagent.models import LLMCall, Span, SpanKind, ToolCall, Trace

# gen_ai.operation.name values → how the span maps
_LLM_OPERATIONS = {"chat", "text_completion", "generate_content", "embeddings"}
_TOOL_OPERATIONS = {"execute_tool"}
_AGENT_OPERATIONS = {"invoke_agent", "create_agent"}


class OtelImportError(ValueError):
    """The file cannot be read as OTel spans at all (not a gate rejection)."""


def load_spans(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Read a JSON export and return a flat list of span dicts.

    OTLP attribute lists ([{key, value: {stringValue: ...}}]) are normalized
    to plain dicts; already-flat exports pass through.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise OtelImportError(f"cannot read '{path}' as JSON: {e}") from e

    if isinstance(data, dict) and "resourceSpans" in data:
        spans: list[dict[str, Any]] = []
        for rs in data.get("resourceSpans") or []:
            for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []:
                spans.extend(ss.get("spans") or [])
        raw_spans = spans
    elif isinstance(data, dict) and isinstance(data.get("spans"), list):
        raw_spans = data["spans"]
    elif isinstance(data, list):
        raw_spans = data
    else:
        raise OtelImportError(
            f"'{path}' is not an OTel span export — expected an OTLP/JSON "
            "envelope (resourceSpans), a {\"spans\": [...]} wrapper, or a "
            "flat list of span objects."
        )

    # LangSmith run exports are their own schema, not OTel spans — route
    # direct callers to the right importer (the CLI dispatcher does this
    # automatically via importers.import_trace_file).
    if raw_spans and all(
        isinstance(s, dict) and "run_type" in s and "attributes" not in s
        for s in raw_spans
    ):
        raise OtelImportError(
            f"'{path}' looks like a LangSmith runs export (run_type objects), "
            "not OTel span data — use ciagent.importers.langsmith (the "
            "`ciagent import` CLI detects and routes this automatically)."
        )

    if not raw_spans:
        raise OtelImportError(f"'{path}' contains no spans.")
    return [_normalize_span(s) for s in raw_spans if isinstance(s, dict)]


def trace_from_otel(spans: list[dict[str, Any]]) -> tuple[Trace, Optional[str]]:
    """Map normalized GenAI spans onto a CIAgent Trace.

    Returns (trace, query_text). Either may be incomplete — completeness is
    judged by the artifact gate, not here.
    """
    root = Span(kind=SpanKind.AGENT, name="imported-agent")
    trace = Trace(agent_name="imported-agent", framework="otel-genai")

    query: Optional[str] = None
    final_output: Optional[str] = None
    start_ns: Optional[int] = None
    end_ns: Optional[int] = None

    for span in spans:
        attrs = span.get("attributes") or {}
        op = str(attrs.get("gen_ai.operation.name") or "").lower()
        start_ns, end_ns = _widen_window(span, start_ns, end_ns)

        # Langfuse SDK spans (v3+) use a langfuse.* attribute dialect with
        # NO gen_ai.* attributes — verified against a live langfuse 4.13
        # capture (tests/fixtures/langfuse_spans_real.json).
        lf_type = str(attrs.get("langfuse.observation.type") or "").lower()
        if lf_type:
            trace.framework = "otel-langfuse"
            query, final_output = _map_langfuse_span(
                span, attrs, lf_type, root, trace, query, final_output,
            )
            continue

        if op in _AGENT_OPERATIONS or (not op and _looks_like_agent(attrs)):
            agent_name = attrs.get("gen_ai.agent.name") or span.get("name")
            if agent_name:
                trace.agent_name = str(agent_name)
                root.name = str(agent_name)

        if op in _TOOL_OPERATIONS:
            root.tool_calls.append(ToolCall(
                tool_name=str(attrs.get("gen_ai.tool.name") or span.get("name") or "unknown-tool"),
                arguments=_parse_json_attr(attrs.get("gen_ai.tool.call.arguments"), default={}),
                result=_parse_json_attr(attrs.get("gen_ai.tool.call.result"), default=None),
                duration_ms=_span_duration_ms(span),
            ))
            continue

        if op in _LLM_OPERATIONS or _looks_like_llm(attrs):
            model = str(
                attrs.get("gen_ai.response.model")
                or attrs.get("gen_ai.request.model")
                or ""
            )
            provider = str(
                attrs.get("gen_ai.provider.name") or attrs.get("gen_ai.system") or ""
            )
            tokens_in = _as_int(attrs.get("gen_ai.usage.input_tokens"))
            tokens_out = _as_int(attrs.get("gen_ai.usage.output_tokens"))
            root.llm_calls.append(LLMCall(
                model=model,
                provider=provider,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                output_text=_last_message_text(attrs, "gen_ai.output.messages", role="assistant") or "",
                cost_usd=compute_cost(provider, model, tokens_in, tokens_out),
                duration_ms=_span_duration_ms(span),
            ))

        # Conversation content can ride on any span kind — keep the last seen
        user_text = _last_message_text(attrs, "gen_ai.input.messages", role="user")
        if user_text:
            query = user_text
        assistant_text = _last_message_text(attrs, "gen_ai.output.messages", role="assistant")
        if assistant_text:
            final_output = assistant_text

    # Tool calls that never got their own span: real chat-instrumentation
    # exports (verified against a live openllmetry 0.62 capture) carry the
    # tool call as a `tool_call` part in the output messages and the tool's
    # RESULT as a `tool_call_response` part in a later span's input messages
    # — the tool itself runs in app code, unspanned. Recover both, paired
    # by call id, so the imported golden keeps tool usage (and the F4
    # retrieval layer can see retriever output). execute_tool spans take
    # precedence; message-derived results backfill them when missing.
    existing = {
        (c.tool_name, json.dumps(c.arguments, sort_keys=True, default=str)): c
        for c in root.tool_calls
    }
    for tc in _tool_calls_from_messages(spans):
        key = (tc.tool_name, json.dumps(tc.arguments, sort_keys=True, default=str))
        match = existing.get(key)
        if match is None:
            root.tool_calls.append(tc)
            existing[key] = tc
        elif match.result is None and tc.result is not None:
            match.result = tc.result

    if final_output:
        trace.metadata["final_output"] = final_output
        root.output_data = final_output
    if query:
        trace.metadata["query"] = query
        trace.test_name = query
    if start_ns is not None and end_ns is not None and end_ns > start_ns:
        trace.total_duration_ms = (end_ns - start_ns) / 1e6

    trace.spans.append(root)
    trace.compute_metrics()
    return trace, query


# ── Internal helpers ───────────────────────────────────────────────────────────


def _normalize_span(span: dict[str, Any]) -> dict[str, Any]:
    """Normalize OTLP attribute lists into a plain attributes dict."""
    attrs = span.get("attributes")
    if isinstance(attrs, list):
        flat: dict[str, Any] = {}
        for item in attrs:
            if isinstance(item, dict) and "key" in item:
                flat[str(item["key"])] = _otlp_value(item.get("value"))
        span = dict(span)
        span["attributes"] = flat
    return span


def _otlp_value(value: Any) -> Any:
    """Unwrap an OTLP AnyValue ({stringValue: ...} etc.)."""
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            v = value[key]
            return int(v) if key == "intValue" and isinstance(v, str) else v
    if "arrayValue" in value:
        return [_otlp_value(v) for v in (value["arrayValue"] or {}).get("values", [])]
    return value


def _looks_like_llm(attrs: dict[str, Any]) -> bool:
    return bool(attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model"))


def _looks_like_agent(attrs: dict[str, Any]) -> bool:
    return bool(attrs.get("gen_ai.agent.name"))


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _span_duration_ms(span: dict[str, Any]) -> float:
    start = _as_int(span.get("startTimeUnixNano"))
    end = _as_int(span.get("endTimeUnixNano"))
    return (end - start) / 1e6 if end > start else 0.0


def _widen_window(
    span: dict[str, Any], start_ns: Optional[int], end_ns: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    s = _as_int(span.get("startTimeUnixNano"))
    e = _as_int(span.get("endTimeUnixNano"))
    if s:
        start_ns = s if start_ns is None else min(start_ns, s)
    if e:
        end_ns = e if end_ns is None else max(end_ns, e)
    return start_ns, end_ns


def _parse_json_attr(value: Any, default: Any) -> Any:
    """GenAI content attributes are JSON-encoded strings; tolerate both."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value if default is None else default
    return default


def _map_langfuse_span(
    span: dict[str, Any],
    attrs: dict[str, Any],
    lf_type: str,
    root: Span,
    trace: Trace,
    query: Optional[str],
    final_output: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Map one Langfuse-dialect span; returns updated (query, final_output).

    Shapes verified against a live langfuse 4.13 capture:
      generation: langfuse.observation.model.name, .usage_details (JSON with
                  prompt_tokens/completion_tokens), .input (JSON with
                  messages), .output (JSON message dict)
      tool/retriever: .input (JSON dict → arguments), .output (result)
      agent/chain root: .input / .output as plain strings (the query and
                  final answer); langfuse.internal.is_app_root marks it
    """
    lf_input = _parse_json_attr(attrs.get("langfuse.observation.input"), default=None)
    lf_output = _parse_json_attr(attrs.get("langfuse.observation.output"), default=None)

    if lf_type == "generation":
        usage = _parse_json_attr(
            attrs.get("langfuse.observation.usage_details"), default={},
        )
        usage = usage if isinstance(usage, dict) else {}
        root.llm_calls.append(LLMCall(
            model=str(attrs.get("langfuse.observation.model.name") or ""),
            tokens_in=_as_int(usage.get("prompt_tokens") or usage.get("input")),
            tokens_out=_as_int(usage.get("completion_tokens") or usage.get("output")),
            output_text=(
                lf_output.get("content") if isinstance(lf_output, dict)
                and isinstance(lf_output.get("content"), str) else ""
            ) or "",
            duration_ms=_span_duration_ms(span),
        ))
        # Fallbacks: a generation-only export still yields query + answer
        if query is None and isinstance(lf_input, dict):
            for msg in reversed(lf_input.get("messages") or []):
                if isinstance(msg, dict) and msg.get("role") == "user" \
                        and isinstance(msg.get("content"), str):
                    query = msg["content"]
                    break
        if isinstance(lf_output, dict) and isinstance(lf_output.get("content"), str) \
                and lf_output["content"].strip():
            final_output = lf_output["content"]
        return query, final_output

    if lf_type in ("tool", "retriever"):
        root.tool_calls.append(ToolCall(
            tool_name=str(span.get("name") or f"unknown-{lf_type}"),
            arguments=lf_input if isinstance(lf_input, dict) else {},
            result=lf_output,
            duration_ms=_span_duration_ms(span),
        ))
        return query, final_output

    # agent / chain / span — the root observation carries the conversation's
    # query and final answer as plain values and names the agent
    if attrs.get("langfuse.internal.is_app_root") or lf_type == "agent":
        name = str(span.get("name") or "").strip()
        if name:
            trace.agent_name = name
            root.name = name
    if isinstance(lf_input, str) and lf_input.strip():
        query = lf_input
    if isinstance(lf_output, str) and lf_output.strip():
        final_output = lf_output
    return query, final_output


def _tool_calls_from_messages(spans: list[dict[str, Any]]) -> list[ToolCall]:
    """Recover tool calls + results from GenAI message content.

    Shapes verified against a live openllmetry 0.62 export:
      output part: {"type": "tool_call", "name", "id", "arguments": {...}}
      input part:  {"type": "tool_call_response", "id", "response": "<json>"}
    Results pair to calls by id; an unpaired call keeps result=None (the
    retrieval layer SKIPs on uncaptured results, never guesses).
    """
    calls: dict[str, ToolCall] = {}   # id → ToolCall (insertion-ordered)
    unkeyed: list[ToolCall] = []
    responses: dict[str, Any] = {}

    for span in spans:
        attrs = span.get("attributes") or {}
        for key in ("gen_ai.output.messages", "gen_ai.input.messages"):
            messages = _parse_json_attr(attrs.get(key), default=None)
            if not isinstance(messages, list):
                continue
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                for part in msg.get("parts") or []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "tool_call" and part.get("name"):
                        call_id = str(part.get("id") or "")
                        if call_id and call_id in calls:
                            continue  # same call echoed in a later span's input
                        tc = ToolCall(
                            tool_name=str(part["name"]),
                            arguments=part.get("arguments")
                            if isinstance(part.get("arguments"), dict)
                            else _parse_json_attr(part.get("arguments"), default={}),
                        )
                        if call_id:
                            calls[call_id] = tc
                        else:
                            unkeyed.append(tc)
                    elif part.get("type") == "tool_call_response":
                        call_id = str(part.get("id") or "")
                        if call_id:
                            responses[call_id] = _parse_json_attr(
                                part.get("response"), default=part.get("response"),
                            )

    for call_id, tc in calls.items():
        if call_id in responses and tc.result is None:
            tc.result = responses[call_id]
    return list(calls.values()) + unkeyed


def _last_message_text(
    attrs: dict[str, Any], key: str, role: str,
) -> Optional[str]:
    """Extract the last `role` message's text from a GenAI messages attribute.

    Tolerates both semconv message shapes:
      [{"role": "user", "parts": [{"type": "text", "content": "..."}]}]
      [{"role": "user", "content": "..."}]
    """
    messages = _parse_json_attr(attrs.get(key), default=None)
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != role:
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        parts = msg.get("parts")
        if isinstance(parts, list):
            texts = [
                str(p.get("content") or p.get("text") or "")
                for p in parts
                if isinstance(p, dict) and p.get("type") in (None, "text")
            ]
            joined = "\n".join(t for t in texts if t)
            if joined.strip():
                return joined
    return None
