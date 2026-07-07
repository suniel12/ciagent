# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
LangSmith run export → CIAgent Trace importer (F7).

Converts LangSmith **Run objects** — what `langsmith run export`, the
`Client.list_runs` API, and the SDK's own tracer produce — into a
`ciagent.models.Trace`. This is LangSmith's native schema, distinct from
OTel spans (which LangSmith can also emit; those go through the OTel
importer).

Shapes verified against a REAL export: a live LangChain tool-calling agent
traced by langchain-core's BaseTracer (the exact machinery LangSmith's
LangChainTracer uses), checked in as tests/fixtures/langsmith_runs_real.json.

Accepted file shapes (JSON / JSONL):
  - a flat list of run dicts (list_runs / CLI export)
  - a {"runs": [...]} wrapper
  - a single root run dict with nested child_runs (RunTree shape)
  - JSONL: one run object per line

Run mapping:
  run_type "llm"    → LLMCall — model from extra.invocation_params,
                      tokens from outputs.llm_output.token_usage,
                      output text from outputs.generations
  run_type "tool"   → ToolCall — name, arguments=inputs, result from
                      outputs.output (ToolMessage dicts unwrap to content;
                      a missing output stays None so the retrieval layer
                      SKIPs, never guesses)
  root run (no parent_run_id) → query from inputs (messages or input),
                      final answer from outputs (messages or output),
                      agent name from the run name

This module only MAPS; fitness to become a golden is the artifact gate's
call (`gate_imported_golden`) — partial exports are rejected there with
the missing fields named.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from ciagent.models import LLMCall, Span, SpanKind, ToolCall, Trace


class LangsmithImportError(ValueError):
    """The file cannot be read as LangSmith runs at all (not a gate rejection)."""


def looks_like_runs(data: Any) -> bool:
    """Whether parsed JSON is LangSmith run data (vs OTel spans etc.)."""
    if isinstance(data, dict):
        if isinstance(data.get("runs"), list):
            return looks_like_runs(data["runs"])
        return "run_type" in data
    if isinstance(data, list) and data:
        return all(
            isinstance(r, dict) and "run_type" in r and "attributes" not in r
            for r in data
        )
    return False


def load_runs(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Read a LangSmith export and return a flat, start-time-ordered run list."""
    text = Path(path).read_text(encoding="utf-8")
    data = _parse_json_or_jsonl(path, text)

    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        data = data["runs"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise LangsmithImportError(f"'{path}' contains no runs.")

    flat: list[dict[str, Any]] = []
    for run in data:
        _flatten_run(run, flat)
    if not all(isinstance(r, dict) and "run_type" in r for r in flat):
        raise LangsmithImportError(
            f"'{path}' is not a LangSmith runs export — entries are missing "
            "run_type."
        )
    flat.sort(key=lambda r: str(r.get("start_time") or r.get("dotted_order") or ""))
    return flat


def trace_from_langsmith(runs: list[dict[str, Any]]) -> tuple[Trace, Optional[str]]:
    """Map LangSmith runs onto a CIAgent Trace.

    Returns (trace, query_text). Either may be incomplete — completeness is
    judged by the artifact gate, not here.
    """
    root_span = Span(kind=SpanKind.AGENT, name="imported-agent")
    trace = Trace(agent_name="imported-agent", framework="langsmith")

    query: Optional[str] = None
    final_output: Optional[str] = None

    roots = [r for r in runs if not r.get("parent_run_id")]
    root = roots[0] if roots else None
    if root is not None:
        name = str(root.get("name") or "").strip()
        if name:
            trace.agent_name = name
            root_span.name = name
        query = _message_text(root.get("inputs"), want="user")
        final_output = _message_text(root.get("outputs"), want="assistant")
        if root.get("error"):
            trace.metadata["error"] = str(root["error"])

    for run in runs:
        run_type = run.get("run_type")
        if run_type == "llm":
            root_span.llm_calls.append(_llm_call_from(run))
        elif run_type == "tool":
            root_span.tool_calls.append(_tool_call_from(run))

    if final_output:
        trace.metadata["final_output"] = final_output
        root_span.output_data = final_output
    if query:
        trace.metadata["query"] = query
        trace.test_name = query
    duration = _window_ms(runs)
    if duration:
        trace.total_duration_ms = duration

    trace.spans.append(root_span)
    trace.compute_metrics()
    return trace, query


# ── Internal helpers ───────────────────────────────────────────────────────────


def _parse_json_or_jsonl(path: Union[str, Path], text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # JSONL: one run per line (langsmith CLI writes .jsonl)
    runs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise LangsmithImportError(
                f"cannot read '{path}' as JSON or JSONL: {e}"
            ) from e
    if not runs:
        raise LangsmithImportError(f"'{path}' contains no runs.")
    return runs


def _flatten_run(run: Any, out: list[dict[str, Any]]) -> None:
    """RunTree shape: nested child_runs flatten into the list."""
    if not isinstance(run, dict):
        return
    children = run.get("child_runs") or []
    slim = {k: v for k, v in run.items() if k != "child_runs"}
    out.append(slim)
    for child in children:
        _flatten_run(child, out)


def _llm_call_from(run: dict[str, Any]) -> LLMCall:
    outputs = run.get("outputs") or {}
    llm_output = outputs.get("llm_output") or {}
    usage = llm_output.get("token_usage") or {}
    params = (run.get("extra") or {}).get("invocation_params") or {}
    model = str(
        llm_output.get("model_name") or params.get("model_name")
        or params.get("model") or run.get("name") or ""
    )
    # "_type" is e.g. "openai-chat" / "anthropic-chat"
    provider = str(params.get("_type") or "").split("-")[0]
    return LLMCall(
        model=model,
        provider=provider,
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        output_text=_generation_text(outputs) or "",
        duration_ms=_run_duration_ms(run),
    )


def _tool_call_from(run: dict[str, Any]) -> ToolCall:
    inputs = run.get("inputs")
    output = (run.get("outputs") or {}).get("output")
    # Tool outputs are often serialized ToolMessage dicts — the payload is
    # their content. Anything else passes through untouched.
    if isinstance(output, dict) and "content" in output:
        output = output["content"]
    return ToolCall(
        tool_name=str(run.get("name") or "unknown-tool"),
        arguments=inputs if isinstance(inputs, dict) else {},
        result=output,
        error=str(run["error"]) if run.get("error") else None,
        duration_ms=_run_duration_ms(run),
    )


def _generation_text(outputs: dict[str, Any]) -> Optional[str]:
    generations = outputs.get("generations")
    if not isinstance(generations, list) or not generations:
        return None
    first = generations[0]
    if isinstance(first, list):  # chat models nest one level deeper
        first = first[0] if first else None
    if isinstance(first, dict):
        text = first.get("text")
        if isinstance(text, str) and text:
            return text
    return None


_ROLE_ALIASES = {
    "user": "user", "human": "user",
    "assistant": "assistant", "ai": "assistant",
}


def _message_text(payload: Any, want: str) -> Optional[str]:
    """Extract the last `want`-role message text from run inputs/outputs.

    Tolerates the shapes real runs carry: message dicts ({type|role,
    content}), ("role", "text") tuples/lists, and plain {"input"|"output":
    str} chains.
    """
    if not isinstance(payload, dict):
        return None

    messages = payload.get("messages")
    if isinstance(messages, list):
        flat_msgs: list[Any] = []
        for m in messages:
            # some exports nest a batch: {"messages": [[msg, msg, ...]]}
            if isinstance(m, list) and m and isinstance(m[0], (dict, list)):
                flat_msgs.extend(m)
            else:
                flat_msgs.append(m)
        for m in reversed(flat_msgs):
            role, content = _role_and_content(m)
            if role == want and isinstance(content, str) and content.strip():
                return content
        return None

    keys = ("input", "question") if want == "user" else ("output", "answer")
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            role, content = _role_and_content(value)
            if role in (None, want) and isinstance(content, str) and content.strip():
                return content
    return None


def _role_and_content(message: Any) -> tuple[Optional[str], Optional[str]]:
    if isinstance(message, dict):
        role = _ROLE_ALIASES.get(str(message.get("role") or message.get("type") or "").lower())
        content = message.get("content")
        return role, content if isinstance(content, str) else None
    if isinstance(message, (list, tuple)) and len(message) == 2:
        role = _ROLE_ALIASES.get(str(message[0]).lower())
        return role, message[1] if isinstance(message[1], str) else None
    return None, None


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        # py3.10 fromisoformat rejects the Z suffix real exports carry
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _run_duration_ms(run: dict[str, Any]) -> float:
    start, end = _parse_ts(run.get("start_time")), _parse_ts(run.get("end_time"))
    if start and end and end > start:
        return (end - start).total_seconds() * 1000
    return 0.0


def _window_ms(runs: list[dict[str, Any]]) -> float:
    starts = [t for t in (_parse_ts(r.get("start_time")) for r in runs) if t]
    ends = [t for t in (_parse_ts(r.get("end_time")) for r in runs) if t]
    if starts and ends and max(ends) > min(starts):
        return (max(ends) - min(starts)).total_seconds() * 1000
    return 0.0
