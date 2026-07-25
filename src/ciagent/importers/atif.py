# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
ATIF trajectory → CIAgent Trace importer.

Converts an Agent Trajectory Interchange Format file — what Harbor
(harbor-framework, from the Terminal-Bench team) writes as
``trajectory.json`` for every trial — into a `ciagent.models.Trace`.

Shapes verified against a REAL export: a Harbor run of the hello-world
task with the claude-code agent (ATIF-v1.7), checked in as
tests/fixtures/atif_real.json. Field names are pinned to Harbor
RFC 0001 (rfcs/0001-trajectory-format.md); fields the RFC does not
define are ignored (forward-compat).

Field map (RFC 0001 → Trace):

  envelope
    schema_version ("ATIF-v*")   → format detection; metadata["atif_schema_version"]
    agent.name                   → trace.agent_name, root span name
    agent.model_name             → LLMCall.model fallback
    final_metrics.total_cost_usd → trace.total_cost_usd (when steps carry no cost)
  step (source == "user")
    message                      → query / trace.metadata["query"] (first user step)
  step (source == "agent")
    step itself (llm_call_count != 0) → LLMCall
    model_name                   → LLMCall.model
    message                      → LLMCall.output_text; last non-empty becomes
                                   trace.metadata["final_output"]
    metrics.prompt_tokens        → LLMCall.tokens_in
    metrics.completion_tokens    → LLMCall.tokens_out
    metrics.cost_usd             → LLMCall.cost_usd
    tool_calls[].function_name   → ToolCall.tool_name
    tool_calls[].arguments       → ToolCall.arguments
    observation.results[].content (matched on source_call_id ==
      tool_call_id)              → ToolCall.result
    timestamp (first → last step) → trace.total_duration_ms

Steps with source == "system" and reasoning_content are not mapped —
diffing operates on tool calls, tokens, and outputs, not prompts.

This module only MAPS; fitness to become a golden is the artifact gate's
call (`gate_imported_golden`) — partial trajectories are rejected there
with the missing fields named.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from ciagent.models import LLMCall, Span, SpanKind, ToolCall, Trace


class AtifImportError(ValueError):
    """The file cannot be read as an ATIF trajectory at all (not a gate rejection)."""


def looks_like_atif(data: Any) -> bool:
    """Whether parsed JSON is an ATIF trajectory envelope (vs runs/spans)."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("schema_version"), str)
        and data["schema_version"].startswith("ATIF-")
        and isinstance(data.get("steps"), list)
    )


def load_trajectory(path: Union[str, Path]) -> dict[str, Any]:
    """Read an ATIF trajectory file and return the parsed envelope."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AtifImportError(f"cannot read '{path}' as JSON: {e}") from e
    if not looks_like_atif(data):
        raise AtifImportError(
            f"'{path}' is not an ATIF trajectory — expected a schema_version "
            "'ATIF-v*' envelope with a steps list."
        )
    return data


def trace_from_atif(data: dict[str, Any]) -> tuple[Trace, Optional[str]]:
    """Map an ATIF trajectory onto a CIAgent Trace.

    Returns (trace, query_text). Either may be incomplete — completeness is
    judged by the artifact gate, not here.
    """
    agent = data.get("agent") or {}
    agent_name = str(agent.get("name") or "").strip() or "imported-agent"
    agent_model = str(agent.get("model_name") or "")

    root_span = Span(kind=SpanKind.AGENT, name=agent_name)
    trace = Trace(agent_name=agent_name, framework="atif")
    trace.metadata["atif_schema_version"] = str(data.get("schema_version") or "")

    steps = [s for s in data.get("steps", []) if isinstance(s, dict)]
    steps.sort(key=lambda s: s.get("step_id") if isinstance(s.get("step_id"), int) else 0)

    query: Optional[str] = None
    final_output: Optional[str] = None

    for step in steps:
        source = step.get("source")
        text = _message_text(step.get("message"))
        if source == "user":
            if query is None and text:
                query = text
            continue
        if source != "agent":
            continue

        for call in step.get("tool_calls") or []:
            if isinstance(call, dict):
                root_span.tool_calls.append(_tool_call_from(call, step))
        # llm_call_count == 0 marks a deterministic step (RFC 0001) — no LLM
        # was invoked, so it contributes no LLMCall.
        if step.get("llm_call_count") != 0:
            root_span.llm_calls.append(_llm_call_from(step, agent_model, text))
        if text:
            final_output = text

    if final_output:
        trace.metadata["final_output"] = final_output
        root_span.output_data = final_output
    if query:
        trace.metadata["query"] = query
        trace.test_name = query
    duration = _window_ms(steps)
    if duration:
        trace.total_duration_ms = duration

    trace.spans.append(root_span)
    trace.compute_metrics()

    # Per-step cost_usd is optional in the RFC and absent in real Harbor
    # output; the trajectory-level total is authoritative when steps carry
    # no cost of their own.
    final_metrics = data.get("final_metrics") or {}
    if not trace.total_cost_usd and isinstance(final_metrics, dict):
        total_cost = final_metrics.get("total_cost_usd")
        if isinstance(total_cost, (int, float)) and total_cost > 0:
            trace.total_cost_usd = float(total_cost)

    return trace, query


# ── Internal helpers ───────────────────────────────────────────────────────────


def _message_text(message: Any) -> Optional[str]:
    """Flatten an ATIF message (string or content-block array) to text."""
    if isinstance(message, str):
        return message.strip() or None
    if isinstance(message, list):
        parts = []
        for block in message:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        joined = "\n".join(p for p in parts if p.strip())
        return joined.strip() or None
    return None


def _tool_call_from(call: dict[str, Any], step: dict[str, Any]) -> ToolCall:
    arguments = call.get("arguments")
    return ToolCall(
        tool_name=str(call.get("function_name") or "unknown-tool"),
        arguments=arguments if isinstance(arguments, dict) else {},
        result=_observation_content(step, call.get("tool_call_id")),
    )


def _observation_content(step: dict[str, Any], call_id: Any) -> Any:
    """The observation result for a tool call, matched on source_call_id.

    A result without source_call_id pairs up only when the step has a single
    tool call — otherwise attribution would be a guess, and the retrieval
    layer treats a None result as SKIP rather than inventing one.
    """
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return None
    results = [r for r in observation.get("results") or [] if isinstance(r, dict)]
    for result in results:
        if result.get("source_call_id") == call_id and call_id is not None:
            return result.get("content")
    if len(results) == 1 and len(step.get("tool_calls") or []) == 1:
        return results[0].get("content")
    return None


def _llm_call_from(step: dict[str, Any], agent_model: str, text: Optional[str]) -> LLMCall:
    metrics = step.get("metrics") if isinstance(step.get("metrics"), dict) else {}
    cost = metrics.get("cost_usd")
    return LLMCall(
        model=str(step.get("model_name") or agent_model),
        tokens_in=int(metrics.get("prompt_tokens") or 0),
        tokens_out=int(metrics.get("completion_tokens") or 0),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else 0.0,
        output_text=text or "",
    )


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        # py3.10 fromisoformat rejects the Z suffix real trajectories carry
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _window_ms(steps: list[dict[str, Any]]) -> float:
    stamps = [t for t in (_parse_ts(s.get("timestamp")) for s in steps) if t]
    if len(stamps) >= 2 and max(stamps) > min(stamps):
        return (max(stamps) - min(stamps)).total_seconds() * 1000
    return 0.0
