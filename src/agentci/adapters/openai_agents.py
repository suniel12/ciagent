"""
OpenAI Agents SDK Adapter for AgentCI.

Implements the SDK's pluggable TracingProcessor interface to natively capture
agent traces — including handoffs, guardrails, tool calls, and LLM generations —
into AgentCI's universal Trace model.

Usage:
    from agentci.adapters.openai_agents import AgentCITraceProcessor
    from agents.tracing import add_trace_processor

    processor = AgentCITraceProcessor()
    add_trace_processor(processor)

    # Run your agent normally...
    result = await Runner.run(triage_agent, "I was charged twice")

    # Retrieve the AgentCI trace
    trace = processor.get_last_trace()
"""

from __future__ import annotations

import time
from typing import Any

from agentci.models import (
    Trace,
    Span,
    SpanKind,
    LLMCall,
    ToolCall,
)


class AgentCITraceProcessor:
    """
    Captures OpenAI Agents SDK traces into AgentCI's Trace model.

    Implements the TracingProcessor protocol:
        on_trace_start, on_trace_end, on_span_start, on_span_end,
        shutdown, force_flush
    """

    def __init__(self) -> None:
        self._current_trace: Trace | None = None
        self._last_trace: Trace | None = None
        self._span_map: dict[str, Span] = {}  # SDK span_id -> AgentCI Span
        self._span_start_times: dict[str, float] = {}

    # ── TracingProcessor protocol ──────────────────────

    def on_trace_start(self, trace: Any) -> None:
        """Called when a new trace begins (one per Runner.run())."""
        self._current_trace = Trace(
            trace_id=getattr(trace, "trace_id", ""),
            agent_name=getattr(trace, "name", ""),
            framework="openai_agents",
        )
        self._span_map.clear()
        self._span_start_times.clear()

    def on_span_start(self, span: Any) -> None:
        """Called when a span begins."""
        span_id = getattr(span, "span_id", "")
        self._span_start_times[span_id] = time.monotonic()

    def on_span_end(self, span: Any) -> None:
        """Called when a span ends. Maps SDK span types to AgentCI model."""
        if self._current_trace is None:
            return

        span_id = getattr(span, "span_id", "")
        parent_id = getattr(span, "parent_id", None)
        span_data = getattr(span, "span_data", None)

        # Calculate duration
        start_time = self._span_start_times.pop(span_id, time.monotonic())
        duration_ms = (time.monotonic() - start_time) * 1000

        # Determine span type and extract data
        agentci_span = self._map_span(span_data, span_id, parent_id, duration_ms)
        if agentci_span:
            self._current_trace.spans.append(agentci_span)
            self._span_map[span_id] = agentci_span

    def on_trace_end(self, trace: Any) -> None:
        """Called when the trace ends. Finalize metrics."""
        if self._current_trace is not None:
            self._current_trace.compute_metrics()
            self._last_trace = self._current_trace
            self._current_trace = None

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass

    def force_flush(self) -> None:
        """Flush any pending data."""
        pass

    # ── Public API ─────────────────────────────────────

    def get_last_trace(self) -> Trace | None:
        """Retrieve the most recently completed AgentCI trace."""
        return self._last_trace

    # ── Internal mapping ───────────────────────────────

    def _map_span(
        self,
        span_data: Any,
        span_id: str,
        parent_id: str | None,
        duration_ms: float,
    ) -> Span | None:
        """Map an OpenAI SDK span_data object to an AgentCI Span."""
        if span_data is None:
            return None

        type_name = type(span_data).__name__

        if type_name == "AgentSpanData":
            return Span(
                span_id=span_id,
                parent_span_id=parent_id,
                kind=SpanKind.AGENT,
                name=getattr(span_data, "name", ""),
                duration_ms=duration_ms,
                metadata={
                    "handoffs": getattr(span_data, "handoffs", []),
                    "tools": getattr(span_data, "tools", []),
                },
            )

        elif type_name == "GenerationSpanData":
            llm_call = LLMCall(
                model=getattr(span_data, "model", ""),
                provider="openai",
                input_messages=getattr(span_data, "input", []) if isinstance(getattr(span_data, "input", None), list) else [],
                output_text=str(getattr(span_data, "output", "")),
                tokens_in=_safe_get_usage(span_data, "input_tokens"),
                tokens_out=_safe_get_usage(span_data, "output_tokens"),
                duration_ms=duration_ms,
            )
            return Span(
                span_id=span_id,
                parent_span_id=parent_id,
                kind=SpanKind.LLM_CALL,
                name="generation",
                llm_calls=[llm_call],
                duration_ms=duration_ms,
            )

        elif type_name == "HandoffSpanData":
            return Span(
                span_id=span_id,
                parent_span_id=parent_id,
                kind=SpanKind.HANDOFF,
                name="handoff",
                from_agent=getattr(span_data, "from_agent", ""),
                to_agent=getattr(span_data, "to_agent", ""),
                duration_ms=duration_ms,
            )

        elif type_name == "FunctionSpanData":
            tool_call = ToolCall(
                tool_name=getattr(span_data, "name", ""),
                arguments=getattr(span_data, "input", {}) if isinstance(getattr(span_data, "input", None), dict) else {},
                result=getattr(span_data, "output", None),
            )
            return Span(
                span_id=span_id,
                parent_span_id=parent_id,
                kind=SpanKind.TOOL_CALL,
                name=getattr(span_data, "name", ""),
                tool_calls=[tool_call],
                duration_ms=duration_ms,
            )

        elif type_name == "GuardrailSpanData":
            return Span(
                span_id=span_id,
                parent_span_id=parent_id,
                kind=SpanKind.GUARDRAIL,
                name=getattr(span_data, "name", ""),
                guardrail_name=getattr(span_data, "name", ""),
                guardrail_triggered=getattr(span_data, "triggered", False),
                duration_ms=duration_ms,
            )

        else:
            # Unknown span type — capture as generic metadata
            return Span(
                span_id=span_id,
                parent_span_id=parent_id,
                kind=SpanKind.AGENT,
                name=type_name,
                duration_ms=duration_ms,
                metadata={"raw_type": type_name},
            )


def _safe_get_usage(span_data: Any, field: str) -> int:
    """Safely extract token usage from GenerationSpanData."""
    usage = getattr(span_data, "usage", None)
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return usage.get(field, 0)
    return getattr(usage, field, 0)
