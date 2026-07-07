# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Trace capture via monkey-patching.

Strategy: Wrap the OpenAI/Anthropic client's .create() methods to
automatically record every LLM call and tool invocation into a Trace
object. The developer doesn't change their agent code at all.

Phase 1: Patch openai.ChatCompletion and anthropic.Messages
Phase 2: Add OTEL span emission for interop with Arize/Langfuse
"""
from __future__ import annotations

import time
import contextvars
from contextlib import contextmanager
from .models import Trace, Span, LLMCall, ToolCall, SpanKind
from .cost import compute_cost

def _tool_result_content(content):
    """Normalize a tool-result payload for ToolCall.result.

    Strings pass through; a list of text blocks joins to text; anything else
    (structured lists/dicts) is kept raw — the retrieval layer parses those
    itself and must see them unmangled.
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get('text') if block.get('type') == 'text' else None
            else:
                text = getattr(block, 'text', None)
            if text is None:
                return content  # not pure text blocks — keep raw
            parts.append(str(text))
        return "\n".join(parts)
    return content


# Global context var — allows nested agent calls to share a trace
_active_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    '_active_trace', default=None
)
_active_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    '_active_span', default=None
)
# Nesting depth of active TraceContexts. SDK patches are installed only by the
# outermost context — stacked patches would record every LLM call once per
# wrapper into the same active span.
_patch_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    '_patch_depth', default=0
)


class TraceContext:
    """Context manager that captures all LLM/tool activity into a Trace.

    Automatically monkey-patches OpenAI and Anthropic client .create() methods
    to record every LLM call and tool invocation. No agent code changes needed.

    Args:
        agent_name: Name of the agent being traced (for identification).
        test_name: Name of the test case (for labeling).

    Attributes:
        trace: The Trace object containing all captured activity.

    Example:
        >>> from ciagent.capture import TraceContext
        >>> with TraceContext(agent_name="booking_agent", test_name="test_booking") as ctx:
        ...     result = my_agent.run("Book a flight to NYC")
        ...     trace = ctx.trace
        >>> print(trace.tool_call_sequence)
        ['search_flights', 'book_flight']

    For LangGraph agents, call attach_langgraph_state() after graph.invoke():
        >>> with TraceContext(agent_name="rag_agent") as ctx:
        ...     result = graph.invoke({"messages": [("user", query)]})
        ...     ctx.attach_langgraph_state(result)
    """

    def __init__(self, agent_name: str = "", test_name: str = ""):
        self.trace = Trace(agent_name=agent_name, test_name=test_name)
        self._patches = []
        self._start_time = 0.0
        self._trace_token = None
        self._span_token = None
        # tool_call_id → ToolCall awaiting its result. Tool outputs come back
        # in the NEXT request's messages (openai role="tool" entries /
        # anthropic tool_result blocks); the patches backfill ToolCall.result
        # from there so the retrieval layer (F4) can read it.
        self._pending_tool_results: dict[str, ToolCall] = {}

    def __enter__(self):
        # Create root span
        root_span = Span(kind=SpanKind.AGENT, name=self.trace.agent_name)
        self.trace.spans.append(root_span)

        # Set context vars, keeping reset tokens so a nested context restores
        # the enclosing context on exit instead of clearing it
        self._trace_token = _active_trace.set(self.trace)
        self._span_token = _active_span.set(root_span)

        # Apply monkey patches only in the outermost context: stacked patches
        # would record every LLM call once per wrapper into the active span
        # (e.g. a Trace-returning runner that uses TraceContext itself, wrapped
        # again by _run_with_retry)
        if _patch_depth.get() == 0:
            self._patch_openai()
            self._patch_anthropic()
        _patch_depth.set(_patch_depth.get() + 1)

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        # Compute duration
        self.trace.total_duration_ms = (time.perf_counter() - self._start_time) * 1000

        # Roll up metrics
        self.trace.compute_metrics()

        # Auto-extract final output if not manually set
        self._auto_extract_final_output()

        # Remove patches (only the outermost context installed any)
        _patch_depth.set(max(_patch_depth.get() - 1, 0))
        for restore_fn in self._patches:
            restore_fn()
        self._patches = []

        # Restore the enclosing context (None only when outermost)
        if self._trace_token is not None:
            _active_trace.reset(self._trace_token)
            self._trace_token = None
        if self._span_token is not None:
            _active_span.reset(self._span_token)
            self._span_token = None

    def _auto_extract_final_output(self) -> None:
        """Auto-extract the agent's final output from the trace.

        Only runs if ``final_output`` has not been manually set in
        ``trace.metadata``.  Extraction priority:

        1. LangGraph state: last AI message's ``.content``
        2. Last span's ``output_data`` (string)
        3. Last span's ``output_data`` dict with common keys
        4. Last LLM call's ``output_text`` from last span
        """
        if "final_output" in self.trace.metadata:
            return

        # 1. LangGraph state
        graph_state = getattr(self.trace, "graph_state", None)
        if graph_state:
            messages = graph_state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", None)
                if content:
                    self.trace.metadata["final_output"] = str(content)
                    return

        # 2-3. Last span output_data
        if self.trace.spans:
            last_span = self.trace.spans[-1]
            output = last_span.output_data

            if output is not None:
                if isinstance(output, str):
                    self.trace.metadata["final_output"] = output
                    return
                if isinstance(output, dict):
                    for key in ("content", "message", "text", "output"):
                        if key in output:
                            self.trace.metadata["final_output"] = str(output[key])
                            return

            # 4. Last LLM call output
            if last_span.llm_calls:
                last_llm = last_span.llm_calls[-1]
                # Handle both LLMCall objects and raw dicts
                if isinstance(last_llm, dict):
                    text = last_llm.get("content") or last_llm.get("output_text", "")
                else:
                    text = getattr(last_llm, "output_text", "")
                if text:
                    self.trace.metadata["final_output"] = str(text)
                    return
    
    def _patch_openai(self):
        """Wrap openai.chat.completions.create to capture LLM calls."""
        try:
            import openai  # type: ignore
            original_create = openai.resources.chat.completions.Completions.create
            
            def patched_create(self_client, *args, **kwargs):
                # Requests carry the results of PREVIOUS tool calls as
                # role="tool" messages — backfill them before anything else.
                self._backfill_openai_tool_results(kwargs.get('messages'))

                start = time.perf_counter()
                response = original_create(self_client, *args, **kwargs)
                duration = (time.perf_counter() - start) * 1000

                span = _active_span.get()
                if span is not None:
                    model = kwargs.get('model', getattr(response, 'model', ''))
                    usage = getattr(response, 'usage', None)
                    tokens_in = getattr(usage, 'prompt_tokens', 0) if usage else 0
                    tokens_out = getattr(usage, 'completion_tokens', 0) if usage else 0

                    llm_call = LLMCall(
                        model=model,
                        provider="openai",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=compute_cost("openai", model, tokens_in, tokens_out),
                        duration_ms=duration,
                    )
                    span.llm_calls.append(llm_call)

                    # Capture tool calls from response
                    choices = getattr(response, 'choices', [])
                    if choices:
                        message = choices[0].message
                        tool_calls = getattr(message, 'tool_calls', None)
                        if tool_calls:
                            for tc in tool_calls:
                                import json
                                tool_args = json.loads(tc.function.arguments)
                                tool_call = ToolCall(
                                    tool_name=tc.function.name,
                                    arguments=tool_args,
                                )
                                span.tool_calls.append(tool_call)
                                call_id = getattr(tc, 'id', None)
                                if call_id:
                                    self._pending_tool_results[call_id] = tool_call
                                # Propagate tool args into span attributes
                                span.attributes[f"tool.args.{tc.function.name}"] = tool_args

                return response
            
            openai.resources.chat.completions.Completions.create = patched_create
            self._patches.append(
                lambda: setattr(
                    openai.resources.chat.completions.Completions, 
                    'create', 
                    original_create
                )
            )
        except ImportError:
            pass  # OpenAI not installed — skip silently
    
    def _patch_anthropic(self):
        """Wrap anthropic.messages.create to capture LLM calls."""
        try:
            import anthropic  # type: ignore
            original_create = anthropic.resources.messages.Messages.create
            
            def patched_create(self_client, *args, **kwargs):
                # Requests carry the results of PREVIOUS tool_use calls as
                # tool_result blocks — backfill them before anything else.
                self._backfill_anthropic_tool_results(kwargs.get('messages'))

                start = time.perf_counter()
                response = original_create(self_client, *args, **kwargs)
                duration = (time.perf_counter() - start) * 1000

                span = _active_span.get()
                if span is not None:
                    model = kwargs.get('model', getattr(response, 'model', ''))
                    usage = getattr(response, 'usage', None)
                    tokens_in = getattr(usage, 'input_tokens', 0) if usage else 0
                    tokens_out = getattr(usage, 'output_tokens', 0) if usage else 0

                    llm_call = LLMCall(
                        model=model,
                        provider="anthropic",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=compute_cost("anthropic", model, tokens_in, tokens_out),
                        duration_ms=duration,
                    )
                    span.llm_calls.append(llm_call)

                    # Capture tool use blocks
                    for block in getattr(response, 'content', []):
                        if getattr(block, 'type', '') == 'tool_use':
                            tool_args = block.input if isinstance(block.input, dict) else {}
                            tool_call = ToolCall(
                                tool_name=block.name,
                                arguments=tool_args,
                            )
                            span.tool_calls.append(tool_call)
                            call_id = getattr(block, 'id', None)
                            if call_id:
                                self._pending_tool_results[call_id] = tool_call
                            # Propagate tool args into span attributes
                            span.attributes[f"tool.args.{block.name}"] = tool_args

                return response
            
            anthropic.resources.messages.Messages.create = patched_create
            self._patches.append(
                lambda: setattr(
                    anthropic.resources.messages.Messages, 
                    'create', 
                    original_create
                )
            )
        except ImportError:
            pass
            
    def _backfill_openai_tool_results(self, messages) -> None:
        """Fill pending ToolCall.result from role="tool" messages.

        OpenAI chat-completions tool outputs are produced by the agent and
        sent back in the next request as {"role": "tool", "tool_call_id",
        "content"} entries — the only place the wire protocol exposes them.
        """
        if not self._pending_tool_results or not messages:
            return
        for msg in messages:
            role = msg.get('role') if isinstance(msg, dict) else getattr(msg, 'role', '')
            if role != 'tool':
                continue
            call_id = (
                msg.get('tool_call_id') if isinstance(msg, dict)
                else getattr(msg, 'tool_call_id', None)
            )
            tool_call = self._pending_tool_results.pop(call_id, None) if call_id else None
            if tool_call is not None and tool_call.result is None:
                content = (
                    msg.get('content') if isinstance(msg, dict)
                    else getattr(msg, 'content', None)
                )
                tool_call.result = _tool_result_content(content)

    def _backfill_anthropic_tool_results(self, messages) -> None:
        """Fill pending ToolCall.result from tool_result content blocks.

        Anthropic tool outputs come back in the next request as
        {"type": "tool_result", "tool_use_id", "content"} blocks inside a
        user message's content list.
        """
        if not self._pending_tool_results or not messages:
            return
        for msg in messages:
            content = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
            if not isinstance(content, list):
                continue
            for block in content:
                btype = block.get('type') if isinstance(block, dict) else getattr(block, 'type', '')
                if btype != 'tool_result':
                    continue
                call_id = (
                    block.get('tool_use_id') if isinstance(block, dict)
                    else getattr(block, 'tool_use_id', None)
                )
                tool_call = self._pending_tool_results.pop(call_id, None) if call_id else None
                if tool_call is not None and tool_call.result is None:
                    raw = (
                        block.get('content') if isinstance(block, dict)
                        else getattr(block, 'content', None)
                    )
                    tool_call.result = _tool_result_content(raw)

    def attach_langgraph_state(self, state: dict) -> None:
        """Parse a LangGraph MessagesState to extract tools and node executions.

        Call this after graph.invoke() to populate the trace with tool calls
        and node executions extracted from the LangGraph state.

        Args:
            state: The LangGraph state dict (must contain a "messages" key).

        Example:
            >>> with TraceContext(agent_name="rag_agent") as ctx:
            ...     result = graph.invoke({"messages": [("user", "What is RAG?")]})
            ...     ctx.attach_langgraph_state(result)
            ...     trace = ctx.trace
        """
        import json
        
        # Save snapshot
        self.trace.graph_state = state
        
        span = _active_span.get()
        if not span:
            return
            
        span.graph_state = state

        # ToolMessages carry each executed tool's output keyed by
        # tool_call_id — pair them so ToolCall.result is populated (the
        # retrieval layer reads it; unpaired calls SKIP, never guess).
        from .adapters.langgraph import _tool_results_by_id

        # Extract reasoning trajectory from LangGraph messages
        messages = state.get("messages", [])
        results_by_id = _tool_results_by_id(messages)
        for msg in messages:
            msg_name = getattr(msg, "name", "")
            
            # If the message is emitted from a distinct node (e.g. grade_artifacts, rewrite_question), 
            # log it as a lightweight ToolCall so it appears in the sequence
            if msg_name and msg_name not in ["retrieve_docs"]:  
                span.tool_calls.append(ToolCall(
                    tool_name=msg_name,
                    arguments={"content": getattr(msg, "content", "")}
                ))
            
            # Extract standard tool calls natively from AIMessage.tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    # Langchain encodes tools as dicts
                    t_name = tc.get("name", "")
                    t_args = tc.get("args", {})

                    if not isinstance(t_args, dict):
                        try:
                            t_args = json.loads(t_args)
                        except:
                            t_args = {"raw": str(t_args)}

                    span.tool_calls.append(ToolCall(
                        tool_name=t_name,
                        arguments=t_args,
                        result=results_by_id.get(tc.get("id")),
                    ))

    def attach(self, state: dict) -> None:
        """Alias for attach_langgraph_state — shorter to type."""
        self.attach_langgraph_state(state)


@contextmanager
def langgraph_trace(agent_name: str = ""):
    """Shortcut context manager for LangGraph agents.

    Usage:
        with langgraph_trace("rag-agent") as ctx:
            output, state = generate_answer_api(query)
            ctx.attach(state)
        trace = ctx.trace
    """
    with TraceContext(agent_name=agent_name) as ctx:
        yield ctx
