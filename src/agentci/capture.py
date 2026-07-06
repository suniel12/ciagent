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
        >>> from agentci.capture import TraceContext
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
                                span.tool_calls.append(ToolCall(
                                    tool_name=tc.function.name,
                                    arguments=tool_args,
                                ))
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
                            span.tool_calls.append(ToolCall(
                                tool_name=block.name,
                                arguments=tool_args,
                            ))
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
        
        # Extract reasoning trajectory from LangGraph messages
        messages = state.get("messages", [])
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
                        arguments=t_args
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
