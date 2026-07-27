# Copyright 2025-2026 The CIAgent Authors
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
import warnings
import threading
import contextvars
from contextlib import contextmanager
from .models import Trace, Span, LLMCall, ToolCall, SpanKind
from .cost import compute_cost, compute_cost_for_model


class CostCaptureWarning(UserWarning):
    """A trace finished with LLM calls recorded but zero tokens and zero cost.

    Cost guardrails silently measure nothing in this state, so it is
    surfaced loudly instead of passing quietly.
    """

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
# Process-global refcount of active TraceContexts. The SDK patches target
# shared class attributes, so installation must be global too: the first
# context in (across ALL threads) installs, the last one out restores.
#
# This was previously a ContextVar, which is per-thread — under a parallel
# runner (ciagent record/test --workers N) every worker thread saw depth 0
# and stacked its own patch onto the shared class attribute, so one real LLM
# call executed N wrappers in the calling thread and was recorded N times
# into that thread's span, inflating tokens/cost by ~Nx (issue #76). Restore
# functions live globally for the same reason: an early-finishing context
# must not rip the patches out from under still-running ones.
_patch_lock = threading.Lock()
_patch_refcount = 0
_patch_restores: list = []
# True while a resource-level patch (Completions.create) is executing the
# original SDK call. The transport-level patch checks this so a call that was
# already recorded at the resource level isn't recorded a second time when it
# funnels through the client's request().
_recording_llm_call: contextvars.ContextVar[bool] = contextvars.ContextVar(
    '_recording_llm_call', default=False
)


def _resolve_openai_parsed(response):
    """Unwrap a raw-response wrapper to the parsed model, if needed.

    with_raw_response.* calls return a LegacyAPIResponse/APIResponse wrapper
    instead of the parsed ChatCompletion/Response. Both wrappers cache
    parse() results by type, so parsing here is idempotent — the caller's own
    later .parse() gets the cached object. Anything unrecognized is returned
    as-is.
    """
    if hasattr(response, 'usage') or not hasattr(response, 'parse'):
        return response
    try:
        return response.parse()
    except Exception:
        return response


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

        # Apply monkey patches only in the first active context process-wide:
        # stacked patches would record every LLM call once per wrapper into
        # the calling thread's active span (e.g. a Trace-returning runner
        # that uses TraceContext itself, wrapped again by _run_with_retry, or
        # N parallel worker threads each entering their own context).
        global _patch_refcount
        with _patch_lock:
            if _patch_refcount == 0:
                self._patch_openai()
                self._patch_anthropic()
                # Restores are global: the last context out runs them, which
                # may not be the one that installed
                _patch_restores.extend(self._patches)
                self._patches = []
            _patch_refcount += 1

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        # Compute duration
        self.trace.total_duration_ms = (time.perf_counter() - self._start_time) * 1000

        # Roll up metrics
        self.trace.compute_metrics()

        # Auto-extract final output if not manually set
        self._auto_extract_final_output()

        # Surface silent-zero cost capture instead of passing quietly
        self._warn_if_usage_missing()

        # Remove patches only when the last active context (process-wide)
        # exits; earlier exits must leave them in place for contexts still
        # running on other threads
        global _patch_refcount
        with _patch_lock:
            _patch_refcount = max(_patch_refcount - 1, 0)
            if _patch_refcount == 0:
                for restore_fn in _patch_restores:
                    restore_fn()
                _patch_restores.clear()

        # Restore the enclosing context (None only when outermost)
        if self._trace_token is not None:
            _active_trace.reset(self._trace_token)
            self._trace_token = None
        if self._span_token is not None:
            _active_span.reset(self._span_token)
            self._span_token = None

    def _warn_if_usage_missing(self) -> None:
        """Warn when LLM calls were recorded but no tokens or cost were.

        This is the failure mode where cost guardrails appear to pass while
        measuring nothing (e.g. an SDK path the patches don't cover), so it
        must be loud rather than silent.
        """
        if not self.trace.total_llm_calls:
            return
        if self.trace.total_tokens or self.trace.total_cost_usd:
            return
        warnings.warn(
            f"Trace '{self.trace.agent_name or self.trace.trace_id}' recorded "
            f"{self.trace.total_llm_calls} LLM call(s) but zero tokens and zero cost. "
            "Token usage was not captured, so cost guardrails will not measure "
            "anything for this run. If the agent uses LangGraph, call "
            "ctx.attach_langgraph_state(result) so usage can be read from "
            "message usage_metadata. Otherwise the SDK call path may not be "
            "supported yet; please report it.",
            CostCaptureWarning,
            stacklevel=3,
        )

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
        """Install both openai capture patches.

        Two interception points are needed because SDK wrappers bypass the
        resource method: langchain-openai calls with_raw_response.create,
        whose wrapper binds the resource method at wrapper-construction time,
        so patching Completions.create alone silently records nothing (zero
        tokens, zero cost). The transport-level request() patch catches every
        variant (with_raw_response, responses.create, parse) because they all
        funnel through the client's request(); the resource-level patch stays
        for callers whose wrapper captured the patched method (a reentrancy
        flag prevents double-recording).
        """
        try:
            import openai  # type: ignore
        except ImportError:
            return  # OpenAI not installed — skip silently
        self._patch_openai_completions(openai)
        self._patch_openai_transport(openai)

    def _record_openai_chat(self, span: Span, parsed, model_hint: str, duration: float) -> None:
        """Record one chat-completions call (usage + tool calls) into span."""
        model = model_hint or getattr(parsed, 'model', '') or ''
        usage = getattr(parsed, 'usage', None)
        tokens_in = getattr(usage, 'prompt_tokens', 0) if usage else 0
        tokens_out = getattr(usage, 'completion_tokens', 0) if usage else 0

        span.llm_calls.append(LLMCall(
            model=model,
            provider="openai",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=compute_cost("openai", model, tokens_in, tokens_out),
            duration_ms=duration,
        ))

        # Capture tool calls from response
        choices = getattr(parsed, 'choices', None) or []
        if choices:
            message = choices[0].message
            tool_calls = getattr(message, 'tool_calls', None)
            if tool_calls:
                import json
                for tc in tool_calls:
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

    def _record_openai_responses(self, span: Span, parsed, model_hint: str, duration: float) -> None:
        """Record one Responses API call (usage + function calls) into span."""
        model = model_hint or getattr(parsed, 'model', '') or ''
        usage = getattr(parsed, 'usage', None)
        tokens_in = getattr(usage, 'input_tokens', 0) if usage else 0
        tokens_out = getattr(usage, 'output_tokens', 0) if usage else 0

        span.llm_calls.append(LLMCall(
            model=model,
            provider="openai",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=compute_cost("openai", model, tokens_in, tokens_out),
            duration_ms=duration,
        ))

        import json
        for item in getattr(parsed, 'output', None) or []:
            if getattr(item, 'type', '') != 'function_call':
                continue
            raw_args = getattr(item, 'arguments', None)
            if isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except (ValueError, TypeError):
                    tool_args = {"raw": raw_args}
            else:
                tool_args = raw_args if isinstance(raw_args, dict) else {}
            tool_call = ToolCall(
                tool_name=getattr(item, 'name', ''),
                arguments=tool_args,
            )
            span.tool_calls.append(tool_call)
            call_id = getattr(item, 'call_id', None)
            if call_id:
                self._pending_tool_results[call_id] = tool_call
            span.attributes[f"tool.args.{tool_call.tool_name}"] = tool_args

    def _patch_openai_completions(self, openai):
        """Wrap openai.chat.completions.create to capture LLM calls."""
        original_create = openai.resources.chat.completions.Completions.create

        def patched_create(self_client, *args, **kwargs):
            # Requests carry the results of PREVIOUS tool calls as
            # role="tool" messages — backfill them before anything else.
            self._backfill_openai_tool_results(kwargs.get('messages'))

            # Mark the call so the transport patch doesn't record it again
            token = _recording_llm_call.set(True)
            start = time.perf_counter()
            try:
                response = original_create(self_client, *args, **kwargs)
            finally:
                _recording_llm_call.reset(token)
            duration = (time.perf_counter() - start) * 1000

            span = _active_span.get()
            if span is not None:
                # A with_raw_response wrapper built after patching binds this
                # patched method, so response may be a raw wrapper — unwrap it
                # for usage extraction (parse() is cached, so the caller's
                # own parse() is unaffected).
                parsed = _resolve_openai_parsed(response)
                self._record_openai_chat(span, parsed, kwargs.get('model', ''), duration)

            return response

        openai.resources.chat.completions.Completions.create = patched_create
        self._patches.append(
            lambda: setattr(
                openai.resources.chat.completions.Completions,
                'create',
                original_create
            )
        )

    def _patch_openai_transport(self, openai):
        """Wrap the openai client's request() to catch SDK paths that bypass
        Completions.create: with_raw_response.create (used by
        langchain-openai), responses.create, and the parse variants."""
        try:
            from openai._base_client import SyncAPIClient  # type: ignore
        except (ImportError, AttributeError):
            return  # Very old SDK layout — resource patch still applies
        original_request = SyncAPIClient.request

        def patched_request(client_self, cast_to, options, *args, **kwargs):
            if _recording_llm_call.get():
                # Already being recorded by a resource-level patch
                return original_request(client_self, cast_to, options, *args, **kwargs)

            url = str(getattr(options, 'url', '') or '')
            is_chat = url.endswith('/chat/completions')
            is_responses = url.endswith('/responses')
            if not (is_chat or is_responses):
                return original_request(client_self, cast_to, options, *args, **kwargs)

            json_data = getattr(options, 'json_data', None)
            json_data = json_data if isinstance(json_data, dict) else {}
            if is_chat:
                self._backfill_openai_tool_results(json_data.get('messages'))
            else:
                self._backfill_openai_responses_tool_results(json_data.get('input'))

            streaming = bool(kwargs.get('stream')) or bool(json_data.get('stream'))

            start = time.perf_counter()
            response = original_request(client_self, cast_to, options, *args, **kwargs)
            duration = (time.perf_counter() - start) * 1000

            span = _active_span.get()
            if span is None or streaming:
                # Streams can't be introspected without consuming them; the
                # LangGraph usage_metadata fallback covers streamed calls.
                return response

            parsed = _resolve_openai_parsed(response)
            model_hint = json_data.get('model', '')
            if is_chat:
                self._record_openai_chat(span, parsed, model_hint, duration)
            else:
                self._record_openai_responses(span, parsed, model_hint, duration)
            return response

        SyncAPIClient.request = patched_request
        self._patches.append(
            lambda: setattr(SyncAPIClient, 'request', original_request)
        )
    
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

    def _backfill_openai_responses_tool_results(self, input_items) -> None:
        """Fill pending ToolCall.result from Responses API input items.

        Responses API tool outputs come back in the next request's input as
        {"type": "function_call_output", "call_id", "output"} items.
        """
        if not self._pending_tool_results or not isinstance(input_items, list):
            return
        for item in input_items:
            item_type = (
                item.get('type') if isinstance(item, dict)
                else getattr(item, 'type', '')
            )
            if item_type != 'function_call_output':
                continue
            call_id = (
                item.get('call_id') if isinstance(item, dict)
                else getattr(item, 'call_id', None)
            )
            tool_call = self._pending_tool_results.pop(call_id, None) if call_id else None
            if tool_call is not None and tool_call.result is None:
                output = (
                    item.get('output') if isinstance(item, dict)
                    else getattr(item, 'output', None)
                )
                tool_call.result = _tool_result_content(output)

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

        # Fallback: recover token usage from AIMessage.usage_metadata when
        # the SDK patches saw nothing (with_raw_response paths, streaming,
        # or providers we don't patch). LangChain populates usage_metadata
        # independently of the SDK call path.
        self._backfill_usage_from_messages(span, messages)

    def _backfill_usage_from_messages(self, span: Span, messages: list) -> None:
        """Populate span.llm_calls token usage from LangChain usage_metadata.

        Only acts when the recorded llm_calls carry no usage at all — if any
        call already has tokens, the SDK patch worked and synthesizing more
        would double-count. When the recorded call count matches the number
        of usage-bearing AI messages, usage is backfilled into the existing
        records in order; when no calls were recorded, new ones are
        synthesized. A mismatched non-empty list is left alone (the exit
        warning will surface it) rather than guessing pairings.
        """
        def _tokens(call, key: str) -> int:
            if isinstance(call, dict):
                return call.get(key, 0) or 0
            return getattr(call, key, 0) or 0

        if any(
            _tokens(c, "tokens_in") or _tokens(c, "tokens_out")
            for c in span.llm_calls
        ):
            return

        usage_msgs = [
            msg for msg in messages
            if getattr(msg, "type", "") == "ai" and getattr(msg, "usage_metadata", None)
        ]
        if not usage_msgs:
            return

        def _call_from(msg) -> LLMCall:
            usage = msg.usage_metadata
            tokens_in = usage.get("input_tokens", 0) or 0
            tokens_out = usage.get("output_tokens", 0) or 0
            meta = getattr(msg, "response_metadata", None) or {}
            model = meta.get("model_name") or meta.get("model") or ""
            provider = meta.get("model_provider") or ""
            return LLMCall(
                model=model,
                provider=provider,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=compute_cost_for_model(model, tokens_in, tokens_out, provider),
            )

        if not span.llm_calls:
            span.llm_calls.extend(_call_from(msg) for msg in usage_msgs)
        elif len(span.llm_calls) == len(usage_msgs):
            for recorded, msg in zip(span.llm_calls, usage_msgs):
                fresh = _call_from(msg)
                if isinstance(recorded, dict):
                    recorded.update(
                        tokens_in=fresh.tokens_in,
                        tokens_out=fresh.tokens_out,
                        cost_usd=fresh.cost_usd,
                    )
                else:
                    recorded.tokens_in = fresh.tokens_in
                    recorded.tokens_out = fresh.tokens_out
                    recorded.cost_usd = fresh.cost_usd
                    if not recorded.model:
                        recorded.model = fresh.model

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
