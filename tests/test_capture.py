"""
Tests for capture.py: TraceContext, attach(), and langgraph_trace().
"""
import pytest
from unittest.mock import patch

from agentci.capture import TraceContext, langgraph_trace
from agentci.models import Trace, Span, LLMCall, SpanKind


class TestTraceContextAttach:
    def test_attach_is_alias_for_attach_langgraph_state(self):
        """TraceContext.attach() delegates to attach_langgraph_state()."""
        with TraceContext(agent_name="test_agent") as ctx:
            called_with = []
            original = ctx.attach_langgraph_state

            def recording_attach(state):
                called_with.append(state)
                return original(state)

            ctx.attach_langgraph_state = recording_attach
            state = {"messages": []}
            ctx.attach(state)
            assert called_with == [state]

    def test_attach_populates_graph_state(self):
        """attach() via attach_langgraph_state sets trace.graph_state."""
        with TraceContext(agent_name="test_agent") as ctx:
            state = {"messages": [], "custom_key": "value"}
            ctx.attach(state)
        assert ctx.trace.graph_state == state


class TestLangGraphTrace:
    def test_returns_context_manager(self):
        """langgraph_trace() is a context manager that yields a TraceContext."""
        with langgraph_trace("my-agent") as ctx:
            assert isinstance(ctx, TraceContext)

    def test_trace_is_accessible_after_context(self):
        """ctx.trace is a Trace object after the context exits."""
        with langgraph_trace("my-agent") as ctx:
            pass
        assert isinstance(ctx.trace, Trace)

    def test_agent_name_is_set(self):
        """langgraph_trace passes agent_name to TraceContext."""
        with langgraph_trace("rag-agent") as ctx:
            pass
        assert ctx.trace.agent_name == "rag-agent"

    def test_empty_agent_name_is_allowed(self):
        """langgraph_trace works with no agent_name argument."""
        with langgraph_trace() as ctx:
            pass
        assert ctx.trace.agent_name == ""

    def test_metrics_computed_after_exit(self):
        """trace.total_cost_usd and total_llm_calls are set after context exits."""
        with langgraph_trace("my-agent") as ctx:
            pass
        # With no real LLM calls, these should be 0 (not None / unset)
        assert ctx.trace.total_cost_usd == 0.0
        assert ctx.trace.total_llm_calls == 0

    def test_attach_inside_context(self):
        """ctx.attach() can be called inside langgraph_trace context."""
        state = {"messages": [], "result": "hello"}
        with langgraph_trace("rag-agent") as ctx:
            ctx.attach(state)
        assert ctx.trace.graph_state == state


# ── Mock helpers for LangGraph-style messages ─────────────────────────────────

class _MockAIMessage:
    """Minimal stand-in for langchain_core.messages.AIMessage."""
    def __init__(self, content: str):
        self.content = content
        self.type = "ai"
        self.tool_calls = []
        self.usage_metadata = None


class _MockHumanMessage:
    def __init__(self, content: str):
        self.content = content
        self.type = "human"


# ── Auto-Extract Final Output Tests ──────────────────────────────────────────


class TestAutoExtractFinalOutput:
    """Tests for TraceContext._auto_extract_final_output()."""

    def test_extract_from_langgraph_state(self):
        """Auto-extracts final_output from LangGraph graph_state messages."""
        with TraceContext(agent_name="test") as ctx:
            state = {
                "messages": [
                    _MockHumanMessage("What is RAG?"),
                    _MockAIMessage("RAG is retrieval-augmented generation."),
                ]
            }
            ctx.attach(state)
        assert ctx.trace.metadata["final_output"] == "RAG is retrieval-augmented generation."

    def test_extract_from_span_output_data_string(self):
        """Auto-extracts from last span's output_data when it's a string."""
        with TraceContext(agent_name="test") as ctx:
            ctx.trace.spans[-1].output_data = "The answer is 42."
        assert ctx.trace.metadata["final_output"] == "The answer is 42."

    def test_extract_from_span_output_data_dict(self):
        """Auto-extracts from output_data dict with common keys."""
        for key in ("content", "message", "text", "output"):
            with TraceContext(agent_name="test") as ctx:
                ctx.trace.spans[-1].output_data = {key: f"value-{key}"}
            assert ctx.trace.metadata["final_output"] == f"value-{key}", (
                f"Failed for dict key '{key}'"
            )

    def test_skips_when_manually_set(self):
        """Does not overwrite a manually set final_output."""
        with TraceContext(agent_name="test") as ctx:
            ctx.trace.metadata["final_output"] = "manual answer"
            ctx.trace.spans[-1].output_data = "auto answer"
        assert ctx.trace.metadata["final_output"] == "manual answer"

    def test_extract_from_llm_call_output_text(self):
        """Falls back to last LLM call's output_text."""
        with TraceContext(agent_name="test") as ctx:
            llm_call = LLMCall(
                model="gpt-4o",
                provider="openai",
                output_text="LLM generated this.",
            )
            ctx.trace.spans[-1].llm_calls.append(llm_call)
        assert ctx.trace.metadata["final_output"] == "LLM generated this."

    def test_extract_from_llm_call_dict(self):
        """Falls back to last LLM call when stored as raw dict (LangGraph adapter).

        Note: We call _auto_extract_final_output() directly because
        compute_metrics() doesn't handle raw dicts in llm_calls.  The
        LangGraph adapter stores dicts; TraceContext.attach() does not.
        """
        ctx = TraceContext(agent_name="test")
        span = Span(kind=SpanKind.AGENT, name="test")
        span.llm_calls.append({"role": "ai", "content": "Dict-based LLM output."})
        ctx.trace.spans.append(span)
        ctx._auto_extract_final_output()
        assert ctx.trace.metadata["final_output"] == "Dict-based LLM output."

    def test_no_extraction_when_trace_empty(self):
        """No final_output set when trace has no spans or data."""
        ctx = TraceContext(agent_name="test")
        ctx.trace = Trace(agent_name="test")  # No spans at all
        ctx._auto_extract_final_output()
        assert "final_output" not in ctx.trace.metadata


class TestNestedTraceContext:
    """Nested TraceContexts (a Trace-returning runner that uses TraceContext
    itself, wrapped again by _run_with_retry) must not stack SDK patches or
    clobber the enclosing context on exit."""

    def test_inner_exit_restores_outer_context(self):
        from agentci.capture import TraceContext, _active_span, _active_trace

        with TraceContext(agent_name="outer") as outer:
            outer_span = _active_span.get()
            with TraceContext(agent_name="inner"):
                assert _active_span.get() is not outer_span
            # Regression: exit used to set(None), losing the outer capture
            assert _active_span.get() is outer_span
            assert _active_trace.get() is outer.trace
        assert _active_span.get() is None
        assert _active_trace.get() is None

    def test_nested_context_does_not_stack_patches(self):
        pytest.importorskip("openai")
        import openai

        from agentci.capture import TraceContext

        original = openai.resources.chat.completions.Completions.create
        with TraceContext(agent_name="outer"):
            patched_once = openai.resources.chat.completions.Completions.create
            assert patched_once is not original
            with TraceContext(agent_name="inner"):
                # Regression: inner enter used to re-patch, wrapping the
                # outer wrapper so one LLM call recorded twice
                assert openai.resources.chat.completions.Completions.create is patched_once
            assert openai.resources.chat.completions.Completions.create is patched_once
        assert openai.resources.chat.completions.Completions.create is original

    def test_patch_depth_recovers_after_exception(self):
        from agentci.capture import TraceContext, _patch_depth

        with pytest.raises(RuntimeError):
            with TraceContext(agent_name="outer"):
                raise RuntimeError("boom")
        assert _patch_depth.get() == 0
