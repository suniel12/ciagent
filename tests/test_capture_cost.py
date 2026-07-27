"""
Regression tests for silent-zero token/cost capture.

Background: langchain-openai calls client.with_raw_response.create(), which
binds the resource method at wrapper-construction time, so the old
Completions.create monkeypatch never fired and every LLM call recorded zero
tokens and zero cost. These tests pin the transport-level patch, the
LangGraph usage_metadata fallback, the metric rollups, and the loud warning
that surfaces any remaining silent-zero traces.
"""
import warnings

import pytest

from ciagent.capture import TraceContext, CostCaptureWarning
from ciagent.cost import compute_cost, compute_cost_for_model, infer_provider
from ciagent.models import LLMCall, Span, SpanKind, Trace

# Captured at collection time, before examples/openai_agent/agent.py (imported
# by test_real_agents.py) replaces Completions.create at ITS import time. The
# transport fixture below pins this pristine method back so those module-level
# mocks can't leak into these tests.
try:
    import openai as _openai_mod
    _PRISTINE_COMPLETIONS_CREATE = (
        _openai_mod.resources.chat.completions.Completions.create
    )
except ImportError:
    _PRISTINE_COMPLETIONS_CREATE = None


# ── Fake OpenAI wire objects ─────────────────────────────────────────────────

class _FakeUsage:
    def __init__(self, tokens_in=120, tokens_out=30):
        self.prompt_tokens = tokens_in
        self.completion_tokens = tokens_out
        self.total_tokens = tokens_in + tokens_out


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content="hello", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeChatCompletion:
    def __init__(self, model="gpt-4o", tool_calls=None, tokens_in=120, tokens_out=30):
        self.model = model
        self.usage = _FakeUsage(tokens_in, tokens_out)
        self.choices = [_FakeChoice(_FakeMessage(tool_calls=tool_calls))]


class _FakeRawResponse:
    """Stand-in for openai's LegacyAPIResponse: parse() yields the model,
    and there is deliberately no .usage attribute on the wrapper itself."""

    def __init__(self, parsed):
        self._parsed = parsed
        self.parse_calls = 0

    def parse(self):
        self.parse_calls += 1
        return self._parsed


def _raw_header_name():
    try:
        from openai._constants import RAW_RESPONSE_HEADER
        return RAW_RESPONSE_HEADER
    except ImportError:
        return "X-Stainless-Raw-Response"


@pytest.fixture
def fake_openai_transport(monkeypatch):
    """Replace the openai HTTP transport with a fake before TraceContext
    patches it, so client calls run the real SDK plumbing but hit no network.

    Returns a dict the test can use to control responses and inspect calls.
    """
    openai = pytest.importorskip("openai")
    from openai._base_client import SyncAPIClient

    state = {"requests": [], "response_factory": lambda options: _FakeChatCompletion()}

    def fake_request(self, cast_to, options, *args, **kwargs):
        state["requests"].append(options)
        parsed = state["response_factory"](options)
        headers = getattr(options, "headers", None) or {}
        if isinstance(headers, dict) and headers.get(_raw_header_name()) == "true":
            return _FakeRawResponse(parsed)
        return parsed

    monkeypatch.setattr(SyncAPIClient, "request", fake_request)
    if _PRISTINE_COMPLETIONS_CREATE is not None:
        monkeypatch.setattr(
            openai.resources.chat.completions.Completions,
            "create",
            _PRISTINE_COMPLETIONS_CREATE,
        )
    state["client"] = openai.OpenAI(api_key="test-key-not-real")
    return state


# ── The langchain-openai regression: with_raw_response.create ────────────────

class TestWithRawResponseCapture:
    def test_raw_response_create_records_tokens_and_cost(self, fake_openai_transport):
        """A with_raw_response wrapper built BEFORE tracing (langchain-openai
        caches it at client setup) must still produce non-zero tokens/cost."""
        client = fake_openai_transport["client"]
        raw = client.chat.completions.with_raw_response  # cached pre-patch

        with TraceContext(agent_name="lc-agent") as ctx:
            response = raw.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )
            response.parse()  # what langchain-openai does with the result

        span = ctx.trace.spans[0]
        assert len(span.llm_calls) == 1
        call = span.llm_calls[0]
        assert call.tokens_in == 120
        assert call.tokens_out == 30
        assert call.cost_usd == compute_cost("openai", "gpt-4o", 120, 30)
        assert call.cost_usd > 0
        assert ctx.trace.total_tokens == 150
        assert ctx.trace.total_cost_usd > 0

    def test_raw_wrapper_built_inside_context_records_once(self, fake_openai_transport):
        """A wrapper built AFTER patching binds the patched create; the call
        must be recorded exactly once (not zero, not twice)."""
        client = fake_openai_transport["client"]

        with TraceContext(agent_name="lc-agent") as ctx:
            raw = client.chat.completions.with_raw_response
            raw.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        span = ctx.trace.spans[0]
        assert len(span.llm_calls) == 1
        assert span.llm_calls[0].tokens_in == 120
        assert span.llm_calls[0].cost_usd > 0

    def test_direct_create_not_double_recorded(self, fake_openai_transport):
        """client.chat.completions.create funnels through both the resource
        patch and the transport patch; exactly one LLMCall must result."""
        client = fake_openai_transport["client"]

        with TraceContext(agent_name="direct") as ctx:
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        span = ctx.trace.spans[0]
        assert len(span.llm_calls) == 1
        assert span.llm_calls[0].tokens_in == 120
        assert span.llm_calls[0].cost_usd > 0

    def test_transport_captures_tool_calls_and_backfills_results(self, fake_openai_transport):
        """Tool calls in a raw-path response are captured, and the next
        request's role=tool messages backfill their results."""
        client = fake_openai_transport["client"]
        raw = client.chat.completions.with_raw_response
        fake_openai_transport["response_factory"] = lambda options: _FakeChatCompletion(
            tool_calls=[_FakeToolCall("call_1", "search", '{"q": "cats"}')]
        )

        with TraceContext(agent_name="tools") as ctx:
            raw.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
            fake_openai_transport["response_factory"] = lambda options: _FakeChatCompletion()
            raw.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "cat facts"},
                ],
            )

        span = ctx.trace.spans[0]
        assert [tc.tool_name for tc in span.tool_calls] == ["search"]
        assert span.tool_calls[0].arguments == {"q": "cats"}
        assert span.tool_calls[0].result == "cat facts"
        assert len(span.llm_calls) == 2

    def test_non_llm_endpoints_not_recorded(self, fake_openai_transport):
        """Calls to non-chat endpoints (e.g. embeddings) pass through the
        transport patch without producing LLMCall records."""
        client = fake_openai_transport["client"]

        class _FakeEmbedding:
            data = []
            model = "text-embedding-3-small"

        fake_openai_transport["response_factory"] = lambda options: _FakeEmbedding()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CostCaptureWarning)
            with TraceContext(agent_name="embed") as ctx:
                client.embeddings.create(
                    model="text-embedding-3-small", input="hello"
                )

        assert ctx.trace.spans[0].llm_calls == []


class _FakeResponsesUsage:
    def __init__(self, tokens_in=200, tokens_out=40):
        self.input_tokens = tokens_in
        self.output_tokens = tokens_out
        self.total_tokens = tokens_in + tokens_out


class _FakeFunctionCallItem:
    type = "function_call"

    def __init__(self, call_id, name, arguments):
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class _FakeResponsesResponse:
    def __init__(self, model="gpt-4o", output=None):
        self.model = model
        self.usage = _FakeResponsesUsage()
        self.output = output or []


class TestResponsesAPICapture:
    def test_responses_create_records_tokens_and_tool_calls(self, fake_openai_transport):
        """The Responses API path (used by langchain-openai for newer models)
        records usage and function calls, and function_call_output input
        items backfill tool results."""
        client = fake_openai_transport["client"]
        if not hasattr(client, "responses"):
            pytest.skip("openai SDK too old for Responses API")
        raw = client.responses.with_raw_response
        fake_openai_transport["response_factory"] = lambda options: _FakeResponsesResponse(
            output=[_FakeFunctionCallItem("call_9", "lookup", '{"id": 7}')]
        )

        with TraceContext(agent_name="responses") as ctx:
            raw.create(model="gpt-4o", input="hi")
            fake_openai_transport["response_factory"] = (
                lambda options: _FakeResponsesResponse()
            )
            raw.create(
                model="gpt-4o",
                input=[
                    {"type": "function_call_output", "call_id": "call_9", "output": "found"},
                ],
            )

        span = ctx.trace.spans[0]
        assert len(span.llm_calls) == 2
        assert span.llm_calls[0].tokens_in == 200
        assert span.llm_calls[0].tokens_out == 40
        assert span.llm_calls[0].cost_usd > 0
        assert [tc.tool_name for tc in span.tool_calls] == ["lookup"]
        assert span.tool_calls[0].arguments == {"id": 7}
        assert span.tool_calls[0].result == "found"


# ── LangGraph usage_metadata fallback ────────────────────────────────────────

class _MockAIMessage:
    type = "ai"

    def __init__(self, content="", usage=None, model="gpt-4o-mini", provider="openai"):
        self.content = content
        self.tool_calls = []
        self.usage_metadata = usage
        self.response_metadata = {"model_name": model, "model_provider": provider}


class _MockHumanMessage:
    type = "human"

    def __init__(self, content):
        self.content = content


class TestLangGraphUsageFallback:
    def test_attach_synthesizes_llm_calls_from_usage_metadata(self):
        """When no SDK patch fired (the langchain-openai case), attach()
        recovers tokens and cost from AIMessage.usage_metadata."""
        state = {
            "messages": [
                _MockHumanMessage("q"),
                _MockAIMessage(
                    "answer",
                    usage={"input_tokens": 500, "output_tokens": 50, "total_tokens": 550},
                ),
            ]
        }
        with TraceContext(agent_name="rag") as ctx:
            ctx.attach(state)

        span = ctx.trace.spans[0]
        assert len(span.llm_calls) == 1
        assert span.llm_calls[0].tokens_in == 500
        assert span.llm_calls[0].tokens_out == 50
        expected_cost = compute_cost("openai", "gpt-4o-mini", 500, 50)
        assert span.llm_calls[0].cost_usd == expected_cost
        assert expected_cost > 0
        assert ctx.trace.total_tokens == 550
        assert ctx.trace.total_tokens_in == 500
        assert ctx.trace.total_tokens_out == 50
        assert ctx.trace.total_cost_usd == expected_cost

    def test_attach_backfills_zero_usage_records_in_order(self):
        """SDK-recorded calls with zero usage get token counts backfilled
        when they pair 1:1 with usage-bearing AI messages."""
        state = {
            "messages": [
                _MockAIMessage("a", usage={"input_tokens": 10, "output_tokens": 1}),
                _MockAIMessage("b", usage={"input_tokens": 20, "output_tokens": 2}),
            ]
        }
        with TraceContext(agent_name="rag") as ctx:
            span = ctx.trace.spans[0]
            span.llm_calls.append(LLMCall(model="gpt-4o", provider="openai"))
            span.llm_calls.append(LLMCall(model="gpt-4o", provider="openai"))
            ctx.attach(state)

        assert [c.tokens_in for c in span.llm_calls] == [10, 20]
        assert [c.tokens_out for c in span.llm_calls] == [1, 2]
        assert span.llm_calls[0].model == "gpt-4o"  # existing model kept
        assert ctx.trace.total_tokens == 33

    def test_attach_does_not_double_count_when_patch_worked(self):
        """If recorded llm_calls already carry usage, the fallback must not
        synthesize duplicates."""
        state = {
            "messages": [
                _MockAIMessage("a", usage={"input_tokens": 10, "output_tokens": 1}),
            ]
        }
        with TraceContext(agent_name="rag") as ctx:
            span = ctx.trace.spans[0]
            span.llm_calls.append(
                LLMCall(model="gpt-4o", provider="openai", tokens_in=10, tokens_out=1)
            )
            ctx.attach(state)

        assert len(span.llm_calls) == 1
        assert ctx.trace.total_tokens == 11


# ── LangGraph adapter: parse_state usage extraction ──────────────────────────

class TestLangGraphAdapterUsage:
    def test_parse_state_populates_tokens_and_cost(self):
        from ciagent.adapters.langgraph import LangGraphAdapter

        state = {
            "messages": [
                _MockHumanMessage("q"),
                _MockAIMessage(
                    "answer",
                    usage={"input_tokens": 1000, "output_tokens": 100},
                ),
            ]
        }
        trace = LangGraphAdapter().parse_state(state)
        trace.compute_metrics()

        span = trace.spans[0]
        assert len(span.llm_calls) == 1
        call = span.llm_calls[0]
        assert call.tokens_in == 1000
        assert call.tokens_out == 100
        assert call.model == "gpt-4o-mini"
        assert call.output_text == "answer"
        assert call.cost_usd > 0
        assert trace.total_tokens == 1100
        assert trace.total_cost_usd > 0

    def test_parse_state_keeps_content_only_messages(self):
        """AI messages without usage_metadata still appear as LLM calls (the
        pre-fix behavior logged their content)."""
        from ciagent.adapters.langgraph import LangGraphAdapter

        message = _MockAIMessage("plain answer", usage=None)
        trace = LangGraphAdapter().parse_state({"messages": [message]})

        span = trace.spans[0]
        assert len(span.llm_calls) == 1
        assert span.llm_calls[0].output_text == "plain answer"


# ── Metric rollups ───────────────────────────────────────────────────────────

class TestMetricRollups:
    def test_span_compute_metrics_tolerates_dict_entries(self):
        span = Span(kind=SpanKind.AGENT, name="s")
        span.llm_calls.append({"role": "ai", "content": "x"})  # legacy stub
        span.llm_calls.append(LLMCall(tokens_in=5, tokens_out=2, cost_usd=0.01))
        span.compute_metrics()
        assert span.total_tokens_in == 5
        assert span.total_tokens_out == 2
        assert span.total_cost_usd == 0.01

    def test_span_compute_metrics_preserves_prepopulated_totals(self):
        """Adapter-populated span totals must not be zeroed when llm_calls
        carry no usage."""
        span = Span(kind=SpanKind.AGENT, name="s")
        span.total_tokens_in = 177
        span.total_tokens_out = 159
        span.total_cost_usd = 0.000122
        span.llm_calls.append({"role": "ai", "content": "x"})
        span.compute_metrics()
        assert span.total_tokens_in == 177
        assert span.total_tokens_out == 159
        assert span.total_cost_usd == 0.000122

    def test_trace_level_token_totals_serialize_as_ints(self):
        """Regression: trace-level total_tokens_in/out used to be missing
        from the serialized trace (read back as null)."""
        trace = Trace()
        span = Span(kind=SpanKind.AGENT, name="s")
        span.llm_calls.append(LLMCall(tokens_in=100, tokens_out=20, cost_usd=0.5))
        trace.spans.append(span)
        trace.compute_metrics()

        dumped = trace.model_dump()
        assert dumped["total_tokens_in"] == 100
        assert dumped["total_tokens_out"] == 20
        assert dumped["total_tokens"] == 120
        assert dumped["total_cost_usd"] == 0.5

        reloaded = Trace.model_validate(dumped)
        assert reloaded.total_tokens_in == 100
        assert reloaded.total_tokens_out == 20

    def test_old_baselines_without_token_split_still_load(self):
        """Golden traces recorded before the fields existed must validate."""
        dumped = Trace().model_dump()
        dumped.pop("total_tokens_in")
        dumped.pop("total_tokens_out")
        reloaded = Trace.model_validate(dumped)
        assert reloaded.total_tokens_in == 0
        assert reloaded.total_tokens_out == 0


# ── Silent-zero warning ──────────────────────────────────────────────────────

class TestCostCaptureWarning:
    def test_warns_on_llm_calls_with_zero_usage(self):
        with pytest.warns(CostCaptureWarning, match="zero tokens and zero cost"):
            with TraceContext(agent_name="silent") as ctx:
                ctx.trace.spans[0].llm_calls.append(LLMCall(model="gpt-4o"))

    def test_no_warning_when_usage_captured(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", CostCaptureWarning)
            with TraceContext(agent_name="ok") as ctx:
                ctx.trace.spans[0].llm_calls.append(
                    LLMCall(model="gpt-4o", tokens_in=10, tokens_out=5, cost_usd=0.01)
                )

    def test_no_warning_when_no_llm_calls(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", CostCaptureWarning)
            with TraceContext(agent_name="idle"):
                pass


# ── Cost helper functions ────────────────────────────────────────────────────

class TestCostHelpers:
    def test_infer_provider(self):
        assert infer_provider("gpt-4o") == "openai"
        assert infer_provider("claude-sonnet-4-20250514") == "anthropic"
        assert infer_provider("o3-mini") == "openai"
        assert infer_provider("mystery-model") == ""
        assert infer_provider("") == ""

    def test_compute_cost_for_model_with_known_provider(self):
        assert compute_cost_for_model("gpt-4o", 1000, 100, "openai") == compute_cost(
            "openai", "gpt-4o", 1000, 100
        )

    def test_compute_cost_for_model_infers_provider(self):
        assert compute_cost_for_model(
            "claude-sonnet-4-20250514", 1000, 100
        ) == compute_cost("anthropic", "claude-sonnet-4-20250514", 1000, 100)

    def test_compute_cost_for_model_unknown_model_is_zero(self):
        assert compute_cost_for_model("mystery-model", 1000, 100) == 0.0
