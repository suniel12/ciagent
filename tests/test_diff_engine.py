import pytest
from ciagent.models import Trace, Span, ToolCall, SpanKind, DiffType
from ciagent.diff_engine import diff_traces

def create_trace(
    tool_calls: list[tuple[str, dict]] = None,
    cost: float = 0.0,
    steps: int = 0
) -> Trace:
    """Helper to create a trace with specific tool calls and metrics."""
    t = Trace()
    t.total_cost_usd = cost
    t.total_llm_calls = steps
    
    if tool_calls:
        # Create a span to hold tool calls
        span = Span(kind=SpanKind.AGENT, name="test_agent")
        for name, args in tool_calls:
            span.tool_calls.append(ToolCall(tool_name=name, arguments=args))
        t.spans.append(span)
        t.total_tool_calls = len(tool_calls)
        
    return t

def test_no_diffs():
    """Identical traces should return empty diff list."""
    t1 = create_trace([("search", {"q": "foo"})], cost=0.01, steps=1)
    t2 = create_trace([("search", {"q": "foo"})], cost=0.01, steps=1)
    
    diffs = diff_traces(t2, t1)
    assert len(diffs) == 0

def test_tools_changed():
    """Detect added/removed tools."""
    golden = create_trace([("search", {}), ("book", {})])
    current = create_trace([("search", {}), ("cancel", {})])
    
    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    d = diffs[0]
    assert d.diff_type == DiffType.TOOLS_CHANGED
    assert d.severity == "error"
    assert "cancel" in d.details["added"]
    assert "book" in d.details["removed"]

def test_sequence_changed():
    """Detect same tools but different order."""
    golden = create_trace([("search", {}), ("book", {})])
    current = create_trace([("book", {}), ("search", {})])
    
    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    assert diffs[0].diff_type == DiffType.SEQUENCE_CHANGED
    assert diffs[0].severity == "warning"

def test_args_changed():
    """Detect changed arguments."""
    golden = create_trace([("search", {"q": "SFO"})])
    current = create_trace([("search", {"q": "JFK"})])
    
    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    d = diffs[0]
    assert d.diff_type == DiffType.ARGS_CHANGED
    assert d.details["tool"] == "search"
    assert d.details["changes"][0]["field"] == "q"
    assert d.details["changes"][0]["golden"] == "SFO"
    assert d.details["changes"][0]["current"] == "JFK"

def test_cost_spike():
    """Detect significant cost increase."""
    golden = create_trace(cost=0.10)
    current = create_trace(cost=0.50)  # 5x increase
    
    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    assert diffs[0].diff_type == DiffType.COST_SPIKE
    assert diffs[0].severity == "error"

def test_steps_increase():
    """Detect step count increase."""
    golden = create_trace(steps=5)
    current = create_trace(steps=10)  # 2x increase

    diffs = diff_traces(current, golden)
    assert len(diffs) == 1
    assert diffs[0].diff_type == DiffType.STEPS_CHANGED


# ── Routing, guardrail, and handoff tests ─────────────


def create_trace_with_handoff(from_agent: str, to_agent: str, **kwargs) -> Trace:
    """Helper to create a trace with a handoff span."""
    t = create_trace(**kwargs)
    t.spans.append(Span(
        kind=SpanKind.HANDOFF,
        name="handoff",
        from_agent=from_agent,
        to_agent=to_agent,
    ))
    return t


def test_routing_changed():
    """Detect when handoff target changes between runs."""
    golden = create_trace_with_handoff("Triage Agent", "Billing Agent")
    current = create_trace_with_handoff("Triage Agent", "Technical Agent")

    diffs = diff_traces(current, golden)
    routing_diffs = [d for d in diffs if d.diff_type == DiffType.ROUTING_CHANGED]
    assert len(routing_diffs) == 1
    assert routing_diffs[0].severity == "error"
    assert "Billing Agent" in routing_diffs[0].message
    assert "Technical Agent" in routing_diffs[0].message


def test_routing_unchanged():
    """Identical handoff targets should not produce a routing diff."""
    golden = create_trace_with_handoff("Triage Agent", "Billing Agent")
    current = create_trace_with_handoff("Triage Agent", "Billing Agent")

    diffs = diff_traces(current, golden)
    routing_diffs = [d for d in diffs if d.diff_type == DiffType.ROUTING_CHANGED]
    assert len(routing_diffs) == 0


def test_stop_reason_changed():
    """Detect when the agent's stop reason changes (e.g. complete -> max_tokens)."""
    golden = Trace()
    golden.spans.append(Span(kind=SpanKind.AGENT, name="agent", stop_reason="complete"))
    current = Trace()
    current.spans.append(Span(kind=SpanKind.AGENT, name="agent", stop_reason="max_tokens"))

    diffs = diff_traces(current, golden)
    stop_diffs = [d for d in diffs if d.diff_type == DiffType.STOP_REASON_CHANGED]
    assert len(stop_diffs) == 1
    assert stop_diffs[0].severity == "error"
    assert "complete" in stop_diffs[0].message
    assert "max_tokens" in stop_diffs[0].message


def test_guardrails_changed():
    """Detect when different guardrails fire between runs."""
    golden = Trace()
    golden.spans.append(Span(
        kind=SpanKind.GUARDRAIL, name="pii_check",
        guardrail_name="pii_check", guardrail_triggered=False,
    ))

    current = Trace()
    current.spans.append(Span(
        kind=SpanKind.GUARDRAIL, name="pii_check",
        guardrail_name="pii_check", guardrail_triggered=True,
    ))

    diffs = diff_traces(current, golden)
    guard_diffs = [d for d in diffs if d.diff_type == DiffType.GUARDRAILS_CHANGED]
    assert len(guard_diffs) == 1
    assert guard_diffs[0].severity == "error"
    assert "pii_check" in guard_diffs[0].message


def test_guardrails_unchanged():
    """Same guardrail state should not produce a diff."""
    golden = Trace()
    golden.spans.append(Span(
        kind=SpanKind.GUARDRAIL, name="pii_check",
        guardrail_name="pii_check", guardrail_triggered=False,
    ))
    current = Trace()
    current.spans.append(Span(
        kind=SpanKind.GUARDRAIL, name="pii_check",
        guardrail_name="pii_check", guardrail_triggered=False,
    ))

    diffs = diff_traces(current, golden)
    guard_diffs = [d for d in diffs if d.diff_type == DiffType.GUARDRAILS_CHANGED]
    assert len(guard_diffs) == 0


def test_available_handoffs_changed():
    """Detect when available handoff options change even if same path taken."""
    golden = Trace()
    golden.spans.append(Span(
        kind=SpanKind.AGENT, name="Triage Agent",
        metadata={"handoffs": ["Billing Agent", "Technical Agent", "Account Agent"]},
    ))
    golden.spans.append(Span(
        kind=SpanKind.HANDOFF, name="handoff",
        from_agent="Triage Agent", to_agent="Billing Agent",
    ))

    current = Trace()
    current.spans.append(Span(
        kind=SpanKind.AGENT, name="Triage Agent",
        metadata={"handoffs": ["Billing Agent", "Technical Agent"]},
    ))
    current.spans.append(Span(
        kind=SpanKind.HANDOFF, name="handoff",
        from_agent="Triage Agent", to_agent="Billing Agent",
    ))

    diffs = diff_traces(current, golden)
    # Routing is the same, but available options changed
    routing_diffs = [d for d in diffs if d.diff_type == DiffType.ROUTING_CHANGED]
    assert len(routing_diffs) == 0

    avail_diffs = [d for d in diffs if d.diff_type == DiffType.AVAILABLE_HANDOFFS_CHANGED]
    assert len(avail_diffs) == 1
    assert avail_diffs[0].severity == "warning"
