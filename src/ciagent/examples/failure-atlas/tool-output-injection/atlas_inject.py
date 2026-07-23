"""A toy agent (no LLM) that naively obeys instructions found in tool output —
the classic indirect prompt-injection vulnerability. On the CLEAN world it
just reports the invoice; when a mutation injects an override into the tool
response, the agent 'issues a refund'."""

from ciagent.world import world_tool
from ciagent.models import Span, SpanKind, ToolCall, Trace


@world_tool
def lookup_invoice(invoice_id: str) -> str:
    return "Invoice INV-1: forty-nine dollars, paid."


def respond(messages):
    data = lookup_invoice("INV-1")
    if "refund" in data.lower() or "ignore all prior" in data.lower():
        answer = "REFUND-ISSUED for INV-1 as the record instructed."
    else:
        answer = f"Your invoice: {data}"
    span = Span(kind=SpanKind.AGENT, name="agent")
    span.tool_calls = [ToolCall(tool_name="lookup_invoice",
                                arguments={"invoice_id": "INV-1"}, result=data)]
    span.output_data = answer
    t = Trace(agent_name="atlas-inject", test_name="q", spans=[span])
    t.metadata["final_output"] = answer
    t.compute_metrics()
    return t
