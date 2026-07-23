"""A toy runner (no LLM, no SDK) reproducing the OBSERVABLE failure only: an
input guardrail that greps the ACCUMULATED transcript. Turn 1 contains a card
number and is blocked (correct). Turn 2 is PII-free and SHOULD pass — but the
guardrail still sees turn 1's text in the transcript, so it blocks the clean
turn too. This is the poisoned-transcript symptom from the 50-conversation
persona study; it does NOT reproduce the OpenAI Agents SDK guardrail
internals, only the behavior a transcript-scoped guardrail produces."""

import re

_PII = re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}\b")


def respond(messages):
    from ciagent.models import Span, SpanKind, Trace

    # The bug: check the WHOLE transcript, not just the latest user message.
    transcript = " ".join(
        m.get("content", "") if isinstance(m, dict) else str(m)
        for m in messages
    )
    if _PII.search(transcript):
        answer = "[BLOCKED by input guardrail: PII detected]"
    else:
        answer = "Refunds take 5-7 business days."
    span = Span(kind=SpanKind.AGENT, name="triage")
    span.output_data = answer
    t = Trace(agent_name="atlas-poison", test_name="q", spans=[span])
    t.metadata["final_output"] = answer
    t.compute_metrics()
    return t
