# transcript-poisoning — guardrail design (LLM01-adjacent)

**Failure:** an input guardrail that checks the ACCUMULATED transcript instead
of just the newest message. Turn 1 contains a card number and is blocked
(correct). Turn 2 is PII-free and should pass — but the guardrail still sees
turn 1's text, so the clean turn stays blocked. The conversation is
unrecoverable.

**Gate:** `per_turn.correctness.not_in_answer: ["[BLOCKED"]` — no turn's answer
may contain a block marker. The clean second turn returns one, firing the gate
(exit 1).

**Run:** `ciagent simulate --yes`

**Real source:** CIAgent's 50-conversation persona study, where 39/50 refund
conversations died mid-flow after one tripped guardrail poisoned the
transcript.

**Honest scope:** the toy runner reproduces the OBSERVABLE failure (a clean
turn still returns blocked), NOT the OpenAI Agents SDK guardrail internals the
original finding used. The lesson transfers: scope input guardrails to the new
message, not the whole transcript.
