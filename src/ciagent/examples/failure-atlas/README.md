# Agent Failure Atlas (seed)

Executable, OWASP-mapped agent failure patterns. Each entry is a runnable
demonstration of a real failure class, using a **toy vulnerable agent** (pure
Python, no LLM, no API key) so you can see the gate fire in seconds, then
adapt the pattern to your own agent.

| slug | OWASP LLM | Failure |
|---|---|---|
| `money-out-no-verification` | LLM06 Excessive Agency | agent takes an irreversible money-out action without the verification the data demands |
| `transcript-poisoning` | Guardrail design (LLM01-adjacent) | a tripped input guardrail poisons the transcript; later clean turns stay blocked |
| `tool-output-injection` | LLM01 Prompt Injection | an adversarial instruction in a tool result redirects the agent |

## Run one

```bash
cd money-out-no-verification && ciagent simulate --yes    # gate fires (exit 1)
cd transcript-poisoning     && ciagent simulate --yes     # gate fires (exit 1)
cd tool-output-injection    && ./run.sh                   # multi-step recipe; final replay exits 1
```

Each entry's toy agent is DESIGNED to fail its scenario's deterministic
check, so a passing exit code would mean the demonstration itself broke.

## What's portable, and what isn't

The **scenario and the check** are the reusable artifact: copy them, point the
`conversation_runner` at your own agent, and you have a gate for that failure
class. The bundled vulnerable agent is a teaching prop, not your agent. For
`tool-output-injection`, the frozen world is agent-specific — you regenerate it
by freezing a run of YOUR agent (`ciagent world freeze`) and mutating it
(`ciagent world mutate --op inject`); the injection *recipe* is what transfers,
not the world file.

This is a seed set, not a community registry. It grows as real failures accrue
(two of these three come from real dogfood findings). Contributions may open
later once the pattern's portability is proven.
