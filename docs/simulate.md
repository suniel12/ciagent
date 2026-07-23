# Simulate — `ciagent simulate`

Single-turn tests catch single-turn bugs. The failures that reach your users
happen three turns into a conversation — the agent forgets what it already
said, a guardrail poisons the transcript, a refund gets promised twice.
`ciagent simulate` drives multi-turn conversation scenarios against your
agent and grades every turn with the same check layers as the rest of
CIAgent.

```bash
ciagent simulate --mock              # validate scenario structure, zero API calls
ciagent simulate                     # run scenarios live (cost estimate + confirm first)
ciagent simulate --runs 5            # stability: run each scenario 5×, attribute flips
ciagent simulate --record            # save each conversation as a golden envelope
ciagent simulate --replay ./golden   # re-drive recorded conversations, diff against golden
```

## Two kinds of scenario

**Scripted** (`turns:` given) — you write the user messages; the conversation
is deterministic and needs no persona LLM. This is the CI path: it runs the
same way every time and works with `--mock` on zero API keys.

**Generative** (`persona:` and/or `goal:`, no `turns:`) — a cheap persona LLM
improvises the user side of the conversation. This is the finder path,
nondeterministic by design: its job is to surface conversations you didn't
think to script. When it finds one, `--record` turns it into a deterministic
regression test (below).

## Spec additions

```yaml
conversation_runner: "myagent.run:respond"   # (messages: list[dict]) -> str | Trace
persona_config: {model: claude-haiku-4-5, temperature: 0.7}   # optional

scenarios:
  - name: refund-flow                        # scripted
    turns:
      - "hi"
      - "i want a refund for order #123"
    max_turns: 8
    per_turn:
      path: {expected_tools: [search_kb]}
    outcome:
      correctness: {any_expected_in_answer: ["refund", "5-7 business days"]}

  - persona: "frustrated customer, discontinued product"   # generative
    goal: "get a refund routed correctly"
    max_turns: 8
    stop_when: {tool_called: process_refund}
    outcome:
      correctness: {any_expected_in_answer: ["refund"]}
```

The `conversation_runner` accepts the full message history
(`[{"role": ..., "content": ...}, ...]`) and returns the agent's reply — a
string, or a `Trace` if you want path/retrieval/cost checks to see tool
calls. Check blocks (`correctness`, `path`, `retrieval`, `cost`) are the
same layers documented in [writing-tests.md](writing-tests.md):

- **`per_turn`** checks run on *every* turn's trace — use them for
  invariants ("the router always calls `search_kb`").
- **`outcome`** checks run *once, at the end of the conversation*, as the
  verdict. They are never a stop condition.

## Termination is deterministic

A conversation ends normally in exactly three ways: scripted turns are
exhausted, `max_turns` is hit, or an explicit `stop_when` event fires (e.g.
`stop_when: {tool_called: process_refund}` — a concrete trace fact, never a
judge opinion or keyword match). The persona never decides to end a
conversation. This matters: if outcome checks or the persona could stop the
session, "the conversation ended" would leak into "the conversation passed,"
and flaky termination would masquerade as agent behavior.

Two abnormal endings also exist: an **agent exception** mid-conversation
(infra-error) and a **`--max-cost` abort**. Both stop the conversation
immediately, skip the outcome verdict (a partial conversation is never
graded), mark the result partial (`is_partial: true`, `outcome: null` in
JSON output), and exit 2. Note that replaying a golden recorded from a
partial conversation can complete — and be graded — where the original
didn't.

## Found bug → regression test, one command each

```bash
ciagent simulate --record            # every conversation saved as a golden envelope
ciagent simulate --replay ./golden   # the suite now gates on it
```

`--record` writes each scenario's conversation — including **failed**
scenarios; that is the point — as a golden envelope under
`<baseline_dir>/<agent>/scenarios/`. `--replay` feeds a recorded envelope's
user turns back to the agent *verbatim*: the persona is never called, only
the agent side can vary, and the run reports a turn-by-turn diff against the
golden. A nondeterministic persona discovery becomes a deterministic CI
gate.

`--record-dir <path>` records somewhere other than the spec's
`baseline_dir` (and implies `--record`). `--replay` accepts a single
envelope `.json` or a directory.

`--record` requires deciding to record up front. To capture failures you
did **not** anticipate, enable [staging](promotion.md): every failing
conversation is auto-staged, and `ciagent promote <id>` turns one into a
golden CI gate after the fact.

## Stability: `--runs N`

Multi-turn sessions have three ways to flip, not two. `--runs N` executes
every scenario N times and attributes each verdict flip:

| Attribution | Meaning | Fix |
|---|---|---|
| `agent-variance` | The agent's output changed across runs | Fix the agent |
| `judge-flake` | Same output, the judge graded it differently | Fix the eval |
| `simulation-variance` | The persona said different things across runs | Persona, not agent — tighten the persona or script the turns |

`simulation-variance` is the one unique to simulate: without it, a chatty
persona would get your agent blamed for flakiness it didn't cause. See
[stability.md](stability.md) for the underlying flip-attribution model.

## Cost controls

Live sessions print a cost estimate (agent turns + persona turns + judged
turns, priced per the spec's models) and ask for confirmation before
running; `--yes` skips the prompt. `--max-cost <usd>` adds a hard ceiling:
when breached, the session aborts mid-conversation and partial results are
reported, clearly marked. `--workers N` caps parallel scenarios (turns
within a conversation always stay sequential).

`--mock` runs the whole thing for free: synthetic conversations in which
each turn satisfies the scenario checks, plus a mock persona for generative
scenarios. Use it to validate scenario structure before spending a cent —
it is also the CI-friendly way to keep scenario specs from rotting.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Every scenario's checks passed |
| 1 | A scenario failed its outcome or a per-turn correctness check |
| 2 | Config error, agent exception, or session aborted on `--max-cost` |

## Output

`--format json` emits the full machine-readable result set for tooling;
the default console report shows per-scenario verdicts, per-turn check
results, and (with `--runs`) the stability table with flip attribution.
