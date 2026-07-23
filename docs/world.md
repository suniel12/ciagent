# Simulated world — `ciagent world` + `simulate --world`

A world is a set of frozen tool fixtures extracted from a recorded failing
run. Replay against it and your agent's tools serve the frozen responses
instead of hitting real backends, so yesterday's repro keeps reproducing even
when the real backend is stateful (the refund already processed), has moved
on, or is gone.

```bash
ciagent simulate --runs 3                 # failing conversation auto-staged
ciagent world freeze <stage-id>           # freeze its tool traffic
ciagent promote <stage-id>                # the repro becomes a golden
ciagent simulate --replay ./golden --world worlds/refund-path.world.json
```

Scope, stated honestly: the world removes **backend** variance from replay.
The model is not frozen, so staying green still depends on the live model
issuing matching tool calls; model-side variance is what stability
attribution and the triage classes already handle.

## Wiring: one decorator per tool

`world_tool` must be the INNERMOST decorator, directly on the plain function:

```python
from agents import function_tool
from ciagent.world import world_tool

@function_tool
@world_tool
def lookup_invoice(customer_email: str) -> str:
    ...
```

With no active world it is a zero-overhead passthrough (normal runs are
byte-identical). During `--world` replay it serves the frozen response for
the offered arguments. Async tools and framework context parameters
(`RunContextWrapper`) are handled. Un-wrapped tools are simply outside the
world and run live.

## Fail-closed: misses, never guesses

A call no fixture matches raises `WorldMiss` and is recorded. A mock that
guesses would be judge-flake wearing a different hat, so the world never
invents a response and never falls through to the real function. Notes:

- Frameworks like openai-agents convert tool exceptions into error strings
  fed back to the model, so after a miss the conversation continues against
  the LIVE model (and spends). The recorded miss list is the authoritative
  signal; the report prints served/miss counts per replay.
- Exit semantics are lifecycle-aware: for `gate` goldens, any recorded miss
  in any run exits 1 (the verdict was not obtained on the frozen world).
  For `xfail` goldens misses never flip the exit code; they suppress XPASS,
  since an xpass on divergent tool traffic proves nothing.
- The miss message shows the nearest fixture's field-level diff and a
  ready-to-paste `"ignore": [...]` suggestion.

## The world file is the authoring surface

```json
{
  "world_schema": 1,
  "tools": {
    "lookup_invoice": {
      "fixtures": [
        {"match": {"customer_email": "redacted-1@example.com"},
         "ignore": [],
         "response": "Invoices for redacted-1@example.com: ..."}
      ],
      "suggested_ignore": ["reason"]
    },
    "process_refund": {
      "sequence": true,
      "fixtures": [
        {"match": {"invoice_id": "INV-1"}, "response": "refund initiated"},
        {"match": {"invoice_id": "INV-1"}, "response": "error: already in progress"}
      ]
    }
  }
}
```

- **`ignore`** marks mutable fields (free text, request ids, timestamps):
  they match any value. Freeze writes `suggested_ignore` hints (fields that
  varied between calls, long free-text strings); suggestions are never
  auto-applied.
- **`sequence: true`** encodes state transitions: matching fixtures are
  consumed in order, so the same call can return different results over
  time. Caution: an extra early call shifts every later response; the `turn`
  field on fixtures says where each came from. Freeze sets `sequence`
  automatically when the same arguments produced different results.
- Matching is tolerant across the framework's validation layer (type
  coercion, defaults filled for omitted optionals) and exact otherwise.
  Loading validates that reusable fixtures stay unambiguous after `ignore`
  edits.
- **`gaps`** records calls frozen without a result (`--allow-gaps`); those
  calls will always miss, and the miss message says so.

## Freeze details

- Sources: a stage id or a golden path. Zero tool calls refuses. Result-less
  calls refuse without `--allow-gaps`. `--tools a,b` filters.
- Redaction: the source envelope is redacted once, envelope-level, before
  extraction, so fixtures inherit consistent placeholders. Freezing an
  UNREDACTED source (pre-0.12 golden) whose values would be scrubbed refuses
  without `--force-redact`, because fixtures would diverge from the golden's
  raw turns and check literals.

## Verify against the world

```bash
ciagent stage verify <id> --world worlds/x.world.json
```

"Does the agent still fail given the frozen backend" — the sharpest verify.
Runs that recorded a world miss are excluded from re-classification (a
divergent run is not a clean agent signal); if every run missed, the staging
block is left untouched and verify exits 1. The block records
`verified_via: replay+world`.

## Mutations: chaos on frozen fixtures

A frozen world captures the backend that happened. `ciagent world mutate`
derives the backends that could have happened, and each derived world flows
through the same replay machinery, so "my agent survives a degraded or
hostile backend" becomes a deterministic CI gate.

```bash
ciagent world operators                       # list operators + payloads
ciagent world mutate w.world.json --op inject --payload-id role-override -o evil.world.json
ciagent simulate --replay ./golden --world evil.world.json
```

Operators: `empty`, `error`, `inject` (adversarial payload into every string
leaf of a tool response), `rewrite` (OLD=NEW over string leaves),
`truncate-sequence`, `swap`. The source world is never modified; the derived
world records `mutated_from` provenance and carries a name suffix.

Two signal channels, stated plainly:

- **Response-changing operators** (`empty`, `error`, `inject`, `rewrite`)
  surface through your scenario's own checks. The agent calls the same tools
  with the same arguments, so there are no world misses; the injected or
  degraded response changes what the agent *does*, and a `not_in_answer` or
  `forbidden_tools` check catches it. This is the injection gate.
- **Designed-miss operators** (`truncate-sequence`, `swap`) make the agent's
  repeat call miss on purpose. Because any gate-lifecycle miss exits 1, these
  are meaningful only under the `xfail` lifecycle (bank the known weakness,
  flip when fixed).

### Prompt injection via tool output

The flagship use: `inject` an override string into a tool's frozen response,
replay, and a money-out `forbidden_tools` gate or a `not_in_answer` check
turns red the moment your agent obeys the tool output. Unlike a robustness
score, the result is a permanent gate: promote it (or `--xfail` it) and every
future change is checked against that exact injection. Built-in payloads are
benign-but-representative; real red-team strings go via `--payload-file` from
your own repo. Payloads are never redacted (a scrubbed payload would silently
neuter the gate).
