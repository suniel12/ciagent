# Import — `ciagent import`

That production failure from last Tuesday becomes a CI test.

`ciagent import` reads an exported production trace and turns it into a
CIAgent golden baseline plus a minimal spec query. You gate on it like any
other golden — so the next regression that would have reproduced the failure
fails your build instead of your users.

```bash
ciagent import trace.json                       # map + gate + write
ciagent import trace.json --dry-run             # gate only; write nothing
ciagent import trace.json --version incident-42 # name the baseline
ciagent import trace.json -c my_spec.yaml       # non-default spec path
```

## Formats (auto-detected)

You do not tell `import` the format; it sniffs the file.

| Emitter | File shape | Reported as |
|---------|-----------|-------------|
| OpenTelemetry GenAI instrumentation (openllmetry, and any GenAI-semconv exporter) | OTLP/JSON envelope, `{"spans": [...]}`, or a flat span list | `otel-genai` |
| Langfuse (v3+ SDK export) | OTel spans in the `langfuse.*` attribute dialect | `otel-langfuse` |
| Google ADK (`google-adk` native OTel) | OTel spans in the `gcp.vertex.agent.*` dialect (llm_request/llm_response, tool_call_args/tool_response) | `otel-adk` |
| LangSmith (`langsmith` run export / `Client.list_runs`) | JSON or JSONL run objects, flat or nested `RunTree` | `langsmith-runs` |

Each dialect is verified against a real export from that tool, not against a
hand-written fixture — "it speaks OTel" is not the same as "its attribute
namespace matches," as Langfuse and Google ADK both proved. Verified against real
captured tool-use traces (query, answer, and tool call **with its result** all
survive import): the **OpenAI** and **Anthropic** providers under openllmetry, a
full **CrewAI** crew (agent + tool + task) whose LLM calls openllmetry traces
through litellm, a **Google ADK** agent via ADK's native OTel export, and a
**Claude Agent SDK** `query()` run instrumented by
`otel-instrumentation-claude-agent-sdk`.
(CrewAI's imported query is its full constructed task prompt — that is the last
user message CrewAI actually sends. The Claude Agent SDK emits one
session-level `invoke_agent` span carrying the messages plus bare
`execute_tool` child spans; the importer merges the two views by tool call
id, so each tool call imports once, with its arguments and result.)

Because these are all OTel (or OTel-derived) paths, any framework whose runs
emit GenAI-semconv spans imports through `otel-genai` without a bespoke
adapter — the runner and importer are framework-agnostic.

## The round-trip gate (always on)

Import will not plant a golden that can never pass. Before anything is
written, the mapped trace must produce a golden that **loads and evaluates
cleanly**. Partial traces — no user input, no final output, no spans — are
rejected with the missing fields named:

```
Rejected by the round-trip gate — nothing written:
  • no user input found in the trace
```

A golden that can never pass is a permanent false regression in your CI; this
command refuses to plant one. This is the same fail-closed philosophy as the
artifact gate everywhere else in CIAgent — a missing signal is a rejection,
never a silent guess.

## What gets written

- **Spec query.** If the trace's query text is new, a minimal query tagged
  `imported` is appended to your spec. Existing queries are **never
  modified**. The spec is backed up to `<spec>.yaml.bak` first.
- **Golden baseline.** A versioned baseline at
  `<baseline_dir>/<agent>/<version>.json`, carrying the recorded tool-call
  sequence and each `ToolCall.result` — the tool state a later replay needs
  to reproduce the failure, not just the final answer.

The default version tag is `imported-<n>` (auto-incrementing); override it
with `--version`.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Imported (or `--dry-run` gate passed) |
| 1 | Trace rejected by the round-trip gate |
| 2 | File or config error |

Exit 1 vs 2 is a real distinction: 1 means the file was a readable trace but
too partial to gate on; 2 means the file couldn't be read as any supported
export, or the spec couldn't be loaded.

## After import

The imported query starts with no assertions — it will pass on any answer
until you add checks. Two ways to give it teeth:

```bash
ciagent generate-checks     # mine deterministic checks from your KB
ciagent test                # gate the suite, imported query included
```

Then the failure you imported is a test that stays failed until the
underlying regression is fixed.
