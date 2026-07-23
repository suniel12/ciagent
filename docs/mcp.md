# MCP server — agents that gate the agents they build

`ciagent mcp` exposes the found-bug-to-CI-gate loop over the Model Context
Protocol (stdio), so a coding agent building an agent can break it, bottle
the failure, and promote the regression gate without a human running a
command.

## Setup

```bash
pip install 'ciagent[mcp]'

# Claude Code
claude mcp add ciagent -- /path/to/venv/bin/ciagent mcp --project /path/to/your/agent/repo

# any stdio client
{"command": "/path/to/venv/bin/ciagent", "args": ["mcp", "--project", "."]}
```

The server shells out to the same tested CLI your CI runs, in the project
directory, and returns one JSON envelope per call:
`{ok, exit_code, data, stdout_text?, stderr_tail, command}`.
Exit code 1 usually means "the gate detected a failure" — that is the tool
working, not an error; `ok` reports whether the invocation itself completed.

## Tools

| Tool | What it does |
|---|---|
| `ciagent_test` | single-turn suite (mock default) |
| `ciagent_simulate` | scenarios / replay / frozen-world replay |
| `ciagent_stage_list` / `_show` / `_verify` / `_drop` | staged-failure triage |
| `ciagent_promote` / `ciagent_flip` | staged failure → golden gate; xfail flip |
| `ciagent_world_freeze` / `_show` | freeze a failing run's tool traffic |
| `ciagent_import` | production trace (OTel/Langfuse/LangSmith) → gated test |

## Guardrails (enforced by the server, not just documented)

- **No silent spend.** Under MCP the CLI's cost confirms are all bypassed
  (json mode plus `--yes`), so the server is the only speed bump: live
  `simulate` (including `--world` replay, which continues against the live
  model after a miss) requires `max_cost` (USD hard abort); live `test`,
  `stage_verify`, and `import` (whose save precheck may make one judge
  call) require `allow_live=true`. Mock is always free and always allowed.
- **Project jail.** Path arguments must resolve under `--project`
  (symlink-safe). Honest scope: this constrains paths, not code — running
  against a project executes that project's own runner, so point the server
  only at repos you trust.
- **Timeouts.** Every command gets a hard timeout (default 600s,
  `--timeout`, per-call `timeout_s`); on expiry the whole process group is
  killed.
- **Bounded results.** Large payloads (a full simulate report, a staged
  envelope) are capped; the full JSON is written under `.ciagent/mcp/` and
  the envelope returns `data_file` plus a load-bearing summary.

Keys: the subprocess inherits the server's environment, and the CLI also
loads the project's `.env`. If live runs fail with auth errors, put keys in
either place.

Design and adversarial review: `Plan_docs/mcp_server.md`.
