---
name: check
description: Run CIAgent regression checks after changing an AI agent's code, prompts, or knowledge base in a repo that has ciagent_spec.yaml, and interpret the results. Covers stability gating with --flaky-sources, staged-failure triage and promotion, and frozen-world replay. Use after editing agent logic, before committing agent changes, or when the user asks whether the agent still works.
allowed-tools: Bash(ciagent *)
---

# Run CIAgent checks on this repo's agent

The repo has `ciagent_spec.yaml` (if it does not, use the `onboard` skill
instead). Your job: run the right check for the change that was just made,
read the result correctly, and never paper over a failure.

## Version skew check

This skill text is written for ciagent 0.16.0. Run `ciagent --version` first.
If it prints a different version, the plugin and the CLI are out of sync and
the commands below may not match the installed CLI. Stop and tell the user to
update the plugin (`/plugin marketplace update ciagent`, then
`/plugin update ciagent@ciagent`) and/or the CLI (`pip install -U ciagent`)
before continuing.

## Which command

| Situation | Command |
|---|---|
| Spec or wiring changed, or no API keys | `ciagent test --mock` |
| Agent code / prompt / retrieval changed | `ciagent test --yes --format json` |
| Result differs from last run, or flakiness suspected | `ciagent test --runs 3 --yes` |
| Gating CI on agent-caused flips only | `ciagent test --runs 3 --flaky-sources=agent --yes` |
| Knowledge base changed | `ciagent generate-checks --dry-run`, review, then apply |
| The LLM judge's verdicts look wrong | `ciagent judge-audit` |
| A failing run was staged and needs triage | `ciagent stage list`, then verify (below) |
| Reproduce a failure without live backends | `ciagent world freeze <stage-id>`, then replay (below) |
| Probe behavior on a degraded or hostile backend | `ciagent world mutate <world> --op <operator>`, then replay |

Live runs (`test` without `--mock`, `judge-audit`, `generate-checks`, and
`stage verify` without `--mock`) call model APIs on the user's keys. Mock mode
is free. If the user has not already approved live runs in this session,
prefer `--mock` or ask.

## Reading results

Exit codes: **0** pass (including flaky-but-passing, unless gated with
`--fail-on-flaky` or `--flaky-sources`), **1** correctness failure (with
`--runs N`: failed in every run, or a gated flip source flipped), **2** infra
or config error: fix the setup, not the agent.

With `--format json`: per-query entries carry layer results (correctness /
path / cost) and the answer text; with `--runs N` a top-level `stability`
block lists flipped queries with `flip_source`, plus suite-level
`flip_sources` counts and `gated_by`.

## Flip sources route the work

- `agent-variance`: the agent's answer changed between runs. Fix the agent
  (prompt, retrieval, temperature).
- `retrieval-variance`: same tool sequence, but the retrieved set changed.
  Fix the retriever, not the prompt.
- `judge-flake`: same answer, the LLM judge changed its verdict. Fix the
  eval (tighten the rubric or replace with a deterministic check).
- `infra-error`: a judge API call failed. Retry; fix nothing.
- `simulation-variance`: the simulated user said different things across
  runs. A persona problem, not an agent problem.
- `world-miss`: a frozen-world replay diverged from the world file. Not a
  clean agent signal; refreeze or fix the world.
- `mixed`: ambiguous; look at the answers yourself.

Attribution is an action, not just a report: `--flaky-sources` makes `test`
exit 1 only when a flip came from the sources you name. The `agent` (or
`real`) alias covers agent-variance plus retrieval-variance, exactly the
flips that mean "fix the agent system"; `judge`, `infra`, and `sim` are the
other aliases, and full source names are accepted. So
`ciagent test --runs 3 --flaky-sources=agent --yes` is the CI gate that fails
on real agent regressions while tolerating judge flake. Bare
`--fail-on-flaky` still gates on any flip.

## Staged failures: triage and promote

Failing live runs auto-stage under `.ciagent/staged/` (on by default,
redacted at capture; disable with `--no-stage`). A staged failure is a
banked repro, not yet a gate:

- `ciagent stage list` shows staged entries best-to-promote first, with a
  triage classification.
- `ciagent stage verify <id>` re-runs one entry and re-classifies it in
  place. `--mock` is zero-key; `--reroll` re-runs the persona fresh (does
  the scenario class reproduce, not just this conversation); `--world <file>`
  verifies against a frozen backend.
- `ciagent promote <id>` turns a verified entry into a permanent golden
  gate. Promotion is the human "yes": ask the user before promoting.
  `--xfail` banks it as an expected-fail bug-golden (CI stays green while
  the bug reproduces, flagged XPASS when it suddenly passes); after the fix
  lands, `ciagent promote --flip <golden>` converts it to a normal gate.
- `ciagent stage drop <id>` discards noise.

## Frozen worlds: deterministic replay and chaos

- `ciagent world freeze <stage-id>` (or a golden envelope) extracts the
  run's tool traffic into a world file: arguments in, frozen response out.
- `ciagent simulate --replay <envelope> --world <file>` replays against the
  frozen backend. Fail-closed: unmatched calls are recorded as world misses
  with a nearest-fixture diff, never guessed.
- `ciagent world mutate <world> --op <operator>` derives a new world (the
  source is never modified) for chaos and injection testing. Operators:
  `empty`, `error`, `inject` (adversarial payload into tool output),
  `rewrite`, `truncate-sequence`, `swap`; `ciagent world operators` lists
  them. Flagship use: `inject` a prompt-injection payload into a frozen tool
  response, replay, and let a `forbidden_tools` / `not_in_answer` check fire
  if the agent obeys it.

For runnable examples of agent failure modes worth gating (excessive agency,
transcript poisoning, tool-output injection), see the Agent Failure Atlas
shipped in the package at `ciagent/examples/failure-atlas/`.

If the user's coding agent should drive this loop itself, `ciagent mcp`
(install `ciagent[mcp]`) runs a stdio MCP server exposing test, simulate,
stage, promote, and world; live runs are refused server-side without an
explicit cost cap.

## Rules

- A correctness failure means the agent lost a fact it used to state. Fix the
  agent, or (only if the check itself is factually wrong) fix the check.
  **Never weaken or delete a correct check or baseline to make a run green**;
  report the failure to the user instead.
- After intentionally changing agent behavior, re-record the affected golden:
  delete its baseline file and rerun
  `ciagent bootstrap --adapter <adapter> --queries <file> --yes` for that query,
  or update the spec's expectations, with the user's confirmation.
- Promotion to a golden gate is the user's call, never automatic. Verify
  first, then ask.
- Report results in one or two sentences: score, what failed and in which
  layer, flip sources if any, and the command you ran.
