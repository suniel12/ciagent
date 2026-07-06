# Stability Testing — `agentci test --runs N`

A suite score that holds steady across runs is not evidence that your system
behaves the same twice. Individual queries can flip verdicts on every run
while the aggregate stays flat, because the errors move around. Stability
mode runs the whole suite N times and reports the difference.

## Usage

```bash
agentci test --runs 3                 # run every query 3 times, live
agentci test --runs 5 --fail-on-flaky # gate CI on verdict stability
AGENTCI_MOCK_FLAKY=1 agentci test --mock --runs 3   # zero-key demo
```

## What you get

Per query:

- **Verdict history** — `✅❌✅` across runs
- **pass rate** — observed fraction of passing runs
- **pass@k** — estimated probability of ≥1 pass in k trials (use when one
  success matters, e.g. retry-able workflows)
- **pass^k** — estimated probability that all k trials pass (use for
  consistency-critical, customer-facing systems). These diverge as k grows —
  a query with pass rate 0.67 has pass@3 ≈ 0.96 but pass^3 ≈ 0.30.
- **Flip source** — see below
- Cost and latency per run (JSON output)

Suite-level: score per run side by side, flip counts by source, and a
`STABLE` / `FLAKY` verdict.

## Flip-source attribution

A verdict flips for one of two reasons, and they demand different fixes:

| Source | What happened | Where the fix lives |
|--------|---------------|---------------------|
| `agent-variance` | The agent produced different output (answer or tool sequence changed) | The agent: prompt, retrieval, temperature |
| `judge-flake` | Output identical across runs, but the LLM judge's verdict changed | The eval: rubric, judge model — or replace the judge with a deterministic check |
| `mixed` | Near-identical paraphrases (similarity ≥ 0.9) with a judge configured | Ambiguous — AgentCI does not guess |

Attribution is structural, not heuristic: deterministic checks evaluate the
answer string and tool sequence, so on identical output they return identical
verdicts *by construction*. A flip on identical output can only have come
from the judge.

Answers are normalized (whitespace, casing) before comparison so formatting
noise doesn't read as agent variance.

## Exit codes

| Condition | Exit |
|-----------|------|
| All verdicts stable and passing | 0 |
| Flaky but every query passed at least once | 0 (warnings) |
| Any query failed in **every** run (consistent failure) | 1 |
| Any flip, with `--fail-on-flaky` | 1 |

Consistent failures are reported separately from flakiness — a query that
fails deterministically is a regression, not noise.

## Output formats

- **console** — compact per-run progress, detail only for consistent
  failures, then the stability section
- **github** — `::warning` annotation per flipped query (source-labelled),
  `::error` for consistent failures
- **json** — `stability` block with per-query verdict histories, estimates,
  flip sources, cost/latency per run
- **html** — stability card in the report dashboard

## Cost

`--runs N` multiplies agent (and judge) calls by N; the pre-run cost estimate
accounts for it. Start with N=3 on a schedule (nightly) rather than every PR
if budget is tight — flakiness doesn't need to be measured on every commit to
be known.
