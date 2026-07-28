# Stability Testing — `ciagent test --runs N`

A suite score that holds steady across runs is not evidence that your system
behaves the same twice. Individual queries can flip verdicts on every run
while the aggregate stays flat, because the errors move around. Stability
mode runs the whole suite N times and reports the difference.

## Usage

```bash
ciagent test --runs 3                 # run every query 3 times, live
ciagent test --runs 5 --fail-on-flaky # gate CI on verdict stability
CIAGENT_MOCK_FLAKY=1 ciagent test --mock --runs 3   # zero-key demo
```

## What you get

Per query:

- **Verdict history** — `✅❌✅` across runs
- **pass rate** — observed fraction of passing runs
- **Flip source** — see below
- Partial-aggregation flag when a query is missing from some runs (adapter
  failures), and a warning when duplicate query texts merge into one record
- In **JSON output only**: pass@k / pass^k *estimates* (probability of ≥1 pass /
  all passes in k trials, computed from the observed pass rate with k = runs),
  plus cost and latency per run. They live in JSON, explicitly labeled, because
  at small k they restate the pass rate — the console shows observed facts only.
- In **JSON output only**: every run's answer text (see below), so `--runs N`
  produces N gradeable answers per query rather than one.

Suite-level: score per run side by side, flip counts by source, and a
`STABLE` / `FLAKY` verdict.

## Flip-source attribution

A verdict flips for one of two reasons, and they demand different fixes:

| Source | What happened | Where the fix lives |
|--------|---------------|---------------------|
| `agent-variance` | The agent produced different output (answer, tool sequence, or a deterministic check's outcome changed) | The agent: prompt, retrieval, temperature |
| `judge-flake` | Every deterministic check agreed across runs — or the output was identical — but the LLM judge's verdict changed | The eval: rubric, judge model — or replace the judge with a deterministic check |
| `infra-error` | A judge API call errored during at least one run (an errored call counts as a fail in the verdict) | Nothing — retry before trusting the flip |
| `mixed` | Near-identical paraphrases (similarity ≥ 0.9) with a judge configured and no clearer signal | Ambiguous — CIAgent does not guess |

Attribution is structural, not heuristic, and checks signals in reliability
order: judge errors first, then per-layer sub-verdicts (if deterministic checks
returned identical outcomes across runs and only the judge's verdict changed,
the flip is the judge's — regardless of answer paraphrase), then output
identity: deterministic checks cannot flip on identical output *by
construction*.

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
  flip sources, cost/latency/tokens/trace ids per run, plus `results[i].answers`
  with every run's answer text
- **html** — stability card in the report dashboard

## Grading the repeats yourself

`--runs N` runs each query N times, and `--format json` hands you all N
answers:

```jsonc
{
  "results": [
    {
      "query": "what is the refund policy?",
      "answer": "refunds within 30 days",          // representative run, unchanged
      "answers": [                                  // present only when runs > 1
        "refunds within 30 days",
        "you can request a refund for a month",
        "refunds within 30 days"
      ]
    }
  ],
  "stability": {
    "queries": [
      {
        "query": "what is the refund policy?",
        "verdicts": [true, false, true],
        "trace_ids": ["...", "...", "..."],
        "total_tokens": [812, 903, 810],
        "cost_usd": [0.004, 0.005, 0.004],
        "latency_ms": [1220, 1355, 1198]
      }
    ]
  }
}
```

`results[i].answers` is index-aligned with `stability.queries[i].verdicts` and
with the per-run `trace_ids`, `total_tokens`, `cost_usd` and `latency_ms`
lists, so run 2's answer, verdict, trace and cost all sit at index 1. Queries
that an adapter failed to answer in some run aggregate over the runs where
they appear, so every list shortens together (the record is flagged
`partial`). `results` lists the representative run's queries, so read
`stability.queries` when you need the full set including a query the adapter
dropped in that run.

Two things this makes possible:

1. **External grading.** If you grade answers with your own metric (an
   LLM judge, gold-aware oracle grading, a human pass), you can now grade all
   N answers from one `ciagent test --runs N` invocation. Previously the
   non-representative answers were generated, evaluated and discarded, so
   getting N gradeable answers meant N separate `--runs 1` invocations merged
   by hand: N times the API spend for the same data the tool already had.
2. **Majority-vote denoising.** Run-to-run variance is large enough that
   single-run external grading is not trustworthy. Two identical
   configurations of the same retrieval setup, graded over the same 20
   questions with nothing changed between them, scored 35% and 25%: 10% of
   questions flipped on noise alone. Grading all N answers and taking the
   majority verdict per query turns repeats you already paid for into a
   materially steadier number.

Backward compatibility: `answer` keeps its meaning (the representative run)
and is always present. `answers` is added only when `runs > 1`, so single-run
output is byte-for-byte what it was.

## Cost

`--runs N` multiplies agent (and judge) calls by N; the pre-run cost estimate
accounts for it. Start with N=3 on a schedule (nightly) rather than every PR
if budget is tight — flakiness doesn't need to be measured on every commit to
be known.


## Source-aware gating: fail on agent-variance, tolerate judge-flake

`--fail-on-flaky` exits 1 on any verdict flip. But CIAgent attributes every
flip to its source (agent-variance, judge-flake, infra-error, mixed,
simulation-variance, retrieval-variance, world-miss), and `--flaky-sources`
turns that attribution into a gate:

```bash
ciagent test --runs 5 --flaky-sources=agent          # fail on agent-variance / retrieval-variance
ciagent test --runs 5 --flaky-sources=agent-variance,judge-flake
```

A team with judge-flake it can't fully eliminate no longer has to choose
between eating red builds on eval noise and disabling the gate entirely: gate
on the sources that mean "fix the agent," tolerate the rest. Aliases: `real`
/ `agent` (the fix-the-agent sources), `judge`, `infra`, `sim`. Sources not
in an alias (world-miss, simulation-variance, mixed) gate only when named
explicitly. This is the thing single-run eval tools structurally cannot do:
they can suppress flakiness (pin the judge, tolerance bands), but they don't
know why a verdict flipped.

The `--format json` stability block carries `flip_sources` (per-source
counts) and `gated_by` (the selected set), so CI scripts and the MCP agent
can act on attribution mechanically.
