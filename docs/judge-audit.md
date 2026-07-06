# Judge Audit — `ciagent judge-audit`

Every eval tool runs an LLM judge. None of them tell you whether the judge
can be trusted. `judge-audit` measures your judge against ground truth you
already have — by re-scoring **recorded** answers (golden baselines). The
agent is never re-run; the only cost is judge calls.

```bash
ciagent judge-audit                        # audit against golden baselines
ciagent judge-audit --repeats 5            # tighter flip-rate measurement
ciagent judge-audit --labels labels.yaml   # add human ground truth
ciagent judge-audit --sample 20 --yes      # cap cost, skip confirmation
```

## The three measurements

### 1. Judge vs. deterministic checks (no labels required)

On queries that have **both** deterministic checks (`expected_in_answer`,
`regex_match`, …) and judge rubrics, both are evaluated independently on the
same recorded answer. The report shows the disagreement matrix; the row that
matters is:

> **judge PASS / check FAIL** — the judge passed an answer that a
> deterministic fact-check failed.

This is the classic shared-context failure: the judge, grading against the
same retrieved context as the agent, sees a fluent, faithful-looking answer
and passes it — while the hard fact is simply wrong or missing.

The reverse cell (judge fails what checks pass) is listed but **not** counted
against the judge — it may be noise, or a real quality problem keywords can't
see.

### 2. Retest stability (`--repeats K`, default 3)

The same answer is judged K times. Any verdict flip on identical input is the
judge's own noise floor — published measurements put mean judge flip rates
around 13.6%, so do not assume zero.

### 3. Hand labels (`--labels FILE`)

A YAML/JSON mapping of query → `pass`/`fail` from your own review:

```yaml
"What's your return window?": pass
"Do you sell gift cards?": fail
```

Reports raw agreement and Cohen's κ. Standard guidance: don't adopt a judge
below 75–90% agreement with human labels; κ < 0.75 is below the trust floor.

## The scoped claim (read this)

Mode 1 measures the judge only on **fact-checkable** queries. The queries you
actually keep a judge for are the judgment-heavy ones, and error rates on the
first population do not automatically transfer to the second.

The honest reading is one-directional: **a judge that fails where you CAN
check it should not be trusted where you can't.** Passing the audit is a
smoke test, not a guarantee, for judgment-only queries — for those, hand-label
a sample and use Mode 3. The report states this scope in its own output.

## Verdicts and exit codes

| Verdict | Meaning | Exit |
|---------|---------|------|
| `TRUSTWORTHY` | No false passes, flip rate ≤ 5%, label agreement ≥ 90% | 0 |
| `NEEDS CALIBRATION` | Any false pass, flip rate 5–20%, or agreement 75–90% | 0 |
| `UNRELIABLE` | False-pass rate > 15%, flip rate > 20%, agreement < 75%, or κ < 0.4 | 1 |
| `ERROR` | Every judge call errored — the judge never ran; no honest verdict | 2 |

With fewer than 5 checkable queries the report flags its rates as anecdotes.

## Where the answers come from

`--baseline-dir` (default: the spec's `baseline_dir`) is scanned recursively
for baseline JSON files; both `ciagent record` output and versioned
`ciagent save` baselines are accepted. Queries are matched to the spec by
query text.
