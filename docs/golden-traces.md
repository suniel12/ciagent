# Golden Traces

Golden traces are the "known good" execution path for your agent — what tools
it called, in what order, at what cost, and what it answered. Record one when
the agent behaves correctly, then diff future runs against it.

## Recording a Golden Trace

```bash
ciagent record
```

This runs every query in `ciagent_spec.yaml` through the spec's adapter and
saves versioned baselines under `<baseline_dir>/<agent>/`. Pass a query (or a
unique substring of one) to record a single baseline, `--version` to tag the
files, and `--force-save` to skip the correctness precheck. `ciagent test`
picks the baselines up automatically.

Use `ciagent save` to store an existing trace JSON as a versioned baseline
and `ciagent baselines` to list the versions you have.

Legacy v1 suites (`ciagent.yaml`) still record a single golden trace with
`ciagent record my_test_case -s ciagent.yaml`; this path is deprecated and
will be removed at 1.0.

## Diffing

Compare two versioned baselines with a three-tier analysis:

```bash
ciagent diff --baseline v1 --compare v2 --agent my-agent
```

Differences are classified and highlighted:

- **Tools Changed** — different tools were called
- **Sequence Changed** — same tools, different order
- **Args Changed** — arguments to tools changed
- **Routing Changed** — a different agent handled the query (multi-agent handoffs)
- **Cost Spike** — the cost increased significantly
- **Stop Reason Changed** — the run terminated differently

Exit code 1 signals a correctness regression (pass → fail), so `ciagent diff`
can gate CI directly.
