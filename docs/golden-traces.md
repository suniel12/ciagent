# Golden Traces

Golden traces are the "known good" execution path for your agent — what tools
it called, in what order, at what cost, and what it answered. Record one when
the agent behaves correctly, then diff future runs against it.

## Recording a Golden Trace

```bash
agentci record my_test_case
```

This runs your agent live and saves the trace to
`golden/my_test_case.golden.json`. Use `agentci save` to store a trace as a
*versioned* baseline (e.g. `v1`, `v2`) and `agentci baselines` to list the
versions you have.

## Diffing

Compare two versioned baselines with a three-tier analysis:

```bash
agentci diff --baseline v1 --compare v2 --agent my-agent
```

Differences are classified and highlighted:

- **Tools Changed** — different tools were called
- **Sequence Changed** — same tools, different order
- **Args Changed** — arguments to tools changed
- **Routing Changed** — a different agent handled the query (multi-agent handoffs)
- **Cost Spike** — the cost increased significantly
- **Stop Reason Changed** — the run terminated differently

Exit code 1 signals a correctness regression (pass → fail), so `agentci diff`
can gate CI directly.
