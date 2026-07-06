# Cost Tracking

CIAgent automatically tracks the cost of every LLM call in a trace and
evaluates it as Layer 3 of every `ciagent test` run. Cost exceedances produce
warnings by default, so they're visible without blocking CI.

## Setting Budgets

Set budgets per query in `agentci_spec.yaml`:

```yaml
queries:
  - query: "What's your return policy?"
    cost:
      max_llm_calls: 8        # max number of LLM API calls
      max_total_tokens: 20000 # input + output tokens across all calls
      max_cost_usd: 0.10      # absolute dollar cap
      max_latency_ms: 15000   # wall-clock latency cap
```

Or once for every query, via `defaults`:

```yaml
defaults:
  cost:
    max_llm_calls: 10
    max_cost_usd: 0.25
```

## Budgets Relative to a Baseline

If you record golden baselines, `max_cost_multiplier` fails a query whose cost
exceeds a multiple of the baseline — this is how you catch cost *spikes* rather
than absolute overruns:

```yaml
defaults:
  cost:
    max_cost_multiplier: 2.0  # warn if a query costs 2× its golden baseline
```

## Don't Guess — Calibrate

Instead of inventing budget numbers, measure your agent and let CIAgent set
them with headroom:

```bash
ciagent calibrate
```

This runs sample queries live, shows measured vs. configured budgets, and
updates the spec (+50% headroom for call counts, +100% for tokens and cost).
