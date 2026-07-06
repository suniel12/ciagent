# AgentCI

**Pytest-native regression testing for AI agents.** Catch routing changes, tool call drift, and cost spikes before production.

[![PyPI](https://img.shields.io/pypi/v/ciagent)](https://pypi.org/project/ciagent/)
[![CI](https://github.com/suniel12/AgentCI/actions/workflows/ci.yml/badge.svg)](https://github.com/suniel12/AgentCI/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![AGENTS.md](https://img.shields.io/badge/AGENTS.md-supported-blue)](AGENTS.md)

You changed a prompt. Your agent broke in production. Three days later, a user complained. You had no tests, no diff, no idea what went wrong.

Works with OpenAI, Anthropic, and LangGraph. Runs inside pytest.

## Add to Your Project

```bash
pip install ciagent
```

Write your golden queries — what should your agent handle, and what should it refuse?

```yaml
# agentci_spec.yaml
agent: my-agent
# runner: any function that takes a query string and returns a response
runner: my_app.agent:run_for_agentci
queries:
  - query: "How do I install AgentCI?"
    correctness:
      any_expected_in_answer: ["pip install", "ciagent"]
    path:
      expected_tools: [retrieve_docs]
    cost:
      max_llm_calls: 8

  - query: "What's the CEO's favorite restaurant?"
    correctness:
      not_in_answer: ["restaurant", "favorite"]
    path:
      expected_tools: []  # expect no tools called for out-of-scope queries
```

Run:

```bash
agentci test --mock       # start here: zero-cost with synthetic traces
agentci test              # run live against your real agent
```

`agentci test` evaluates each query through 3 layers — correctness, path, and cost:

```
============================================================

Query: How do I install AgentCI?
Answer: To install AgentCI, you can use pip with the following command:
        pip install ciagent. Make sure you have Python 3.10 or later.

  ✅  CORRECTNESS: PASS
       ✓ Found keywords: "pip install ciagent"
       ✓ LLM judge passed (score: 5 ≥ 0.6)
  📈  PATH: PASS
       ✓ Tool recall: 1.000 (expected: [retrieve_docs])
       ✓ Tool precision: 0.500
       ✓ No loops detected
  💰  COST: PASS
       ✓ LLM calls: 8 ≤ max 8

============================================================

Query: What Python version does AgentCI require and what frameworks does it support?
Answer: AgentCI currently does not specify a required Python version
        in the provided context, so I don't have that information...

  ❌  CORRECTNESS: FAIL
       • Expected '3.10' not found in answer
  📈  PATH: PASS
       ✓ Tool recall: 1.000 (expected: [retrieve_docs])
       ✓ Loops: 1 ≤ max 3
  💰  COST: PASS
       ✓ LLM calls: 4 ≤ max 5

============================================================
```

Don't have golden queries yet? `agentci init --generate` scans your code and generates a starter spec.

## Check facts in code. Save the judge for judgment.

Most agent failures that matter involve a hard fact — a product name, a price, a version number. Those are checkable deterministically, for free. And an LLM judge grading against the same context as your agent inherits your agent's blind spots: when retrieval comes up empty, the agent answers from nothing and the judge — reading the same nothing — passes it.

So AgentCI runs deterministic checks first and treats the judge as the last resort, not the default:

1. **Fact checks in code** — `expected_in_answer`, `not_in_answer`, `regex_match`, `json_schema`. Zero LLM calls, zero flakiness, same verdict every run.
2. **Path checks** — did the agent call the tools it should have? A missing expected tool warns; a forbidden tool fails.
3. **Cost budgets** — LLM calls, tokens, dollars per query.
4. **LLM judge** (`llm_judge` rubrics, optional) — only for answers that genuinely need judgment, evaluated after every deterministic check has run.

## Demo

Here's a RAG agent demo where someone "optimizes for latency" by reducing retriever docs from 8 to 1. AgentCI catches the correctness regression:

![AgentCI Demo](demo/agentci-rag-demo.gif)

## CLI

```bash
agentci init --generate        # Scan project, generate test spec
agentci init                   # Generate GitHub Actions workflow + pre-push hook
agentci test --mock --yes      # Zero-cost synthetic traces, CI-friendly (no keys, no prompts)
agentci test                   # Run 3-layer evaluation (correctness → path → cost)
agentci test --format html -o report.html  # HTML report with per-query details
agentci calibrate              # Measure real agent metrics, auto-tune spec budgets
agentci doctor                 # Health check: spec, deps, API keys
agentci record <test>          # Record golden baseline
agentci diff                   # Diff against baseline
agentci report -i results.json # Generate HTML report from JSON results
```
## Docs

- [Quickstart](docs/quickstart.md) — install to first green run
- [Writing tests](docs/writing-tests.md) — the full spec reference
- [Cost tracking](docs/cost-tracking.md) — budgets and spike detection
- [Golden traces](docs/golden-traces.md) — record baselines, diff regressions
- [CI/CD integration](docs/ci-cd.md) — GitHub Actions setup
- [LangGraph](docs/langgraph.md) — graph-based agent support
- [Metrics reference](docs/metrics_reference.md) — every metric, defined

## Why not just an LLM judge?

Judge-only evals are expensive, flaky, and blind to their own context. AgentCI is pytest-native regression testing: deterministic checks catch the factual failures, golden traces catch behavioral drift, cost budgets catch spend regressions — and the judge handles only what genuinely needs judgment. Mock mode (`agentci test --mock`) runs the whole suite with zero API keys and zero cost, so it can gate every PR.

## Contributing

[GitHub Issues](https://github.com/suniel12/AgentCI/issues) ·
[DemoAgents](https://github.com/suniel12/DemoAgents) — working examples for OpenAI, Anthropic, and LangGraph agents

Apache 2.0. If you build an agent and test it with AgentCI, I'd love to hear about it.

---
