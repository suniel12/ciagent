# Quickstart

Install to first green run in under two minutes.

## 1. Install

```bash
pip install ciagent
```

## 2. Write a spec

Create `agentci_spec.yaml` next to your agent code. The `runner` is any function
that takes a query string and returns a response:

```yaml
agent: my-agent
runner: my_app.agent:run_for_agentci
queries:
  - query: "How do I install AgentCI?"
    correctness:
      any_expected_in_answer: ["pip install", "ciagent"]
    path:
      expected_tools: [retrieve_docs]
    cost:
      max_llm_calls: 8
```

Don't have queries yet? Generate a starter spec from your codebase:

```bash
agentci init --generate
```

## 3. Validate with zero cost

Mock mode runs your whole spec against synthetic traces — no API keys, no spend:

```bash
agentci test --mock
```

## 4. Run live

```bash
agentci test
```

Each query is evaluated through three layers: **correctness** (deterministic
checks first, LLM judge only if configured), **path** (expected/forbidden
tools, loop detection), and **cost** (LLM calls, tokens, dollars).

## 5. Next steps

- `agentci calibrate` — measure your agent's real metrics and auto-tune spec budgets
- `agentci init` — generate a GitHub Actions workflow ([CI/CD guide](ci-cd.md))
- `agentci doctor` — health-check your setup
- [Writing tests](writing-tests.md) — the full spec reference
