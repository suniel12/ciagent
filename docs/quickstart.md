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
  - query: "How do I install CIAgent?"
    correctness:
      any_expected_in_answer: ["pip install", "ciagent"]
    path:
      expected_tools: [retrieve_docs]
    cost:
      max_llm_calls: 8
```

Don't have queries yet? Generate a starter spec from your codebase:

```bash
ciagent init --generate
```

## 3. Validate with zero cost

Mock mode runs your whole spec against synthetic traces — no API keys, no spend:

```bash
ciagent test --mock
```

## 4. Run live

```bash
ciagent test
```

Each query is evaluated through four layers: **correctness** (deterministic
checks first, LLM judge only if configured), **path** (expected/forbidden
tools, loop detection), **retrieval** (deterministic assertions on the
retriever tool's captured output — empty-retrieval gate, expected sources,
count floors), and **cost** (LLM calls, tokens, dollars).

## 5. Next steps

- `ciagent calibrate` — measure your agent's real metrics and auto-tune spec budgets
- `ciagent init` — generate a GitHub Actions workflow ([CI/CD guide](ci-cd.md))
- `ciagent doctor` — health-check your setup
- [Writing tests](writing-tests.md) — the full spec reference
