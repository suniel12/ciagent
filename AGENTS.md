# AGENTS.md

> Machine-readable reference for coding agents (Claude Code, Cursor, Codex, Copilot).
> For human-friendly docs, see [README.md](README.md).

**Version**: 0.6.0 | **Package**: `pip install ciagent` | **License**: Apache-2.0

## Overview

AgentCI is a trace-based regression testing framework for AI agents. It captures LLM calls, tool invocations, routing decisions, and costs, then diffs them against known-good baselines to catch semantic drift before production.

## Installation

```bash
pip install ciagent
```

Optional framework-specific extras:

```bash
pip install ciagent[openai]      # OpenAI Agents SDK support
pip install ciagent[anthropic]   # Anthropic Claude support
pip install ciagent[langgraph]   # LangGraph/LangChain support
pip install ciagent[all]         # All frameworks
```

## CLI Commands

```bash
# ── Setup & Scaffolding ──────────────────────────────────────────────
agentci init                          # Scaffold GitHub Actions workflow + optional pre-push hook
agentci init --hook                   # Also install .git/hooks/pre-push
agentci init --force                  # Overwrite existing files
agentci init --generate               # Guided interview: auto-scan, generate agentci_spec.yaml
agentci init --generate --mode mock   # Non-interactive mock mode
agentci init --generate --mode mock --golden-file qa.json  # Zero-API-key spec from Q&A file
agentci init --generate --kb-path ./docs  # Specify knowledge base directory

agentci doctor                        # Health check: spec, deps, API keys, KB, CI workflow
agentci doctor --config path.yaml     # Check a specific config file

agentci validate agentci_spec.yaml    # Validate spec against schema (no execution)

agentci bootstrap                     # Quick setup from queries file + runner path
agentci bootstrap --queries q.txt --runner myagent:run --output spec.yaml

agentci calibrate                     # Run sample queries, measure actuals, auto-tune spec budgets
agentci calibrate --samples 3         # Number of sample queries per spec entry
agentci calibrate --dry-run           # Show proposed changes without writing
agentci calibrate --yes               # Skip confirmation prompt

# ── Testing & Evaluation ────────────────────────────────────────────────
agentci test                          # 3-layer evaluation (Correctness → Path → Cost)
agentci test --mock                   # Zero-cost synthetic traces — no API keys needed
agentci test --yes                    # Skip cost-estimate confirmation (CI-friendly)
agentci test --workers 4              # Parallel execution
agentci test --tags routing           # Filter queries by tag
agentci test --format json            # Machine-readable JSON output
agentci test --format html -o report.html  # HTML report with per-query details
agentci test --sample-ensemble 3      # LLM judge ensemble (majority vote)

agentci eval                          # Standalone correctness evaluation (no golden baselines)
agentci eval --config spec.yaml       # Evaluate a specific spec
agentci eval --tags safety            # Filter by tag

# ── Golden Baselines ─────────────────────────────────────────────────
agentci record <test_name>            # Run agent live, save golden baseline
agentci record <test_name> -o path/   # Specify output path

agentci save --agent my-agent --version v1 --trace-file trace.json  # Save versioned baseline
agentci save --agent my-agent --version v2 --trace-file t.json --force-save  # Skip precheck

agentci baselines --agent my-agent    # List saved baseline versions for an agent

agentci diff --agent my-agent --baseline v1 --compare v2  # Diff two baseline versions
agentci diff --agent my-agent --baseline v1 --compare v2 --format json  # JSON output
agentci diff --spec-path spec.yaml --baseline-dir baselines/  # With custom paths

# ── Legacy & Reporting ────────────────────────────────────────────────
agentci run                           # Legacy test suite runner (pytest-compatible)
agentci run -s path/to/suite.yaml     # Specify suite file
agentci run -n 5                      # Statistical mode: run 5 times
agentci run -t routing -t cost        # Filter tests by tag
agentci run --no-diff                 # Skip golden trace comparison
agentci run --fail-on-cost 0.50       # Fail if total cost exceeds $0.50
agentci run --ci                      # CI mode: exit code 1 on any failure
agentci run --json                    # Machine-readable JSON output

agentci report -i results.json -o report.html  # Generate HTML report
```

## Running Tests

AgentCI is a pytest plugin. Tests can be run with either:

```bash
agentci run                          # Via AgentCI CLI
pytest                               # Via pytest directly (AgentCI auto-discovers)
```

## Core Imports

```python
# Data models
from agentci.models import Trace, Span, LLMCall, ToolCall, SpanKind, DiffType

# Trace capture
from agentci.capture import TraceContext

# Assertions (used in agentci.yaml, evaluated by runner)
from agentci.assertions import (
    evaluate_assertion,
    assert_golden_match,
    assert_budget,
    truncate_tokens,
)

# Mocks for zero-cost testing
from agentci.mocks import MockTool, MockToolkit, AnthropicMocker, OpenAIMocker

# Diff engine
from agentci.diff_engine import diff_traces, DiffReport

# Public API (top-level)
from agentci import TraceContext, test, diff, load_baseline
```

## Writing Tests

### Trace Capture

```python
from agentci.capture import TraceContext

with TraceContext(agent_name="my_agent", test_name="test_routing") as ctx:
    result = my_agent.run("I need help with billing")
    trace = ctx.trace

# Inspect trace properties
print(trace.tool_call_sequence)     # ["lookup_account", "check_billing"]
print(trace.total_cost_usd)         # 0.0023
print(trace.total_llm_calls)        # 3
print(trace.total_tool_calls)       # 2
print(trace.metadata["final_output"])  # Auto-captured from trace
```

> **Note (v0.6.0):** `final_output` is now auto-captured from traces. Extraction priority: LangGraph state messages > span `output_data` > last LLM call output. Manual `trace.metadata["final_output"] = str(result)` still works and takes precedence if set.

### Common Assertion Patterns

```python
# Check which tools were called
assert "vector_search" in trace.tool_call_sequence

# Check tool was NOT called
assert "dangerous_tool" not in trace.tool_call_sequence

# Check routing decisions (multi-agent)
handoffs = trace.get_handoffs()
assert len(handoffs) == 1
assert handoffs[-1].to_agent == "Billing Agent"

# Check cost
assert trace.total_cost_usd < 0.01

# Check LLM call count
assert trace.total_llm_calls <= 5

# Check guardrails
assert "pii_guardrail" not in trace.guardrails_triggered

# Check output content
assert "confirmation" in str(trace.spans[-1].output_data)

# Check agents involved (multi-agent)
assert trace.agents_involved == ["Triage Agent", "Billing Agent"]

# Golden trace comparison
from agentci.assertions import assert_golden_match
assert_golden_match(trace, "golden_traces/test_routing.json")

# Budget decorator
from agentci.assertions import assert_budget

@assert_budget(max_cost=0.10, max_tokens=50000)
def test_my_agent():
    ...
```

### YAML-Based Test Configuration (agentci.yaml)

```yaml
name: my-agent-tests
agent: myapp.agent:run_agent          # import path (module:function)
framework: generic                     # generic | langgraph | openai_agents
mocks: tests/mocks.yaml               # Optional mock responses

tests:
  - name: test_billing_routing
    input: "I have a billing question"
    assertions:
      - type: handoff_target
        value: "Billing Agent"
      - type: tool_called
        tool: lookup_account
      - type: cost_under
        threshold: 0.05
    golden_trace: golden/billing.json
    tags: [routing, billing]

  - name: test_stays_under_budget
    input: "Simple question"
    assertions:
      - type: cost_under
        threshold: 0.01
      - type: steps_under
        threshold: 3
    tags: [cost]
```

### Assertion Types (for agentci.yaml)

| Type | Fields | Description |
|------|--------|-------------|
| `tool_called` | `tool` | Tool was called |
| `tool_not_called` | `tool` | Tool was NOT called |
| `tool_call_count` | `tool`, `value` | Exact call count |
| `arg_equals` | `tool`, `field`, `value` | Argument equals value |
| `arg_contains` | `tool`, `field`, `value` | Argument contains substring |
| `cost_under` | `threshold` | Total cost <= threshold |
| `steps_under` | `threshold` | LLM calls <= threshold |
| `output_contains` | `value` | Output contains text |
| `output_not_contains` | `value` | Output excludes text |
| `llm_judge` | `value` | LLM evaluates qualitative rule |
| `handoff_target` | `value` | Final handoff routed to agent |
| `handoff_targets_available` | `value` (list) | All expected agents reachable |
| `handoff_count` | `threshold` | Exact handoff count |

### Mocking (Zero-Cost Testing)

```python
# Anthropic mock
from agentci.mocks import AnthropicMocker

mocker = AnthropicMocker(mock_responses=[
    {"tool": "search_flights", "input": {"origin": "SFO"}},
    {"tool": "book_flight", "input": {"id": 123}},
    {"text": "Booked your flight! Confirmation ABC."},
])
my_agent.client = mocker.client

# OpenAI mock
from agentci.mocks import OpenAIMocker

mocker = OpenAIMocker(mock_responses=[
    {"tool": "search_flights", "arguments": {"origin": "SFO"}},
    {"text": "Found flights from SFO."},
])
# Inject: mocker.client.chat.completions.create or mocker.client.responses.create

# YAML-based toolkit
from agentci.mocks import MockToolkit

toolkit = MockToolkit.from_yaml("tests/mocks.yaml")
tool = toolkit.get("search_flights")
result = tool.call(origin="SFO")
```

### Golden Baseline Workflow

```bash
# 1. Record a golden baseline from a live run
agentci record test_billing_routing -o golden/billing.json

# 2. Run tests with automatic diffing against golden
agentci run  # Compares if golden_trace is set in agentci.yaml

# 3. Diff categories detected:
#    TOOLS_CHANGED, ARGS_CHANGED, SEQUENCE_CHANGED, OUTPUT_CHANGED,
#    COST_SPIKE, LATENCY_SPIKE, STEPS_CHANGED, STOP_REASON_CHANGED,
#    ROUTING_CHANGED, GUARDRAILS_CHANGED, AVAILABLE_HANDOFFS_CHANGED
```

## Project Structure Convention

```
my-agent-project/
├── agentci.yaml              # Test suite configuration
├── tests/
│   ├── conftest.py           # AgentCI fixtures and mock setup
│   ├── fixtures/             # Recorded mock responses
│   ├── golden_traces/        # Baseline traces for regression
│   ├── test_routing.py       # Test files
│   └── test_tools.py
└── .github/
    └── workflows/
        └── agentci.yml       # Generated by `agentci init`
```

## Framework-Specific Setup

### OpenAI Agents SDK

```python
from agentci.adapters.openai_agents import AgentCITraceProcessor
from agents.tracing import add_trace_processor

add_trace_processor(AgentCITraceProcessor())
```

### LangGraph / LangChain

```python
from agentci.capture import TraceContext

with TraceContext(agent_name="rag_agent") as ctx:
    result = graph.invoke({"messages": [("user", query)]})
    ctx.attach_langgraph_state(result)
    trace = ctx.trace
```

### Raw Anthropic

```python
from agentci.mocks import AnthropicMocker

mocker = AnthropicMocker(mock_responses=[...])
# AnthropicMocker patches anthropic.Anthropic.messages.create
```

## Data Model Reference

```
Trace
├── trace_id: str
├── spans: list[Span]
├── total_cost_usd: float
├── total_tokens: int
├── total_llm_calls: int
├── total_tool_calls: int
├── tool_call_sequence: list[str]       # property
├── tool_call_details: list[ToolCall]   # property
├── get_handoffs() -> list[Span]        # method
├── guardrails_triggered: list[str]     # property
├── agents_involved: list[str]          # property
└── available_handoffs: list[list[str]] # property

Span
├── kind: SpanKind  (AGENT | LLM_CALL | TOOL_CALL | HANDOFF | GUARDRAIL)
├── name: str
├── tool_calls: list[ToolCall]
├── llm_calls: list[LLMCall]
├── from_agent: str | None   (handoff source)
├── to_agent: str | None     (handoff target)
├── guardrail_name: str | None
└── guardrail_triggered: bool

ToolCall
├── tool_name: str
├── arguments: dict[str, Any]
├── result: Any | None
└── error: str | None

LLMCall
├── model: str
├── provider: str  ("openai" | "anthropic")
├── tokens_in: int
├── tokens_out: int
├── cost_usd: float
└── stop_reason: str | None
```

## Error Messages

AgentCI errors include actionable fix suggestions. Examples:

- `"No agent import path provided in test suite."` — Set `agent: myapp.agent:run_agent` in agentci.yaml
- `"Could not import agent function 'path': error"` — Check the module:function import path
- `"Mock tool 'name' not found. Available: [...]"` — Use one of the listed mock tools
- `"Golden trace not found: path. Run with --update-golden to create it."` — Record a baseline first
- `"Budget Exceeded: Agent cost $X > max allowed $Y"` — Reduce token usage or increase budget
- `"Tool 'name' was NOT called. Tools called: [...]"` — Check agent routing logic

## CI/CD Setup

```bash
agentci init         # Generates .github/workflows/agentci.yml
agentci init --hook  # Also generates .git/hooks/pre-push
```

The generated workflow runs `pytest` and `agentci diff` on every push/PR.
