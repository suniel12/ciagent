# Making AgentCI Agent-Friendly: Research & Action Plan

## The Strategic Insight

You're right that the primary *writer* of agent code in 2025-2026 isn't a human reading your README — it's Claude Code, Cursor, Codex, or Copilot interpreting your docs programmatically. This means AgentCI has two distinct user personas:

1. **The human** who decides to adopt AgentCI (reads the README, evaluates the tool)
2. **The coding agent** who actually writes the `test_*.py` files, configures AgentCI, and runs it (reads structured docs, AGENTS.md, llms.txt, type hints, error messages)

The human decides. The agent executes. You need to optimize for both — but almost nobody in the testing tools space is optimizing for the agent yet. That's your opening.

---

## Part 1: Research Findings

### 1.1 The Emerging Standards Landscape

Three standards have emerged for making projects agent-friendly:

**AGENTS.md** (the most important one for you)
- Open standard stewarded by the Agentic AI Foundation under the Linux Foundation
- Already adopted by 60,000+ GitHub repos
- Supported by Claude Code, OpenAI Codex, Google Jules, Cursor, Aider, RooCode, Zed
- Think of it as "README for agents" — structured, predictable instructions on how to work with your project
- Claude Code reads `CLAUDE.md` (proprietary equivalent), but also supports AGENTS.md via the Agent Skills standard
- Key content: setup commands, testing workflows, coding style, PR guidelines, project structure

**llms.txt** (useful for docs sites, less critical at launch)
- Proposed standard for making website content LLM-friendly
- 844,000+ websites implemented it (Stripe, Anthropic, Cloudflare, etc.)
- Provides clean markdown for AI agents to consume instead of parsing HTML
- Two files: `llms.txt` (index) and `llms-full.txt` (comprehensive reference)
- Most relevant when you have a docs site; less critical for a GitHub-only project at launch
- Context7 MCP server already indexes llms.txt files for coding agents

**Agent Skills** (Anthropic's open standard)
- Open standard released by Anthropic in late 2025, adopted by OpenAI, Cursor, GitHub Copilot
- SKILL.md files with YAML frontmatter describing capabilities
- Agents load skills dynamically based on description matching
- AgentCI could ship as a *skill* that coding agents discover automatically

### 1.2 How Coding Agents Actually Use Libraries

Based on research into Claude Code, Codex, and Cursor workflows, here's what happens when a coding agent needs to integrate a testing library:

**Step 1: Discovery.** The agent reads the project's AGENTS.md/CLAUDE.md, which says "run tests with `agentci run`" or "use AgentCI for regression testing." Without this, the agent doesn't know AgentCI exists.

**Step 2: Installation.** The agent runs `pip install agentci`. If this fails or requires extra steps, the agent gets stuck. Zero-config installation is critical.

**Step 3: API Understanding.** The agent reads type hints, docstrings, and error messages to understand how to use the library. It does NOT typically read the README — it reads the *code*. Good type annotations and docstrings are more important than good prose for agent consumers.

**Step 4: Writing Tests.** The agent generates test files. If your API has consistent patterns (always `trace.X` or `assert_X(trace, ...)`) the agent generalizes quickly. Inconsistent APIs cause the agent to hallucinate wrong function signatures.

**Step 5: Running & Debugging.** The agent runs tests and reads error output. If errors are structured and actionable ("Expected tool 'vector_search' in trace, got: ['generate']"), the agent can self-correct. If errors are vague ("AssertionError"), the agent flounders.

**Step 6: CI Integration.** The agent generates GitHub Actions YAML. If `agentci init` produces a working workflow, the agent is done. If manual config is needed, friction multiplies.

### 1.3 What Makes a Library Agent-Hostile vs Agent-Friendly

**Agent-Hostile Patterns:**
- Implicit configuration (env vars that must be set, config files that must exist)
- Untyped APIs (bare dicts, string arguments where enums should be)
- Generic error messages ("Error" instead of "AgentCITraceError: No spans found. Did you forget to attach the trace processor?")
- Complex multi-step setup (create config → set env → init project → then write tests)
- Documentation in formats agents can't easily parse (videos, images without alt text, interactive tutorials)
- Global state that makes tests non-deterministic
- Different APIs for different frameworks (LangGraph uses X, OpenAI uses Y)

**Agent-Friendly Patterns:**
- Single entry point (`pip install agentci` → import → use)
- Rich type annotations (Pydantic models, typed return values, enums for options)
- Structured, actionable error messages with fix suggestions
- Consistent API surface across frameworks (same `trace.tools_called` everywhere)
- Docstrings that include usage examples (agents read these!)
- JSON/structured output from CLI (not just human-readable text)
- `py.typed` marker for type checker support
- Convention-over-configuration (sensible defaults, zero required config)

### 1.4 The CLAUDE.md / AGENTS.md Pattern for Libraries

The key insight from the HumanLayer blog post: "If your CLI commands are complex and verbose, don't write paragraphs of documentation to explain them. That's patching a human problem. Instead, write a simple bash wrapper with a clear, intuitive API."

Applied to AgentCI: When a coding agent is working on a project that uses AgentCI, it needs to know:
- How to run tests: `agentci run` or `pytest`
- How to record baselines: `agentci record`
- How to diff: `agentci diff --baseline golden/`
- What assertions are available: `trace.tools_called`, `trace.get_handoffs()`, etc.
- What the import looks like: `from agentci import Trace, assert_budget`

This information should live in a place that coding agents automatically discover.

---

## Part 2: Action Plan

### Priority 1: AGENTS.md File (Do This First — 1 hour)

Create an `AGENTS.md` at the root of the AgentCI repo. This is the single highest-leverage change. Every coding agent that touches a project using AgentCI will read this.

```markdown
# AGENTS.md

## Overview
AgentCI is a trace-based regression testing framework for AI agents. It captures LLM calls, tool invocations, routing decisions, and costs, then diffs them against known-good baselines.

## Installation
pip install agentci

## Running Tests
agentci run                    # Run all tests (pytest-compatible)
agentci run tests/test_rag.py  # Run specific test file
pytest                         # Also works — AgentCI is a pytest plugin

## Recording Baselines
agentci record --name "v1-baseline" --output golden/

## Diffing Against Baselines
agentci diff --baseline golden/v1-baseline.json

## CI Setup
agentci init  # Generates .github/workflows/agentci.yml and pre-push hook

## Writing Tests

### Core Imports
```python
from agentci import Trace, Span, LLMCall, ToolCall
from agentci.assertions import (
    tool_called, cost_under, output_contains,
    assert_llm_judge, assert_handoff_target,
    assert_handoff_count, assert_guardrails_triggered
)
from agentci.mocks import AnthropicMocker, OpenAIMocker
from agentci.decorators import assert_budget
```

### Common Assertion Patterns
```python
# Check which tools were called
assert "vector_search" in trace.tools_called

# Check routing decisions (multi-agent)
handoffs = trace.get_handoffs()
assert handoffs[0].to_agent == "Billing Agent"

# Check cost
assert trace.total_cost < 0.01

# Check guardrails
assert "pii_guardrail" not in trace.guardrails_triggered

# LLM-as-judge for output quality
assert_llm_judge(trace, "answer is relevant to the question")

# Budget decorator
@assert_budget(max_cost=0.10)
def test_my_agent():
    ...
```

### Mocking (Zero-Cost Testing)
```python
# Anthropic
with AnthropicMocker(fixtures="tests/fixtures/") as mocker:
    trace = run_agent("test query")

# OpenAI
with OpenAIMocker(fixtures="tests/fixtures/") as mocker:
    trace = run_agent("test query")
```

## Project Structure Convention
```
tests/
├── conftest.py          # AgentCI fixtures and mock setup
├── fixtures/            # Recorded mock responses
├── golden_traces/       # Baseline traces for regression
├── test_routing.py      # Test files
└── test_tools.py
```

## Error Messages
AgentCI errors include fix suggestions. Example:
- "No spans found in trace. Did you register AgentCITraceProcessor?"
- "Tool 'vector_search' not found. Tools called: ['generate', 'summarize']"
- "Cost $0.15 exceeds budget $0.10 (1.5x over)"

## Framework-Specific Setup

### LangGraph
```python
from agentci.integrations.langgraph import attach_langgraph_state
ctx.attach_langgraph_state(graph)
```

### OpenAI Agents SDK
```python
from agentci.integrations.openai import AgentCITraceProcessor
from agents.tracing import add_trace_processor
add_trace_processor(AgentCITraceProcessor())
```

### Raw Anthropic
```python
from agentci.mocks import AnthropicMocker
# AnthropicMocker patches anthropic.Anthropic.messages.create
```
```

**Also create this file in the DemoAgents repo** — adapted to show how to run the specific demos.

### Priority 2: Type Annotations & Docstrings (2-3 hours)

The biggest impact on agent usability is making the Python API fully typed with comprehensive docstrings. Agents read these, not your README.

**Action items:**

a) **Add `py.typed` marker** to the package. This tells type checkers (and agents) that your package has type annotations.
```bash
touch agentci/py.typed
```
Add to your `pyproject.toml`:
```toml
[tool.hatchling.build]
include = ["agentci/py.typed"]
```

b) **Ensure all public functions have typed signatures with docstrings containing examples:**
```python
def tool_called(trace: Trace, tool_name: str) -> bool:
    """Check if a specific tool was called during the agent run.
    
    Args:
        trace: The captured execution trace
        tool_name: Name of the tool to check for
        
    Returns:
        True if the tool was called, False otherwise
        
    Example:
        >>> trace = run_agent("search for documents")
        >>> assert tool_called(trace, "vector_search")
    """
```

c) **Use Enums instead of strings where possible:**
```python
# Instead of:
span.kind = "handoff"  # agent might typo this

# Use:
from agentci import SpanKind
span.kind = SpanKind.HANDOFF  # autocomplete-friendly, agent can discover valid values
```

d) **Make Pydantic models for all data structures.** You already do this (Trace, Span, etc. are Pydantic). Make sure `.model_json_schema()` produces clean output — agents can use this to understand the data model without reading docs.

### Priority 3: Structured CLI Output (2 hours)

Right now your CLI likely outputs human-readable text. Add a `--json` flag for agent consumption.

```bash
# Human-readable (default)
agentci run
# ======================== AgentCI Test Results ========================
# tests/test_routing.py::test_billing  ✅ PASSED
# ...

# Machine-readable (for agents)
agentci run --json
# {"tests": [{"name": "test_billing", "status": "passed", "duration": 0.3}], ...}

agentci diff --baseline golden/v1.json --json
# {"diffs": [{"type": "ROUTING_CHANGED", "query": "...", "baseline": "Account", "current": "Billing"}], ...}
```

This lets a coding agent programmatically check test results and take action (fix the code, update the baseline, etc.).

### Priority 4: Actionable Error Messages (1-2 hours)

Audit every error path in AgentCI and ensure errors tell the agent *what to do*, not just what went wrong.

**Pattern:**
```python
class AgentCIError(Exception):
    """Base exception with fix suggestion."""
    def __init__(self, message: str, fix: str = ""):
        self.fix = fix
        super().__init__(f"{message}\n  Fix: {fix}" if fix else message)

# Usage:
raise AgentCIError(
    "No spans found in trace",
    fix="Register AgentCITraceProcessor before running your agent: "
        "add_trace_processor(AgentCITraceProcessor())"
)
```

**Key error scenarios to make actionable:**
- No trace processor registered → show exact import + setup code
- Tool not found in trace → show which tools *were* called
- Cost over budget → show exact cost and threshold
- Golden baseline file not found → show `agentci record` command
- Mock fixture missing → show how to record fixtures
- Framework not detected → show supported frameworks and setup

### Priority 5: Agent Skill File (1 hour)

Create an AgentCI skill that coding agents can discover and load automatically. This goes in the AgentCI repo.

```
.claude/skills/agentci-testing/SKILL.md
```

```markdown
---
name: agentci-testing
description: |
  Add trace-based regression tests for AI agents using AgentCI.
  Use when writing tests for agents built with LangGraph, Anthropic, 
  or OpenAI Agents SDK. Covers tool call assertions, routing checks,
  cost guards, guardrail verification, and golden baseline diffing.
---

# AgentCI Testing Skill

When adding AgentCI tests to an agent project:

1. Install: `pip install agentci`
2. Create `tests/conftest.py` with framework-specific fixtures
3. Write test functions using `trace.tools_called`, `trace.get_handoffs()`, etc.
4. Record golden baselines with `agentci record`
5. Add CI with `agentci init`

## Test File Template
```python
import pytest
from agentci import Trace
from agentci.assertions import tool_called, cost_under, assert_handoff_target

def test_agent_routes_correctly():
    trace = run_agent("I need help with billing")
    handoffs = trace.get_handoffs()
    assert len(handoffs) == 1
    assert_handoff_target(trace, "Billing Agent")

def test_agent_stays_under_budget():
    trace = run_agent("simple query")
    assert cost_under(trace, 0.01)

def test_agent_calls_expected_tools():
    trace = run_agent("search the knowledge base")
    assert tool_called(trace, "vector_search")
```

See AGENTS.md in the project root for full API reference.
```

### Priority 6: llms.txt (30 minutes — do after docs site exists)

When you eventually have a docs site, add `llms.txt` to point agents to the right documentation pages. For now, since everything is on GitHub, the AGENTS.md + README covers this. But create a minimal one in the repo:

```markdown
# AgentCI
> Trace-based regression testing for AI agents. Catch semantic drift, tool call changes, and cost spikes before production.

## Docs
- [README](https://github.com/YOUR_USERNAME/AgentCI/blob/main/README.md): Overview, quickstart, and core concepts
- [AGENTS.md](https://github.com/YOUR_USERNAME/AgentCI/blob/main/AGENTS.md): Machine-readable API reference and setup instructions for coding agents
- [DemoAgents](https://github.com/YOUR_USERNAME/DemoAgents): Three working demo agents with full test suites

## API Reference
- [Trace Model](https://github.com/YOUR_USERNAME/AgentCI/blob/main/agentci/models.py): Trace, Span, LLMCall, ToolCall, HandoffSpan, GuardrailSpan
- [Assertions](https://github.com/YOUR_USERNAME/AgentCI/blob/main/agentci/assertions.py): tool_called, cost_under, assert_handoff_target, assert_llm_judge
- [Mocks](https://github.com/YOUR_USERNAME/AgentCI/blob/main/agentci/mocks.py): AnthropicMocker, OpenAIMocker
- [CLI](https://github.com/YOUR_USERNAME/AgentCI/blob/main/agentci/cli.py): run, record, diff, init
```

### Priority 7: README Additions for Agent Context (30 minutes)

Add a small section to the README that addresses the agent-as-user directly:

```markdown
## For Coding Agents

If you're a coding agent (Claude Code, Cursor, Codex, etc.) integrating AgentCI 
into a project, see [`AGENTS.md`](AGENTS.md) for structured setup instructions, 
import patterns, and assertion API reference.

Quick setup for any agent project:
1. `pip install agentci`
2. Copy the test template from AGENTS.md
3. Run `agentci init` to generate CI config
4. Run `agentci run` to execute tests
```

### Priority 8: MCP Server (Future — Post-Launch)

The ultimate agent-friendly move: build an AgentCI MCP server that coding agents can connect to directly. This would let agents:
- Run tests without shelling out to the CLI
- Get structured diff results
- Record baselines programmatically
- Query trace data

This is a post-launch initiative, but flag it in your roadmap because it positions AgentCI as a native tool in the MCP ecosystem alongside Playwright's MCP server, Selenium's MCP server, etc.

---

## Part 3: Impact Matrix

| Change | Effort | Impact on Agents | Impact on Humans | Priority |
|--------|--------|-----------------|-----------------|----------|
| AGENTS.md | 1 hour | ★★★★★ | ★★ (also useful as quick reference) | Do first |
| Type annotations + docstrings | 2-3 hours | ★★★★★ | ★★★ (IDE autocomplete) | Do second |
| Structured CLI output (--json) | 2 hours | ★★★★ | ★★ (scriptability) | Do third |
| Actionable error messages | 1-2 hours | ★★★★ | ★★★★ (humans benefit too) | Do fourth |
| Agent Skill file | 1 hour | ★★★ | ★ (agent-only feature) | Do fifth |
| llms.txt | 30 min | ★★ | ★ (minimal until docs site) | Do after launch |
| README "For Coding Agents" section | 30 min | ★★ | ★★ (signals awareness) | Do with README |
| MCP server | 2-3 days | ★★★★★ | ★★★ (programmatic access) | Post-launch |

**Total pre-launch effort: ~8 hours for priorities 1-5 + 7**

---

## Part 4: The Competitive Moat

Here's why this matters strategically: nobody else in the agent testing space is doing this. The existing testing tools (pytest, unittest, Hypothesis) were all designed before coding agents existed. They have no AGENTS.md, no structured CLI output, no agent skill files.

By making AgentCI agent-native from day one, you create a flywheel:

1. Developer uses Claude Code to build an agent
2. Claude Code discovers AgentCI (via AGENTS.md or Agent Skills)
3. Claude Code writes AgentCI tests automatically
4. Tests catch regressions the developer didn't know about
5. Developer becomes an AgentCI advocate
6. More projects add AgentCI to their AGENTS.md
7. More coding agents discover AgentCI

The agent isn't just your user — it's your distribution channel. Every AGENTS.md file that mentions AgentCI is organic discovery for the next coding agent that reads that project.

This is the same dynamic that made pytest win over unittest: pytest didn't just have better features, it had better *developer experience*. You're doing the same thing for the next era — better *agent experience*.

---

## Part 5: Changes to the README Instructions

Update the README creation instructions (from the previous document) with these additions:

1. **Add a "For Coding Agents" section** (Priority 7 above) — place it after "CI/CD" and before "Assertions"

2. **In the "Status" section**, add to the roadmap:
   - "MCP server for native coding agent integration"
   - "Agent Skills distribution for auto-discovery"

3. **In every code block**, ensure imports are complete (agents copy-paste code blocks literally — incomplete imports cause errors they can't resolve)

4. **Add AGENTS.md badge** to the README badges:
   ```markdown
   [![AGENTS.md](https://img.shields.io/badge/AGENTS.md-supported-blue)](AGENTS.md)
   ```

---

## Summary: What to Do and When

**Before README (today):**
- [ ] Create AGENTS.md in AgentCI repo
- [ ] Create AGENTS.md in DemoAgents repo
- [ ] Add `py.typed` marker
- [ ] Audit and add docstrings with examples to all public functions

**During README sprint:**
- [ ] Add "For Coding Agents" section to README
- [ ] Add AGENTS.md badge
- [ ] Ensure all README code blocks have complete imports

**Day after launch:**
- [ ] Add `--json` flag to CLI commands
- [ ] Audit error messages for actionable fix suggestions
- [ ] Create Agent Skill file (.claude/skills/agentci-testing/SKILL.md)
- [ ] Create minimal llms.txt

**Post-launch (week 2-3):**
- [ ] Build AgentCI MCP server prototype
- [ ] Submit to MCP Registry
- [ ] Submit to llms-txt-hub directory