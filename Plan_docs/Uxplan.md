# AgentCI 2.0: Developer Experience Analysis & UX Roadmap

**Date:** February 27, 2026  
**Scope:** Research-grounded assessment of the attached UX plan + additional high-impact proposals  
**Research basis:** Current DX patterns from promptfoo, DeepEval, Braintrust, Langfuse, Arize Phoenix, and 15+ developer tool onboarding studies

---

## Assessment of the Attached UX/DX Plan

The five proposals in the attached document correctly identify the **real friction points** in the current AgentCI developer journey. Here's a per-proposal verdict grounded in what the competitive landscape actually does:

### Proposal 1: Auto-Generate `agentci_spec.yaml` during `init`

**Verdict: Strongly Agree — this is the single highest-ROI change.**

The research is unambiguous here. promptfoo's `init` command creates a working `promptfooconfig.yaml` with dummy prompts, providers, and test cases on day zero. DeepEval's quickstart generates a runnable test file with a single `deepeval test run` command. InfluxDB's CLI onboarding wizard was explicitly built to eliminate the "hours heads-down in documentation" problem. The Evil Martians devtools UX study (Dec 2024, analysis of 40+ tools) identifies the "a-ha moment" — the instant a new user understands how the tool solves their problem — as the critical onboarding goal. Currently, `agentci init` creates CI infrastructure but not the actual test spec, which delays that a-ha moment by an entire manual authoring step.

**Recommendation:** Implement exactly as described. The interactive runner prompt ("What is the import path for your agent runner function?") is the right call — it mirrors promptfoo's interactive init, which asks for provider selection and creates a complete runnable config. Add a `--example` flag (like promptfoo's `init --example getting-started`) to generate a pre-populated spec with the RAG agent demo queries, so users can see a working evaluation immediately.

**Priority: P0 — ship with launch.**

### Proposal 2: "Zero-to-Golden" Interactive Bootstrapper

**Verdict: Agree on the concept, with a significant refinement.**

This is the right idea but the execution needs sharpening. The proposal describes an interactive loop where the user enters queries, approves outputs, and AgentCI auto-generates the spec + baselines. This is powerful, but the research reveals a better pattern.

Braintrust's killer workflow is "production traces become eval cases with one click." Langfuse has a dedicated UI flow where you select production traces, click "+ Add to dataset," and the trace becomes a golden test case. The key insight: **the best test cases come from real agent runs, not from a developer typing queries into a terminal.** The bootstrapper should capture the trace structure (tools called, handoffs made, token count, latency) and auto-infer assertions from what actually happened — not just from the final text output.

**Refined recommendation:** `agentci bootstrap` should:

1. Accept queries interactively OR from a file (`--queries queries.txt`)
2. Run the agent live and display a **rich summary** of each trace (not just the output — show tools called, tokens used, latency, handoffs)
3. On user approval, auto-generate both the baseline JSON AND the spec assertions:
   - `expected_tools` populated from the observed tool sequence
   - `max_tool_calls` set to observed count + 1 (buffer)
   - `max_total_tokens` set to observed tokens × 1.5
   - `not_in_answer` left empty (user adds manually)
   - `llm_judge` rubric auto-generated from the query semantics ("Response should address installation instructions" inferred from "How do I install X?")
4. Write the complete `agentci_spec.yaml` with all queries, save baselines, and print "Run `agentci test` to verify"

The LLM-assisted assertion inference (step 3d) is the differentiator. promptfoo already has `promptfoo generate assertions` which uses an LLM to auto-generate assertions. AgentCI should do this from the start — it turns a 30-minute manual spec-authoring session into a 2-minute interactive flow.

**Priority: P0 — ship with launch. This is the "a-ha moment" feature.**

### Proposal 3: "Fix My Spec" AI Assistant

**Verdict: Agree, but reframe as `agentci doctor` and scope it tighter.**

The `--auto-fix` flag concept is sound, but it should be a standalone command rather than a flag on the test pipeline. The reason: developers need to understand what changed before accepting fixes. Embedding auto-fix into the test pipeline conflates "run tests" with "modify test definitions," which violates the principle of least surprise.

**Refined recommendation:** Create `agentci doctor` command:

- Runs after a test failure or on demand
- Loads the failed spec + the actual trace
- Uses an LLM to identify the mismatch (e.g., "Spec expects tool `search_docs`, but agent called `retriever_tool` — likely a rename")
- Outputs a diff-style suggestion: "Replace `search_docs` with `retriever_tool` in path.expected_tools? [y/N]"
- On confirmation, edits the YAML in place

This is less aggressive than `--auto-fix` (which silently modifies your spec) and more useful than raw error messages. promptfoo's `generate assertions` already proves LLM-assisted spec writing works; `agentci doctor` is the natural complement for spec maintenance.

**Priority: P1 — ship in first update. Requires the LLM judge infrastructure to already be stable.**

### Proposal 4: Interactive Trace Inspector

**Verdict: Agree, but don't overinvest in CLI visualization. The high-value version is simpler than described.**

Rich terminal trace trees are appealing in demos but rarely used in practice. Braintrust's and LangSmith's dominance in debugging comes from their **web UIs**, not terminal rendering. The Evil Martians study explicitly notes that CLI tools should optimize for "efficiency" rather than "immersive" experiences.

That said, a lightweight version adds real value. The current experience of opening a raw JSON trace file is genuinely painful.

**Refined recommendation:** Two-tier approach:

**Tier 1 (P0):** On test failure, print a **structured summary** to the console — not a full interactive tree, but a condensed trace timeline:

```
❌ FAIL: "What's the weather in Tokyo?"
   Correctness: FAIL — Forbidden term "degrees" found in answer
   Path:        WARN — 11 tool calls (max: 0)
   Cost:        WARN — 4,200 tokens (max: 500)
   
   Trace: tavily_search → tavily_search → tavily_search → ... (11 calls)
   Answer: "The weather in Tokyo is currently 15 degrees..."
   
   Baseline: baselines/rag-agent/v1-broken.json
   Full trace: .agentci/traces/2026-02-27T14:30:00Z.json
```

This is implementable in ~50 lines with `rich` and gives 90% of the debugging value.

**Tier 2 (P2):** `agentci view <trace-file>` opens a local HTML report in the browser (like promptfoo's `promptfoo view`) with the full trace tree, tool call details, and layer-by-layer breakdown. This is the web viewer pattern, not a terminal tree. A single-file HTML export (like pytest-html) is simpler to build and more useful than a terminal TUI.

**Priority: Tier 1 at P0, Tier 2 at P2.**

### Proposal 5: Drop-in pytest Plugin

**Verdict: Strongly Agree — this is the second highest-ROI change after init.**

The research is definitive on this point. DeepEval's entire positioning is "Pytest for LLMs" — their `deepeval test run` command wraps pytest, auto-discovers test files, and reports results natively in pytest's output format. It's their most-cited feature. The pytest-play project proves that YAML-driven test generation via pytest plugins is a well-established pattern with dedicated community support. The pytest docs even include a canonical example of "specifying tests in YAML files."

AgentCI already has the runner wired (`resolve_runner` + `run_spec_parallel`). The plugin would:

1. Auto-discover `agentci_spec.yaml` files via `pytest_collect_file` hook
2. Generate one `pytest.Item` per query in the spec
3. Report correctness failures as pytest `FAILED`, path/cost warnings as pytest `WARNING` 
4. Support standard pytest flags: `-k "weather"` for filtering, `-x` for fail-fast, `--tb=short` for output control
5. Work with pytest-xdist for parallelism out of the box

The key insight from the DeepEval ecosystem: **pytest is the interface most Python developers already know.** Making AgentCI a pytest plugin removes an entire category of onboarding friction — users don't learn new CLI commands, they just run `pytest`.

**Refined recommendation:** Ship as `pytest-agentci` on PyPI. Entry point: `pytest11` in `pyproject.toml`. Minimal implementation:

```python
# pytest_agentci/plugin.py
def pytest_collect_file(parent, file_path):
    if file_path.name == "agentci_spec.yaml":
        return AgentCIFile.from_parent(parent, path=file_path)

class AgentCIFile(pytest.File):
    def collect(self):
        spec = load_spec(self.path)
        for i, query in enumerate(spec.queries):
            yield AgentCIItem.from_parent(self, name=query.query[:60], spec=spec, query=query)

class AgentCIItem(pytest.Item):
    def runtest(self):
        result = evaluate_query(self.query, trace, baseline, self.spec)
        if result.hard_fail:
            raise AgentCITestFailure(result)
    
    def repr_failure(self, excinfo):
        return f"Correctness FAIL: {excinfo.value.result.correctness.messages}"
```

**Priority: P0 — ship with launch. This is how most developers will actually interact with AgentCI.**

---

## Additional Proposals (Research-Driven)

Beyond the five proposals in the attached document, the competitive research reveals six additional high-impact UX improvements that no competitor has fully nailed in the open-source CLI space.

### Proposal 6: Watch Mode with Eval Caching

**Source:** promptfoo's "live reload" is their most-cited developer experience feature. Their docs highlight it as the primary reason developers prefer promptfoo over competitors.

**The problem:** Every `agentci test` run re-executes all queries against the live agent, which is slow (LLM API calls) and expensive (tokens). During iterative prompt development, you want instant feedback on changes.

**Implementation:**

- `agentci test --watch` monitors `agentci_spec.yaml` and the agent source files for changes
- On change, re-run only the affected queries (spec change = re-run changed queries; agent source change = re-run all)
- Cache traces locally in `.agentci/cache/` keyed by `hash(query + agent_source_hash)`
- `agentci test --cache` reuses cached traces for queries where neither the spec nor the source changed
- DeepEval's `-c` flag does exactly this: "If you're running 1000 test cases and encounter an error on the 999th, the cache lets you skip all previously evaluated 999 test cases"

**Why this matters:** The edit-test-debug loop is where developers spend 80% of their time. Watch mode + caching turns a 2-minute feedback cycle into a 5-second one for unchanged queries.

**Priority: P1 — ship in first update.**

### Proposal 7: `agentci eval` — Run Without Baselines

**Source:** Every competitor (promptfoo, DeepEval, Langfuse, Braintrust) allows running evaluations without a pre-existing baseline. AgentCI currently requires baselines for diff operations, which creates a chicken-and-egg problem for new users.

**The problem:** A developer who just wants to check if their agent's answers are correct shouldn't need to save baselines first. The current workflow forces: write spec → run agent → save baseline → run test against baseline. For correctness-only assertions (string matching, LLM judge), no baseline is needed.

**Implementation:**

- `agentci eval` runs the spec's correctness + absolute cost assertions (e.g., `max_total_tokens`, `max_cost_usd`) without loading baselines
- Relative assertions that require baselines (`max_cost_multiplier`, `min_sequence_similarity`) are skipped with a `SKIP` status and a message: "Requires baseline — run `agentci save` first"
- This gives new users an immediate working evaluation with zero setup beyond writing the spec
- `agentci test` remains the full pipeline (with baselines, diffs, all three layers)

**Why this matters:** Eliminates the chicken-and-egg problem. A developer can go from `pip install agentci` to seeing their first pass/fail evaluation in under 5 minutes, without ever touching baselines.

**Priority: P0 — ship with launch.**

### Proposal 8: `agentci promote` — Production Traces to Test Cases

**Source:** Braintrust's signature feature is "production traces become eval cases with one click." Langfuse's "Add to dataset" from production traces is their most recommended workflow. This is the pattern that turns operational monitoring into regression testing.

**The problem:** The best test cases come from production failures, not from a developer's imagination. Currently, there's no way to take a trace from a live agent run and add it to the AgentCI spec.

**Implementation:**

- `agentci promote <trace-file> --agent rag-agent` takes any trace JSON (from AgentCI's own tracing, or imported from LangSmith/Langfuse/OpenTelemetry) and:
  1. Extracts the query, tool sequence, answer, token count, latency
  2. Runs the LLM judge to generate a correctness rubric from the query
  3. Appends the query to `agentci_spec.yaml` with auto-inferred assertions
  4. Saves the trace as a versioned baseline
- `agentci promote --from-langfuse <trace-id>` and `--from-langsmith <trace-id>` for direct platform imports (future)

**Why this matters:** Closes the "observability → testing" loop that Braintrust charges $249/month for. AgentCI becomes the open-source alternative for turning production insights into regression tests.

**Priority: P1 — ship in first major update.**

### Proposal 9: `--dry-run` for Cost Estimation

**Source:** promptfoo shows "estimated probe count and runtime" before executing scans. DeepEval's cache flag exists specifically because evaluation costs can be significant.

**The problem:** Running `agentci test` against 20 queries with LLM judge ensembles could cost $5-15 in API calls. Developers need to know the cost before committing.

**Implementation:**

- `agentci test --dry-run` parses the spec, counts queries, estimates:
  - Number of agent runs required
  - Number of LLM judge calls (based on correctness specs × ensemble config)
  - Estimated token count (based on baseline traces or conservative defaults)
  - Estimated cost in USD (based on model pricing)
- Prints a summary: "This evaluation will run 20 queries, make ~60 LLM judge calls, use ~150K tokens, and cost approximately $0.45. Proceed? [Y/n]"
- `agentci test --budget 1.00` sets a hard cost ceiling — evaluation halts if the running total exceeds the budget

**Why this matters:** Cost anxiety is a real adoption barrier for developer tools that make LLM API calls. Making costs transparent and controllable before execution builds trust.

**Priority: P1.**

### Proposal 10: Structured Error Messages with Fix Suggestions

**Source:** The Rust compiler's error messages are the gold standard for developer-facing diagnostics: they show the error, the context, and a concrete suggestion for fixing it. The Evil Martians study specifically identifies "clear, actionable errors" as a key onboarding accelerator.

**The problem:** When an `agentci_spec.yaml` has a validation error, Pydantic dumps a raw error stack. When a test fails, the output shows what failed but not always why or what to do about it.

**Implementation:** Wrap every user-facing error with:

1. **What happened** (the error)
2. **Where it happened** (file + line number in the YAML)
3. **Why it matters** (which layer is affected)
4. **How to fix it** (concrete suggestion)

```
error: Unknown field 'notes' in query #3
  --> agentci_spec.yaml:42
   |
42 |     notes: "This tests billing routing"
   |     ^^^^^ 'notes' is not a valid GoldenQuery field
   |
  help: Did you mean 'description'? This field accepts a human-readable test description.
  hint: Run 'agentci validate agentci_spec.yaml' to check for other issues.
```

For test failures:

```
FAIL: "What's the weather in Tokyo?"
  Correctness → not_in_answer violated
  
  The answer contains forbidden term "degrees":
    "...The weather in Tokyo is currently 15 degrees Celsius..."
                                            ^^^^^^^
  
  This means the agent is answering out-of-scope queries instead of declining.
  
  Suggested fix: Add a system prompt check that declines weather queries.
  Related: See demos/rag-agent/FIXED_PROMPT.md for an example.
```

**Why this matters:** Every error message is a teaching moment. Good error messages are the difference between "I'm stuck" and "I know exactly what to do next."

**Priority: P0 — weave into launch quality.**

### Proposal 11: MCP Server for IDE Integration

**Source:** promptfoo now ships an MCP server (`promptfoo mcp`) that exposes evaluation tools to AI agents and development environments. This allows Claude, Cursor, and other AI coding assistants to run evaluations directly.

**The problem:** As AI-assisted development becomes standard, AgentCI should be accessible from within AI coding environments, not just the terminal.

**Implementation:**

- `agentci mcp` starts a local MCP server exposing tools:
  - `validate_spec(path)` — validate a spec file
  - `run_evaluation(path, tags?)` — run evaluation and return structured results
  - `explain_failure(result)` — use LLM to explain why a test failed
  - `suggest_fix(spec, trace)` — suggest spec corrections based on trace data
- This lets Claude Code or Cursor's agent run `agentci test` inside the IDE and interpret results natively
- Future: `agentci` as a VS Code extension using the MCP server as the backend

**Why this matters:** The frontier of developer tools is IDE integration via AI agents. By shipping an MCP server early, AgentCI positions itself as AI-native tooling that works with the developer's existing workflow — whether that's a terminal, pytest, or an AI coding assistant.

**Priority: P2 — ship after core DX is solid.**

---

## Prioritized Roadmap Summary

### Launch (P0) — Ship with v2.0 public release

| # | Feature | Impact | Effort |
|---|---------|--------|--------|
| 1 | Enhanced `agentci init` (generates spec + interactive runner prompt) | Eliminates blank-page problem | 1 day |
| 2 | `agentci bootstrap` (run queries → approve → auto-generate spec + baselines) | "A-ha moment" in 2 minutes | 2 days |
| 5 | `pytest-agentci` plugin (auto-discover YAML, generate pytest items) | Meet developers where they are | 1-2 days |
| 7 | `agentci eval` (run without baselines, correctness-only mode) | Eliminates chicken-and-egg problem | 0.5 days |
| 10 | Structured error messages with fix suggestions | Every error teaches | 1 day |
| 4a | Console trace summary on failure (rich, condensed) | 90% debugging value, minimal effort | 0.5 days |

**Total P0 effort: ~6-7 days**

### First Update (P1) — Ship within 2 weeks of launch

| # | Feature | Impact | Effort |
|---|---------|--------|--------|
| 6 | Watch mode + eval caching | 10x faster edit-test loop | 2 days |
| 3 | `agentci doctor` (LLM-assisted spec repair) | Self-healing spec maintenance | 1 day |
| 8 | `agentci promote` (trace → test case pipeline) | Close observability → testing loop | 2 days |
| 9 | `--dry-run` + `--budget` for cost control | Build trust with transparent costs | 1 day |

### Future (P2)

| # | Feature | Impact | Effort |
|---|---------|--------|--------|
| 4b | `agentci view` local HTML report (browser-based trace viewer) | Full trace debugging | 3 days |
| 11 | MCP server for IDE integration | AI-native tooling | 2 days |
| — | `agentci generate` (LLM auto-generates test cases from agent source code) | Zero-effort spec creation | 3 days |
| — | OpenTelemetry trace import (Langfuse, Arize, LangSmith compat) | Ecosystem interop | 3 days |

---

## Strategic Positioning

The research reveals a clear pattern in the market: **observability platforms (Langfuse, Braintrust, LangSmith) are adding evaluation features, and evaluation frameworks (DeepEval, promptfoo) are adding observability.** Everyone is converging toward the same "observe → evaluate → improve" loop.

AgentCI's differentiation is not trying to be a full-stack platform. Instead, it occupies a specific niche that nobody else fills:

**AgentCI is the open-source, CI/CD-native, declarative regression testing framework for AI agents.**

The UX improvements above reinforce this positioning:

1. **pytest plugin** → meets developers in their existing test infrastructure (not another dashboard)
2. **GitHub annotations** → CI/CD native (not a separate web UI you have to check)
3. **YAML specs** → version-controlled, reviewable in PRs (not stored in a platform)
4. **`agentci promote`** → interoperates with observability platforms rather than replacing them
5. **MCP server** → works inside the developer's AI coding environment

The goal is not to compete with Braintrust's dashboard or Langfuse's tracing. The goal is to be the **testing layer** that sits alongside any observability platform — the pytest of the AI agent ecosystem, just as DeepEval claims to be for LLMs but with trajectory evaluation, layered severity, and true CI/CD integration that DeepEval lacks.