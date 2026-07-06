# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — stability report hardening (pre-launch fixes from eng review)
- **Flip attribution now compares per-layer sub-verdicts first**: if every
  deterministic check returned the same outcome across runs and only the LLM judge's
  verdict changed, the flip is `judge-flake` even when the answer text was
  paraphrased (previously mislabeled `agent-variance`)
- **New flip source `infra-error`**: a judge API failure counted as a fail no longer
  reads as `judge-flake` — one transient hiccup must not say "fix your rubric"
- **Console shows observed facts only** (verdict history + pass rate); pass@k /
  pass^k move to JSON output as clearly labeled estimates (at small k they restate
  the pass rate)
- Duplicate query texts in a spec are flagged (they merge into one stability
  record); queries missing from some runs are marked `partial (k/N runs)`
- `agentci run` (legacy suite runner) now prints a deprecation warning pointing at
  `agentci test`; removal planned for 0.9.0
- Removed stale `src/agentci/_version.py` (said 0.1.0; version comes from package
  metadata)

## [0.7.0] - 2026-07-05

### Added

#### KB-Derived Fact Checks — `agentci generate-checks`
Mines the knowledge base for hard facts (prices, rates, SKUs, versions,
explicit quantities) and proposes them as deterministic assertions on
existing spec queries. One LLM call at authoring time; the checks run
deterministically forever at zero cost.

- **Brittleness gate**: every candidate is validated against recorded golden
  answers before it is offered — a check that would fail a known-good answer
  is rejected automatically with the failing answer shown
- Only non-paraphrasable facts; prose facts become variant sets
  (`any_expected_in_answer`), never single literal strings; only
  `any_expected_in_answer` / `not_in_answer` / `regex_match` are proposed
- Candidates without a golden answer are `unvalidated` and never
  auto-accepted, even with `--yes`; interactive review for everything else
- Merge never overwrites user-written assertions; `.bak` backup on write;
  `--dry-run` mode
- New module `agentci.engine.check_generator`; 21 new tests

#### Judge Audit — `agentci judge-audit`
Meta-evaluation of the LLM judge against ground truth you already have,
re-scoring recorded golden baselines (the agent is never re-run):

- **Mode 1 — judge vs. deterministic checks**: independent verdicts on the
  same recorded answer; reports the disagreement matrix. The killer row:
  answers the judge PASSED that a deterministic fact-check FAILED
  (shared-context judge blindness, detected automatically)
- **Mode 2 — retest stability** (`--repeats K`, default 3): same answer
  judged K times; verdict flips on identical input are the judge's noise floor
- **Mode 3 — hand labels** (`--labels FILE`): agreement + Cohen's κ against
  human review, with standard trust thresholds
- Scoped claim stated in the report itself: measured on fact-checkable
  queries; one-directional (disqualifying) signal for judgment-only queries
- Verdicts: TRUSTWORTHY / NEEDS CALIBRATION / UNRELIABLE (exit 1) /
  ERROR when the judge never ran (exit 2 — a judge that couldn't run is
  never scored)
- `--sample N` cost cap; console + JSON output
- New module `agentci.engine.judge_audit`; 27 new tests

#### Stability Report — `agentci test --runs N`
A stable suite score can hide per-query verdict flips: the aggregate holds
because the errors move around. `--runs N` executes the whole suite N times
and reports what a single run cannot show:

- Per-query verdict history (✅❌✅), pass rate, pass@k and pass^k estimates
- Suite score per run printed side by side with the queries that flipped
- **Flip-source attribution** — every flip is labelled:
  - `agent-variance`: the agent's output changed → fix the agent
  - `judge-flake`: same output, the LLM judge changed its verdict → fix the eval
  - `mixed`: near-identical paraphrase with a judge configured — not guessed
  Attribution rests on a structural fact: deterministic checks cannot flip on
  identical output, so identical answer + tools + flipped verdict = judge.
- Exit semantics: flaky-but-passing exits 0 (warnings only); queries failing
  in EVERY run exit 1; `--fail-on-flaky` escalates flips to exit 1
- Works in every format: console section, GitHub `::warning` annotations,
  `stability` block in JSON, stability card in the HTML report
- Mock mode support: `AGENTCI_MOCK_FLAKY=1 agentci test --mock --runs 3`
  demonstrates the report with zero API keys
- New module `agentci.engine.stability` (`build_stability_report`,
  `StabilityReport`, `QueryStability`, `FlipSource`); 21 new tests

### Changed
- **`expected_tools` now asserts by default**: a missing expected tool produces a WARN
  (tool recall gates at 1.0 unless `min_tool_recall` explicitly loosens it). Previously,
  without `min_tool_recall`, a recall of 0.0 displayed as PASS with a checkmark.
- **`expected_tools: []` now asserts that no tools are called**: an explicit empty list
  produces a WARN if the agent called any tool. Previously it was silently skipped.
- `agentci doctor` no longer reports numpy as a required dependency (it was never used
  by AgentCI, so every fresh install showed a false failure).

### Docs
- README: fixed the quickstart spec example (was missing the required `agent:` field),
  added CI badge, "Check facts in code" section, docs index
- Rewrote `docs/quickstart.md`, `docs/ci-cd.md`, `docs/cost-tracking.md`, and
  `docs/golden-traces.md` to match the current `agentci test` workflow
- Fixed dead clone URLs in CONTRIBUTING.md and quickstart
- Removed unused demo GIF variants and cast recordings (6 files, ~2 MB)

## [0.6.0] - 2026-03-05

### Added

#### `final_output` Auto-Capture
- `TraceContext._auto_extract_final_output()` called in `__exit__()` — automatically extracts the agent's answer from traces
- Extraction priority: LangGraph state messages > span `output_data` (string) > span `output_data` (dict with `content`/`message`/`text`/`output` keys) > last LLM call `output_text`
- Manual `trace.metadata["final_output"]` still takes precedence (no overwrite)
- LangGraph adapter: auto-sets `final_output` from last AI message in `parse_state()`
- OpenAI Agents adapter: auto-sets `final_output` from last span output in `on_trace_end()`

#### `agentci calibrate` Command
- Runs N sample queries against the live agent, measures actual metrics, shows Rich comparison table
- Updates spec budgets with headroom: +50% for LLM/tool calls, +100% for tokens/cost
- Flags: `--samples N` (default 2), `--dry-run`, `--yes`, `--spec PATH`

#### Strict Tool Sequence Assertions
- `PathSpec.expected_tool_sequence: Optional[list[str]]` — strict ordered tool call check
- Mismatch = WARN (soft warning) with position-level diff via `_format_sequence_diff()`

#### HTML Trace Report
- Self-contained `report.html.j2` Jinja2 template with dark theme
- Summary dashboard (pass/fail/warn counts, total cost)
- Per-query cards with status badges, answer preview, three-layer details
- Collapsible trace tree with JS toggle
- Available via `agentci test --format html --output report.html` or `agentci report -i results.json`

### Changed
- `max_llm_calls` default in spec generator raised from 8 to 10 (better headroom for real agents)
- `max_llm_calls` fallback in mock runner raised from 3 to 10
- Calibrate command floor raised from 8 to 10
- `--format` choices in `test` and `eval` commands now include `html`
- `--output / -o` option added to `test` and `eval` commands for HTML file path
- `agentci report` command fully implemented (was stub) — converts JSON results to HTML
- 22 new tests added (570 total, up from 548 in v0.5.1)

## [0.5.0] - 2026-03-01

### Added

#### Three-Layer Evaluation Engine
- **Correctness layer** (Layer 1 — hard fail): keyword matching, LLM-as-a-judge, safety checks, hallucination checks, regex/exact match, JSON schema validation
- **Path layer** (Layer 2 — soft warn): tool trajectory validation, loop detection (default `max_loops=3`), routing assertions, handoff expectations
- **Cost layer** (Layer 3 — soft warn): token budgets, cost caps, LLM call limits, latency thresholds
- `runner.py` orchestrates all three layers per query
- `parallel.py` for parallel query execution across specs

#### OR-Logic Keywords
- `any_expected_in_answer` field — at least one keyword must match (complementing `expected_in_answer` which requires all)

#### LLM Judge Enhancements
- `context_file` support in `JudgeRubric` — doc-grounded judging against reference documents
- `refutes_premise` flag — injects built-in premise-correction rubric for trick questions

#### Span Assertions
- `SpanAssertionSpec` schema model for span-level assertions
- `Span.attributes: dict[str, Any]` — OTel-style span-level data propagation
- Span-level LLM judge support

#### Mock Testing Mode
- `agentci test --mock` — generates synthetic traces, zero API cost
- `mock_runner.py` — synthetic trace generation from spec expectations
- `--golden-file` flag — load Q&A pairs from JSON/CSV for mock mode

#### Cost Estimator
- `cost_estimator.py` — pre-execution cost estimates with pricing table
- Cost estimate shown before live test runs; `--yes`/`-y` skips confirmation

#### CLI Improvements
- `agentci init --generate` — AI-assisted spec generation with guided interview
- `agentci doctor` — health-check command (spec, runner, API keys, deps, CI)
- Scan-first flow: auto-scan project before questions, show summary
- Agent type auto-detected via `_detect_agent_type_from_code()`
- Skeleton template generation with TODO placeholders for zero-API-key usage
- Context-aware "Next Steps" based on mode (mock vs live)
- Non-interactive flags: `--kb-path`, `--mode`

#### Diff Engine
- Three-tier diff engine for baseline comparison
- 11 `DiffType` categories including `ROUTING_CHANGED`, `GUARDRAILS_CHANGED`
- `agentci diff` CLI command

#### Reporting
- GitHub annotations with budget cap (`MAX_INLINE_ANNOTATIONS = 10` for warnings; errors uncapped)
- JSON output format
- Prometheus metrics export

#### Trace Helpers
- `Trace.called(tool)` / `never_called(tool)` / `loop_count(tool)` — readable assertion helpers
- `Trace.cost_under(usd)` / `llm_calls_under(n)` — budget assertion helpers
- `langgraph_trace(agent_name)` — context manager shortcut for LangGraph
- `TraceContext.attach(state)` — alias for `attach_langgraph_state()`

#### Adapters
- OpenAI Agents SDK adapter (`openai_agents.py`)
- LangGraph adapter (`langgraph.py`)

#### Other
- `python-dotenv` added as core dependency
- Deep KB sampling: 2000 chars/file for spec generation
- Progressive spec building: smoke queries (3) then full queries (10-12)
- Pytest plugin entry point (`pytest11: agentci`)
- GUARDRAIL span type, HANDOFF span type

### Changed
- `PathSpec.max_loops` now defaults to `3` (was `None`)
- Development status upgraded from Alpha to Beta
- Package version bumped to 0.5.0

## [0.4.1] - 2026-02-20

### Added
- Initial project structure and core models
- Basic trace capture and assertion framework
- CLI scaffolding with `agentci init`
- PyPI publishing as `ciagent`
