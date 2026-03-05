# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
