 
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 AgentCI v2 Implementation Plan — Phase 1 (A+B)                                                                                                                
                                                        
 Context

 AgentCI v1 uses a flat assertion/diff system (assertions.py with 12 assertion types, diff_engine.py with 11 diff categories). v2 transitions to a declarative
  YAML spec with a three-layer evaluation engine (Correctness → Path → Cost), GitHub annotations, and baseline versioning. This is AgentCI's key
 differentiator: no existing framework combines YAML specs + trajectory evaluation + CI-native severity-layered feedback.

 Branch strategy: Create v2 branch from clean main. Keep existing v1 modules (assertions.py, diff_engine.py, runner.py) intact for backward compatibility. New
  engine runs as a parallel code path. Old run command preserved; new test command uses v2 engine.

 Package location: All code in /Users/sunilpandey/startup/github/Agents/AgentCI/src/agentci/

 ---
 Step 0: Branch Setup & Directory Scaffolding

 Actions:
 1. Verify main branch is clean (git status)
 2. Create and checkout v2 branch
 3. Create directories: src/agentci/schema/, src/agentci/engine/, tests/integration/

 ---
 Step 1: Schema Definition (schema/spec_models.py)

 Foundation — every other module depends on these models.

 Create:
 - src/agentci/schema/__init__.py (~5 lines) — exports all spec models
 - src/agentci/schema/spec_models.py (~150 lines) — Pydantic models:
   - MatchMode(str, Enum) — strict, unordered, subset, superset
   - JudgeRubric(BaseModel) — rule, scale, threshold (0-1), few_shot_examples
   - CorrectnessSpec(BaseModel) — expected_in_answer, not_in_answer, exact_match, regex_match, json_schema, llm_judge, safety_check, hallucination_check
   - PathSpec(BaseModel) — max_tool_calls, expected_tools, forbidden_tools, max_loops, match_mode, min_tool_recall, min_tool_precision,
 min_sequence_similarity, expected_handoff, expected_handoffs_available, max_handoff_count
   - CostSpec(BaseModel) — max_cost_multiplier, max_total_tokens, max_llm_calls, max_latency_ms, max_cost_usd
   - GoldenQuery(BaseModel) — query (validated non-empty), description, tags, correctness, path, cost
   - AgentCISpec(BaseModel) — version, agent, baseline_dir, defaults, judge_config, queries (min_length=1)
 - src/agentci/schema/generate_schema.py (~15 lines) — generates agentci_spec.schema.json from Pydantic
 - src/agentci/schema/agentci_spec.schema.json — auto-generated

 Test: tests/test_schema_validation.py (~200 lines, 15+ tests)
 - Valid minimal/full specs pass
 - Empty query, missing agent, missing queries, empty queries list, invalid match_mode, threshold out of range, negative max_tool_calls all fail
 - Nested correctness + path + cost on single query works

 ---
 Step 2: Engine Results Model (engine/results.py)

 Create:
 - src/agentci/engine/__init__.py (~5 lines)
 - src/agentci/engine/results.py (~55 lines):
   - LayerStatus(str, Enum) — PASS, FAIL, WARN, SKIP
   - LayerResult (dataclass) — status, details (dict), messages (list[str])
   - QueryResult (dataclass) — query, correctness, path, cost + properties hard_fail, has_warnings

 Test: Tested implicitly through engine tests.

 ---
 Step 3: Shared Metrics Module (engine/metrics.py)

 Pure functions, zero agentci dependencies. Can be built in parallel with Steps 1-2.

 Create: src/agentci/engine/metrics.py (~90 lines)
 - compute_tool_recall(expected: set, used: set) -> float
 - compute_tool_precision(expected: set, used: set) -> float
 - compute_sequence_lcs(seq_a: list, seq_b: list) -> float — normalized LCS: 2×|LCS|/(|A|+|B|)
 - compute_edit_distance_similarity(seq_a: list, seq_b: list) -> float — 1 - ED/max(|A|,|B|)
 - detect_loops(tool_sequence: list) -> int — count consecutive repeated tool invocations

 Test: tests/test_metrics.py (~150 lines)
 - Edge cases per function: empty inputs, disjoint, full overlap, partial overlap, single element, reversed sequences

 ---
 Step 4: LLM Judge Module (engine/judge.py)

 Depends on Step 1 (JudgeRubric model).

 Create: src/agentci/engine/judge.py (~150 lines)
 - JudgeVerdict(BaseModel) — score (1-5), label (pass/fail/borderline), rationale
 - run_judge(answer, rubric, config, context) -> dict — returns {passed, score, label, rationale, model}
 - _run_ensemble(system, user, config, rubric) -> dict — majority vote across models
 - _call_judge(model, system, user, temperature) -> JudgeVerdict — calls Anthropic/OpenAI based on model name prefix
 - _build_judge_system_prompt(rubric) -> str — rubric-driven prompt with scale anchors + few-shot
 - _build_judge_user_prompt(answer, rubric, context) -> str
 - _score_threshold(threshold: float) -> int — converts 0-1 to 1-5 scale

 Design: temp=0 always, structured JSON output, deterministic checks gate judge calls

 Test: tests/test_judge.py (~150 lines)
 - Threshold conversion, prompt building, mocked judge pass/fail, ensemble voting logic

 Live test: tests/integration/test_judge_live.py (~60 lines)
 - Gated by AGENTCI_LIVE_TESTS=1 env var, real API call with simple rubric

 ---
 Step 5: Correctness Engine — Layer 1 (engine/correctness.py)

 Depends on Steps 1, 2, 4. Hard pass/fail — failures block CI.

 Create: src/agentci/engine/correctness.py (~100 lines)
 - evaluate_correctness(answer: str, spec: CorrectnessSpec, trace: dict, judge_config: dict | None) -> LayerResult
 - Eval order (cost optimization): expected_in_answer → not_in_answer → exact_match → regex_match → json_schema → llm_judge (skipped if already failed) →
 safety_check → hallucination_check

 Test: tests/test_correctness_engine.py (~200 lines)
 - Each check type: pass and fail cases
 - Short-circuit: deterministic fail skips judge (verify mock not called)
 - Case-insensitive string matching

 ---
 Step 6: Path Engine — Layer 2 (engine/path.py)

 Depends on Steps 1, 2, 3. Soft warnings, except forbidden_tools → hard FAIL.

 Create: src/agentci/engine/path.py (~130 lines)
 - evaluate_path(trace: Trace, spec: PathSpec, baseline_trace: Trace | None) -> LayerResult
 - evaluate_match_mode(used_tools: list, reference_tools: list, mode: MatchMode) -> dict
 - Checks: max_tool_calls, forbidden_tools (→ FAIL), tool recall/precision, sequence LCS, loop detection, match modes, handoff assertions
 - Uses trace.tool_call_sequence (existing property on models.Trace)
 - Uses trace.get_handoffs() (existing method returning list[Span] with .to_agent)

 Test: tests/test_path_engine.py (~250 lines)
 - Each check: pass, warn, fail (forbidden_tools)
 - All 4 match modes
 - Build Trace objects using existing Trace/Span/ToolCall Pydantic models

 ---
 Step 7: Cost Engine — Layer 3 (engine/cost.py)

 Depends on Steps 1, 2. Soft warnings only.

 Create: src/agentci/engine/cost.py (~70 lines)
 - evaluate_cost(trace: Trace, spec: CostSpec, baseline_trace: Trace | None) -> LayerResult
 - Maps: trace.total_cost_usd, trace.total_tokens, trace.total_llm_calls, trace.total_duration_ms

 Test: tests/test_cost_engine.py (~120 lines)
 - Each bound: under (pass), over (warn), no baseline for multiplier (skip)

 ---
 Step 8: YAML Loader (loader.py)

 Depends on Step 1. Replaces config.py for v2 specs (old config.py kept for v1 compat).

 Create: src/agentci/loader.py (~65 lines)
 - load_spec(spec_path: str | Path) -> AgentCISpec
 - filter_by_tags(spec: AgentCISpec, tags: list[str]) -> AgentCISpec
 - _merge_defaults(query: GoldenQuery, defaults: dict) -> GoldenQuery
 - _deep_merge(base: dict, override: dict) -> dict

 Test: tests/test_yaml_loader.py (~150 lines)
 - Load valid YAML, file not found, invalid content
 - Defaults merging: applied, overridden, deep nested
 - Tag filtering: match, no match, empty tags

 ---
 Step 9: Engine Runner (Orchestrator) (engine/runner.py)

 Depends on Steps 5, 6, 7, 8. Wires all three layers.

 Create: src/agentci/engine/runner.py (~100 lines)
 - evaluate_query(query: GoldenQuery, trace: Trace, baseline_trace: Trace | None, judge_config: dict | None) -> QueryResult
 - evaluate_spec(spec: AgentCISpec, traces: dict[str, Trace], baselines: dict[str, Trace] | None) -> list[QueryResult]
 - _extract_answer(trace: Trace) -> str — gets final answer from last span's output_data

 Test: Integration-style tests using real engine modules with mocked judge.

 ---
 Step 10: Reporter (engine/reporter.py)

 Depends on Step 2. Output layer with GitHub annotations.

 Create: src/agentci/engine/reporter.py (~150 lines)
 - report_results(results: list[QueryResult], format: str, spec_file: str) -> int — returns exit code
 - Exit codes: 0=pass, 1=correctness fail, 2=infra error. Warnings → annotations only, not exit code.
 - _emit_console(results) — rich formatted output
 - _emit_github_annotations(results, spec_file) — ::error file=...:: for correctness, ::warning file=...:: for path/cost
 - _emit_json(results) — structured JSON with summary + per-result details
 - _emit_prometheus(results) — gauge lines for Grafana
 - _is_github_actions() -> bool — auto-detect via GITHUB_ACTIONS env var

 Test: tests/test_reporter.py (~150 lines)
 - Exit code mapping, annotation format (capsys), JSON structure, GitHub env detection

 ---
 Step 11: Baselines Module (baselines.py)

 Depends on Steps 1, 5.

 Create: src/agentci/baselines.py (~120 lines)
 - save_baseline(trace, agent, version, spec, baseline_dir, force, query_text) -> Path
   - Precheck: runs evaluate_correctness before saving (unless force=True)
   - Metadata: model, spec_hash (SHA256), capture time, precheck_passed
 - load_baseline(agent, version, baseline_dir) -> dict
 - list_baselines(agent, baseline_dir) -> list[dict]

 Test: tests/test_baselines.py (~150 lines)
 - Save/load round-trip, precheck pass/fail, force bypass, list versions

 ---
 Step 12: Update Existing Files

 src/agentci/exceptions.py — Add new exception classes

 - JudgeError(AgentCIError) — LLM judge failures
 - SchemaError(AgentCIError) — YAML validation failures
 - EngineError(AgentCIError) — evaluation engine errors

 src/agentci/cli.py — Add new commands (~450 lines total)

 - agentci validate <spec_path> — validate YAML spec, exit 0/1
 - agentci test --config --tags --format — new v2 evaluation pipeline
 - agentci save --agent --version --config --force-save — save versioned baseline
 - agentci diff --baseline --compare --agent --config — three-tier diff
 - agentci baselines --agent --config — list baseline versions
 - Keep existing run and record commands for backward compat (deprecated)

 src/agentci/__init__.py — Add v2 exports (~30 lines)

 # v2 exports (alongside existing v1 exports)
 from .schema.spec_models import AgentCISpec, GoldenQuery
 from .loader import load_spec
 from .engine.runner import evaluate_query, evaluate_spec
 from .engine.results import QueryResult, LayerResult, LayerStatus

 src/agentci/pytest_plugin.py — Add v2 fixture path

 - New agentci_trace behavior: uses evaluate_query when _agentci_v2_config present
 - Old path preserved for backward compat

 pyproject.toml

 - Add jsonschema>=4.0 to optional dependencies
 - Bump version to 0.2.0
 - Add agentci.schema and agentci.engine to packages

 ---
 Step 13: Documentation & Demo

 Create:
 - docs/sample_spec.yaml (~80 lines) — fully commented reference spec
 - docs/metrics_reference.md (~100 lines) — formulas for recall, precision, LCS, edit distance, cost multiplier

 Create in DemoAgents:
 - DemoAgents/examples/rag-agent/agentci_spec.yaml (~60 lines) — 3 queries (install, weather, AWS)
 - DemoAgents/examples/rag-agent/baselines/ directory — for versioned baselines

 ---
 Build Order (Optimized for Incremental Testability)

 Step 0:  Branch + directories
     ├── Step 1:  Schema (spec_models.py) + tests     ← RUN TESTS
     ├── Step 2:  Results model (results.py)
     ├── Step 3:  Metrics (metrics.py) + tests         ← RUN TESTS
     │
     ├── Step 4:  Judge (judge.py) + tests             ← RUN TESTS
     ├── Step 5:  Correctness engine + tests           ← RUN TESTS
     ├── Step 6:  Path engine + tests                  ← RUN TESTS
     ├── Step 7:  Cost engine + tests                  ← RUN TESTS
     │
     ├── Step 8:  YAML loader + tests                  ← RUN TESTS
     ├── Step 9:  Engine runner + tests                ← RUN TESTS
     ├── Step 10: Reporter + tests                     ← RUN TESTS
     ├── Step 11: Baselines + tests                    ← RUN TESTS
     │
     ├── Step 12: Update existing files (exceptions, cli, init, pytest_plugin, pyproject)
     └── Step 13: Docs + RAG demo spec                 ← FULL TEST SUITE

 ---
 Backward Compatibility

 Existing DemoAgent tests require ZERO changes. They import:
 - agentci.capture.TraceContext — unchanged
 - agentci.models.Trace/Span/ToolCall/Assertion/SpanKind/DiffType — unchanged
 - agentci.mocks.OpenAIMocker/AnthropicMocker — unchanged
 - agentci.assertions.evaluate_assertion — unchanged (file kept)
 - agentci.diff_engine.diff/diff_traces/load_baseline — unchanged (file kept)
 - agentci.adapters.* — unchanged

 Old assertions.py, diff_engine.py, runner.py, config.py stay in the package but are not used by the new v2 CLI commands.

 ---
 Verification Plan

 1. Per-step: Run conda activate agentci && python -m pytest tests/test_<step>.py -v after each step
 2. After Step 11: Run full new test suite: python -m pytest tests/ -v --ignore=tests/integration
 3. After Step 12: Run existing tests to verify no breakage: python -m pytest tests/ -v
 4. Live judge test: AGENTCI_LIVE_TESTS=1 python -m pytest tests/integration/test_judge_live.py -v
 5. CLI smoke test: agentci validate docs/sample_spec.yaml
 6. Existing DemoAgent tests: cd DemoAgents/examples/rag-agent && python -m pytest tests/ -v

 ---
 File Summary

 ┌───────────────────┬───────────┬──────────────┐
 │     Category      │ New Files │    ~Lines    │
 ├───────────────────┼───────────┼──────────────┤
 │ Schema            │ 4         │ 170          │
 ├───────────────────┼───────────┼──────────────┤
 │ Engine            │ 7         │ 600          │
 ├───────────────────┼───────────┼──────────────┤
 │ Judge             │ 1         │ 150          │
 ├───────────────────┼───────────┼──────────────┤
 │ Loader            │ 1         │ 65           │
 ├───────────────────┼───────────┼──────────────┤
 │ Reporter          │ 1         │ 150          │
 ├───────────────────┼───────────┼──────────────┤
 │ Baselines         │ 1         │ 120          │
 ├───────────────────┼───────────┼──────────────┤
 │ Unit Tests        │ 9         │ 1,520        │
 ├───────────────────┼───────────┼──────────────┤
 │ Integration Tests │ 1         │ 60           │
 ├───────────────────┼───────────┼──────────────┤
 │ Modified Existing │ 5         │ ~120 net new │
 ├───────────────────┼───────────┼──────────────┤
 │ Docs + Demo       │ 3         │ 240          │
 ├───────────────────┼───────────┼──────────────┤
 │ Total             │ 33 files  │ ~3,195 lines │
 └───────────────────┴───────────┴──────────────┘