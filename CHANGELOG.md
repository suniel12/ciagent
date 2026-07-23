# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Agent Failure Atlas (seed) + a safety-gap fix it surfaced
- New `src/ciagent/examples/failure-atlas/`: runnable, OWASP-mapped agent
  failure patterns, each a toy vulnerable agent (no LLM, no API key) that
  fails a deterministic gate. Seed: money-out-no-verification (LLM06
  Excessive Agency), transcript-poisoning (guardrail design, from the
  50-conversation study), tool-output-injection (LLM01, multi-step recipe).
  A conformance test runs every entry live and asserts its gate fires, so
  the atlas can't rot into prose. Ships in the wheel
- Fixed a real safety gap the atlas dogfooding surfaced: a `forbidden_tools`
  violation printed "PATH: FAIL" but the run exited 0 — QueryResult.hard_fail
  read correctness alone and ignored the path layer's only hard-fail state
  (forbidden tools). A documented safety boundary now actually gates the
  build. Regression-tested

### Added — source-aware flaky gating (lean into the wedge)
- New `--flaky-sources` on `ciagent test`: gate only on selected flip
  sources (e.g. `--flaky-sources=agent` fails on agent-variance /
  retrieval-variance but tolerates judge-flake). Attribution becomes an
  ACTION, not just a report — the thing single-run eval tools can't do
  (they suppress flakiness; they don't know why a verdict flipped). Aliases
  real/agent/judge/infra/sim derive from the promotion classifier so they
  can't drift; bare `--fail-on-flaky` is unchanged (gates any flip)
- The stability JSON gains suite-level `flip_sources` counts and `gated_by`;
  the console flip-source summary now shows all seven sources (retrieval-
  variance previously hid inside "mixed"). MCP `ciagent_test` forwards
  `fail_on_flaky`/`flaky_sources`
- v1 is `test`-only (simulate's flip gating is entangled with its
  lifecycle/world-miss exit fold — separate future work). Design:
  Plan_docs/flip_attribution_deepening.md

### Added — published world-file format + JSON Schema
- The world-file format (the frozen-tool-state artifact with no standard
  equivalent) is now documented at docs/world-file-schema.md and published
  as a machine-readable JSON Schema (2020-12) shipped in the package:
  `from ciagent.world import world_file_schema`. A conformance test enforces
  that every world the suite produces (freeze + mutate output) AND a minimal
  hand-authored world validate, and that malformed worlds are rejected — so
  the schema is real, not prose
- Framed as documentation, not a standards campaign: consume it if useful.
  Compat promise is narrow and matches the loader: unknown keys within
  world_schema 1 are ignored; a version bump is a hard incompatibility
- Design and scope decisions (ATIF import blocked on a real fixture; OTel
  export deferred as speculative + lossy): Plan_docs/standards_adoption.md

## [0.14.0] - 2026-07-23

### Added — world mutations: chaos engineering on frozen fixtures
- `ciagent world mutate <world> --op <operator>` derives a NEW world file
  (source never modified) that flows through the existing replay machinery,
  so "my agent survives a degraded or hostile backend" is a deterministic
  gate, not a robustness score. Operators: `empty`, `error`, `inject`
  (adversarial payload into every string leaf), `rewrite` (OLD=NEW),
  `truncate-sequence`, `swap`. `ciagent world operators` lists them
- Flagship: prompt injection via tool output. `inject` an override into a
  tool's frozen response, replay, and a `forbidden_tools`/`not_in_answer`
  gate fires the moment the agent obeys it — promotable as a permanent gate
  or xfail. Built-in payloads are benign-but-representative; real ones via
  --payload-file; payloads are never redacted
- Two signal channels documented: response-changing operators surface via
  check verdicts (same call args → no misses); truncate/swap are designed
  misses, xfail-only for gate lifecycles. `mutated_from` provenance is a
  first-class world field (survives round-trips, shown by `world show`)
- MCP: `ciagent_world_mutate` + `ciagent_world_operators`
- Design: Plan_docs/world_mutations.md (adversarial review, M1-M10 folded in)

### Added — MCP server: agents that gate the agents they build
- `ciagent mcp` (new `ciagent[mcp]` extra) runs a stdio MCP server exposing
  the loop to coding agents: test, simulate (incl. frozen-world replay),
  stage list/show/verify/drop, promote/flip, world freeze/show, import.
  One JSON envelope per call; exit 1 is "the gate detected a failure",
  reported as such, not as an error
- Server-enforced guardrails: live runs refused without `max_cost`
  (simulate) or `allow_live` (test/verify/import) — under MCP the CLI's
  cost confirms are all bypassed, so the server is the only speed bump;
  symlink-safe project jail for path arguments; process-group-killing
  timeouts; oversized results written to `.ciagent/mcp/` with a summary
  returned
- `python -m ciagent` now works (new `__main__.py`); the server invokes
  the CLI via the same interpreter, PATH-independent
- Design: Plan_docs/mcp_server.md (adversarial review, 13 findings folded
  in — incl. the per-command flag-capability matrix and the finding that
  4 of 11 commands report outcomes on stdout without a JSON mode)

## [0.13.0] - 2026-07-22

### Added — Simulated World MVP: world-from-failure
- `ciagent world freeze <stage-id|golden>` extracts a failing run's tool
  traffic (arguments → frozen response) into a versioned, human-editable
  world file; `ciagent world show` prints the tool surface. Envelope-level
  redaction on freeze; unredacted sources refuse without `--force-redact`;
  result-less calls refuse without `--allow-gaps` (recorded as gaps)
- `@world_tool` (innermost decorator, framework-agnostic, async-aware):
  zero-overhead passthrough normally; during `simulate --replay --world`
  it serves frozen responses instead of hitting real backends. Fail-closed:
  unmatched calls raise + record `WorldMiss` with a nearest-fixture diff
  and a ready-to-paste `ignore:` suggestion — never a guess
- Matching crosses the framework validation layer (type coercion, defaults
  filled for omitted optionals); `ignore` marks mutable fields;
  `sequence: true` encodes state transitions (FIFO consumption, set
  automatically at freeze when the same args produced different results);
  ambiguity is validated at load
- Exit semantics are lifecycle-aware: for gate goldens any recorded miss in
  any run exits 1 (the verdict was not on the frozen world); for xfail
  goldens misses suppress XPASS but never flip the exit. `--format json`
  adds per-scenario and summary `world_misses`
- `stage verify --world`: runs with world misses are excluded from
  re-classification; all-missed leaves the staging block untouched (exit 1);
  block records `verified_via: replay+world`. New `world-miss` flip source
  maps to `held`
- Scope, honestly: the world removes BACKEND variance; the model is not
  frozen. Design: Plan_docs/world_sim_mvp.md (adversarial review A1-A14
  folded in)

## [0.12.0] - 2026-07-22

### Added — capture-time redaction; staging is now ON by default
- New deterministic redactor (`ciagent.redaction`) scrubs staged
  conversations before they hit disk: known secret-key prefixes (OpenAI,
  Anthropic, AWS, GitHub, Slack, Google, Stripe), key/Bearer contexts, a
  key-aware walk (`api_key`, `token`, `password`, ... values redacted
  whatever their shape), emails, phones, Luhn-checked card numbers, plus
  spec-configured `staging.redact_patterns`. No LLM, no entropy scanning.
  Placeholders are shape-preserving and stable within an envelope, so
  replays keep behaving like the original conversation; re-redaction is a
  byte-level no-op
- BREAKING-ish default: `staging.enabled` now defaults to **true** (the
  0.11.0 changelog promised default-ON with the redaction release). Disable
  with `staging: false` or `--no-stage`. Note: a bare `staging:` key (null)
  coerces to defaults and is now enabled. The `.ciagent/staged/` gitignore
  entry is scaffolded on the first auto-stage for repos that never re-ran
  `init`
- `stage show` (console and json) and `--export` now re-redact with the
  current config on every read, retroactively covering staged files written
  by 0.11.0; the staging block records `redaction: {applied, counts}`;
  `stage verify` preserves it. `stage verify`/`promote` warn when a
  scenario's check literals reference redacted values (a `not_in_answer`
  leak-gate on a redacted literal is vacuous; use regex checks, which
  redaction does not rewrite)
- Design: Plan_docs/redaction_capture.md (adversarial architect pass folded
  in: staging block inside the walk, parse-safety guards, degraded fallback)

### Added — xfail bug-golden lifecycle
- `ciagent promote --xfail`: promote a repro as an expected-fail golden.
  Replay stays green (exit 0) while the bug reproduces, reported as XFAIL;
  the repro is banked in CI without blocking merges. When a replay suddenly
  passes it is flagged XPASS (still green, pytest non-strict semantics)
  with the exact flip command
- `ciagent promote --flip <golden>`: converts a passing xfail golden to a
  normal gate golden (stamps `flipped_at`), so replay exits 1 if the bug
  regresses. Refused on non-xfail goldens. State machine:
  staged → promoted(gate|xfail) → fixed(flip)
- `simulate --replay` exit fold is lifecycle-aware: only `gate` failures
  block. `--format json` adds per-scenario `lifecycle`/`xpass` and summary
  `xfail_expected`/`xpass` counts

### Added — single-turn `test` staging
- Failing `ciagent test` queries now stage too (live runs only): a failing
  query becomes a one-turn envelope (mode `single`) carrying its
  correctness checks as a replayable scenario spec, so `stage verify`,
  `promote`, and replay work on it unchanged. New `--stage/--no-stage` and
  `--staged-dir` on `test`, same defaults and redaction as simulate
- Classification comes from `QueryStability` with `--runs N`, including the
  `mixed` attribution state that only exists in single-turn stability
  (mapped to `held`)
- Mock failures are synthetic and never stage (test data is never promoted
  toward production goldens)

### Added — `stage verify --reroll`
- `ciagent stage verify --reroll` re-runs the persona FRESH from the
  original `persona:`/`goal:` instead of replaying recorded turns —
  answering "does this scenario class reproduce," not "does this exact
  conversation reproduce." The staging block records which mode produced
  the classification (`verified_via: replay | reroll`); scripted scenarios
  degenerate to the verbatim replay with a note. Zero-key via --mock
  (mock persona)

### Fixed
- In GitHub Actions, `--format json` printed `::error` workflow-command
  annotation lines to stdout before the JSON payload, corrupting the
  one-JSON-document contract from #39. Annotations still emit for
  console/github/html/prometheus formats, never in json mode

## [0.11.0] - 2026-07-22

### Added — Golden Promotion Pipeline v1: auto-stage → triage → one-command promote
- Failing `ciagent simulate` conversations are auto-staged under
  `.ciagent/staged/` when staging is enabled (`--stage` or spec
  `staging.enabled`), so a nondeterministic persona repro is never lost at
  the moment it is produced. Staging is opt-in in v1 (no redactor exists
  yet — raw conversation text hits disk) and best-effort: a staging error
  never changes the run's exit code
- Each staged failure is triaged from the existing stability attribution:
  `consistent` (reproducible — NOT attributed to the agent), `flaky-agent`,
  `unverified`, `held`, `held-infra`. New `ciagent stage` group:
  `list` (best-to-promote first), `show` (`--export` for sharing, clearly
  labeled unredacted), `verify` (re-run N×, re-classify in place), `drop`,
  `gc` (per-scenario cap + age cutoff + global 500-file/50 MB cap)
- `ciagent promote <id>` moves one staged conversation into
  `<baseline_dir>/<agent>/scenarios/` — where `--record` writes — swapping
  `staging:` for an additive `provenance:` block. Gated: held/unverified
  classes refuse without `--force`, the envelope must pass the same
  structural replay gate the record path uses, and the staged copy is
  consumed only on success. No bulk promote by design
- Envelope schema: additive optional `staging:` and `provenance:` fields;
  files without them are byte-identical to pre-0.11 goldens
- `ciagent init` now gitignores `.ciagent/staged/`

### Fixed
- `--format json` wrote the console banner to stdout, corrupting the JSON
  stream (`json.load` failed on line 1). All rich chrome now routes to
  stderr in JSON mode — stdout carries exactly one JSON document — across
  `test`, `simulate`, `eval`, `diff`, `judge-audit`, `stage list/show`, and
  `promote` (#39)

## [0.10.0] - 2026-07-07

### Added — `ciagent import`: a production trace becomes a regression test (F7)
- New `ciagent import <trace_file>` command: convert an exported production
  trace into a spec query (tagged `imported`) plus a versioned golden
  baseline you can gate on. The golden carries the recorded tool-call
  sequence and each `ToolCall.result`, not just the final answer — the tool
  state a later replay needs to reproduce the failure
- Format is auto-detected, no flag: OTel GenAI spans (OTLP/JSON envelope,
  `{"spans": [...]}` wrapper, or a flat span list — what OTel GenAI
  instrumentation and openllmetry emit), the Langfuse v3+ `langfuse.*`
  attribute dialect, and LangSmith run exports (JSON or JSONL, flat or
  nested `RunTree`). Reported source formats: `otel-genai`, `otel-langfuse`,
  `langsmith-runs`
- Each dialect verified against a **real export** from that tool, not a
  hand-written fixture: openllmetry (GenAI semconv, with tool calls
  recovered from message content), a real Langfuse 4.13 capture, and a real
  LangSmith SDK export. "It speaks OTel" is not "its attribute namespace
  matches" — Langfuse proved the gap
- Round-trip artifact gate (always on): the mapped trace must produce a
  golden that loads and evaluates cleanly *before* anything is written.
  Partial traces (no user input, no final output, no spans) are rejected
  with the missing fields named — a golden that can never pass is a
  permanent false regression, and import refuses to plant one. Exit codes:
  `0` imported/`--dry-run` gate passed, `1` gate rejection, `2` file/config
  error. `--dry-run` maps and gates but writes nothing
- The spec gains a minimal `imported`-tagged query only when the query text
  is new (with a `.yaml.bak` backup); existing queries are never modified —
  only the golden is written

### Added — judge-audit answer sources: audit on fresh answers, gate on goldens
- `ciagent judge-audit --live` re-runs the agent (spec needs a `runner:`)
  for exactly the judged queries and scores the fresh answers; the confirm
  prompt counts both agent and judge calls
- `ciagent judge-audit --answers results.json` reuses a
  `ciagent test --format json` run you already paid for
- Closes the circularity when checks came from `generate-checks`: those
  checks are validated against the same goldens, so "judge PASS / check
  FAIL" cannot fire against goldens by construction and Mode-1 agreement is
  inflated. The rule the report now enforces in guidance: audit on fresh
  answers, gate on goldens

### Fixed — from the DemoAgents 0.9 sync
- Assertions read `metadata.final_output` first and never grade the literal
  string `"None"` as an answer
- Retired the dead default judge model that could silently mis-route judge
  calls
- `AgentCITraceProcessor` now subclasses the OpenAI Agents SDK's
  `TracingProcessor` base, so the adapter registers cleanly against current
  SDK versions

### Fixed — `ciagent import` hardening (found dogfooding F7 on our own agent)
- Import no longer crashes rewriting a spec that carries a `path:` assertion.
  The spec rewrite now serializes with `model_dump(mode="json")`, so enum
  fields (e.g. `path.match_mode`) become their plain-string values instead of
  an un-representable `Enum` object — the previous behavior raised
  `yaml.representer.RepresenterError` on essentially any real spec
- `ciagent import --force-save`: import a trace that FAILS its query's own
  correctness assertions — the found failure itself becomes the golden, which
  is the point of F7. Without the flag the save prechecks and stops with a
  clear message (exit 1) rather than a bare stack trace, so a bad capture is
  never planted silently

## [0.9.0] - 2026-07-06

### Added — retrieval layer 2.5: `retrieval:` assertions (F4, F6 Phase 4)
- New `retrieval:` block on queries and scenario `per_turn:`/`outcome:`
  checks — deterministic assertions on the retriever tool's captured
  `ToolCall.result`: `forbid_empty` (empty retrieval + confident answer →
  WARN "ungrounded answer"; a refusal passes), `min_results`,
  `expected_sources`, `facts_in_context` (informational-only in v1),
  `result_format` hint, `empty_markers`/`refusal_markers` overrides
- Binding result-interpretation contract: the layer SKIPs on uncaptured or
  unparseable results — it never guesses. Empty means None/[]/""/whitespace
  plus spec markers. Retrieval never hard-fails (WARN tier, like path/cost)
- Fifth stability flip source `retrieval-variance` (single-turn and
  scenario): verdict flipped, same tool sequence, retrieved set differed →
  blames the retriever, not the prompt; attribution skips when any run's
  retriever output wasn't captured
- Judge audit gains a "judged against EMPTY retrieval" row (console + JSON):
  the judge graded an answer whose ground truth was already lost
- Diff engine gains `RETRIEVAL_CHANGED`: retrieved source set changed vs
  golden, emitted only when a source set is extractable from both traces
- Mock runner synthesizes retrieval results satisfying the spec — the
  zero-key path exercises layer 2.5 end to end
- Adapter capture verification and fixes: langgraph now pairs `ToolMessage`
  outputs onto tool calls by `tool_call_id`; the openai/anthropic
  monkey-patches backfill `ToolCall.result` from the next request's
  `role="tool"` messages / `tool_result` blocks (openai-agents adapter
  already captured results)

### Added — generative personas, cost guardrails, stability (F6 Phase 3)
- Scenarios with `persona:`/`goal:` and no `turns:` get user turns from a
  persona LLM (cheap haiku-class default, `persona_config` spec key) — the
  finder path. Termination rules unchanged: max_turns / stop_when only; a
  derailed persona (empty/unusable output) marks the scenario infra-error
  with partial turns kept rather than silently grading the agent
- `--max-cost` session budget: hard-aborts mid-conversation at the next turn
  boundary; partial results are clearly marked and the outcome verdict is
  never evaluated on a partial conversation. Pre-run cost estimate with a
  confirm gate; per-scenario cost line
- `--runs N` scenario stability with new `simulation-variance` flip source:
  the simulated user said different things across runs — the persona varied,
  not the agent. `--workers` runs scenarios in parallel (turns stay
  sequential); a mock persona gives a zero-key generative path

### Added — record + replay: found bug → regression test (F6 Phase 2)
- `--record` / `--record-dir`: save any driven conversation as a golden
  envelope at `<baseline_dir>/<agent>/scenarios/<slug>.json`. Recording
  never prechecks — a FAILED scenario records too (`checks_passed: false`),
  because converting a found bug into a regression test is the point
- `--replay`: recorded user turns are fed back verbatim (the persona is
  never consulted); the scenario spec is embedded in the envelope so goldens
  are self-contained. Replaying a deterministic agent twice yields
  byte-identical verdicts (the `scenario_verdict` contract)
- Conversation-aware diff (`diff_envelopes`): per-turn tool-sequence and
  answer changes between a golden conversation and a fresh run

### Added — `ciagent simulate` Phase 1: scripted multi-turn scenarios (F6)
- New `scenarios:` spec block: `turns:` (scripted user messages), `max_turns`,
  `per_turn:` checks (evaluated on every turn), `outcome:` checks (evaluated
  once at the END as the verdict — never a stop condition), and explicit
  `stop_when: {tool_called: X}` early exit. Termination is deterministic only:
  script exhausted, max_turns, or the stop_when event — no judge, no keywords
- New `conversation_runner:` spec key — `(messages: list[{role, content}]) →
  str | Trace`, history passed explicitly, same capture/coercion as `test`
- `ciagent simulate` CLI: `--mock` runs scenarios on synthetic traces with
  zero API keys (the CI path); live runs confirm turn count first. Exit codes:
  0 pass / 1 outcome or per-turn correctness failure / 2 config or agent error
- Agent exception mid-conversation marks the scenario `infra-error` and keeps
  the completed turns; conversations convert to schema_version-2 envelopes
- Spec change: `queries:` is no longer required when `scenarios:` is present
  (at least one of the two must exist)
- 18 new tests, incl. the ADR checklist items: both termination causes,
  agent-raises-mid-conversation, per_turn + outcome evaluation, and
  outcome-never-stops (a turn-1 keyword match must not end the scenario)

### Added — schema_version + conversation envelope (F6 Phase 0)
- `schema_version: 1` written into new single-trace baselines; unversioned
  files read as legacy; envelopes are `schema_version: 2`; newer-than-reader
  files rejected by name. `ConversationEnvelope`: one loader for envelope,
  wrapper, and bare-trace shapes (single-turn = 1-turn degenerate case)

### Changed — brand and story
- **Brand: AgentCI → CIAgent** (display name), standardizing on the `ciagent`
  package name. Spec/runner filenames (`agentci_spec.yaml`), `AGENTCI_*` env
  vars, and `AgentCISpec`/`AgentCITraceProcessor` class names unchanged.
- **BREAKING: Python module renamed `agentci` → `ciagent`** (`from ciagent
  import ...`); the `agentci` CLI entry point is removed (`ciagent` is the
  command), and the pytest plugin entry point is now `ciagent`. Flag-day, no
  deprecation shim (pre-adoption). Motive is a verified collision: the
  unrelated PyPI package `agentci` 0.1.1 (Agent-CI) installs a top-level
  `agentci/` module and an `agentci` console script — installing both tools
  in one environment broke ours.
- **GitHub repo renamed** to `suniel12/ciagent` (old URLs redirect).
- **Claude Code plugin identity renamed**: install is now
  `/plugin marketplace add suniel12/ciagent` + `/plugin install ciagent@ciagent`;
  skills are `ciagent:onboard` / `ciagent:check`. (The 0.8.0 identity below had
  zero installs.)
- **README hero, PyPI description, and GitHub About repositioned to the
  eval-reliability wedge** ("Your eval score is stable. Your system isn't."),
  per the strategy ADR; regression testing remains as supporting capability.

## [0.8.0] - 2026-07-06

### Added — Claude Code plugin (F5: your coding agent onboards and operates CIAgent)
- The repo is now a Claude Code plugin marketplace
  (`/plugin marketplace add suniel12/AgentCI`, `/plugin install agentci@agentci`)
  with two skills: **onboard** (write the runner, record goldens via
  `bootstrap --yes`, generate + tighten the spec, verify with `test --runs 3`)
  and **check** (run the right check after agent changes; route failures by
  flip source; never weaken a correct check to go green)
- `agentci bootstrap --yes`: fully non-interactive golden recording (requires
  `--queries`); bootstrap now runs the runner through the same TraceContext
  capture path as `agentci test`, so **runners returning plain strings work**
  (previously crashed on anything but a Trace)
- `--format json` output now includes the agent's `answer` text per query
  (eng-review work item: JSON consumers need what the agent said, not just
  verdicts; also unblocks JSON as a judge-audit answers source)
- Tests: skills are lint-gated — every `agentci` command and flag a skill
  teaches is asserted to exist in the CLI, so skill docs cannot rot silently

### Fixed — from Codex cross-model review of this branch
- **Nested `TraceContext` no longer double-records or loses capture**: a
  Trace-returning runner that uses `TraceContext` internally, wrapped again by
  the executor, used to stack SDK patches (every LLM call recorded once per
  wrapper) and its exit cleared the outer context entirely. Patches now install
  only in the outermost context, and exits restore the enclosing context via
  contextvar tokens. Affected `agentci test` as well as the new bootstrap path
- JSON `answer` field now uses the correctness layer's extractor (metadata →
  last span output), so trace shapes the evaluator can grade also serialize
  their answer instead of `null`

### Added — zero-key demo (`uvx ciagent test --mock --runs 3`)
- `agentci test --mock` with no `agentci_spec.yaml` in the working directory now
  falls back to a bundled demo spec (8 synthetic support-agent queries), clearly
  labeled as demo mode with synthetic data. An explicitly passed `--config` that
  is missing remains an error — the fallback only applies to the default path
- Demo multi-run sessions simulate a pseudo-flaky agent by default with the new
  `"spread"` style (one query breaks per run), so the aggregate score stays
  constant while individual verdicts flip — the report the demo exists to show.
  `AGENTCI_MOCK_FLAKY=0` turns the simulation off; non-demo specs are unaffected
  (flakiness stays env-var opt-in, `"alternate"` style unchanged)
- Missing spec without `--mock` now exits with a hint pointing at both
  `agentci init` and the zero-key demo command

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
