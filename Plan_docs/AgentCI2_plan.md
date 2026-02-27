# AgentCI 2.0: Production Implementation Plan

**Version:** Final Spec v1.0  
**Date:** February 26, 2026  
**Author:** Sunil Pandey (Founder, AgentCI)  
**Purpose:** Coding-agent-ready implementation specification

---

## Executive Summary

AgentCI 2.0 transitions the framework from binary unit-style trace diffing to a **declarative, probabilistic, CI/CD-native evaluation system** for AI agents. The architecture rests on three pillars: validated YAML specs (Pydantic + JSONSchema), a three-layer assertion engine (Correctness → Path → Cost), and granular CI feedback via GitHub annotations — a capability no existing framework offers.

### Competitive Position

Across 15+ frameworks analyzed (LangSmith, Braintrust, DeepEval, promptfoo, Langfuse, Arize Phoenix, Patronus AI, Inspect AI, RAGAS, and others), **no tool combines declarative YAML test specs with trajectory evaluation and CI/CD-native severity-layered feedback**. The closest competitors each cover a piece:

| Competitor | Strength | Gap AgentCI 2.0 Fills |
|---|---|---|
| **promptfoo** | Only YAML-native framework; JSON Schema validation | No trajectory eval; no trace-based regression |
| **DeepEval** | 50+ metrics; pytest-native; ToolCorrectness metric | Code-only (no declarative specs); no layered severity |
| **LangSmith** | Best trajectory eval (5 match modes via `agentevals`) | No CI tooling; no declarative specs; no annotations |
| **Braintrust** | Best CI polish (GitHub Action + PR comments) | No trajectory metrics; no YAML specs; no annotations |
| **RAGAS** | Canonical RAG metrics + ToolCallF1 | Research-oriented; no CI integration |
| **Inspect AI** | Statistical rigor (epochs, stderr, power analysis) | No CI tooling; discourages model-graded scoring |

**AgentCI 2.0's unique value: pytest-style trace regression testing with declarative YAML specs, three-layer severity (hard correctness / soft path / soft cost), and inline GitHub annotations.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        agentci CLI                              │
│  init │ validate │ test │ diff │ save │ export-metrics           │
└───────┬─────────┬───────┬──────┬──────┬────────────────────────┘
        │         │       │      │      │
        ▼         ▼       ▼      ▼      ▼
┌───────────┐ ┌───────┐ ┌────────────────────────────────────────┐
│  Schema   │ │ YAML  │ │         Evaluation Engine              │
│ Validator │ │Loader │ │  ┌────────────┬──────────┬───────────┐ │
│ (Pydantic │ │(with  │ │  │Correctness │  Path    │   Cost    │ │
│  + JSON   │ │$ref + │ │  │  Engine    │ Engine   │  Engine   │ │
│  Schema)  │ │inherit│ │  │(hard fail) │(soft warn│(soft warn)│ │
│           │ │ance)  │ │  └──────┬─────┴────┬─────┴─────┬─────┘ │
└───────────┘ └───────┘ │        │           │           │       │
                        │        ▼           ▼           ▼       │
                        │  ┌──────────────────────────────────┐  │
                        │  │        Result Aggregator         │  │
                        │  │  Exit Codes + GitHub Annotations │  │
                        │  │  + JSON/Prometheus Export         │  │
                        │  └──────────────────────────────────┘  │
                        └────────────────────────────────────────┘
                                         │
                        ┌────────────────┼────────────────────┐
                        ▼                ▼                    ▼
                   ┌─────────┐   ┌──────────────┐   ┌──────────────┐
                   │ Baseline│   │  Judge        │   │  Trace       │
                   │ Manager │   │  Safeguards   │   │  Adapters    │
                   │(version,│   │(temp=0,       │   │(LangGraph,   │
                   │ precheck│   │ ensemble,     │   │ OpenAI SDK,  │
                   │ --force)│   │ structured)   │   │ raw traces)  │
                   └─────────┘   └──────────────┘   └──────────────┘
```

---

## Phase 1: Core Framework (MVP — Target: 3-4 days)

Phase 1 delivers the validated YAML spec, three-layer assertion engine, and the RAG agent demo wired end-to-end. Subdivided into two sub-phases to derisk.

### Phase 1A: Schema + Correctness + Basic Cost + RAG Demo

Ship a working end-to-end pipeline with correctness checks, basic cost gating, and the RAG weather query demo.

---

### Step 0.5: Schema Definition & Validation

**Objective:** Fail fast on invalid specs. No runtime should ever encounter malformed YAML.

**Files to create/modify:**
- `agentci/schema/spec_models.py` — Pydantic models
- `agentci/schema/agentci_spec.schema.json` — Generated JSON Schema
- `agentci/cli.py` — Add `--validate` command
- `docs/sample_spec.yaml` — Reference spec with inline comments
- `docs/metrics_reference.md` — Mathematical definitions

**Pydantic Model Hierarchy:**

```python
# agentci/schema/spec_models.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum

class MatchMode(str, Enum):
    STRICT = "strict"           # Exact match: same tools, same order
    UNORDERED = "unordered"     # Same tools, any order
    SUBSET = "subset"           # Reference tools must appear (extras OK)
    SUPERSET = "superset"       # All used tools must be in reference

class JudgeRubric(BaseModel):
    """Structured rubric for LLM-as-a-judge evaluation."""
    rule: str = Field(..., description="Natural language evaluation criterion")
    scale: Optional[list[str]] = Field(
        None,
        description="Score anchors, e.g. ['1: Off-topic', '2: Partially relevant', '3: Fully correct']"
    )
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum passing score (normalized)")
    few_shot_examples: Optional[list[dict]] = Field(
        None,
        description="Example input/output/score triples for calibration"
    )

class CorrectnessSpec(BaseModel):
    """Layer 1: Hard pass/fail. Failures block the CI pipeline."""
    expected_in_answer: Optional[list[str]] = None
    not_in_answer: Optional[list[str]] = None
    exact_match: Optional[str] = None
    regex_match: Optional[str] = None
    json_schema: Optional[dict] = None
    llm_judge: Optional[list[JudgeRubric]] = None
    # Safety/hallucination as sub-checks (not top-level)
    safety_check: Optional[JudgeRubric] = Field(
        None,
        description="Rubric for safety evaluation (treated as correctness-tier)"
    )
    hallucination_check: Optional[JudgeRubric] = Field(
        None,
        description="Rubric for hallucination/grounding evaluation"
    )

class PathSpec(BaseModel):
    """Layer 2: Trajectory evaluation. Exceedances produce warnings, not failures."""
    max_tool_calls: Optional[int] = Field(None, ge=0)
    expected_tools: Optional[list[str]] = Field(
        None,
        description="Tools that should be called (for recall calculation)"
    )
    forbidden_tools: Optional[list[str]] = Field(
        None,
        description="Tools that must NOT be called (safety boundary)"
    )
    max_loops: Optional[int] = Field(
        None, ge=1,
        description="Maximum allowed repeated tool invocations (cycle detection)"
    )
    match_mode: MatchMode = Field(
        MatchMode.SUBSET,
        description="How to compare tool sequences against the golden baseline"
    )
    min_tool_recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_tool_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_sequence_similarity: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Minimum normalized LCS similarity (0=disjoint, 1=identical order)"
    )
    # Multi-agent routing assertions
    expected_handoff: Optional[str] = Field(
        None,
        description="Expected handoff target agent name"
    )
    expected_handoffs_available: Optional[list[str]] = None
    max_handoff_count: Optional[int] = Field(None, ge=0)

class CostSpec(BaseModel):
    """Layer 3: Efficiency budget. Exceedances produce warnings, not failures."""
    max_cost_multiplier: Optional[float] = Field(
        None, gt=0,
        description="Max allowed cost as multiple of golden baseline (e.g. 2.0 = 2x)"
    )
    max_total_tokens: Optional[int] = Field(None, ge=0)
    max_llm_calls: Optional[int] = Field(None, ge=0)
    max_latency_ms: Optional[int] = Field(
        None, ge=0,
        description="Max wall-clock latency in milliseconds"
    )
    max_cost_usd: Optional[float] = Field(None, ge=0)

class GoldenQuery(BaseModel):
    """Single test case in the AgentCI spec."""
    query: str = Field(..., description="The input stimulus to the agent")
    description: Optional[str] = Field(None, description="Human-readable test description")
    tags: Optional[list[str]] = Field(None, description="Tags for filtering (e.g. ['smoke', 'edge-case'])")
    correctness: Optional[CorrectnessSpec] = None
    path: Optional[PathSpec] = None
    cost: Optional[CostSpec] = None

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('query must not be empty')
        return v

class AgentCISpec(BaseModel):
    """Root schema for agentci_spec.yaml."""
    version: int = Field(1, description="Schema version for forward compatibility")
    agent: str = Field(..., description="Agent identifier (e.g. 'rag-agent', 'support-router')")
    baseline_dir: str = Field(
        "./baselines",
        description="Directory containing golden baseline trace files"
    )
    defaults: Optional[dict] = Field(
        None,
        description="Default correctness/path/cost applied to all queries unless overridden"
    )
    judge_config: Optional[dict] = Field(
        None,
        description="Global LLM judge settings: model, temperature, ensemble, structured_output"
    )
    queries: list[GoldenQuery] = Field(..., min_length=1)
```

**CLI `--validate` Command:**

```python
# In agentci/cli.py
@cli.command()
@click.argument('spec_path', type=click.Path(exists=True))
def validate(spec_path: str):
    """Validate an agentci_spec.yaml file against the schema."""
    import yaml
    from agentci.schema.spec_models import AgentCISpec
    from pydantic import ValidationError

    with open(spec_path) as f:
        raw = yaml.safe_load(f)

    try:
        spec = AgentCISpec(**raw)
        click.echo(f"✅ Valid: {len(spec.queries)} queries, agent='{spec.agent}'")
        sys.exit(0)
    except ValidationError as e:
        click.echo(f"❌ Validation failed:\n{e}", err=True)
        sys.exit(1)
```

**JSON Schema Generation** (auto-generated from Pydantic for IDE autocomplete):

```python
# agentci/schema/generate_schema.py
import json
from agentci.schema.spec_models import AgentCISpec

schema = AgentCISpec.model_json_schema()
with open("agentci/schema/agentci_spec.schema.json", "w") as f:
    json.dump(schema, f, indent=2)
```

**Sample Spec (`docs/sample_spec.yaml`):**

```yaml
# yaml-language-server: $schema=../agentci/schema/agentci_spec.schema.json
version: 1
agent: rag-agent
baseline_dir: ./baselines/rag

# Defaults applied to all queries unless overridden
defaults:
  correctness:
    hallucination_check:
      rule: "Answer is grounded in retrieved context only; no fabricated facts"
      threshold: 0.8
  cost:
    max_cost_multiplier: 2.0

# Global judge settings
judge_config:
  model: claude-sonnet-4-5-20250929
  temperature: 0
  structured_output: true
  ensemble:
    enabled: false          # Enable via --sample-ensemble 0.2 in CI
    models: ["claude-sonnet-4-5-20250929", "gpt-4o-mini", "gpt-4o-mini"]
    strategy: majority_vote

queries:
  - query: "How do I install AgentCI?"
    description: "Core in-scope question — should retrieve from docs"
    tags: [smoke, happy-path]
    correctness:
      expected_in_answer: ["pip install", "agentci"]
      llm_judge:
        - rule: "Response provides clear, actionable installation steps"
          threshold: 0.7

  - query: "What's the weather in Tokyo?"
    description: "Out-of-scope — agent must decline gracefully"
    tags: [edge-case, out-of-scope]
    correctness:
      not_in_answer: ["Tokyo", "degrees", "forecast"]
      llm_judge:
        - rule: "Agent politely declines and explains this is outside its knowledge domain"
          threshold: 0.8
    path:
      max_tool_calls: 0       # Should decline immediately, no tools
      forbidden_tools: [tavily_search, web_search]
    cost:
      max_llm_calls: 2
      max_total_tokens: 500

  - query: "Explain the LangGraph state management pattern"
    description: "In-scope technical question"
    tags: [technical, in-scope]
    correctness:
      llm_judge:
        - rule: "Response accurately describes LangGraph state channels and reducers"
          threshold: 0.7
    path:
      expected_tools: [retriever_tool]
      min_tool_recall: 1.0    # Must use the retriever
      max_tool_calls: 5
      match_mode: subset
```

**Deliverables:**
- [ ] `agentci/schema/spec_models.py` with full Pydantic models
- [ ] `agentci/schema/agentci_spec.schema.json` auto-generated
- [ ] `agentci validate` CLI command (exit 0 = valid, exit 1 = invalid)
- [ ] `docs/sample_spec.yaml` with inline documentation
- [ ] `docs/metrics_reference.md` with formulas for every path/cost metric
- [ ] Unit tests for validation: valid specs pass, 10+ invalid spec variants fail correctly

---

### Step 1: YAML Loader with Defaults Inheritance

**Objective:** Parse validated YAML into executable evaluation plans, merging per-query specs with global defaults.

**Files to create/modify:**
- `agentci/loader.py` — YAML loading + defaults merging
- `agentci/schema/spec_models.py` — Add `merge_with_defaults()` method

**Implementation:**

```python
# agentci/loader.py
import yaml
from pathlib import Path
from agentci.schema.spec_models import AgentCISpec, GoldenQuery

def load_spec(spec_path: str | Path) -> AgentCISpec:
    """Load and validate an AgentCI spec file."""
    with open(spec_path) as f:
        raw = yaml.safe_load(f)
    spec = AgentCISpec(**raw)
    if spec.defaults:
        spec.queries = [_merge_defaults(q, spec.defaults) for q in spec.queries]
    return spec

def _merge_defaults(query: GoldenQuery, defaults: dict) -> GoldenQuery:
    """Deep merge defaults into query, with query values taking precedence."""
    query_dict = query.model_dump(exclude_none=True)
    merged = _deep_merge(defaults, query_dict)
    return GoldenQuery(**merged)

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

**Deliverables:**
- [ ] `load_spec()` function with defaults merging
- [ ] Support for `$ref`-style file includes (future: `correctness: !include shared/hallucination.yaml`)
- [ ] Tag-based filtering: `agentci test --tags smoke`
- [ ] Tests: verify defaults merging, override behavior, tag filtering

---

### Step 2: Three-Layer Assertion Engine

**Objective:** Evaluate each golden query against its spec, producing structured results per layer.

**Files to create/modify:**
- `agentci/engine/correctness.py` — Layer 1: hard pass/fail
- `agentci/engine/path.py` — Layer 2: trajectory metrics
- `agentci/engine/cost.py` — Layer 3: efficiency metrics
- `agentci/engine/runner.py` — Orchestrator
- `agentci/engine/metrics.py` — Shared metric calculations
- `agentci/models.py` — Update result models

**Result Model:**

```python
# agentci/engine/results.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class LayerStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"      # Path/cost exceeded but correctness passed
    SKIP = "skip"      # No assertions defined for this layer

@dataclass
class LayerResult:
    status: LayerStatus
    details: dict       # Metric-specific details
    messages: list[str] # Human-readable explanations

@dataclass
class QueryResult:
    query: str
    correctness: LayerResult
    path: LayerResult
    cost: LayerResult

    @property
    def hard_fail(self) -> bool:
        return self.correctness.status == LayerStatus.FAIL

    @property
    def has_warnings(self) -> bool:
        return (self.path.status == LayerStatus.WARN or
                self.cost.status == LayerStatus.WARN)
```

#### Correctness Engine (Layer 1 — Hard Fail)

```python
# agentci/engine/correctness.py
from agentci.schema.spec_models import CorrectnessSpec
from agentci.engine.results import LayerResult, LayerStatus
from agentci.engine.judge import run_judge
import re

def evaluate_correctness(
    answer: str,
    spec: CorrectnessSpec,
    trace: dict,
    judge_config: dict | None = None
) -> LayerResult:
    """Evaluate correctness assertions. Any failure = hard fail."""
    failures = []
    details = {}

    # Deterministic checks FIRST (fast, free)
    if spec.expected_in_answer:
        for term in spec.expected_in_answer:
            if term.lower() not in answer.lower():
                failures.append(f"Expected '{term}' not found in answer")
        details["expected_in_answer"] = {"checked": spec.expected_in_answer,
                                          "all_found": len(failures) == 0}

    if spec.not_in_answer:
        for term in spec.not_in_answer:
            if term.lower() in answer.lower():
                failures.append(f"Forbidden term '{term}' found in answer")
        details["not_in_answer"] = {"checked": spec.not_in_answer,
                                     "none_found": len(failures) == 0}

    if spec.exact_match and answer.strip() != spec.exact_match.strip():
        failures.append("Exact match failed")

    if spec.regex_match and not re.search(spec.regex_match, answer):
        failures.append(f"Regex '{spec.regex_match}' did not match")

    if spec.json_schema:
        # Validate answer parses as JSON matching schema
        import json, jsonschema
        try:
            parsed = json.loads(answer)
            jsonschema.validate(parsed, spec.json_schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            failures.append(f"JSON schema validation failed: {e}")

    # LLM judge checks LAST (expensive, only if deterministic checks pass)
    if spec.llm_judge and not failures:  # Skip if already failed
        for rubric in spec.llm_judge:
            result = run_judge(answer=answer, rubric=rubric, config=judge_config)
            details[f"judge_{rubric.rule[:30]}"] = result
            if not result["passed"]:
                failures.append(f"Judge failed: {rubric.rule}")

    # Safety and hallucination sub-checks
    if spec.safety_check and not failures:
        result = run_judge(answer=answer, rubric=spec.safety_check, config=judge_config)
        details["safety"] = result
        if not result["passed"]:
            failures.append(f"Safety check failed: {spec.safety_check.rule}")

    if spec.hallucination_check and not failures:
        result = run_judge(answer=answer, rubric=spec.hallucination_check, config=judge_config)
        details["hallucination"] = result
        if not result["passed"]:
            failures.append(f"Hallucination check failed")

    return LayerResult(
        status=LayerStatus.FAIL if failures else LayerStatus.PASS,
        details=details,
        messages=failures if failures else ["All correctness checks passed"]
    )
```

#### Path Engine (Layer 2 — Soft Warning)

**Metric Definitions (Mathematical):**

```
tool_recall    = |expected_tools ∩ used_tools| / |expected_tools|
tool_precision = |expected_tools ∩ used_tools| / |used_tools|
tool_f1        = 2 × (precision × recall) / (precision + recall)

sequence_similarity (normalized LCS):
  LCS = longest common subsequence of (predicted_tools, reference_tools)
  similarity = 2 × |LCS| / (|predicted| + |reference|)

sequence_edit_distance (normalized Levenshtein):
  ED = min edits (insert/delete/substitute) to transform predicted → reference
  similarity = 1 - ED / max(|predicted|, |reference|)
```

```python
# agentci/engine/path.py
from agentci.schema.spec_models import PathSpec
from agentci.engine.results import LayerResult, LayerStatus
from agentci.engine.metrics import (
    compute_tool_recall, compute_tool_precision,
    compute_sequence_lcs, detect_loops
)

def evaluate_path(
    trace: "Trace",
    spec: PathSpec,
    baseline_trace: "Trace | None" = None
) -> LayerResult:
    """Evaluate trajectory/path assertions. Exceedances = warnings."""
    warnings = []
    details = {}
    used_tools = trace.get_tool_names()

    # Tool count bound
    if spec.max_tool_calls is not None:
        actual = len(used_tools)
        details["tool_calls"] = {"actual": actual, "max": spec.max_tool_calls}
        if actual > spec.max_tool_calls:
            warnings.append(
                f"Tool calls: {actual} > max {spec.max_tool_calls}"
            )

    # Forbidden tools (safety boundary — escalate to FAIL if violated)
    hard_fail = False
    if spec.forbidden_tools:
        used_set = set(used_tools)
        violations = used_set & set(spec.forbidden_tools)
        details["forbidden_tools"] = {"violations": list(violations)}
        if violations:
            warnings.append(f"Forbidden tools used: {violations}")
            hard_fail = True  # Safety violations are hard fails

    # Tool recall
    if spec.expected_tools:
        expected_set = set(spec.expected_tools)
        used_set = set(used_tools)
        recall = compute_tool_recall(expected_set, used_set)
        details["tool_recall"] = round(recall, 3)
        if spec.min_tool_recall is not None and recall < spec.min_tool_recall:
            warnings.append(
                f"Tool recall: {recall:.2f} < min {spec.min_tool_recall}"
            )

    # Tool precision
    if spec.expected_tools and used_tools:
        precision = compute_tool_precision(set(spec.expected_tools), set(used_tools))
        details["tool_precision"] = round(precision, 3)
        if spec.min_tool_precision is not None and precision < spec.min_tool_precision:
            warnings.append(
                f"Tool precision: {precision:.2f} < min {spec.min_tool_precision}"
            )

    # Sequence similarity (LCS against baseline)
    if spec.min_sequence_similarity is not None and baseline_trace:
        baseline_tools = baseline_trace.get_tool_names()
        similarity = compute_sequence_lcs(used_tools, baseline_tools)
        details["sequence_similarity"] = round(similarity, 3)
        if similarity < spec.min_sequence_similarity:
            warnings.append(
                f"Sequence similarity: {similarity:.2f} < min {spec.min_sequence_similarity}"
            )

    # Loop detection
    if spec.max_loops is not None:
        loops = detect_loops(used_tools)
        details["loops_detected"] = loops
        if loops > spec.max_loops:
            warnings.append(f"Loops detected: {loops} > max {spec.max_loops}")

    # Match mode evaluation (against baseline)
    if baseline_trace and spec.match_mode:
        baseline_tools = baseline_trace.get_tool_names()
        match_result = evaluate_match_mode(used_tools, baseline_tools, spec.match_mode)
        details["match_mode"] = match_result
        if not match_result["matched"]:
            warnings.append(f"Match mode '{spec.match_mode.value}' failed: {match_result['reason']}")

    # Multi-agent routing assertions
    if spec.expected_handoff:
        handoffs = trace.get_handoffs()
        actual_targets = [h.target for h in handoffs]
        details["handoffs"] = {"expected": spec.expected_handoff, "actual": actual_targets}
        if spec.expected_handoff not in actual_targets:
            warnings.append(
                f"Expected handoff to '{spec.expected_handoff}', got {actual_targets}"
            )

    if hard_fail:
        status = LayerStatus.FAIL  # Forbidden tool = hard fail
    elif warnings:
        status = LayerStatus.WARN
    else:
        status = LayerStatus.PASS

    return LayerResult(status=status, details=details, messages=warnings or ["Path OK"])
```

**Shared Metrics Module:**

```python
# agentci/engine/metrics.py

def compute_tool_recall(expected: set[str], used: set[str]) -> float:
    """Recall = |expected ∩ used| / |expected|"""
    if not expected:
        return 1.0
    return len(expected & used) / len(expected)

def compute_tool_precision(expected: set[str], used: set[str]) -> float:
    """Precision = |expected ∩ used| / |used|"""
    if not used:
        return 1.0 if not expected else 0.0
    return len(expected & used) / len(used)

def compute_sequence_lcs(seq_a: list[str], seq_b: list[str]) -> float:
    """Normalized LCS similarity: 2 × |LCS| / (|A| + |B|)"""
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0

    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i-1] == seq_b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    return (2 * lcs_len) / (m + n)

def compute_edit_distance_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """Normalized edit distance similarity: 1 - ED / max(|A|, |B|)"""
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0

    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i-1] == seq_b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    ed = dp[m][n]
    return 1.0 - ed / max(m, n)

def detect_loops(tool_sequence: list[str]) -> int:
    """Count consecutive repeated tool invocations."""
    if not tool_sequence:
        return 0
    loops = 0
    for i in range(1, len(tool_sequence)):
        if tool_sequence[i] == tool_sequence[i-1]:
            loops += 1
    return loops
```

#### Cost Engine (Layer 3 — Soft Warning)

```python
# agentci/engine/cost.py
from agentci.schema.spec_models import CostSpec
from agentci.engine.results import LayerResult, LayerStatus

def evaluate_cost(
    trace: "Trace",
    spec: CostSpec,
    baseline_trace: "Trace | None" = None
) -> LayerResult:
    """Evaluate cost/efficiency assertions. Exceedances = warnings."""
    warnings = []
    details = {}

    actual_cost = trace.total_cost
    actual_tokens = trace.total_tokens
    actual_llm_calls = trace.llm_call_count
    actual_latency = trace.latency_ms

    details["actual"] = {
        "cost_usd": round(actual_cost, 6),
        "total_tokens": actual_tokens,
        "llm_calls": actual_llm_calls,
        "latency_ms": actual_latency,
    }

    if spec.max_cost_multiplier is not None and baseline_trace:
        baseline_cost = baseline_trace.total_cost
        if baseline_cost > 0:
            multiplier = actual_cost / baseline_cost
            details["cost_multiplier"] = round(multiplier, 2)
            if multiplier > spec.max_cost_multiplier:
                warnings.append(
                    f"Cost {multiplier:.1f}x baseline (max {spec.max_cost_multiplier}x): "
                    f"${actual_cost:.4f} vs ${baseline_cost:.4f}"
                )

    if spec.max_total_tokens is not None and actual_tokens > spec.max_total_tokens:
        warnings.append(f"Tokens: {actual_tokens} > max {spec.max_total_tokens}")

    if spec.max_llm_calls is not None and actual_llm_calls > spec.max_llm_calls:
        warnings.append(f"LLM calls: {actual_llm_calls} > max {spec.max_llm_calls}")

    if spec.max_latency_ms is not None and actual_latency > spec.max_latency_ms:
        warnings.append(f"Latency: {actual_latency}ms > max {spec.max_latency_ms}ms")

    if spec.max_cost_usd is not None and actual_cost > spec.max_cost_usd:
        warnings.append(f"Cost: ${actual_cost:.4f} > max ${spec.max_cost_usd:.4f}")

    return LayerResult(
        status=LayerStatus.WARN if warnings else LayerStatus.PASS,
        details=details,
        messages=warnings or ["Cost within bounds"]
    )
```

**Deliverables:**
- [ ] `agentci/engine/correctness.py` — deterministic-first, then LLM judge
- [ ] `agentci/engine/path.py` — all five match modes + precision/recall/LCS
- [ ] `agentci/engine/cost.py` — multiplier, tokens, calls, latency, USD
- [ ] `agentci/engine/metrics.py` — pure functions with docstrings and formulas
- [ ] `agentci/engine/runner.py` — orchestrates all three layers per query
- [ ] 40+ unit tests covering every metric edge case

---

### Step 2.5: LLM Judge Safeguards

**Objective:** Make judge evaluations reliable, structured, and cost-controlled.

**Files to create/modify:**
- `agentci/engine/judge.py` — Judge execution with safeguards

**Structured Output Schema:**

```python
# agentci/engine/judge.py
from pydantic import BaseModel
from typing import Literal

class JudgeVerdict(BaseModel):
    """Structured output required from all LLM judges."""
    score: int              # 1-5 scale
    label: Literal["pass", "fail", "borderline"]
    rationale: str          # Required short explanation (forces chain-of-thought)

def run_judge(
    answer: str,
    rubric: "JudgeRubric",
    config: dict | None = None,
    context: str | None = None,  # Retrieved docs for grounding checks
) -> dict:
    """Execute an LLM-as-a-judge evaluation with safeguards."""
    config = config or {}
    model = config.get("model", "claude-sonnet-4-5-20250929")
    temperature = config.get("temperature", 0)  # ALWAYS default to 0
    ensemble_config = config.get("ensemble", {})

    # Build the judge prompt with structured output requirement
    system_prompt = _build_judge_system_prompt(rubric)
    user_prompt = _build_judge_user_prompt(answer, rubric, context)

    if ensemble_config.get("enabled", False):
        return _run_ensemble(system_prompt, user_prompt, ensemble_config, rubric)

    # Single judge call
    verdict = _call_judge(model, system_prompt, user_prompt, temperature)
    passed = verdict.score >= _score_threshold(rubric.threshold)

    return {
        "passed": passed,
        "score": verdict.score,
        "label": verdict.label,
        "rationale": verdict.rationale,
        "model": model,
    }

def _run_ensemble(system: str, user: str, config: dict, rubric) -> dict:
    """Majority vote across multiple judge models."""
    models = config.get("models", [
        "claude-sonnet-4-5-20250929", "gpt-4o-mini", "gpt-4o-mini"
    ])
    verdicts = [_call_judge(m, system, user, temperature=0) for m in models]
    votes = [v.label for v in verdicts]
    majority = max(set(votes), key=votes.count)
    avg_score = sum(v.score for v in verdicts) / len(verdicts)
    passed = majority != "fail" and avg_score >= _score_threshold(rubric.threshold)

    return {
        "passed": passed,
        "score": round(avg_score, 2),
        "label": majority,
        "rationale": f"Ensemble ({len(models)} judges): {votes}",
        "individual_verdicts": [v.model_dump() for v in verdicts],
    }

def _build_judge_system_prompt(rubric: "JudgeRubric") -> str:
    """Construct a rubric-driven system prompt."""
    prompt = (
        "You are an evaluation judge. Assess the given response against the rubric.\n"
        "You MUST respond with valid JSON matching this schema:\n"
        '{"score": <1-5>, "label": "<pass|fail|borderline>", "rationale": "<string>"}\n\n'
        f"RUBRIC: {rubric.rule}\n"
    )
    if rubric.scale:
        prompt += "\nSCORING ANCHORS:\n"
        for anchor in rubric.scale:
            prompt += f"  - {anchor}\n"
    if rubric.few_shot_examples:
        prompt += "\nEXAMPLES:\n"
        for ex in rubric.few_shot_examples:
            prompt += f"  Input: {ex.get('input', 'N/A')}\n"
            prompt += f"  Output: {ex.get('output', 'N/A')}\n"
            prompt += f"  Score: {ex.get('score', 'N/A')}\n\n"
    return prompt

def _score_threshold(threshold: float) -> int:
    """Convert 0-1 threshold to 1-5 scale. 0.5 → 3, 0.8 → 4, etc."""
    return max(1, min(5, round(threshold * 5)))
```

**Key Design Decisions (Research-Backed):**
1. **Temperature = 0 always** (Li et al. 2025; even at temp=0, GPU non-determinism means aggregation is still needed)
2. **Structured JSON output** (Pydantic AI approach — rationale field forces chain-of-thought)
3. **Additive rubric pattern** (outperforms holistic scoring in benchmarks)
4. **Cross-family ensembles** recommended (models from same family show internal agreement bias)
5. **`--sample-ensemble N` flag** for CI cost control (e.g., `--sample-ensemble 0.2` = ensemble on 20% of tests)

**Deliverables:**
- [ ] `agentci/engine/judge.py` with structured output, ensembles, rubric templates
- [ ] 3 built-in rubric templates: `polite_refusal`, `factual_grounded`, `actionable_steps`
- [ ] `--sample-ensemble` CLI flag with percentage-based sampling
- [ ] Tests: mock judge calls, ensemble voting logic, threshold conversion

---

### Step 3: Granular Exit Codes & GitHub Annotations

**Objective:** First-in-class CI/CD integration with severity-mapped exit codes and inline PR annotations.

**Exit Code Specification:**

| Exit Code | Meaning | CI Effect | GitHub Annotation |
|---|---|---|---|
| `0` | All layers pass | ✅ Pipeline passes | None |
| `1` | Correctness failure (or forbidden tool) | ❌ Pipeline fails, blocks merge | `::error file=...::` |
| `0` (with annotations) | Correctness passes, Path/Cost warns | ✅ Pipeline passes | `::warning file=...::` |
| `2` | Runtime/infrastructure error | ❌ Pipeline fails (infra) | `::error::` |

**Implementation:**

```python
# agentci/engine/reporter.py
import sys
import json
from agentci.engine.results import QueryResult, LayerStatus

def report_results(
    results: list[QueryResult],
    format: str = "console",        # console | github | json | prometheus
    spec_file: str = "agentci.yaml"
) -> int:
    """Generate output and return appropriate exit code."""

    has_hard_failures = any(r.hard_fail for r in results)
    has_warnings = any(r.has_warnings for r in results)

    if format == "github" or _is_github_actions():
        _emit_github_annotations(results, spec_file)

    if format == "json":
        _emit_json(results)
    elif format == "prometheus":
        _emit_prometheus(results)
    else:
        _emit_console(results)

    # Exit code logic
    if has_hard_failures:
        return 1
    return 0  # Warnings are annotations, not exit code failures

def _is_github_actions() -> bool:
    import os
    return os.environ.get("GITHUB_ACTIONS") == "true"

def _emit_github_annotations(results: list[QueryResult], spec_file: str):
    """Emit GitHub Actions annotations for PR inline feedback."""
    for r in results:
        query_short = r.query[:60]

        if r.correctness.status == LayerStatus.FAIL:
            for msg in r.correctness.messages:
                # ::error prints red in the PR Files Changed tab
                print(f"::error file={spec_file}::"
                      f"[CORRECTNESS FAIL] {query_short}: {msg}")

        if r.path.status == LayerStatus.WARN:
            for msg in r.path.messages:
                # ::warning prints yellow — visible but non-blocking
                print(f"::warning file={spec_file}::"
                      f"[PATH] {query_short}: {msg}")

        if r.cost.status == LayerStatus.WARN:
            for msg in r.cost.messages:
                print(f"::warning file={spec_file}::"
                      f"[COST] {query_short}: {msg}")

def _emit_console(results: list[QueryResult]):
    """Rich console output with the three-tier report."""
    for r in results:
        print(f"\n{'='*60}")
        print(f"Query: {r.query}")
        _print_layer("✅" if r.correctness.status == LayerStatus.PASS else "❌",
                     "Correctness", r.correctness)
        _print_layer("📈", "Path", r.path)
        _print_layer("💰", "Cost", r.cost)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if not r.hard_fail)
    warned = sum(1 for r in results if r.has_warnings and not r.hard_fail)
    failed = sum(1 for r in results if r.hard_fail)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {warned} warnings, {failed} failures")

def _emit_json(results: list[QueryResult]):
    """JSON export for dashboards and external tooling."""
    output = {
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if not r.hard_fail),
            "failed": sum(1 for r in results if r.hard_fail),
            "warnings": sum(1 for r in results if r.has_warnings),
        },
        "results": [_serialize_result(r) for r in results]
    }
    print(json.dumps(output, indent=2))

def _emit_prometheus(results: list[QueryResult]):
    """Prometheus exposition format for Grafana dashboards."""
    for r in results:
        query_label = r.query[:40].replace('"', '\\"')
        # Correctness as boolean gauge
        val = 1 if r.correctness.status == LayerStatus.PASS else 0
        print(f'agentci_correctness_pass{{query="{query_label}"}} {val}')
        # Path metrics
        if "tool_recall" in r.path.details:
            print(f'agentci_tool_recall{{query="{query_label}"}} '
                  f'{r.path.details["tool_recall"]}')
        if "tool_precision" in r.path.details:
            print(f'agentci_tool_precision{{query="{query_label}"}} '
                  f'{r.path.details["tool_precision"]}')
        # Cost metrics
        if "actual" in r.cost.details:
            actual = r.cost.details["actual"]
            print(f'agentci_cost_usd{{query="{query_label}"}} {actual["cost_usd"]}')
            print(f'agentci_latency_ms{{query="{query_label}"}} {actual["latency_ms"]}')
            print(f'agentci_total_tokens{{query="{query_label}"}} {actual["total_tokens"]}')
```

**GitHub Actions Workflow Template** (updated `agentci init`):

```yaml
# .github/workflows/agentci.yaml
name: AgentCI Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install agentci
          pip install -r requirements.txt

      - name: Validate specs
        run: agentci validate agentci_spec.yaml

      - name: Run AgentCI evaluation
        run: agentci test --config agentci_spec.yaml --format github
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          # Annotations appear automatically in PR "Files Changed" tab

      - name: Export metrics (optional)
        if: always()
        run: agentci test --config agentci_spec.yaml --format json > eval_results.json

      - name: Upload results artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: agentci-results
          path: eval_results.json
```

**Deliverables:**
- [ ] `agentci/engine/reporter.py` with console, GitHub annotations, JSON, Prometheus output
- [ ] Exit code logic: 0 (pass), 1 (correctness fail), 2 (runtime error)
- [ ] Auto-detection of GitHub Actions environment
- [ ] Updated `agentci init` Jinja2 template with new workflow
- [ ] Tests: verify annotation format, exit code mapping

---

### Step 3.5: Baseline Guardrails & Versioning

**Objective:** Prevent saving broken baselines; enable A/B comparison across versions.

**Commands:**

```bash
# Save with automatic correctness precheck
agentci save --agent rag-agent --query "How do I install AgentCI?" --version v2-fixed

# Bypass precheck for intentional "broken" demos
agentci save --agent rag-agent --query "..." --force-save --version v1-broken

# Compare two baseline versions
agentci diff --baseline v1-broken --compare v2-fixed

# List available baseline versions
agentci baselines --agent rag-agent
```

**Baseline File Format:**

```json
{
  "version": "v2-fixed",
  "agent": "rag-agent",
  "captured_at": "2026-02-26T14:30:00Z",
  "metadata": {
    "model": "gpt-4o-mini",
    "spec_hash": "sha256:abc123...",
    "judge_rubric_ids": ["polite_refusal_v1"],
    "precheck_passed": true
  },
  "trace": { /* full trace object */ }
}
```

**Implementation:**

```python
# agentci/baselines.py
import hashlib, json
from pathlib import Path
from agentci.engine.correctness import evaluate_correctness

def save_baseline(
    trace: "Trace",
    agent: str,
    version: str,
    spec: "AgentCISpec",
    baseline_dir: str = "./baselines",
    force: bool = False
) -> Path:
    """Save a trace as a versioned golden baseline."""
    if not force:
        # Run precheck: quick correctness evaluation
        query_spec = _find_query_spec(trace.input_query, spec)
        if query_spec and query_spec.correctness:
            result = evaluate_correctness(
                trace.final_answer, query_spec.correctness, trace.raw,
                spec.judge_config
            )
            if result.status.value == "fail":
                raise ValueError(
                    f"Precheck failed — baseline does not pass correctness:\n"
                    f"{result.messages}\n"
                    f"Use --force-save to bypass."
                )

    # Compute spec hash for traceability
    spec_hash = hashlib.sha256(
        json.dumps(spec.model_dump(), sort_keys=True).encode()
    ).hexdigest()[:12]

    baseline = {
        "version": version,
        "agent": agent,
        "captured_at": _now_iso(),
        "metadata": {
            "model": trace.model_name,
            "spec_hash": f"sha256:{spec_hash}",
            "precheck_passed": not force,
        },
        "trace": trace.to_dict()
    }

    out_dir = Path(baseline_dir) / agent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{version}.json"
    out_path.write_text(json.dumps(baseline, indent=2))
    return out_path
```

**Deliverables:**
- [ ] `agentci/baselines.py` with save, load, list, precheck
- [ ] `--precheck-correctness` (default on) and `--force-save` flags
- [ ] `--version` flag on save command
- [ ] `agentci diff --baseline v1 --compare v2` for A/B comparison
- [ ] Rich metadata (model, spec hash, capture time) in baseline files

---

## Phase 1B: Path Metrics + Sequence Similarity

Add the full path evaluation engine once correctness and cost are validated on real traces.

### Step 2B: Advanced Path Metrics

**Objective:** Implement tool precision/recall, LCS sequence similarity, and match modes using real trace data to tune defaults.

**Implementation:** Already specified in Step 2 above. This sub-phase activates and tunes:
- `compute_tool_recall()` and `compute_tool_precision()` — set-based metrics
- `compute_sequence_lcs()` — primary sequence similarity (default)
- `compute_edit_distance_similarity()` — optional alternative
- Match modes: strict, unordered, subset (default), superset
- Loop detection with `max_loops` threshold

**Tuning Process:**
1. Run the RAG agent demo traces through the path engine
2. Observe natural variance across 5+ runs to set sensible defaults
3. Document recommended thresholds in `docs/metrics_reference.md`

---

## Phase 2: Refactored Diff Engine + Demo Sequence

### Step 4: Refactor the Diff Engine

**Objective:** Replace the flat structural diff with a three-tiered comparison report.

**Updated Diff Output Format:**

```
$ agentci diff --baseline v1-broken --compare v2-fixed

╔══════════════════════════════════════════════════════════════╗
║  AgentCI Diff: rag-agent (v1-broken → v2-fixed)            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ✅ Correctness: Unchanged (PASS → PASS)                    ║
║                                                              ║
║  📈 Path:                                                    ║
║     Tool calls:        11 → 0  (▼ 100%)                     ║
║     Tool precision:    0.09 → 1.00  (▲)                     ║
║     Tool recall:       1.00 → 1.00  (unchanged)             ║
║     Sequence LCS:      0.00 → 1.00                          ║
║     Loops detected:    3 → 0                                ║
║                                                              ║
║  💰 Cost:                                                    ║
║     Total cost:        $0.0080 → $0.0001  (▼ 98.8%)        ║
║     Total tokens:      4,200 → 180  (▼ 95.7%)              ║
║     LLM calls:         11 → 1  (▼ 90.9%)                   ║
║     Latency:           8,200ms → 1,100ms  (▼ 86.6%)        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**Implementation updates to existing `agentci/diff.py`:**

```python
# agentci/diff.py (updated)
from agentci.engine.correctness import evaluate_correctness
from agentci.engine.path import evaluate_path
from agentci.engine.cost import evaluate_cost

def diff_baselines(
    baseline_path: str,
    compare_path: str,
    spec: "AgentCISpec"
) -> "DiffReport":
    """Three-tiered diff comparing two baseline traces."""
    baseline = load_baseline(baseline_path)
    compare = load_baseline(compare_path)

    report = DiffReport(
        agent=baseline["agent"],
        from_version=baseline["version"],
        to_version=compare["version"],
    )

    # Correctness: evaluate both against spec
    query_spec = _find_query_spec(baseline, spec)
    if query_spec and query_spec.correctness:
        base_result = evaluate_correctness(
            baseline["trace"]["final_answer"], query_spec.correctness, baseline["trace"]
        )
        comp_result = evaluate_correctness(
            compare["trace"]["final_answer"], query_spec.correctness, compare["trace"]
        )
        report.correctness_delta = {
            "before": base_result.status.value,
            "after": comp_result.status.value,
            "changed": base_result.status != comp_result.status
        }

    # Path: compare metrics between traces
    report.path_delta = _compute_path_delta(baseline["trace"], compare["trace"])

    # Cost: compare efficiency
    report.cost_delta = _compute_cost_delta(baseline["trace"], compare["trace"])

    return report
```

**Also preserve existing DiffTypes** from the current engine:
- `DiffType.TOOLS_CHANGED` → now maps to Path layer detail
- `DiffType.COST_CHANGED` → now maps to Cost layer detail
- `DiffType.OUTPUT_CHANGED` → now maps to Correctness layer detail
- `DiffType.ROUTING_CHANGED` → now maps to Path layer handoff detail

**Deliverables:**
- [ ] Updated `agentci/diff.py` with three-tier report
- [ ] Console + JSON + GitHub annotation output modes
- [ ] Backward compatibility with existing DiffType enums
- [ ] `agentci diff --baseline v1 --compare v2` command

---

### Step 5: RAG Agent Demo Sequence

**Objective:** Prove the framework's value with the weather query before/after story.

**Demo Script:**

```bash
# 1. Validate the spec
agentci validate demos/rag/agentci_spec.yaml

# 2. Run against the BROKEN agent (11 tool calls for weather)
agentci test --config demos/rag/agentci_spec.yaml --format console
# Expected: Correctness PASS, Path WARN (11 tools > max 0), Cost WARN

# 3. Save the broken trace as v1 (bypass precheck for demo)
agentci save --agent rag-agent --version v1-broken --force-save

# 4. Apply the system prompt fix (pre-retrieval qualification)
# Edit the agent's system prompt to decline out-of-scope queries

# 5. Run again — should now decline immediately
agentci test --config demos/rag/agentci_spec.yaml --format console
# Expected: Correctness PASS, Path PASS (0 tools), Cost PASS

# 6. Save the fixed trace as v2
agentci save --agent rag-agent --version v2-fixed

# 7. Show the dramatic diff
agentci diff --baseline v1-broken --compare v2-fixed
# Shows: 11→0 tools, $0.008→$0.0001, 8200ms→1100ms
```

**RAG Demo Spec (`demos/rag/agentci_spec.yaml`):**

```yaml
version: 1
agent: rag-agent
baseline_dir: ./demos/rag/baselines

defaults:
  correctness:
    hallucination_check:
      rule: "Answer is grounded in retrieved context only"
      threshold: 0.8

judge_config:
  model: claude-sonnet-4-5-20250929
  temperature: 0

queries:
  - query: "How do I install AgentCI?"
    tags: [smoke, in-scope]
    correctness:
      expected_in_answer: ["pip install"]
      llm_judge:
        - rule: "Provides clear installation instructions"
          threshold: 0.7
    path:
      expected_tools: [retriever_tool]
      min_tool_recall: 1.0
      max_tool_calls: 5

  - query: "What's the weather in Tokyo?"
    tags: [edge-case, out-of-scope]
    correctness:
      not_in_answer: ["degrees", "forecast", "sunny", "rain"]
      llm_judge:
        - rule: "Agent politely declines, explaining this is outside its scope"
          threshold: 0.8
    path:
      max_tool_calls: 0
      forbidden_tools: [tavily_search, web_search]
    cost:
      max_llm_calls: 2
      max_total_tokens: 500

  - query: "How do I configure an AWS load balancer?"
    tags: [edge-case, out-of-scope, anti-hallucination]
    correctness:
      not_in_answer: ["ALB", "target group", "listener"]
      llm_judge:
        - rule: "Agent does NOT hallucinate AWS instructions from pre-trained knowledge"
          threshold: 0.9
    path:
      max_tool_calls: 3
```

**Deliverables:**
- [ ] `demos/rag/agentci_spec.yaml` — validated spec
- [ ] `demos/rag/baselines/v1-broken.json` — 11-tool weather trace
- [ ] `demos/rag/baselines/v2-fixed.json` — 0-tool weather trace
- [ ] Updated `manual_testing_playbook.md` with new demo steps

---

## Phase 3: Agent Porting & Scalability

### Step 6: Port Support Router & DevAgent

**Objective:** Validate that the YAML spec generalizes to multi-agent routing and deterministic analysis.

**Support Router Spec (`demos/support-router/agentci_spec.yaml`):**

```yaml
version: 1
agent: support-router
baseline_dir: ./demos/support-router/baselines

defaults:
  cost:
    max_cost_multiplier: 3.0

judge_config:
  model: claude-sonnet-4-5-20250929
  temperature: 0

queries:
  - query: "I want to cancel my subscription"
    tags: [billing, routing]
    correctness:
      llm_judge:
        - rule: "Response addresses subscription cancellation professionally"
          threshold: 0.7
    path:
      expected_handoff: BillingAgent
      expected_tools: [cancel_subscription]
      min_tool_recall: 1.0
      match_mode: subset

  - query: "My app keeps crashing on login"
    tags: [technical, routing]
    path:
      expected_handoff: TechnicalAgent
      expected_tools: [check_service_status, get_technical_documentation]
      match_mode: subset

  - query: "Tell me about the meaning of life"
    tags: [guardrail, off-topic]
    correctness:
      llm_judge:
        - rule: "Agent declines the off-topic request politely"
          threshold: 0.8
    path:
      forbidden_tools: [cancel_subscription, reset_password, update_account]

  # ... port remaining 29 queries from existing test_support_router.py
```

**Parallelization Strategy:**

```python
# agentci/engine/runner.py — parallel execution
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def run_spec_parallel(
    spec: "AgentCISpec",
    agent_runner: callable,
    max_workers: int = 4,         # Conservative default
    retry_on_infra_error: int = 2  # Retries for rate limits / timeouts
) -> list["QueryResult"]:
    """Execute spec queries in parallel with retry logic."""
    semaphore = asyncio.Semaphore(max_workers)

    async def run_with_limit(query):
        async with semaphore:
            for attempt in range(retry_on_infra_error + 1):
                try:
                    return await _evaluate_query(query, agent_runner, spec)
                except (RateLimitError, TimeoutError) as e:
                    if attempt < retry_on_infra_error:
                        wait = 2 ** attempt
                        logger.warning(f"Infra error on '{query.query[:30]}', "
                                      f"retry {attempt+1} in {wait}s: {e}")
                        await asyncio.sleep(wait)
                    else:
                        return _infra_error_result(query, e)

    results = await asyncio.gather(
        *[run_with_limit(q) for q in spec.queries]
    )
    return results
```

**pytest Integration:**

```python
# tests/test_with_agentci.py — pytest-native pattern
import pytest
from agentci import load_spec, run_spec

@pytest.fixture
def spec():
    return load_spec("agentci_spec.yaml")

@pytest.mark.parametrize("query_idx", range(32))  # pytest-xdist compatible
def test_support_router(spec, query_idx):
    if query_idx >= len(spec.queries):
        pytest.skip("Query index out of range")
    result = run_spec(spec, query_indices=[query_idx])
    assert not result[0].hard_fail, f"Correctness failed: {result[0].correctness.messages}"
```

**GitHub Actions Matrix Strategy (for large suites):**

```yaml
# .github/workflows/agentci.yaml (scaled)
jobs:
  evaluate:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        agent: [rag-agent, support-router, dev-agent]
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - run: pip install agentci
      - run: agentci test --config demos/${{ matrix.agent }}/agentci_spec.yaml --format github
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Deliverables:**
- [ ] `demos/support-router/agentci_spec.yaml` — all 32 queries ported
- [ ] `demos/dev-agent/agentci_spec.yaml` — deterministic tests ported
- [ ] Parallel execution with `--workers N` flag (default: 4)
- [ ] Retry/backoff for rate limits with `[INFRA]` labeling in output
- [ ] pytest-xdist compatibility via parametrized test pattern
- [ ] GitHub Actions matrix strategy template

---

## Implementation Priority & Timeline

```
Week 1 (Days 1-4): Phase 1A — Ship MVP
  Day 1: Step 0.5 (Schema + Validation) + Step 1 (YAML Loader)
  Day 2: Step 2 (Three-Layer Engine — correctness + basic path + cost)
  Day 3: Step 2.5 (Judge Safeguards) + Step 3 (Exit Codes + Annotations)
  Day 4: Step 3.5 (Baselines) + Step 5 (RAG Demo)

Week 1 (Days 5-7): Phase 1B + Phase 2
  Day 5: Step 2B (Advanced Path Metrics — tune on real traces)
  Day 6: Step 4 (Refactored Diff Engine)
  Day 7: Step 6 (Port Support Router + DevAgent)

Post-launch (Week 2+): Polish & Scale
  - pytest-xdist integration docs
  - Prometheus export + Grafana dashboard template
  - Auto-generate specs from existing test suites
  - pass^k reliability metric (multiple baseline captures)
```

---

## Testing Strategy for the Framework Itself

```
agentci/tests/
├── test_schema_validation.py      # 15+ tests: valid specs pass, invalid fail
├── test_yaml_loader.py            # Defaults merging, tag filtering, file refs
├── test_correctness_engine.py     # Deterministic checks, judge mocking
├── test_path_engine.py            # Every metric function + match modes
├── test_cost_engine.py            # Multiplier, token, latency bounds
├── test_metrics.py                # Edge cases: empty sequences, single items
├── test_judge_safeguards.py       # Ensemble voting, structured output parsing
├── test_reporter.py               # GitHub annotations format, exit codes
├── test_baselines.py              # Save, load, precheck, version comparison
├── test_diff_engine.py            # Three-tier diff output, backward compat
└── integration/
    ├── test_rag_demo.py           # Full RAG agent end-to-end
    ├── test_support_router.py     # Full Support Router end-to-end
    └── test_dev_agent.py          # Full DevAgent end-to-end
```

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| **Spec format** | YAML with Pydantic + JSONSchema | Only promptfoo is YAML-native; code-first excludes non-devs |
| **Default path metric** | Normalized LCS (sequence similarity) | Order-preserving, tolerates insertions, interpretable [0,1] |
| **Default match mode** | `subset` | Reference tools must appear, extras allowed — least brittle |
| **Judge temperature** | 0 (non-negotiable default) | Li et al. 2025; reduces but doesn't eliminate non-determinism |
| **Judge output** | Structured JSON (score/label/rationale) | Pydantic AI pattern; rationale forces chain-of-thought |
| **Ensemble strategy** | Majority vote, 3 judges, cross-family | 80%+ agreement at ≥2/3 judges; same-family bias documented |
| **Eval ordering** | Deterministic → semantic → LLM judge | Cost optimization; fast checks gate expensive judge calls |
| **Exit codes** | 0 (pass), 1 (correctness fail), 2 (infra error) | promptfoo pattern; warnings via annotations not exit codes |
| **GitHub annotations** | `::error` / `::warning` mapped to layers | **First-in-class** — no existing framework does this |
| **Hallucination/safety** | Sub-layer of Correctness (not top-level) | Prevents YAML bloat; these are hard-fail concerns |
| **Baseline precheck** | On by default, `--force-save` to bypass | Prevents propagating broken baselines; demo flexibility |
| **Parallelism default** | 4 workers with retry/backoff | Conservative; avoids rate limits and port conflicts |

---

## Appendix A: Metric Formulas Quick Reference

```
CORRECTNESS (Boolean):
  pass = all(deterministic_checks) AND all(judge_checks)

PATH (Continuous, mapped to PASS/WARN/FAIL):
  tool_recall       = |E ∩ U| / |E|        where E=expected, U=used
  tool_precision    = |E ∩ U| / |U|
  tool_f1           = 2·P·R / (P+R)
  sequence_lcs      = 2·|LCS(P,R)| / (|P|+|R|)
  sequence_edit     = 1 - ED(P,R) / max(|P|,|R|)
  loop_count        = Σ(consecutive repeated tool calls)

COST (Continuous, compared to baseline or absolute):
  cost_multiplier   = actual_cost / baseline_cost
  token_ratio       = actual_tokens / max_tokens
  latency_ratio     = actual_ms / max_ms
```

## Appendix B: Competitive Feature Matrix (Research-Backed)

| Feature | AgentCI 2.0 | promptfoo | DeepEval | LangSmith | Braintrust |
|---|---|---|---|---|---|
| YAML specs | ✅ | ✅ | ❌ | ❌ | ❌ |
| Schema validation | ✅ Pydantic | ✅ JSON Schema | ❌ | ❌ | ❌ |
| Trajectory metrics | ✅ P/R/F1/LCS | ❌ | ✅ ToolCorrectness | ✅ 5 match modes | ❌ |
| Layered severity | ✅ 3 layers | ❌ | ❌ | ❌ | ❌ |
| GitHub annotations | ✅ First-in-class | ❌ PR comments | ❌ | ❌ | ❌ PR comments |
| Trace regression | ✅ Golden baselines | ❌ | ❌ | ✅ Datasets | ✅ Experiments |
| Baseline versioning | ✅ | ❌ | ❌ | ❌ | ✅ |
| Judge ensembles | ✅ | ❌ | ❌ | ❌ | ❌ |
| Structured judge output | ✅ | ❌ | ✅ DAG/G-Eval | ❌ | ❌ |
| Multi-agent routing | ✅ Handoff assertions | ❌ | ❌ | ❌ | ❌ |
| Prometheus export | ✅ | ❌ | ❌ | ❌ | ❌ |
| Open source | ✅ | ✅ (MIT) | ✅ (Apache 2.0) | ❌ (open-core) | ❌ |
| Self-hosted | ✅ (CLI) | ✅ | ✅ | Enterprise only | Enterprise only |