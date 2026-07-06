# CIAgent v2 Metrics Reference

Mathematical definitions for every metric in the three-layer evaluation engine.

---

## Layer 1: Correctness (Hard Fail)

Correctness checks are boolean — any single failure causes `LayerStatus.FAIL`.

| Check | Formula / Rule |
|-------|---------------|
| `expected_in_answer` | `∀ term ∈ expected: term.lower() in answer.lower()` |
| `not_in_answer` | `∀ term ∈ forbidden: term.lower() not in answer.lower()` |
| `exact_match` | `answer.strip() == expected.strip()` |
| `regex_match` | `re.search(pattern, answer) is not None` |
| `json_schema` | `answer` parses as JSON conforming to the JSON Schema |
| `llm_judge` | `score ≥ threshold × 5` (converted to 1–5 scale) |
| `safety_check` | Same as `llm_judge` sub-check |
| `hallucination_check` | Same as `llm_judge` sub-check |

### Evaluation Order (Cost Optimisation)

Deterministic checks always run first. LLM judge calls are skipped if any deterministic check has already failed.

```
1. expected_in_answer  (O(1), free)
2. not_in_answer       (O(1), free)
3. exact_match         (O(1), free)
4. regex_match         (O(n), free)
5. json_schema         (O(n), free)
6. llm_judge           (API call, ~$0.001 per check)
7. safety_check        (API call, skipped if 6 failed)
8. hallucination_check (API call, skipped if 7 failed)
```

---

## Layer 2: Path (Soft Warning)

Path metrics produce `LayerStatus.WARN` (not FAIL) on exceedance.
**Exception:** `forbidden_tools` violations escalate to `LayerStatus.FAIL`.

### Set-Based Metrics

Let **E** = set of expected tools, **U** = set of used tools.

```
tool_recall    = |E ∩ U| / |E|
              → "What fraction of expected tools were used?"
              → 1.0 when all expected tools appear; 0.0 when none do

tool_precision = |E ∩ U| / |U|
              → "What fraction of used tools were expected?"
              → 1.0 when only expected tools used; lower when extras added

tool_f1        = 2 × (precision × recall) / (precision + recall)
              → Harmonic mean; 1.0 when both are perfect
```

**Edge cases:**
- Empty expected set → recall = 1.0 (no expectations to violate)
- Empty used set + empty expected → precision = 1.0, recall = 1.0

### Sequence-Based Metrics

Let **P** = predicted tool sequence (list), **R** = reference tool sequence.

#### Normalised LCS Similarity (default)

```
LCS(P, R) = length of Longest Common Subsequence

sequence_lcs = 2 × |LCS(P, R)| / (|P| + |R|)

Range: [0, 1]
  0.0 → completely disjoint sequences
  1.0 → identical sequences

Properties:
  - Order-preserving (rewards correct ordering)
  - Tolerates insertions/deletions without full penalty
  - Symmetric: LCS(P, R) = LCS(R, P)
```

**Example:**
- P = [search, rerank, generate], R = [search, generate] → LCS=2, sim = 2×2/(3+2) = 0.80

#### Normalised Edit Distance Similarity (alternative)

```
ED(P, R) = minimum edits (insert / delete / substitute) to transform P → R
           (Levenshtein distance)

sequence_edit = 1 - ED(P, R) / max(|P|, |R|)

Range: [0, 1]
  0.0 → completely different (max edits required)
  1.0 → identical (zero edits)
```

### Match Modes

| Mode | Rule | Use Case |
|------|------|----------|
| `strict` | `P == R` (exact list equality) | Deterministic agents; exact sequence required |
| `unordered` | `set(P) == set(R)` | Same tools, order doesn't matter |
| `subset` *(default)* | `set(R) ⊆ set(P)` | Reference tools must appear; extras OK |
| `superset` | `set(P) ⊆ set(R)` | All used tools must be in reference set |

### Loop Detection

```
loop_count = number of consecutive identical tool invocations

Example: [search, search, grade, grade, grade]
         → search-search (1) + grade-grade (1) + grade-grade (1) = 3 loops
```

---

## Layer 3: Cost (Soft Warning)

All cost checks produce `LayerStatus.WARN` on exceedance — they never block CI.

| Metric | Formula |
|--------|---------|
| `max_cost_multiplier` | `actual_cost / baseline_cost ≤ max_multiplier` |
| `max_total_tokens` | `trace.total_tokens ≤ max_total_tokens` |
| `max_llm_calls` | `trace.total_llm_calls ≤ max_llm_calls` |
| `max_latency_ms` | `trace.total_duration_ms ≤ max_latency_ms` |
| `max_cost_usd` | `trace.total_cost_usd ≤ max_cost_usd` |

**Cost multiplier** requires a baseline trace. If no baseline is provided or baseline cost is 0, the check is skipped.

---

## LLM Judge Score Mapping

Rubric thresholds are specified in [0, 1] and converted to the 1–5 judge scoring scale:

```
threshold_score = int(threshold × 5 + 0.5)   # standard rounding (not banker's)

Examples:
  0.0 → 1    (any non-zero score passes)
  0.2 → 1
  0.5 → 3
  0.7 → 4
  0.8 → 4
  1.0 → 5    (only a perfect score passes)
```

A judge verdict passes when `score ≥ threshold_score`.

---

## Recommended Thresholds

Based on empirical testing with Anthropic models:

| Use Case | Metric | Recommended Value |
|----------|--------|------------------|
| Polite refusal | `llm_judge` threshold | 0.8 |
| Factual grounding | `hallucination_check` threshold | 0.8 |
| Actionable steps | `llm_judge` threshold | 0.7 |
| Tool recall (must-use tools) | `min_tool_recall` | 1.0 |
| Sequence similarity | `min_sequence_similarity` | 0.6–0.8 |
| Cost regression guard | `max_cost_multiplier` | 2.0–3.0 |
| Out-of-scope queries | `max_tool_calls` | 0 |
