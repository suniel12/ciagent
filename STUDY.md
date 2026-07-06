# STUDY.md: Provenance for the Numbers

AgentCI's launch materials quote three findings. This document is where they come from:
the system, the data, the hand-grading protocol, and the derivation of each figure,
including the parts of the study that went wrong before they went right.

**The claims:**

1. **~1 in 7 answers the LLM judge passed were wrong** (8 of 53 judge-passed answers
   in a blind, hand-graded sample).
2. **Same question, same knowledge base, three runs hours apart: the agent answered
   correctly once. The judge passed all three.**
3. **Deterministic checks caught 8 of 8 of those silent failures** with zero LLM calls,
   while covering 36 of the 58 sampled answers (the other 22 had no checkable fact).

Study conducted July 4–5, 2026 on eval data recorded March–April 2026. Author: Sunil
Pandey, who is both the maintainer of AgentCI and the builder of the system under test.
That conflict is real and disclosed; the protocol below exists to contain it.

---

## 1. The system under test

**ActDesk**, my AI customer-support agent, running against **Alpine Gear Co.**
(`actdeskai.myshopify.com`), a demo Shopify store I operate. The store is synthetic;
the pipeline is not: the eval harness exercises the production agent code, retrieval,
and tools.

- **Agent model:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`). Temperature was
  not set, so the provider default (~1.0) applied. That default is the mechanical
  source of the run-to-run variance in §6.
- Up to 3 tool steps per question. Read-only tools: knowledge-base search (RAG over
  store policies), order lookup, product search. Fresh context per question, no chat
  history.
- Production ActDesk can escalate to Claude Sonnet 4.5; escalation was disabled
  during eval. Every result here describes the Haiku-only path.

## 2. The automated judge

Cross-vendor by design: **OpenAI GPT-5.4-nano** (`gpt-5.4-nano`, temperature 0,
structured outputs) grades the Claude agent, which rules out "Claude grading Claude"
self-bias as an explanation for leniency.

The judge is two model calls plus a deterministic formula, not a single 1–5 rating:

- **Faithfulness:** decompose the answer into atomic claims; verify each against the
  retrieved context and the golden answer. `faith_score` = supported ÷ total claims.
- **Task completion:** boolean, "did it address the need?"
- **Path:** deterministic tool recall/precision. No model.
- `composite = clamp(faith_score × 3 + (task_completed ? 1.5 : 0) + forbidden_penalty
  + 0.5, 1, 5)`. Label: ≥ 4 pass, ≤ 2 fail, 3 borderline.

## 3. The data and its scoping

```
1,234 answers graded by the judge (17 runs, 416 unique questions)
   └─ scoped to one fixed KB snapshot ("hash C"): 260 answers
      = 252 judge-passes + 5 judge-fails + 3 borderline
        └─ blind sample: 58 answers = 53 judge-passes + 5 judge-fails
```

The 17 runs span three store-data generations; policies, products, and golden answers
all changed between generations. Hand-grading is only defensible against a knowledge
base you can hold fixed, so everything below is scoped to the final generation:
four runs on one KB snapshot (six policies plus a product catalog; the FAQ section was
empty). Three of those runs are a same-day rerun triple from April 10, 2026; the
fourth ran April 12.

Two ground-truth corrections applied before grading, both documented in the review
files:

- A few golden answers predated the shipping policy and claimed rates were "not in
  policies." The rates entered the KB around April 1, so those goldens are stale.
  **Grading used the KB as ground truth wherever a golden conflicted with it.**
- Golden answers were regenerated on every store change. Reviewing them side by side
  showed paraphrase variation with stable meaning, so they are usable as ground truth
  with the correction above.

## 4. The blind hand-grading protocol

A stratified sample of 58 answers, built by a script (`blind-check.ts`, seed 42):

1. every answer from the April 10 triple whose judge verdict flipped across the
   triple's three runs,
2. all 5 judge-fails in the fixed-KB slice (there were only 5), and
3. a seeded random draw of judge-passes.

That yields 53 judge-passes and 5 judge-fails, weighted toward passes on purpose: a
false pass can only surface when you regrade something the judge passed.

The script shows the question, the agent's answer, and the golden answer. It hides
the judge's verdict and shuffles the order so the stratum cannot be inferred. I graded
all 58 myself in one sitting on July 5, 2026 (29 minutes, resume-safe log with
timestamps), after studying the fixed KB. The grading rule, verbatim from the review
doc:

> **CORRECT** if the answer either (a) gives the right answer available in the KB /
> via a lookup the agent has, or (b) for genuinely-unknowable questions, correctly
> says so and points to the right next step.
> **INCORRECT** if it (a) states something false, or (b) fails to give an answer that
> was actually available (in the KB, standard Shopify behavior, or a tool it has),
> i.e., confident evasion.

**What got discarded along the way, and why:**

- **A first grading round of 8 answers (July 4) was thrown out.** I graded them before
  studying the KB and caught myself scoring against wrong assumptions about what the
  store's policies contained. The discarded grades are preserved in a separate log
  file; the 58 grades above started from an empty file after the KB review.
- **An earlier, non-blind review pointed the wrong direction and was superseded.**
  Before the blind protocol I reviewed 40 judge-fails and 40 judge-passes with the
  verdicts visible, and came away believing most failures were fluent hallucinations.
  The blind check reversed that: many of the "fails" I had endorsed were correct
  answers, and the real asymmetry ran the other way (a lenient judge, §5). No number
  from the non-blind review appears in AgentCI's materials.

## 5. Results: auditing the judge

| Measure | Value |
|---|---|
| Agreement, 3-way (pass/borderline/fail) | 44/58 = 75.9% |
| Agreement, binary (pass vs not-pass) | 45/58 = 77.6% |
| False passes (judge passed, I failed) | **8** |
| False fails (judge failed, I passed) | 3 |
| Judge-passes in sample | 53 |
| **Silent-failure rate among judge-passes** | **8/53 = 15.1% ≈ 1 in 7** |

Where we disagreed, the judge was lenient roughly 3:1. All 9 answers I failed were
fluent, non-empty (the shortest ran 211 characters), and error-free; in a production
log, every one records as a success. Hence "silent failure."

5 of the 8 false passes were product questions. The pattern: the agent's product
search misses an item, the agent states a wrong catalog fact with confidence, and the
judge, which never sees the live catalog, verifies claims against the same retrieved
nothing. One verbatim example:

> **Q:** "whats the price range for your snowboards?"
> **Agent:** "It looks like we don't currently have snowboards in stock. Our catalog
> doesn't show any snowboard products available at the moment. …"
> **Golden:** snowboard range is $600–$2,629.95.
> **Judge:** score 5, pass.

**Citation rail:** quote this as "~1 in 7 (8 of 53)." The sample is 53 judge-passes;
treating 15.1% as a precise population rate would overclaim.

## 6. Results: a stable score over an unstable system

The April 10 triple ran the same 62 questions against the same KB three times, hours
apart (run ids `6dc5c356`, `da8d13b9`, `64acad29`):

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Mean judge score | 95.2% | 96.1% | 96.1% |
| Judge pass rate | 96.8% | 98.4% | 96.8% |

The aggregate reads rock-solid. Underneath it: **all 62 questions produced three
textually distinct answers** (62 of 62), **4 of 62 changed judge verdict** across
runs, and 2 of those were hard pass↔fail flips.

The centerpiece, "how much does shipping cost?" The $9.95 / free-over-$100 rate was
in the KB for all three runs:

| Run | Agent answer (trimmed) | Correct? | Judge |
|---|---|---|---|
| 1 | "I searched our knowledge base for shipping cost details, but unfortunately our knowledge base doesn't include…" | wrong | pass (5) |
| 2 | "Great question! Here are our shipping rates: … Standard $9.95 … free over $100 …" | right | pass (5) |
| 3 | "I don't have access to specific shipping rates in our knowledge base…" | wrong | pass (5) |

Same input, same KB: right once, and the judge passed all three. Retrieval
nondeterminism produced the variance; the judge, sharing the agent's context,
rubber-stamped both evasions.

The two hard flips split cleanly across failure sources, which is why AgentCI's
stability report attributes flips instead of just counting them: "do you process
orders on holidays?" scored `[2, 5, 5]` from the judge while I graded the underlying
answers `[fail, pass, pass]` (the agent's answers changed: agent variance). "can I
have multiple addresses on my account?" scored `[5, 2, 2]` while I graded `[pass,
borderline, pass]` (judge noise on similar answers).

## 7. Results: the deterministic replay

On July 5, 2026 I replayed all 58 recorded answers through ciagent's deterministic
correctness layer (AgentCI v0.6.0). Zero LLM calls; the agent was never re-run.

**Check-authorship rule:** every check derives from the golden answers and the KB
document only. I did not consult agent answers while writing checks. Hard-fact
questions (prices, rates, dates, the support email, product variant names) got
`expected_in_answer` / `any_expected_in_answer` / `regex_match` checks. Judgment-only
questions ("should offer to look up the order") got no deterministic check; those
belong to a judge or a human.

| Measure | Value |
|---|---|
| Answers with ≥ 1 deterministic check | 36 of 58 |
| Answers with no checkable fact | 22 of 58 |
| **Silent failures caught (of §5's 8)** | **8 of 8** |
| Answers I failed that were caught | 9 of 9 |
| False alarms among the 46 answers I passed | 2 (~4%) |

All 8 silent failures fell inside the checkable 36, and that is not luck: this agent
failed by stating, or confidently evading, checkable facts, and a presence check fails
"I don't have that information" exactly as it fails a wrong number.

The 2 false alarms were the checks being stricter than me rather than wrong: one
answer omitted a fulfillment window the golden requires; one deferred to support
without giving the support email address (the LLM judge failed that one too). Also
honest: 2 of the judge's 3 false-fails sat in the uncheckable 22, where only a judge
or a human can rule.

A real `agentci test` run against the worst run of the triple (14 checked queries)
returned 8 pass / 6 fail; the 6 failures were that run's 5 silent failures plus its
1 borderline. The CLI output is preserved with the study artifacts.

**Citation rails:** "8 of 8" is this sample, one agent, one store. The checks
required golden answers containing hard facts, which these had. Deterministic checks
shrank the surface that had to be delegated to an LLM judge from 58 answers to 22;
they do not close the judge problem, and AgentCI does not claim they do.

## 8. Limitations

- **One agent, one judge, one store.** The judge architecture (claim decomposition,
  cross-vendor, temperature 0) is a reasonable one; other judges may behave better or
  worse. The store is a demo store, though the code path is production.
- **Small n everywhere.** 58 graded answers, 53 judge-passes, 8 false passes. The
  fractions are exact for this sample; the rates are estimates with wide error bars.
- **The grader is the author.** I was blind to the judge's verdicts but not to the
  study's purpose, and I maintain the tool whose results §7 reports. The mitigations
  are the blind shuffle, the written grading rule, the discarded-round and
  superseded-review disclosures in §4, and raw grade logs with timestamps.
- **Checks were authored after the grades existed, on the same 58 answers.** The
  containment was mechanical derivation from goldens + KB only; the 2 false alarms
  are evidence the checks were not tuned to reproduce my grades. A pre-registered
  version of §7 on fresh data would be stronger.
- **Temperature default (~1.0) inflates agent variance** relative to a temperature-0
  deployment. It does not explain the judge findings: the judge graded at
  temperature 0, and §5's false passes are same-run, single-answer events.
- **Retrieval-miss vs reasoning-miss is not separated.** The shipping example shows
  the information was present and not retrieved, but per-answer retrieved context was
  not exported for the full sample.

## 9. Reproduction

The raw artifacts live in the ActDesk application repo (private, since it is a
commercial codebase): the 1,234-row eval export, the KB snapshot rendered to
markdown, the blind-grading script and both grade logs (kept and discarded rounds),
the golden-answer review, and the ciagent replay spec, per-answer results, and CLI
output. Ask (GitHub issue or DM) and I will share the export and grade logs.

Every number in this document was recomputed from those raw files with a fresh
stdlib-Python script on July 6, 2026, before publication: row counts, the scoping
funnel, both agreement figures, the false-pass/false-fail split, the triple's
per-run scores, the 62/62 text-distinctness count, the verdict-change and hard-flip
counts, the centerpiece verdicts, and every figure in §7's table. All matched the
working notes.
