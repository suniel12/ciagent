# KB-Derived Fact Checks — `ciagent generate-checks`

Most agent failures that matter involve a hard fact — a price, a rate, a
product, a version. Those are checkable in code, deterministically, for
free. `generate-checks` mines your knowledge base for those facts and
proposes them as deterministic assertions on your existing spec queries,
reserving the LLM judge for answers with nothing checkable.

```bash
ciagent generate-checks                  # interactive review of proposals
ciagent generate-checks --dry-run        # look, don't touch
ciagent generate-checks --yes            # accept everything the gate validated
ciagent generate-checks --kb ./docs      # explicit KB directory
```

Extraction uses an LLM **once, at authoring time**. The generated checks run
deterministically forever, at zero cost — that asymmetry is the point.

## The brittleness gate

A tool that flags flaky evals must not generate brittle string checks that
fail correct paraphrases. Three rules are built in:

1. **Only non-paraphrasable facts are extracted** — numbers, rates, SKUs,
   codes, versions, explicit quantities ("30 days"). Prose facts become
   variant sets (`any_expected_in_answer: ["4.5%", "4.5 percent"]`), never a
   single literal string.
2. **Every candidate is validated against your recorded golden answers
   before you see it.** A check that would fail a known-good answer is
   rejected automatically, with the failing answer shown. Record baselines
   first (`ciagent record`) — the gate is only as strong as the answers it
   can test against.
3. **Nothing is written silently.** Gate survivors go through interactive
   review; `--yes` accepts validated checks only. Candidates whose query has
   no golden answer can't be gated — they are flagged `unvalidated` and are
   **never** auto-accepted, even with `--yes`.

## What it will and won't do

- Fills only **empty** fields — a user-written assertion is never
  overwritten (skips are reported).
- Appends new variants to existing keyword lists, deduplicated.
- Proposes only `any_expected_in_answer`, `not_in_answer`, and
  `regex_match` (`expected_in_answer`'s AND-logic is too brittle to
  generate).
- Queries with no checkable hard fact get **nothing** — "no checks" is a
  valid outcome; judgment-only queries belong to the judge
  (see [judge-audit](judge-audit.md)).
- Writing rewrites the spec YAML (a `.bak` backup is kept; comments are not
  preserved). Use `--dry-run` first if that matters.

## Suggested workflow

```bash
ciagent record <tests>        # 1. record known-good baselines
ciagent generate-checks       # 2. mine KB facts, gate against baselines
ciagent test --mock --yes     # 3. sanity-check spec structure, zero cost
ciagent test --runs 3         # 4. confirm checks are stable across runs
```
