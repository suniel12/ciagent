# Golden promotion — `ciagent stage` + `ciagent promote`

`--record` requires you to decide to record *before* you know there's a bug.
Generative personas are nondeterministic by design: if a failure surfaces in a
run without `--record`, the repro is gone, and a re-run may never rediscover
the same conversation. Staging closes that gap. With staging enabled, every
failing `simulate` conversation is captured automatically; promoting one to a
permanent CI gate stays a single human "yes".

```bash
ciagent simulate --runs 3 --stage   # failing conversations auto-staged + triaged
ciagent stage list                  # best-to-promote first
ciagent promote <id>                # one staged failure becomes a golden CI gate
ciagent simulate --replay ./golden  # exit 1 while the bug reproduces
```

## Staging is on by default, with capture-time redaction

Since 0.12, staging is **on by default**: every failing simulate conversation
is captured automatically, and staged files are scrubbed of secrets and PII
before they hit disk. Spec surface:

```yaml
staging:
  enabled: true        # default; `staging: false` disables
  redact: true         # default; false writes RAW text and warns
  redact_patterns: []  # extra regexes to scrub, optional
  cap: 10              # staged conversations kept per scenario (newest win)
  max_age_days: 30     # age cutoff for retention GC
```

Staging covers both surfaces: multi-turn `simulate` scenarios and single-turn
`test` queries (live runs only — mock failures are synthetic and never
stage). A failing query stages as a one-turn envelope carrying its
correctness checks, so verify, replay, and promote work on it unchanged.

`staging: false` is accepted as a bool shorthand. Files land under
`.ciagent/staged/<agent>/<scenario-id>/<run-ts>-<hash>.json`; the gitignore
entry for `.ciagent/staged/` is scaffolded by `ciagent init` and on the
first auto-stage, so staged files are never committed. When a failure is
found with staging explicitly disabled, simulate prints a one-line notice
instead of writing anything.

Staging is best-effort by contract: a staging error prints a warning and
never changes the run's exit code.

### What redaction scrubs

Deterministic patterns only (no LLM, no entropy scanning), applied over every
string in the envelope with a key-aware walk (a value under `api_key`,
`token`, `password`, and similar keys is redacted whatever its shape):

- Known key prefixes: OpenAI/Anthropic `sk-…`, AWS `AKIA…`, GitHub `ghp_…`,
  Slack `xox…`, Google `AIza…`, Stripe `sk_live_…`.
- `key=value` / `Authorization: Bearer …` contexts inside strings.
- Emails, phone numbers, card numbers (Luhn-checked). Placeholders are
  shape-preserving (`redacted-1@example.com`, `+1-555-0100`) so replays keep
  behaving like the original conversation; the same value always maps to the
  same placeholder within an envelope.
- Your own `redact_patterns` regexes.

The staging block records what happened
(`redaction: {applied, counts, degraded?}`). `stage show` and `--export`
re-redact with the current config on every read, which also covers staged
files written before 0.12. One caveat: if a scenario's check literals get
redacted (say a `not_in_answer` leak-gate on a real key), `stage verify` and
`promote` warn that the check semantics are degraded; express leak-gates as
regex checks, which redaction does not rewrite. Full design:
`Plan_docs/redaction_capture.md`.

## Triage classification

With `--runs N` (N > 1), each staged failure carries a classification derived
from the existing stability attribution:

| Class | Meaning | Promotable? |
|---|---|---|
| `consistent` | reproducible failure, every run failed | yes |
| `flaky-agent` | real distribution bug (agent or retrieval variance) | yes |
| `unverified` | seen in a single run only | after `stage verify`, or `--force` |
| `held` | not a clean agent signal (simulation variance, judge flake, mixed) | `--force` only |
| `held-infra` | infra noise (timeouts, 5xx) | `--force` only |

One deliberate limitation: `consistent` promises **reproducibility, not fault
location**. A broken rubric, a deterministic-eval bug, and a real agent bug
all reproduce identically. The classification tells you a human should look;
it never claims the agent is at fault.

## The stage group

```bash
ciagent stage list [--agent A] [--classification consistent] [--format json]
ciagent stage show <id> [--export shared.json]
ciagent stage verify <id> [--runs 3] [--mock]
ciagent stage drop <id> | --held | --all
ciagent stage gc
```

- **`list`** sorts best-to-promote first: `consistent` → `flaky-agent` →
  `unverified` → `held` → `held-infra`, then newest first. Ids accept unique
  prefixes everywhere.
- **`show --export`** writes a copy for sharing in an issue or PR. v1 applies
  **no redaction**; the export contains the raw conversation text, and the
  command says so. Review before sharing.
- **`verify`** re-runs the scenario N times, replaying the staged user turns
  verbatim (the persona is never re-rolled — it verifies agent-side
  reproducibility, not simulation luck), then re-classifies in place. This is
  the cheap path from `unverified` to `consistent`.
- **`gc`** applies retention: the per-scenario `cap`, the `max_age_days`
  cutoff, and a global cap (500 files / 50 MB) so staging can never leak disk.
  The same GC also runs automatically on every write.

## Promote

```bash
ciagent promote            # interactive picker, best-first
ciagent promote <id>       # one staged failure → golden
ciagent promote <id> --force
```

`promote` moves the envelope into `<baseline_dir>/<agent>/scenarios/` —
exactly where `--record` writes — swapping the `staging:` block for an
additive `provenance:` block (staged run id, classification at promotion,
timestamp, lifecycle). The staged copy is consumed on success; a refused or
failed promote leaves it untouched.

Two gates protect the golden suite:

1. **Classification gate.** `held`, `held-infra`, and `unverified` entries are
   refused without `--force`, with the reason printed. Bulk promote does not
   exist — promoting one id at a time is the design, because rubber-stamping
   a directory of snapshots is the failure mode this feature replaces.
2. **Replay gate.** The envelope is re-validated with the same structural gate
   the record path uses before anything is written. A golden that cannot
   replay is never created.

### Lifecycles: gate and xfail

The state machine is `staged → promoted(gate|xfail) → fixed(flip)`:

- **`gate`** (default): `simulate --replay` exits 1 while the bug still
  reproduces. CI is red until the fix lands, and goes green on its own once
  the agent is fixed. Honest, but harsh if the fix isn't landing this sprint.
- **`xfail`** (`promote --xfail`): the failure is EXPECTED, so replay stays
  green (exit 0) while the bug reproduces, shown as `XFAIL` in the report.
  The repro is banked in CI without blocking merges. When a replay suddenly
  passes, it is flagged `XPASS` with the exact flip command; CI stays green
  (non-strict xpass, same semantics as pytest).
- **`promote --flip <golden>`**: the human acknowledgment that the fix
  landed. Converts the xfail golden to a normal `gate` golden (stamping
  `flipped_at`), so replay exits 1 if the bug ever comes back. Flipping a
  non-xfail golden is refused.

`--format json` carries the fold: each scenario reports `lifecycle` and
`xpass`, and the summary adds `xfail_expected` and `xpass` counts.

## Envelope compatibility

`staging:` and `provenance:` are additive optional fields on the schema-v2
envelope. Files without them are written byte-identical to pre-0.11 goldens,
and old loaders ignore the new keys. The golden trace format remains
backward-compatible.
