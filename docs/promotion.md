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

## Enabling staging (opt-in in v1)

Staged files contain the raw conversation text (possibly PII) and no redactor
is wired yet, so staging is **off by default**. Turn it on per run with
`--stage`, or in the spec:

```yaml
staging:
  enabled: true
  cap: 10           # staged conversations kept per scenario (newest win)
  max_age_days: 30  # age cutoff for retention GC
```

`staging: true` is accepted as a bool shorthand. Files land under
`.ciagent/staged/<agent>/<scenario-id>/<run-ts>-<hash>.json`; `ciagent init`
adds `.ciagent/staged/` to `.gitignore` so they are never committed. When a
failure is found with staging off, simulate prints a one-line notice instead
of writing anything.

Staging is best-effort by contract: a staging error prints a warning and
never changes the run's exit code.

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

Promoted goldens use the `gate` lifecycle: `simulate --replay` exits 1 while
the bug still reproduces, and goes green on its own once the agent is fixed.
An `--xfail` lifecycle (expected-fail until fixed, then flip) is deferred to
v2.

## Envelope compatibility

`staging:` and `provenance:` are additive optional fields on the schema-v2
envelope. Files without them are written byte-identical to pre-0.11 goldens,
and old loaders ignore the new keys. The golden trace format remains
backward-compatible.
