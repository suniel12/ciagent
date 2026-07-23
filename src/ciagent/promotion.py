# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Golden Promotion Pipeline (v1): auto-stage → triage → one-command promote.

Three collaborators, split by responsibility (SOLID):

- ``StageStore``      — storage only: atomic writes, retention GC, redaction
                        hook. Knows nothing about classification.
- ``TriageClassifier``— a pure function mapping the *existing* stability/flip
                        attribution onto a promotion class. No I/O.
- ``PromotionService``— move + provenance stamping + lifecycle. Depends on
                        ``StageStore`` and the baseline-dir seam, never on
                        concrete CLI paths.

Every failing ``simulate`` conversation is captured automatically (opt-in in
v1) so a nondeterministic repro is never lost; promoting one to a permanent CI
gate stays a single human "yes".
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .conversation import (
    ConversationEnvelope,
    ConversationTurn,
    load_envelope,
    normalize_to_envelope,
    save_envelope,
)
from .engine.simulate import scenario_slug
from .engine.stability import FlipSource

DEFAULT_STAGED_DIR = ".ciagent/staged"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _identity(env: ConversationEnvelope) -> ConversationEnvelope:
    """Default no-op redactor. Replaced by the real capture-time redactor when
    the redaction ADR lands — this is a declared stub, not done work."""
    return env


# ── Triage classification ───────────────────────────────────────────────────────


class Classification(str, Enum):
    """Promotion triage class for a staged failing conversation."""
    CONSISTENT = "consistent"      # reproducible failure — ready to promote
    FLAKY_AGENT = "flaky-agent"    # real distribution bug (agent or retriever)
    HELD = "held"                  # not a clean agent signal — not a candidate
    HELD_INFRA = "held-infra"      # infra noise — excluded
    UNVERIFIED = "unverified"      # single run — verify or --force to promote


# Meaning shown to the user. `consistent` promises REPRODUCIBILITY, never fault
# location: a broken rubric, a deterministic-eval bug, or a failing retrieval
# assertion all land here too — it is NOT a verdict that the agent is at fault.
CLASSIFICATION_MEANING: dict[Classification, str] = {
    Classification.CONSISTENT: "reproducible failure — a human should look at it "
                               "(NOT attributed to the agent)",
    Classification.FLAKY_AGENT: "real distribution bug — promote as a stability scenario",
    Classification.HELD: "not a clean agent signal — not a promote candidate",
    Classification.HELD_INFRA: "infra noise — excluded, retry before trusting",
    Classification.UNVERIFIED: "single run — verify or --force before promoting",
}


class StabilityLike(Protocol):
    """The narrow surface both ScenarioStability and QueryStability expose."""

    @property
    def always_failed(self) -> bool: ...

    @property
    def flipped(self) -> bool: ...

    flip_source: Optional[FlipSource]


# The flip_source → class map is the open-closed extension point: future
# world-sim fault sources add one entry here, no branching change.
FLIP_SOURCE_TO_CLASS: dict[FlipSource, Classification] = {
    FlipSource.AGENT_VARIANCE: Classification.FLAKY_AGENT,
    FlipSource.RETRIEVAL_VARIANCE: Classification.FLAKY_AGENT,
    FlipSource.SIMULATION_VARIANCE: Classification.HELD,
    FlipSource.JUDGE_FLAKE: Classification.HELD,
    FlipSource.MIXED: Classification.HELD,
    FlipSource.INFRA_ERROR: Classification.HELD_INFRA,
    FlipSource.WORLD_MISS: Classification.HELD,
}


class TriageClassifier:
    """Pure mapping from existing stability output → promotion class."""

    @staticmethod
    def classify(stability: Optional[StabilityLike], *, runs: int) -> Classification:
        if stability is None or runs <= 1:
            return Classification.UNVERIFIED
        if not stability.flipped:
            # every run failed the same way — reproducible, not necessarily a
            # bug in the agent (see CLASSIFICATION_MEANING).
            if stability.always_failed:
                return Classification.CONSISTENT
            # not flipped and not always-failed means every run passed — nothing
            # to stage. Treat as unverified (should not reach staging anyway).
            return Classification.UNVERIFIED
        source = stability.flip_source
        if source is None:
            return Classification.HELD
        return FLIP_SOURCE_TO_CLASS.get(source, Classification.HELD)


def build_staging_block(
    *,
    run_id: str,
    scenario_id: str,
    source: str,
    classification: Classification,
    stability: Optional[StabilityLike],
    runs_observed: int,
    failure_summary: str,
    now: Callable[[], datetime] = _utcnow,
) -> dict[str, Any]:
    """Assemble the additive `staging:` block for one staged conversation."""
    verdicts = list(getattr(stability, "verdicts", []) or [])
    flip_source = getattr(stability, "flip_source", None)
    return {
        "run_id": run_id,
        "staged_at": now().isoformat(),
        "scenario_id": scenario_id,
        "source": source,
        "classification": classification.value,
        "runs_observed": runs_observed,
        "verdicts": verdicts,
        "flip_source": flip_source.value if flip_source is not None else None,
        "flip_reason": getattr(stability, "flip_reason", "") or "",
        "failure_summary": failure_summary,
    }


def query_result_to_envelope(query: str, trace: Any, *, agent: str,
                             checks: Any = None) -> ConversationEnvelope:
    """Adapt a single-turn `test` failure (query + Trace) to an envelope.

    `test_cmd` lives in QueryResult/Trace space; staging and promotion live in
    envelope space. The embedded scenario spec reconstructs a one-turn
    scripted scenario (turns=[query]) carrying the query's correctness checks
    as the outcome, so verify/replay/promote work on it unchanged. `checks`
    is any pydantic model with model_dump (duck-typed to avoid a schema
    import here).
    """
    spec_dict: dict[str, Any] = {"name": query, "turns": [query]}
    if checks is not None:
        dumped = checks.model_dump(exclude_none=True)
        if dumped:
            spec_dict["outcome"] = {"correctness": dumped}
    return ConversationEnvelope(
        mode="single",
        agent=agent,
        scenario={"name": query, "spec": spec_dict},
        turns=[ConversationTurn(turn_index=0, user_message=query, trace=trace)],
    )


# ── Staging store ────────────────────────────────────────────────────────────────


@dataclass
class StagedEntry:
    """One staged conversation, addressable by ``stage_id``."""
    stage_id: str            # "<agent>/<scenario_id>/<run-ts+hash>"
    path: Path
    agent: str
    scenario_id: str
    staging: dict[str, Any]  # the envelope's staging block

    @property
    def classification(self) -> str:
        return self.staging.get("classification", Classification.UNVERIFIED.value)


# Sort weight for `stage list` — best-to-promote first.
_CLASS_SORT_ORDER = {
    Classification.CONSISTENT.value: 0,
    Classification.FLAKY_AGENT.value: 1,
    Classification.UNVERIFIED.value: 2,
    Classification.HELD.value: 3,
    Classification.HELD_INFRA.value: 4,
}


class StageStore:
    """Storage for staged failing conversations. Storage ONLY — no triage."""

    def __init__(
        self,
        staged_root: Path,
        *,
        cap: int = 10,
        max_age_days: int = 30,
        global_max_files: int = 500,
        global_max_bytes: int = 50 * 1024 * 1024,
        redactor: Callable[[ConversationEnvelope], ConversationEnvelope] = _identity,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.root = Path(staged_root)
        self.cap = cap
        self.max_age_days = max_age_days
        self.global_max_files = global_max_files
        self.global_max_bytes = global_max_bytes
        self._redactor = redactor
        self._now = now

    # -- write path ------------------------------------------------------------

    def stage(self, env: ConversationEnvelope, *, staging_block: dict[str, Any]) -> Path:
        """Write one staged envelope atomically, then GC.

        Path: ``<root>/<agent>/<scenario_id>/<run-ts+hash>.json``. The id is
        collision-resistant (run-ts + a short content hash) because simulate
        stages scenarios concurrently. GC runs at write time so staging can
        never leak disk.
        """
        agent = env.agent or "agent"
        scenario_id = staging_block.get("scenario_id") or scenario_slug(
            (env.scenario or {}).get("name", "") or "scenario"
        )
        scenario_dir = self.root / agent / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Ordering matters (ADR A1): attach the staging block BEFORE redacting
        # so its failure_summary (which embeds a raw answer preview) is inside
        # the walk, then merge the redaction metadata into the redacted block.
        work = env.model_copy(deep=True)
        work.staging = dict(staging_block)
        if hasattr(self._redactor, "redact_with_counts"):
            redacted, counts, degraded = self._redactor.redact_with_counts(work)
            redaction_meta: dict[str, Any] = {"applied": True, "counts": counts}
            if degraded:
                redaction_meta["degraded"] = True
        else:
            redacted = self._redactor(work)
            redaction_meta = {"applied": False, "counts": {}}
        block = dict(redacted.staging or {})
        block["redaction"] = redaction_meta
        redacted.staging = block

        stamp = self._now().strftime("%Y%m%dT%H%M%S")
        suffix = self._content_hash(redacted, staging_block)
        name = f"{stamp}-{suffix}.json"
        final_path = scenario_dir / name

        self._atomic_write(redacted, final_path)

        self._gc_scenario(scenario_dir)
        self._gc_global()
        return final_path

    def _content_hash(self, env: ConversationEnvelope, block: dict[str, Any]) -> str:
        h = hashlib.sha1()
        h.update(env.model_dump_json().encode("utf-8"))
        h.update(json.dumps(block, sort_keys=True, default=str).encode("utf-8"))
        # os.urandom keeps two concurrent stages of an identical scenario from
        # colliding even within the same second on identical content.
        h.update(os.urandom(8))
        return h.hexdigest()[:8]

    def _atomic_write(self, env: ConversationEnvelope, final_path: Path) -> None:
        tmp = final_path.with_suffix(".json.tmp")
        save_envelope(env, tmp)
        os.replace(tmp, final_path)

    # -- retention -------------------------------------------------------------

    def _gc_scenario(self, scenario_dir: Path) -> None:
        files = sorted(scenario_dir.glob("*.json"))
        cutoff = self._now() - timedelta(days=self.max_age_days)
        # Age eviction: drop anything older than the cutoff by run-ts stamp.
        survivors: list[Path] = []
        for f in files:
            if self._file_age_ok(f, cutoff):
                survivors.append(f)
            else:
                f.unlink(missing_ok=True)
        # Cap eviction: keep the newest `cap` by name (name starts with run-ts).
        survivors.sort()
        for f in survivors[: max(0, len(survivors) - self.cap)]:
            f.unlink(missing_ok=True)

    def _file_age_ok(self, path: Path, cutoff: datetime) -> bool:
        stamp = path.name.split("-", 1)[0]
        try:
            ts = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return True  # unparseable — never evict on age, fail open
        return ts >= cutoff

    def _gc_global(self) -> None:
        all_files = sorted(
            self.root.rglob("*.json"), key=lambda p: p.name, reverse=True
        )
        kept_bytes = 0
        for idx, f in enumerate(all_files):
            try:
                size = f.stat().st_size
            except OSError:
                continue
            over_file_cap = idx >= self.global_max_files
            over_byte_cap = kept_bytes + size > self.global_max_bytes
            if over_file_cap or over_byte_cap:
                f.unlink(missing_ok=True)
            else:
                kept_bytes += size

    def gc(self) -> int:
        """Run age + global GC across every scenario. Returns files removed."""
        before = sum(1 for _ in self.root.rglob("*.json")) if self.root.exists() else 0
        for scenario_dir in self._scenario_dirs():
            self._gc_scenario(scenario_dir)
        self._gc_global()
        after = sum(1 for _ in self.root.rglob("*.json")) if self.root.exists() else 0
        return max(0, before - after)

    def _scenario_dirs(self) -> list[Path]:
        if not self.root.exists():
            return []
        dirs: set[Path] = set()
        for f in self.root.rglob("*.json"):
            dirs.add(f.parent)
        return sorted(dirs)

    # -- read / mutate ---------------------------------------------------------

    def list(self, *, agent: Optional[str] = None) -> list[StagedEntry]:
        entries: list[StagedEntry] = []
        for f in self._all_files():
            try:
                env = load_envelope(f)
            except Exception:
                continue
            entry_agent, scenario_id = self._parts(f)
            if agent is not None and entry_agent != agent:
                continue
            entries.append(
                StagedEntry(
                    stage_id=self._stage_id(f),
                    path=f,
                    agent=entry_agent,
                    scenario_id=scenario_id,
                    staging=env.staging or {},
                )
            )
        entries.sort(
            key=lambda e: (
                _CLASS_SORT_ORDER.get(e.classification, 99),
                _negated_stamp(e.path.name),
            )
        )
        return entries

    def load(self, stage_id: str) -> tuple[Path, ConversationEnvelope]:
        path = self._resolve(stage_id)
        return path, load_envelope(path)

    def drop(self, stage_id: str) -> None:
        path = self._resolve(stage_id)
        path.unlink(missing_ok=True)

    def update_staging_block(self, stage_id: str, block: dict[str, Any]) -> Path:
        """Rewrite one staged file's staging block in place (used by verify).

        The incumbent `redaction` record is carried forward when the new block
        lacks one — verify rebuilds blocks from scratch and must not erase the
        transparency record (ADR A7; this is the single choke point)."""
        path = self._resolve(stage_id)
        env = load_envelope(path)
        new_block = dict(block)
        incumbent = (env.staging or {}).get("redaction")
        if "redaction" not in new_block and incumbent is not None:
            new_block["redaction"] = incumbent
        env.staging = new_block
        self._atomic_write(env, path)
        return path

    # -- id resolution ---------------------------------------------------------

    def _all_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.rglob("*.json"))

    def _stage_id(self, path: Path) -> str:
        rel = path.relative_to(self.root)
        return str(rel.with_suffix(""))

    def _parts(self, path: Path) -> tuple[str, str]:
        rel = path.relative_to(self.root).parts
        agent = rel[0] if len(rel) >= 1 else ""
        scenario_id = rel[1] if len(rel) >= 2 else ""
        return agent, scenario_id

    def _resolve(self, stage_id: str) -> Path:
        """Resolve a full id or a unique prefix (e.g. the run-ts alone)."""
        sid = stage_id.rstrip("/")
        candidates: list[Path] = []
        for f in self._all_files():
            fid = self._stage_id(f)
            if fid == sid:
                return f
            if fid.endswith("/" + sid) or f.stem == sid or f.stem.startswith(sid) or sid in fid:
                candidates.append(f)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise StageNotFound(stage_id)
        raise StageAmbiguous(stage_id, [self._stage_id(c) for c in candidates])


def _negated_stamp(name: str) -> str:
    """Sort key that puts the newest run-ts first (lexicographic descending)."""
    # invert each char so a plain ascending sort yields newest-first
    return "".join(chr(255 - ord(c)) for c in name)


class StageNotFound(LookupError):
    def __init__(self, stage_id: str) -> None:
        super().__init__(f"no staged entry matches '{stage_id}'")
        self.stage_id = stage_id


class StageAmbiguous(LookupError):
    def __init__(self, stage_id: str, matches: list[str]) -> None:
        super().__init__(
            f"'{stage_id}' matches {len(matches)} staged entries: {matches}"
        )
        self.stage_id = stage_id
        self.matches = matches


# ── Promotion ────────────────────────────────────────────────────────────────────


_GATED_CLASSES = {
    Classification.HELD.value,
    Classification.HELD_INFRA.value,
    Classification.UNVERIFIED.value,
}


class PromotionRefused(Exception):
    """Raised when a staged entry's class is gated and --force was not given."""


# Bug-golden lifecycle state machine (ADR):
#   staged → promoted(gate|xfail) → fixed(flip: xfail → gate + flipped_at)
# `gate`: replay exits 1 while the bug reproduces (CI red until the fix).
# `xfail`: replay treats the failure as expected (CI green); a passing replay
# is XPASS — flag it, and `promote --flip` converts to a normal gate golden.
LIFECYCLES = ("gate", "xfail")


class PromotionService:
    """Move a staged conversation into the golden dir with provenance.

    Depends on ``StageStore`` and the baseline-dir string seam, never on
    concrete CLI paths. Reuses the record path's conventions (scenario_slug,
    gate_conversation_envelope, save_envelope) rather than duplicating them.
    """

    def __init__(self, store: StageStore, *, now: Callable[[], datetime] = _utcnow) -> None:
        self._store = store
        self._now = now

    def promote(
        self,
        stage_id: str,
        *,
        baseline_dir: str,
        lifecycle: str = "gate",
        force: bool = False,
    ) -> Path:
        from .engine.artifact_gate import gate_conversation_envelope

        if lifecycle not in LIFECYCLES:
            raise ValueError(f"unknown lifecycle '{lifecycle}' (one of {LIFECYCLES})")

        path, env = self._store.load(stage_id)
        staging = env.staging or {}
        classification = staging.get("classification", Classification.UNVERIFIED.value)

        if classification in _GATED_CLASSES and not force:
            raise PromotionRefused(
                f"'{stage_id}' is classified '{classification}' — "
                f"{CLASSIFICATION_MEANING.get(Classification(classification), '')}. "
                "Re-run `ciagent stage verify` or pass --force to promote anyway."
            )

        # Never write an un-replayable golden: re-run the structural gate.
        gate = gate_conversation_envelope(env)
        if gate.rejected:
            raise PromotionRefused(
                f"refusing to promote '{stage_id}' — {gate.summary()}"
            )

        promoted = self._stamp_provenance(env, staging, classification, lifecycle)

        agent = env.agent or "agent"
        name = (env.scenario or {}).get("name") or staging.get("scenario_id") or "scenario"
        out_dir = Path(baseline_dir) / agent / "scenarios"
        out = save_envelope(promoted, out_dir / f"{scenario_slug(name)}.json")
        # A promote is a MOVE: the golden (with provenance) is now the source of
        # truth, so the staged copy is consumed. Only after a successful write —
        # any failure above leaves the staged file untouched.
        path.unlink(missing_ok=True)
        return out

    def flip(self, golden_ref: str, *, baseline_dir: str) -> Path:
        """Flip a promoted xfail golden to a normal gate golden after its fix.

        `golden_ref` is a path to the golden, or a unique substring of one
        under `<baseline_dir>` (e.g. the scenario slug). Refuses when the
        golden's lifecycle is not `xfail` — there is nothing to flip on a
        gate golden, and re-flipping is a no-op worth surfacing.
        """
        path = self._resolve_golden(golden_ref, baseline_dir)
        env = load_envelope(path)
        prov = dict(env.provenance or {})
        if prov.get("lifecycle") != "xfail":
            raise PromotionRefused(
                f"'{golden_ref}' is not an xfail golden "
                f"(lifecycle: {prov.get('lifecycle') or 'none'}) — nothing to flip."
            )
        prov["lifecycle"] = "gate"
        prov["flipped_at"] = self._now().isoformat()
        env.provenance = prov
        return save_envelope(env, path)

    @staticmethod
    def _resolve_golden(golden_ref: str, baseline_dir: str) -> Path:
        direct = Path(golden_ref)
        if direct.is_file():
            return direct
        root = Path(baseline_dir)
        candidates = [
            p for p in (sorted(root.rglob("*.json")) if root.exists() else [])
            if golden_ref in str(p.relative_to(root)) or p.stem == golden_ref
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise StageNotFound(golden_ref)
        raise StageAmbiguous(golden_ref, [str(c) for c in candidates])

    def _stamp_provenance(
        self,
        env: ConversationEnvelope,
        staging: dict[str, Any],
        classification: str,
        lifecycle: str,
    ) -> ConversationEnvelope:
        """Drop `staging:`, stamp `provenance:` — a promoted golden reads clean."""
        provenance = {
            "staged_run_id": staging.get("run_id", ""),
            "classification_at_promotion": classification,
            "promoted_at": self._now().isoformat(),
            "lifecycle": lifecycle,
        }
        # Reconstruct so we don't mutate the staged object in place.
        data = json.loads(env.model_dump_json())
        data.pop("staging", None)
        data["provenance"] = provenance
        promoted = normalize_to_envelope(data)
        promoted.provenance = provenance
        return promoted
