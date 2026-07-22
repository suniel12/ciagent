# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Conversation envelope: the multi-turn golden format for `ciagent simulate`.

Design rule (eng review, 2026-07-05, binding): a single-turn baseline is the
1-turn degenerate case of a conversation, so ONE loader serves every shape on
disk. No second format. `normalize_to_envelope` accepts:

1. schema_version 2 envelope (written by the simulate flow)
2. schema_version 1 / unversioned wrapper (``ciagent save`` / bootstrap
   baselines: {version, agent, query, metadata, trace})
3. bare Trace dict (``ciagent record`` output)

and returns a ConversationEnvelope in all three cases. Files newer than this
reader (schema_version > 2) are rejected by name, never guessed at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from .exceptions import BaselineError
from .models import Trace

ENVELOPE_SCHEMA_VERSION = 2


class ConversationTurn(BaseModel):
    """One user turn and the agent's traced response to it."""

    turn_index: int = 0
    user_message: str = ""
    trace: Trace


class ConversationEnvelope(BaseModel):
    """A recorded conversation: envelope of per-turn Traces.

    ``mode`` records how the user turns were produced:
    - ``single``:    degenerate 1-turn case (a classic single-query baseline)
    - ``scripted``:  fixed user turns from the scenario spec (CI path)
    - ``simulated``: persona LLM generated the turns (finder path)
    - ``replay``:    recorded turns fed back verbatim (deterministic gate)
    """

    schema_version: int = ENVELOPE_SCHEMA_VERSION
    mode: str = "single"
    agent: str = ""
    version: str = ""
    captured_at: str = ""
    scenario: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    turns: list[ConversationTurn] = Field(default_factory=list)
    # Additive, optional (schema stays version 2). Present on staged files only
    # (`staging:`) and on promoted goldens only (`provenance:`). Old loaders
    # ignore unknown keys by construction, so the backward-compat contract holds.
    staging: Optional[dict[str, Any]] = None
    provenance: Optional[dict[str, Any]] = None

    @property
    def is_single_turn(self) -> bool:
        return len(self.turns) == 1

    def final_trace(self) -> Optional[Trace]:
        return self.turns[-1].trace if self.turns else None


def normalize_to_envelope(data: dict[str, Any], source: str = "") -> ConversationEnvelope:
    """Normalize any on-disk baseline shape into a ConversationEnvelope."""
    where = f" ({source})" if source else ""

    declared = data.get("schema_version")
    if isinstance(declared, int) and declared > ENVELOPE_SCHEMA_VERSION:
        raise BaselineError(
            f"Baseline{where} has schema_version {declared}, newer than this "
            f"ciagent understands (max {ENVELOPE_SCHEMA_VERSION}).",
            fix="Upgrade ciagent to read this file.",
        )

    # Shape 1: envelope (schema_version 2)
    if "turns" in data:
        turns = [
            ConversationTurn(
                turn_index=t.get("turn_index", i),
                user_message=t.get("user_message", ""),
                trace=Trace.model_validate(t["trace"]),
            )
            for i, t in enumerate(data["turns"])
        ]
        return ConversationEnvelope(
            schema_version=declared or ENVELOPE_SCHEMA_VERSION,
            mode=data.get("mode", "scripted"),
            agent=data.get("agent", ""),
            version=data.get("version", ""),
            captured_at=data.get("captured_at", ""),
            scenario=data.get("scenario") or {},
            metadata=data.get("metadata") or {},
            turns=turns,
            staging=data.get("staging"),
            provenance=data.get("provenance"),
        )

    # Shape 2: versioned single-trace wrapper (schema_version 1 or legacy/unversioned)
    if "trace" in data:
        trace = Trace.model_validate(data["trace"])
        return ConversationEnvelope(
            mode="single",
            agent=data.get("agent", ""),
            version=data.get("version", ""),
            captured_at=data.get("captured_at", ""),
            metadata=data.get("metadata") or {},
            turns=[
                ConversationTurn(
                    turn_index=0,
                    user_message=data.get("query", "")
                    or (data["trace"].get("metadata") or {}).get("query", ""),
                    trace=trace,
                )
            ],
        )

    # Shape 3: bare Trace dict (ciagent record output)
    if "spans" in data or "trace_id" in data:
        trace = Trace.model_validate(data)
        meta = data.get("metadata") or {}
        return ConversationEnvelope(
            mode="single",
            agent=data.get("agent_name", ""),
            turns=[
                ConversationTurn(
                    turn_index=0,
                    user_message=meta.get("query", "") or data.get("test_name", ""),
                    trace=trace,
                )
            ],
        )

    raise BaselineError(
        f"Unrecognized baseline shape{where}: expected an envelope with 'turns', "
        "a wrapper with 'trace', or a bare trace with 'spans'/'trace_id'. "
        f"Top-level keys found: {sorted(data.keys())[:8]}",
        fix="Re-record the baseline with 'ciagent save' or 'ciagent bootstrap'.",
    )


def load_envelope(path: Union[str, Path]) -> ConversationEnvelope:
    """Load any baseline file (legacy or envelope) as a ConversationEnvelope."""
    p = Path(path)
    if not p.exists():
        raise BaselineError(f"Baseline not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BaselineError(f"Baseline is not valid JSON: {p}: {e}") from e
    return normalize_to_envelope(data, source=str(p))


def save_envelope(
    envelope: ConversationEnvelope,
    path: Union[str, Path],
) -> Path:
    """Serialize an envelope to disk (schema_version 2)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(envelope.model_dump_json())
    payload["schema_version"] = ENVELOPE_SCHEMA_VERSION
    # Additive optional blocks are dropped when absent so files without them
    # stay byte-identical to pre-0.11 goldens (record/replay regression).
    for optional_key in ("staging", "provenance"):
        if payload.get(optional_key) is None:
            payload.pop(optional_key, None)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
