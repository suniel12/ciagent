# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
CIAgent v2 Baseline Manager.

Provides versioned golden baseline save/load/list with an optional
correctness precheck to prevent saving broken baselines.

File format:
    baselines/<agent>/<version>.json

Baseline JSON structure:
    {
        "version": "v2-fixed",
        "agent": "rag-agent",
        "captured_at": "2026-02-26T14:30:00Z",
        "query": "How do I install CIAgent?",
        "metadata": {
            "model": "gpt-4o-mini",
            "spec_hash": "sha256:abc123...",
            "precheck_passed": true
        },
        "trace": { ... }
    }
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ciagent.exceptions import BaselineError

if TYPE_CHECKING:
    from ciagent.engine.results import LayerResult
    from ciagent.models import Trace
    from ciagent.schema.spec_models import CIAgentSpec, GoldenQuery


# ── Public API ─────────────────────────────────────────────────────────────────


def save_baseline(
    trace: "Trace",
    agent: str,
    version: str,
    spec: "CIAgentSpec",
    query_text: str = "",
    baseline_dir: str = "./golden",
    force: bool = False,
) -> Path:
    """Save a trace as a versioned golden baseline.

    Args:
        trace:        Execution trace to save.
        agent:        Agent identifier (matches spec.agent).
        version:      Version tag, e.g. "v1-broken" or "v2-fixed".
        spec:         The CIAgentSpec (used for precheck and spec_hash).
        query_text:   The query this baseline corresponds to.
        baseline_dir: Root directory for baseline files.
        force:        If True, skips the correctness precheck.

    Returns:
        Path to the saved baseline JSON file.

    Raises:
        ValueError: If precheck fails and force=False.
    """
    if not force and query_text:
        query_spec = _find_query_spec(query_text, spec)
        if query_spec and query_spec.correctness:
            from ciagent.engine.correctness import evaluate_correctness
            from ciagent.engine.runner import _extract_answer

            answer = _extract_answer(trace)
            result = evaluate_correctness(
                answer=answer,
                spec=query_spec.correctness,
                trace=trace,
                judge_config=spec.judge_config,
            )
            if result.status.value == "fail":
                raise ValueError(
                    f"Precheck failed — baseline does not pass correctness:\n"
                    f"  {result.messages}\n"
                    f"Use --force-save to bypass."
                )

    spec_hash = _compute_spec_hash(spec)
    model_name = _extract_model_name(trace)

    baseline_data: dict[str, Any] = {
        # Format version of the baseline file itself (1 = single-trace wrapper;
        # 2 = conversation envelope, written by the simulate flow). Files
        # without the field read as 1 (pre-0.9 legacy).
        "schema_version": 1,
        "version": version,
        "agent": agent,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "query": query_text,
        "metadata": {
            "model": model_name,
            "spec_hash": f"sha256:{spec_hash}",
            "precheck_passed": not force,
        },
        "trace": json.loads(trace.model_dump_json()),
    }

    out_dir = Path(baseline_dir) / agent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{version}.json"
    out_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
    return out_path


def load_baseline(
    agent: str,
    version: str,
    baseline_dir: str = "./golden",
) -> dict[str, Any]:
    """Load a versioned baseline JSON file.

    Args:
        agent:        Agent identifier.
        version:      Version tag to load.
        baseline_dir: Root directory for baseline files.

    Returns:
        Parsed baseline dict with keys: version, agent, captured_at, trace, metadata.

    Raises:
        BaselineError: If the file does not exist.
    """
    path = Path(baseline_dir) / agent / f"{version}.json"
    if not path.exists():
        raise BaselineError(
            f"Baseline not found: {path}",
            fix=f"Run 'ciagent baselines --agent {agent}' to list available versions.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def discover_baselines(
    baseline_dir: str = "./golden",
    agent: Optional[str] = None,
) -> dict[str, "Trace"]:
    """Map query text → baseline Trace for every baseline under baseline_dir.

    Scans both storage layouts in use:
        <dir>/*.json            — flat files (written by `ciagent init --generate`)
        <dir>/<agent>/*.json    — versioned files (written by save_baseline)

    Files are keyed by their top-level "query" field. Conversation envelopes
    (schema_version 2) and files without a query are skipped, as are files
    that fail to parse. When the same query appears in both layouts, the
    versioned per-agent file wins.

    Args:
        baseline_dir: Root directory for baseline files.
        agent:        If given, only scan this agent's subdirectory (flat
                      files are always scanned).

    Returns:
        dict mapping query_text → Trace. Empty if the directory is missing.
    """
    from ciagent.models import Trace

    root = Path(baseline_dir)
    if not root.is_dir():
        return {}

    if agent:
        subdirs = [root / agent]
    else:
        subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]

    # Flat layout first so per-agent versioned files override it.
    files: list[Path] = sorted(root.glob("*.json"))
    for d in subdirs:
        files.extend(sorted(d.glob("*.json")))

    baselines: dict[str, "Trace"] = {}
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("schema_version", 1) != 1:
            continue  # conversation envelopes belong to the simulate flow
        query = data.get("query")
        trace_data = data.get("trace")
        if not query or not isinstance(trace_data, dict):
            continue
        try:
            baselines[query] = Trace.model_validate(trace_data)
        except Exception:  # noqa: BLE001 — one bad trace must not kill the run
            continue

    return baselines


def list_baselines(
    agent: str,
    baseline_dir: str = "./golden",
) -> list[dict[str, Any]]:
    """List all available baseline versions for an agent.

    Args:
        agent:        Agent identifier.
        baseline_dir: Root directory for baseline files.

    Returns:
        List of baseline metadata dicts (without the full trace), sorted by version.
    """
    agent_dir = Path(baseline_dir) / agent
    if not agent_dir.exists():
        return []

    result = []
    for json_file in sorted(agent_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            result.append({
                "version": data.get("version", json_file.stem),
                "agent": data.get("agent", agent),
                "captured_at": data.get("captured_at"),
                "query": data.get("query", ""),
                "metadata": data.get("metadata", {}),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return result


# ── Internal helpers ───────────────────────────────────────────────────────────


def _find_query_spec(
    query_text: str,
    spec: "CIAgentSpec",
) -> Optional["GoldenQuery"]:
    """Find the GoldenQuery that matches query_text (exact string match)."""
    for q in spec.queries:
        if q.query == query_text:
            return q
    return None


def _compute_spec_hash(spec: "CIAgentSpec") -> str:
    """Compute a short SHA-256 hash of the spec for traceability."""
    canonical = json.dumps(spec.model_dump(), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _extract_model_name(trace: "Trace") -> str:
    """Extract the model name from the first LLM call in the trace."""
    for span in trace.spans:
        for llm_call in span.llm_calls:
            if llm_call.model:
                return llm_call.model
    return "unknown"
