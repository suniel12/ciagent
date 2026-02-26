"""
AgentCI v2 Baseline Manager.

Provides versioned golden baseline save/load/list with an optional
correctness precheck to prevent saving broken baselines.

File format:
    baselines/<agent>/<version>.json

Baseline JSON structure:
    {
        "version": "v2-fixed",
        "agent": "rag-agent",
        "captured_at": "2026-02-26T14:30:00Z",
        "query": "How do I install AgentCI?",
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

from agentci.exceptions import BaselineError

if TYPE_CHECKING:
    from agentci.engine.results import LayerResult
    from agentci.models import Trace
    from agentci.schema.spec_models import AgentCISpec, GoldenQuery


# ── Public API ─────────────────────────────────────────────────────────────────


def save_baseline(
    trace: "Trace",
    agent: str,
    version: str,
    spec: "AgentCISpec",
    query_text: str = "",
    baseline_dir: str = "./golden",
    force: bool = False,
) -> Path:
    """Save a trace as a versioned golden baseline.

    Args:
        trace:        Execution trace to save.
        agent:        Agent identifier (matches spec.agent).
        version:      Version tag, e.g. "v1-broken" or "v2-fixed".
        spec:         The AgentCISpec (used for precheck and spec_hash).
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
            from agentci.engine.correctness import evaluate_correctness
            from agentci.engine.runner import _extract_answer

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
            fix=f"Run 'agentci baselines --agent {agent}' to list available versions.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


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
    spec: "AgentCISpec",
) -> Optional["GoldenQuery"]:
    """Find the GoldenQuery that matches query_text (exact string match)."""
    for q in spec.queries:
        if q.query == query_text:
            return q
    return None


def _compute_spec_hash(spec: "AgentCISpec") -> str:
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
