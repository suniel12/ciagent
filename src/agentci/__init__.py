"""
Agent CI — Continuous Integration for AI Agents.

Catch cost spikes and logic regressions before production.

v1 exports (backward compatible):
    test, TraceContext, diff, load_baseline

v2 exports (new declarative evaluation engine):
    AgentCISpec, GoldenQuery, load_spec
    evaluate_query, evaluate_spec
    QueryResult, LayerResult, LayerStatus
    save_baseline (v2), load_versioned_baseline, list_baselines
    diff_baselines, DiffReport (v2 three-tier diff)
"""

try:
    from importlib.metadata import version
    __version__ = version("agentci")
except Exception:
    __version__ = "0.0.0"

# ── v1 exports (preserved for backward compatibility) ─────────────────────────
from .pytest_plugin import test
from .capture import TraceContext
from .diff_engine import diff, load_baseline

# ── v2 exports ─────────────────────────────────────────────────────────────────
from .schema.spec_models import AgentCISpec, GoldenQuery
from .loader import load_spec
from .engine.results import QueryResult, LayerResult, LayerStatus
from .baselines import (
    save_baseline as save_baseline,
    load_baseline as load_versioned_baseline,
    list_baselines,
)

__all__ = [
    # v1
    "test",
    "TraceContext",
    "diff",
    "load_baseline",
    # v2
    "AgentCISpec",
    "GoldenQuery",
    "load_spec",
    "QueryResult",
    "LayerResult",
    "LayerStatus",
    "save_baseline",
    "load_versioned_baseline",
    "list_baselines",
    "diff_baselines",
    "DiffReport",
    "run_spec",
    "resolve_runner",
]


def __getattr__(name: str):
    """Lazy-load v2 engine functions to avoid circular imports."""
    if name == "evaluate_query":
        from .engine.runner import evaluate_query
        return evaluate_query
    if name == "evaluate_spec":
        from .engine.runner import evaluate_spec
        return evaluate_spec
    if name in ("diff_baselines", "DiffReport", "MetricDelta"):
        from .engine.diff import diff_baselines, DiffReport, MetricDelta  # noqa: F401
        return locals()[name]
    if name == "run_spec":
        from .engine.parallel import run_spec
        return run_spec
    if name == "resolve_runner":
        from .engine.parallel import resolve_runner
        return resolve_runner
    raise AttributeError(f"module 'agentci' has no attribute {name!r}")
