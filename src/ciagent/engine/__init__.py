# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
AgentCI v2 Evaluation Engine.

Three-layer evaluation: Correctness (hard fail) → Path (soft warn) → Cost (soft warn).
"""

from .results import LayerResult, LayerStatus, QueryResult

__all__ = [
    "LayerResult",
    "LayerStatus",
    "QueryResult",
    "evaluate_query",
    "evaluate_spec",
    "diff_baselines",
    "DiffReport",
]


def __getattr__(name: str):
    if name in ("evaluate_query", "evaluate_spec"):
        from .runner import evaluate_query, evaluate_spec  # noqa: F401
        return locals()[name]
    if name in ("diff_baselines", "DiffReport", "MetricDelta"):
        from .diff import diff_baselines, DiffReport, MetricDelta  # noqa: F401
        return locals()[name]
    raise AttributeError(f"module 'ciagent.engine' has no attribute {name!r}")
