# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Result data classes for the AgentCI v2 evaluation engine.

Each query evaluation produces one QueryResult containing one LayerResult
per layer. Layer severity:
  - Correctness: FAIL blocks the CI pipeline (exit code 1)
  - Path / Cost:  WARN emits GitHub annotations but does not fail CI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LayerStatus(str, Enum):
    """Evaluation outcome for a single layer."""
    PASS = "pass"   # All assertions satisfied
    FAIL = "fail"   # Hard failure (correctness or forbidden tool)
    WARN = "warn"   # Soft exceedance (path/cost thresholds breached)
    SKIP = "skip"   # No assertions defined for this layer


@dataclass
class LayerResult:
    """Result from evaluating one layer (correctness, path, or cost)."""
    status: LayerStatus
    details: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


def _skip_layer() -> LayerResult:
    return LayerResult(
        status=LayerStatus.SKIP,
        details={},
        messages=["No assertions configured"],
    )


@dataclass
class QueryResult:
    """Aggregated result for a single golden query across all layers.

    Layer order of evaluation: correctness (1), path (2), retrieval (2.5),
    cost (3). Retrieval defaults to SKIP so pre-F4 construction sites and
    serialized results stay valid unchanged.
    """
    query: str
    correctness: LayerResult
    path: LayerResult
    cost: LayerResult
    trace: Any = None  # Using Any to avoid circular imports without TYPE_CHECKING string annotations parsing issues
    retrieval: LayerResult = field(default_factory=_skip_layer)

    @property
    def hard_fail(self) -> bool:
        """True if correctness failed, or a forbidden tool was used.

        The path layer is otherwise soft (warn-only); it reaches FAIL status
        only on a forbidden_tools violation (engine/path.py), which is a
        documented safety boundary and must gate the build. Historically
        this property read correctness alone, so a forbidden-tools violation
        printed 'PATH: FAIL' yet the run exited 0 — a silent safety gap
        surfaced by dogfooding the failure atlas."""
        return (
            self.correctness.status == LayerStatus.FAIL
            or self.path.status == LayerStatus.FAIL
        )

    @property
    def has_warnings(self) -> bool:
        """True if path, retrieval, or cost layers produced warnings."""
        return (
            self.path.status == LayerStatus.WARN
            or self.retrieval.status == LayerStatus.WARN
            or self.cost.status == LayerStatus.WARN
        )
