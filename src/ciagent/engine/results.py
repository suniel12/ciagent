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


@dataclass
class QueryResult:
    """Aggregated result for a single golden query across all three layers."""
    query: str
    correctness: LayerResult
    path: LayerResult
    cost: LayerResult
    trace: Any = None  # Using Any to avoid circular imports without TYPE_CHECKING string annotations parsing issues

    @property
    def hard_fail(self) -> bool:
        """True if correctness failed (or a forbidden tool was used)."""
        return self.correctness.status == LayerStatus.FAIL

    @property
    def has_warnings(self) -> bool:
        """True if path or cost layers produced warnings."""
        return (
            self.path.status == LayerStatus.WARN
            or self.cost.status == LayerStatus.WARN
        )
