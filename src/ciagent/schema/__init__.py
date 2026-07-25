# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
CIAgent v2 Schema — Pydantic models for ciagent_spec.yaml.
"""

from .spec_models import (
    CIAgentSpec,
    GoldenQuery,
    CorrectnessSpec,
    PathSpec,
    CostSpec,
    MatchMode,
    JudgeRubric,
)

__all__ = [
    "CIAgentSpec",
    "GoldenQuery",
    "CorrectnessSpec",
    "PathSpec",
    "CostSpec",
    "MatchMode",
    "JudgeRubric",
]
