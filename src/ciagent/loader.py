# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
AgentCI v2 YAML Spec Loader.

Loads an agentci_spec.yaml file, validates it via Pydantic, and merges
global defaults into each query. Replaces v1's config.py for v2 specs.

Public API:
    load_spec(path)          → AgentCISpec
    filter_by_tags(spec, tags) → AgentCISpec
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import ValidationError

from ciagent.exceptions import ConfigError
from ciagent.schema.spec_models import AgentCISpec, GoldenQuery


def load_spec(spec_path: Union[str, Path]) -> AgentCISpec:
    """Load and validate an agentci_spec.yaml file.

    Args:
        spec_path: Path to the YAML spec file.

    Returns:
        Validated AgentCISpec with defaults merged into each query.

    Raises:
        ConfigError: If the file cannot be read or fails Pydantic validation.
    """
    path = Path(spec_path)
    if not path.exists():
        raise ConfigError(
            f"Spec file not found: {path}",
            fix=f"Create {path} or run 'ciagent init' to scaffold one.",
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Invalid YAML in {path}: {e}",
            fix="Fix the YAML syntax error and try again.",
        ) from e

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Spec file must be a YAML mapping, got {type(raw).__name__}",
            fix="Ensure the file starts with top-level keys like 'agent:' and 'queries:'.",
        )

    try:
        spec = AgentCISpec(**raw)
    except ValidationError as e:
        raise ConfigError(
            f"Spec validation failed for {path}:\n{e}",
            fix="Run 'ciagent validate <path>' for detailed error messages.",
        ) from e

    if spec.defaults:
        spec.queries = [_merge_defaults(q, spec.defaults) for q in spec.queries]

    return spec


def filter_by_tags(spec: AgentCISpec, tags: list[str]) -> AgentCISpec:
    """Return a copy of the spec containing only queries that match any given tag.

    Args:
        spec: The loaded AgentCISpec.
        tags: List of tag strings to filter by. Empty list returns all queries.

    Returns:
        New AgentCISpec with filtered queries list.
    """
    if not tags:
        return spec
    filtered = [q for q in spec.queries if q.tags and set(q.tags) & set(tags)]
    return spec.model_copy(update={"queries": filtered})


# ── Internal helpers ────────────────────────────────────────────────────────────


def _merge_defaults(query: GoldenQuery, defaults: dict[str, Any]) -> GoldenQuery:
    """Deep-merge defaults into a query, with query values taking precedence."""
    query_dict = query.model_dump(exclude_none=True)
    merged = _deep_merge(defaults, query_dict)
    return GoldenQuery(**merged)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dicts. Override values take precedence over base."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
