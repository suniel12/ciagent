# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
CIAgent YAML loader — the single entry point for both spec formats.

Formats:
    v2 (current)  ciagent_spec.yaml, `queries:` key  → load_spec() → CIAgentSpec
    v1 (legacy)   ciagent.yaml, `tests:` key         → load_suite() → TestSuite

Public API:
    load_spec(path)            → CIAgentSpec
    load_suite(path)           → TestSuite (v1, legacy)
    detect_format(path)        → "v1" | "v2"
    filter_by_tags(spec, tags) → CIAgentSpec
"""

from __future__ import annotations

import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import ValidationError

from ciagent.exceptions import ConfigError
from ciagent.models import TestSuite
from ciagent.schema.spec_models import CIAgentSpec, GoldenQuery

# Top-level keys that only appear in the v2 spec format. Used to reject v2
# files handed to the v1 loader (Pydantic would silently drop them and the
# suite would load with zero tests) and to classify files in detect_format.
# The *_runner entries are deprecated aliases accepted until 1.0.
_V2_SPEC_KEYS = frozenset({
    "queries", "baseline_dir",
    "adapter", "conversation_adapter",
    "runner", "conversation_runner",
})

# Legacy default spec filename, accepted with a deprecation warning until 1.0.
DEFAULT_SPEC_FILENAME = "ciagent_spec.yaml"
LEGACY_SPEC_FILENAME = "agentci_spec.yaml"

# Deprecation warnings fire once per CLI invocation (process), on stderr,
# regardless of --runs N or worker count. Keyed by warning topic.
_deprecation_warned: set[str] = set()


def _warn_deprecated_once(topic: str, message: str) -> None:
    if topic in _deprecation_warned:
        return
    _deprecation_warned.add(topic)
    import sys
    print(f"DEPRECATED: {message}", file=sys.stderr)


def resolve_default_spec_path(directory: Union[str, Path] = ".") -> Path:
    """Resolve the default spec path, honoring the legacy filename until 1.0.

    Prefers ciagent_spec.yaml; falls back to agentci_spec.yaml with a one-line
    deprecation warning. When neither exists, returns the new default so the
    caller's not-found error names the current convention.
    """
    d = Path(directory)
    new = d / DEFAULT_SPEC_FILENAME
    old = d / LEGACY_SPEC_FILENAME
    if new.exists():
        return new
    if old.exists():
        _warn_deprecated_once(
            "spec-filename",
            f"{LEGACY_SPEC_FILENAME} is deprecated; rename it to "
            f"{DEFAULT_SPEC_FILENAME} (accepted until 1.0).",
        )
        return old
    return new


def _read_yaml_mapping(path: Path, expected_keys: str) -> dict[str, Any]:
    """Read a YAML file and require a top-level mapping.

    Args:
        path: Path to the YAML file (must exist).
        expected_keys: Human-readable hint for error messages,
            e.g. "'agent:' and 'queries:'".

    Raises:
        ConfigError: On YAML syntax errors or non-mapping content.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Invalid YAML in {path}: {e}",
            fix="Fix the YAML syntax error and try again.",
        ) from e

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Spec file {path} must be a YAML mapping, got {type(raw).__name__}",
            fix=f"Ensure the file starts with top-level keys like {expected_keys}.",
        )
    return raw


def detect_format(path: Union[str, Path]) -> str:
    """Classify a YAML config file as "v1" (tests:) or "v2" (queries:).

    Raises:
        ConfigError: If the file is missing, malformed, or matches neither
            format.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"Config file not found: {p}",
            fix="Run 'ciagent init' to scaffold a spec, or pass --config with "
                "the correct path.",
        )
    raw = _read_yaml_mapping(p, "'agent:' and 'queries:' (or 'tests:')")

    if _V2_SPEC_KEYS & raw.keys():
        return "v2"
    if "tests" in raw:
        return "v1"
    raise ConfigError(
        f"Cannot determine format of {p}: found neither a 'queries:' key "
        f"(v2 spec) nor a 'tests:' key (v1 suite).",
        fix="Add your test cases under 'queries:' (ciagent_spec.yaml, "
            "recommended) or 'tests:' (legacy ciagent.yaml).",
    )


def load_spec(spec_path: Union[str, Path]) -> CIAgentSpec:
    """Load and validate an ciagent_spec.yaml file.

    Args:
        spec_path: Path to the YAML spec file.

    Returns:
        Validated CIAgentSpec with defaults merged into each query.

    Raises:
        ConfigError: If the file cannot be read or fails Pydantic validation.
    """
    path = Path(spec_path)
    if not path.exists() and path.name == DEFAULT_SPEC_FILENAME:
        legacy = path.with_name(LEGACY_SPEC_FILENAME)
        if legacy.exists():
            _warn_deprecated_once(
                "spec-filename",
                f"{LEGACY_SPEC_FILENAME} is deprecated; rename it to "
                f"{DEFAULT_SPEC_FILENAME} (accepted until 1.0).",
            )
            path = legacy
    if not path.exists():
        raise ConfigError(
            f"Spec file not found: {path}",
            fix=f"Create {path} or run 'ciagent init' to scaffold one.",
        )

    raw = _read_yaml_mapping(path, "'agent:' and 'queries:'")

    for legacy_key, new_key in (("runner", "adapter"),
                                ("conversation_runner", "conversation_adapter")):
        if legacy_key in raw:
            _warn_deprecated_once(
                f"spec-key-{legacy_key}",
                f"spec key '{legacy_key}:' is deprecated; rename it to "
                f"'{new_key}:' (accepted until 1.0).",
            )

    try:
        spec = CIAgentSpec(**raw)
    except ValidationError as e:
        raise ConfigError(
            f"Spec validation failed for {path}:\n{e}",
            fix="Run 'ciagent validate <path>' for detailed error messages.",
        ) from e

    if spec.defaults:
        spec.queries = [_merge_defaults(q, spec.defaults) for q in spec.queries]

    return spec


def load_suite(path: Union[str, Path] = "ciagent.yaml") -> TestSuite:
    """Load and validate a v1 ciagent.yaml suite (legacy `tests:` format).

    The v1 format is superseded by ciagent_spec.yaml (use load_spec). This
    loader stays strict: a v2 spec or a file with no recognizable test key
    raises instead of silently loading as zero tests.

    Args:
        path: Path to the YAML suite file (default: ciagent.yaml).

    Returns:
        A validated TestSuite with golden_trace paths resolved relative to
        the suite file.

    Raises:
        ConfigError: If the file is missing, invalid, is a v2 spec, or
            contains only keys this loader does not understand.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"Configuration file not found: {p}",
            fix="Run 'ciagent init' to generate a default ciagent.yaml, "
                "or create one manually. See AGENTS.md for the expected format.",
        )

    config_dict = _read_yaml_mapping(p, "'agent:' and 'tests:'")

    v2_keys = _V2_SPEC_KEYS & config_dict.keys()
    if v2_keys:
        raise ConfigError(
            f"{p} looks like a v2 spec (found {sorted(v2_keys)}), "
            f"not a v1 ciagent.yaml suite. Loading it as a v1 suite would "
            f"silently produce zero tests.",
            fix="Run it with 'ciagent test' (which uses the v2 loader), or "
                "load it with ciagent.loader.load_spec(). If you meant to "
                "write a v1 suite, put test cases under a 'tests:' key.",
        )

    # "defaults" is not a model field: a TestSuite validator consumes it and
    # maps it onto the default_* fields.
    unknown_keys = config_dict.keys() - TestSuite.model_fields.keys() - {"defaults"}
    if unknown_keys and "tests" not in config_dict:
        raise ConfigError(
            f"{p} has no 'tests:' key and contains unrecognized keys "
            f"{sorted(unknown_keys)}. Loading it would silently produce an "
            f"empty suite.",
            fix="Put test cases under a 'tests:' key (v1 format), or use "
                "'ciagent test' / ciagent.loader.load_spec() for v2 specs.",
        )
    if unknown_keys:
        warnings.warn(
            f"Ignoring unrecognized keys in {p}: {sorted(unknown_keys)}",
            UserWarning,
            stacklevel=2,
        )

    suite = TestSuite(**config_dict)

    # Resolve relative paths relative to the suite file
    base_dir = os.path.dirname(os.path.abspath(str(p)))
    for test in suite.tests:
        if test.golden_trace and not os.path.isabs(test.golden_trace):
            test.golden_trace = os.path.join(base_dir, test.golden_trace)

    return suite


def filter_by_tags(spec: CIAgentSpec, tags: list[str]) -> CIAgentSpec:
    """Return a copy of the spec containing only queries that match any given tag.

    Args:
        spec: The loaded CIAgentSpec.
        tags: List of tag strings to filter by. Empty list returns all queries.

    Returns:
        New CIAgentSpec with filtered queries list.
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
