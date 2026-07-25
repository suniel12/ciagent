# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
DEPRECATED shim for the old v1 config loader.

All loading now lives in ciagent.loader:
    load_spec(path)   — v2 ciagent_spec.yaml (`queries:` key)
    load_suite(path)  — v1 ciagent.yaml (`tests:` key, legacy)

This module remains only so `from ciagent.config import load_config`
keeps working. It will be removed together with the v1 format in 0.9.0.
"""

import warnings

from .loader import load_suite
from .models import TestSuite


def load_config(path: str = "ciagent.yaml") -> TestSuite:
    """DEPRECATED: use ciagent.loader.load_suite (v1) or load_spec (v2)."""
    warnings.warn(
        "ciagent.config.load_config is deprecated and will be removed in "
        "0.9.0. Use ciagent.loader.load_suite for v1 ciagent.yaml suites, "
        "or migrate to ciagent_spec.yaml and ciagent.loader.load_spec.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_suite(path)
