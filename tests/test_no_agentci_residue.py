# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""Residue guard: the old brand (agentci / AgentCI) must not creep back.

The 0.16.0 sweep purged the pre-rename brand from every user-facing surface.
`agentci` is also a competitor's PyPI package, so regressions here hand users
a competitor's name. Scan is `git grep` over tracked text files only.

Allowlist rationale:
- CHANGELOG.md, Plan_docs/: history, stays as written
- demo/*.cast: recorded terminal sessions (re-record, never rewrite)
- tests/fixtures/legacy/, test_legacy_fallbacks.py: exercise the 1.0
  deprecation fallbacks on purpose
- test_plugin_and_bootstrap.py: guards AGAINST the removed `agentci` CLI alias
- loader.py, cli.py, pytest_plugin.py: contain the legacy-fallback literals
- this file
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ALLOWED = (
    "CHANGELOG.md",
    "Plan_docs/",
    "demo/",
    "tests/fixtures/legacy/",
    "tests/test_legacy_fallbacks.py",
    "tests/test_no_agentci_residue.py",
    "tests/test_plugin_and_bootstrap.py",
    "src/ciagent/loader.py",
    "src/ciagent/cli.py",
    "src/ciagent/pytest_plugin.py",
)


def test_no_agentci_residue_outside_allowlist():
    out = subprocess.run(
        ["git", "grep", "-Ili", "agentci"],
        cwd=REPO, capture_output=True, text=True,
    )
    # exit 1 = no matches at all, which is also a pass
    offenders = [
        line for line in out.stdout.splitlines()
        if line and not line.startswith(ALLOWED)
    ]
    assert not offenders, (
        "old brand 'agentci' found outside the allowlist "
        f"(rename to ciagent/CIAgent): {offenders}"
    )


def test_allowlisted_code_files_only_contain_fallback_literals():
    """The three src allowlist entries may mention agentci ONLY as the legacy
    fallback literal, never as a generated/default artifact name."""
    for rel in ("src/ciagent/loader.py", "src/ciagent/cli.py",
                "src/ciagent/pytest_plugin.py"):
        text = (REPO / rel).read_text().lower()
        # every remaining mention must be the legacy spec filename
        assert text.count("agentci") == text.count("agentci_spec.yaml"), (
            f"{rel}: 'agentci' appears outside the legacy-filename fallback"
        )
