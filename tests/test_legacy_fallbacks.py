# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""Deprecation fallbacks for the 0.16.0 rename (accepted until 1.0).

Covers: legacy spec filename agentci_spec.yaml, legacy spec keys
runner:/conversation_runner:, warn-once semantics, and pytest collection of
the legacy filename. The legacy fixture lives in tests/fixtures/legacy/ and
is the residue guard's only sanctioned home for old names.
"""
import shutil
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

import ciagent.loader as loader_mod
from ciagent.loader import load_spec, resolve_default_spec_path

FIXTURES = Path(__file__).parent / "fixtures" / "legacy"


@pytest.fixture(autouse=True)
def _reset_warn_once():
    loader_mod._deprecation_warned.clear()
    yield
    loader_mod._deprecation_warned.clear()


def test_legacy_keys_load_as_adapter_fields(capsys):
    spec = load_spec(FIXTURES / "agentci_spec.yaml")
    assert spec.adapter == "myagent.run:run_for_agentci"
    assert spec.conversation_adapter == "myagent.run:respond"
    err = capsys.readouterr().err
    assert err.count("'runner:' is deprecated") == 1
    assert err.count("'conversation_runner:' is deprecated") == 1


def test_new_keys_produce_no_warning(tmp_path, capsys):
    p = tmp_path / "ciagent_spec.yaml"
    p.write_text(
        "agent: modern\n"
        "adapter: myagent.run:run_for_ciagent\n"
        "queries:\n  - query: hi\n"
    )
    spec = load_spec(p)
    assert spec.adapter == "myagent.run:run_for_ciagent"
    assert "DEPRECATED" not in capsys.readouterr().err


def test_legacy_filename_fallback_warns_once(tmp_path, capsys):
    shutil.copy(FIXTURES / "agentci_spec.yaml", tmp_path / "agentci_spec.yaml")
    requested = tmp_path / "ciagent_spec.yaml"  # does not exist
    spec1 = load_spec(requested)
    spec2 = load_spec(requested)  # second load: no second filename warning
    assert spec1.agent == spec2.agent == "legacy-agent"
    assert capsys.readouterr().err.count("agentci_spec.yaml is deprecated") == 1


def test_resolve_default_prefers_new_name(tmp_path):
    (tmp_path / "ciagent_spec.yaml").write_text("agent: a\nqueries: []\n")
    (tmp_path / "agentci_spec.yaml").write_text("agent: b\nqueries: []\n")
    assert resolve_default_spec_path(tmp_path).name == "ciagent_spec.yaml"


def test_pytest_collects_legacy_filename(pytester):
    shutil.copy(FIXTURES / "agentci_spec.yaml", pytester.path / "agentci_spec.yaml")
    result = pytester.runpytest("--collect-only")
    result.stdout.fnmatch_lines(["*agentci_spec.yaml*"])


def test_mock_test_prefers_legacy_spec_over_demo(tmp_path, monkeypatch):
    """With only agentci_spec.yaml present, `ciagent test --mock` must run it
    (via the filename fallback), never silently switch to the bundled demo."""
    from click.testing import CliRunner
    from ciagent.cli import cli

    shutil.copy(FIXTURES / "agentci_spec.yaml", tmp_path / "agentci_spec.yaml")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["test", "--mock", "--yes"])
    assert "Demo mode" not in result.output
    assert "legacy-agent" in result.output
