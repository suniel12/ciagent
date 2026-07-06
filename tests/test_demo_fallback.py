# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the zero-key demo fallback: `agentci test --mock` with no
agentci_spec.yaml in the working directory runs the bundled demo spec.

Guarantees under test:
- The fallback triggers ONLY for the default config path + --mock; an
  explicitly passed --config that is missing stays an error.
- Demo multi-run sessions default the flaky simulation ON with the "spread"
  style, so the aggregate score stays constant while verdicts flip (the
  money screenshot); an explicit AGENTCI_MOCK_FLAKY always wins.
- A local spec always takes precedence over the bundled demo.
"""

from __future__ import annotations

from importlib.resources import files

import pytest
from click.testing import CliRunner

from agentci.cli import cli
from agentci.loader import load_spec

MINIMAL_SPEC = """
agent: local-agent
queries:
  - query: "hello"
    correctness:
      expected_in_answer: ["documentation"]
"""


@pytest.fixture()
def runner():
    return CliRunner()


def _invoke_isolated(runner, args, env=None):
    with runner.isolated_filesystem():
        return runner.invoke(cli, args, env=env or {})


# ── Bundled spec integrity ──────────────────────────────────────────────────────


class TestBundledDemoSpec:
    def test_demo_spec_loads_and_validates(self):
        path = files("agentci").joinpath("examples", "demo_spec.yaml")
        spec = load_spec(str(path))
        assert spec.agent == "demo-support-agent"
        # ≥4 queries so the spread style (breaks queries 0-2) leaves the
        # majority stable and the demo reads as flaky-suite, not broken-suite.
        assert len(spec.queries) >= 4
        for q in spec.queries:
            assert q.correctness is not None

    def test_spread_style_breaks_one_query_per_run(self):
        from agentci.engine.mock_runner import run_mock_spec

        path = files("agentci").joinpath("examples", "demo_spec.yaml")
        spec = load_spec(str(path))
        for run_index in range(3):
            traces = run_mock_spec(
                spec, run_index=run_index, flaky=True, flaky_style="spread"
            )
            broken = [
                q.query
                for q in spec.queries
                if "flaky variant" in traces[q.query].metadata["final_output"]
            ]
            assert broken == [spec.queries[run_index % 3].query]

    def test_alternate_style_unchanged(self):
        from agentci.engine.mock_runner import run_mock_spec

        path = files("agentci").joinpath("examples", "demo_spec.yaml")
        spec = load_spec(str(path))
        traces = run_mock_spec(spec, run_index=1, flaky=True)
        broken = [
            i
            for i, q in enumerate(spec.queries)
            if "flaky variant" in traces[q.query].metadata["final_output"]
        ]
        assert broken == [i for i in range(len(spec.queries)) if i % 2 == 0]


# ── CLI fallback behavior ───────────────────────────────────────────────────────


class TestDemoFallback:
    def test_multi_run_demo_shows_flaky_stability_report(self, runner):
        result = _invoke_isolated(runner, ["test", "--mock", "--runs", "3"])
        assert result.exit_code == 0, result.output
        assert "Demo mode" in result.output
        # rich wraps at 80 cols, so match a single word of the banner
        assert "synthetic" in result.output
        assert "FLAKY" in result.output
        assert "agent-variance" in result.output

    def test_multi_run_demo_score_is_stable_across_runs(self, runner):
        # The demo's whole point: aggregate score identical, verdicts flip.
        result = _invoke_isolated(runner, ["test", "--mock", "--runs", "3"])
        assert result.exit_code == 0, result.output
        scores = [
            line for line in result.output.splitlines() if "Suite score" in line
        ]
        assert len(scores) == 1
        percents = [tok for tok in scores[0].split() if tok.endswith("%")]
        assert len(percents) == 3
        assert len(set(percents)) == 1, scores[0]

    def test_single_run_demo_passes_and_hints_runs(self, runner):
        result = _invoke_isolated(runner, ["test", "--mock"])
        assert result.exit_code == 0, result.output
        assert "Demo mode" in result.output
        assert "--runs 3" in result.output

    def test_env_var_off_makes_demo_stable(self, runner):
        result = _invoke_isolated(
            runner,
            ["test", "--mock", "--runs", "3"],
            env={"AGENTCI_MOCK_FLAKY": "0"},
        )
        assert result.exit_code == 0, result.output
        assert "STABLE" in result.output
        assert "FLAKY —" not in result.output

    def test_no_mock_no_spec_errors_with_demo_hint(self, runner):
        result = _invoke_isolated(runner, ["test"])
        assert result.exit_code == 2, result.output
        assert "ciagent init" in result.output
        assert "--mock --runs 3" in result.output

    def test_explicit_missing_config_is_error_not_demo(self, runner):
        result = _invoke_isolated(
            runner, ["test", "--mock", "--config", "nope.yaml"]
        )
        assert result.exit_code == 2, result.output
        assert "Demo mode" not in result.output

    def test_explicit_default_named_config_is_error_not_demo(self, runner):
        # Even the default filename, when typed by the user, must not
        # silently become the demo.
        result = _invoke_isolated(
            runner, ["test", "--mock", "--config", "agentci_spec.yaml"]
        )
        assert result.exit_code == 2, result.output
        assert "Demo mode" not in result.output

    def test_local_spec_takes_precedence(self, runner):
        with runner.isolated_filesystem():
            with open("agentci_spec.yaml", "w") as f:
                f.write(MINIMAL_SPEC)
            result = runner.invoke(cli, ["test", "--mock"])
        assert result.exit_code == 0, result.output
        assert "Demo mode" not in result.output
        assert "local-agent" in result.output

    def test_local_spec_multi_run_flaky_stays_opt_in(self, runner):
        # Outside demo mode, flakiness must remain env-var opt-in.
        with runner.isolated_filesystem():
            with open("agentci_spec.yaml", "w") as f:
                f.write(MINIMAL_SPEC)
            result = runner.invoke(cli, ["test", "--mock", "--runs", "2"])
        assert result.exit_code == 0, result.output
        assert "STABLE" in result.output
