"""
Tests for suite-level `defaults:` handling in v1 ciagent.yaml configs.

Covers three layers:
1. Parsing: TestSuite accepts a `defaults: {max_cost_usd, max_steps}` block
   and maps it onto default_max_cost_usd / default_max_steps.
2. Propagation: suite defaults fill into tests that don't set their own
   budgets; explicit per-test budgets win.
3. Enforcement: TestRunner fails a test whose trace exceeds the (defaulted)
   max_cost_usd or max_steps budget.

Also asserts the four shipped example suites load cleanly, with no
UserWarning about unrecognized keys.
"""

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from ciagent.capture import _active_span
from ciagent.loader import load_suite
from ciagent.models import LLMCall
from ciagent.models import TestResult as Result
from ciagent.models import TestSuite as Suite
from ciagent.runner import TestRunner as Runner

PROJECT_ROOT = Path(__file__).parent.parent

# (relative config path, expected default_max_cost_usd, expected default_max_steps)
EXAMPLE_SUITES = [
    ("examples/anthropic_agent/ciagent.yaml", 0.05, 5),
    ("examples/langgraph_example/ciagent.yaml", 0.05, 10),
    ("examples/demo_agent/ciagent.yaml", 0.10, 5),
    ("examples/openai_agent/ciagent.yaml", 0.05, 5),
]


# ── Example suites load cleanly with defaults applied ────


@pytest.mark.parametrize("rel_path,max_cost,max_steps", EXAMPLE_SUITES)
def test_example_suite_loads_without_warnings(rel_path, max_cost, max_steps):
    """Each shipped example loads with zero warnings and its defaults parsed."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        suite = load_suite(str(PROJECT_ROOT / rel_path))

    assert suite.default_max_cost_usd == max_cost
    assert suite.default_max_steps == max_steps


@pytest.mark.parametrize("rel_path,max_cost,max_steps", EXAMPLE_SUITES)
def test_example_suite_defaults_reach_every_test(rel_path, max_cost, max_steps):
    """Every test case ends up with a budget, either its own or the suite default."""
    suite = load_suite(str(PROJECT_ROOT / rel_path))

    assert suite.tests, f"{rel_path} defines no tests"
    for test in suite.tests:
        assert test.max_cost_usd is not None
        assert test.max_steps is not None


def test_per_test_budget_overrides_suite_default():
    """The anthropic cost_check test keeps its tighter explicit budget."""
    suite = load_suite(
        str(PROJECT_ROOT / "examples" / "anthropic_agent" / "ciagent.yaml")
    )

    by_name = {t.name: t for t in suite.tests}
    assert by_name["cost_check"].max_cost_usd == 0.005      # explicit
    assert by_name["summarize_basic"].max_cost_usd == 0.05  # inherited
    assert by_name["summarize_basic"].max_steps == 5        # inherited


# ── defaults: block parsing ──────────────────────────────


def test_defaults_block_maps_to_default_fields():
    suite = Suite(
        name="s",
        defaults={"max_cost_usd": 0.02, "max_steps": 3},
        tests=[{"name": "t", "input": "hi"}],
    )

    assert suite.default_max_cost_usd == 0.02
    assert suite.default_max_steps == 3
    assert suite.tests[0].max_cost_usd == 0.02
    assert suite.tests[0].max_steps == 3


def test_explicit_default_fields_win_over_defaults_block():
    suite = Suite(
        name="s",
        default_max_cost_usd=0.5,
        defaults={"max_cost_usd": 0.02, "max_steps": 3},
    )

    assert suite.default_max_cost_usd == 0.5
    assert suite.default_max_steps == 3


def test_unknown_key_inside_defaults_raises():
    with pytest.raises(ValidationError, match="max_tokens"):
        Suite(name="s", defaults={"max_tokens": 100})


def test_non_mapping_defaults_raises():
    with pytest.raises(ValidationError, match="mapping"):
        Suite(name="s", defaults=[0.05])


def test_defaults_block_does_not_mutate_caller_dict():
    config = {"name": "s", "defaults": {"max_steps": 3}}
    Suite(**config)
    assert config["defaults"] == {"max_steps": 3}


def test_unrecognized_top_level_key_warns(tmp_path):
    config = tmp_path / "ciagent.yaml"
    config.write_text(
        "name: typo_suite\n"
        "agent: 'mod:fn'\n"
        "defautls:\n"
        "  max_steps: 3\n"
        "tests: []\n"
    )

    with pytest.warns(UserWarning, match="defautls"):
        suite = load_suite(str(config))

    assert suite.default_max_steps is None


# ── Runner enforcement of defaulted budgets ──────────────


def _make_suite(**defaults) -> Suite:
    return Suite(
        name="budget_suite",
        agent="mod:fn",
        defaults=defaults,
        tests=[{"name": "budget_test", "input": "go"}],
    )


def _agent_with_llm_calls(count: int, cost_each: float = 0.0):
    """Return a fake agent that records `count` LLM calls into the trace."""
    def agent(prompt):
        span = _active_span.get()
        for _ in range(count):
            span.llm_calls.append(
                LLMCall(model="fake-model", provider="fake", cost_usd=cost_each)
            )
        return "done"
    return agent


def test_runner_enforces_default_max_steps():
    suite = _make_suite(max_steps=5)
    runner = Runner(suite)

    result = runner.run_test(suite.tests[0], _agent_with_llm_calls(6))

    assert result.result == Result.FAILED
    assert any(
        "LLM calls" in r["message"] for r in result.assertion_results
    )


def test_runner_enforces_default_max_cost():
    suite = _make_suite(max_cost_usd=0.01)
    runner = Runner(suite)

    result = runner.run_test(
        suite.tests[0], _agent_with_llm_calls(2, cost_each=0.02)
    )

    assert result.result == Result.FAILED
    assert any(
        "exceeds budget" in r["message"] for r in result.assertion_results
    )


def test_runner_passes_within_default_budgets():
    suite = _make_suite(max_cost_usd=0.05, max_steps=5)
    runner = Runner(suite)

    result = runner.run_test(
        suite.tests[0], _agent_with_llm_calls(3, cost_each=0.001)
    )

    assert result.result == Result.PASSED
