"""
Integration tests for real-world agent examples.

Tests all three agent types (OpenAI, Anthropic, LangGraph) through the
full AgentCI pipeline: config loading -> runner -> trace capture -> assertions -> diffing.

These tests run in mock mode by default (no API keys required).
Set AGENTCI_LIVE=1 with appropriate API keys for live testing.
"""

import os
import pytest
from pathlib import Path

from ciagent.config import load_config
from ciagent.runner import TestRunner
from ciagent.models import TestResult

PROJECT_ROOT = Path(__file__).parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import agent modules to access their mock activation functions.
# The import order matters: the last OpenAI-patching agent imported
# will have its mock active. We use _activate_mock/_deactivate_mock
# to switch between them for cross-agent tests.
import examples.openai_agent.agent as openai_agent_mod
import examples.anthropic_agent.agent as anthropic_agent_mod
import examples.langgraph_example.agent as langgraph_agent_mod


# ── OpenAI Agent Tests ───────────────────────────────────


class TestOpenAIAgent:
    """Integration tests for the OpenAI weather agent."""

    @pytest.fixture(autouse=True)
    def activate_openai_mock(self):
        """Ensure the OpenAI agent's mock is active for these tests."""
        openai_agent_mod._activate_mock()
        yield
        openai_agent_mod._deactivate_mock()

    @pytest.fixture
    def suite(self):
        config_path = str(PROJECT_ROOT / "examples" / "openai_agent" / "agentci.yaml")
        return load_config(config_path)

    @pytest.fixture
    def runner(self, suite):
        return TestRunner(suite)

    def test_suite_loads(self, suite):
        """Config loads and parses correctly."""
        assert suite.name == "openai_weather_agent_tests"
        assert len(suite.tests) == 3
        assert suite.agent == "examples.openai_agent.agent:run_agent"

    def test_all_tests_pass(self, runner):
        """All test cases pass through the full pipeline."""
        result = runner.run_suite()
        assert result.total_passed == 3
        assert result.total_failed == 0
        assert result.total_errors == 0

    def test_weather_basic_captures_tool_calls(self, runner, suite):
        """The weather_basic test captures get_weather tool call."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "weather_basic")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.result == TestResult.PASSED
        assert "get_weather" in run_result.trace.tool_call_sequence

    def test_weather_forecast_captures_tool_calls(self, runner, suite):
        """The forecast test captures get_forecast tool call."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "weather_forecast")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.result == TestResult.PASSED
        assert "get_forecast" in run_result.trace.tool_call_sequence

    def test_cost_is_computed(self, runner, suite):
        """Costs are computed from mock token usage."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "weather_basic")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.trace.total_cost_usd > 0

    def test_llm_calls_are_recorded(self, runner, suite):
        """LLM calls are captured in the trace spans."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "weather_basic")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.trace.total_llm_calls > 0
        assert len(run_result.trace.spans) > 0
        assert len(run_result.trace.spans[0].llm_calls) > 0

    def test_golden_trace_diffing(self, runner, suite):
        """Golden trace diffing produces no diffs for deterministic mock."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "weather_basic")

        if test_case.golden_trace and os.path.exists(test_case.golden_trace):
            run_result = runner.run_test(test_case, agent_fn)
            assert len(run_result.diffs) == 0

    def test_statistical_mode(self, runner):
        """Statistical mode runs multiple iterations correctly."""
        result = runner.run_suite(runs=3)
        assert result.total_passed == 9  # 3 tests x 3 runs
        assert result.total_failed == 0


# ── Anthropic Agent Tests ─────────────────────────────────


class TestAnthropicAgent:
    """Integration tests for the Anthropic summarizer agent."""

    @pytest.fixture(autouse=True)
    def activate_anthropic_mock(self):
        """Ensure the Anthropic agent's mock is active for these tests."""
        anthropic_agent_mod._activate_mock()
        yield
        anthropic_agent_mod._deactivate_mock()

    @pytest.fixture
    def suite(self):
        config_path = str(PROJECT_ROOT / "examples" / "anthropic_agent" / "agentci.yaml")
        return load_config(config_path)

    @pytest.fixture
    def runner(self, suite):
        return TestRunner(suite)

    def test_suite_loads(self, suite):
        """Config loads and parses correctly."""
        assert suite.name == "anthropic_summarizer_tests"
        assert len(suite.tests) == 3

    def test_all_tests_pass(self, runner):
        """All test cases pass through the full pipeline."""
        result = runner.run_suite()
        assert result.total_passed == 3
        assert result.total_failed == 0
        assert result.total_errors == 0

    def test_summarize_basic_captures_both_tools(self, runner, suite):
        """Summarize basic test captures fetch_article and save_summary."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "summarize_basic")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.result == TestResult.PASSED
        tools = run_result.trace.tool_call_sequence
        assert "fetch_article" in tools
        assert "save_summary" in tools

    def test_no_save_respects_instruction(self, runner, suite):
        """When asked not to save, save_summary should not be called."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "summarize_no_save")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.result == TestResult.PASSED
        tools = run_result.trace.tool_call_sequence
        assert "fetch_article" in tools
        assert "save_summary" not in tools

    def test_anthropic_cost_computed(self, runner, suite):
        """Costs use Anthropic pricing model."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "summarize_basic")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.trace.total_cost_usd > 0
        for span in run_result.trace.spans:
            for llm_call in span.llm_calls:
                assert llm_call.provider == "anthropic"

    def test_golden_trace_diffing(self, runner, suite):
        """Golden trace diffing works for Anthropic agent."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "summarize_basic")

        if test_case.golden_trace and os.path.exists(test_case.golden_trace):
            run_result = runner.run_test(test_case, agent_fn)
            assert len(run_result.diffs) == 0


# ── LangGraph Agent Tests ─────────────────────────────────


class TestLangGraphAgent:
    """Integration tests for the LangGraph research agent."""

    @pytest.fixture(autouse=True)
    def activate_langgraph_mock(self):
        """Ensure the LangGraph agent's mock is active for these tests."""
        langgraph_agent_mod._activate_mock()
        yield
        langgraph_agent_mod._deactivate_mock()

    @pytest.fixture
    def suite(self):
        config_path = str(PROJECT_ROOT / "examples" / "langgraph_example" / "agentci.yaml")
        return load_config(config_path)

    @pytest.fixture
    def runner(self, suite):
        return TestRunner(suite)

    def test_suite_loads(self, suite):
        """Config loads and parses correctly."""
        assert suite.name == "langgraph_research_agent_tests"
        assert len(suite.tests) == 3

    def test_all_tests_pass(self, runner):
        """All test cases pass through the full pipeline."""
        result = runner.run_suite()
        assert result.total_passed == 3
        assert result.total_failed == 0
        assert result.total_errors == 0

    def test_research_basic_uses_web_search(self, runner, suite):
        """Basic research question triggers web_search."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "research_basic")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.result == TestResult.PASSED
        assert "web_search" in run_result.trace.tool_call_sequence

    def test_math_uses_both_search_and_calculator(self, runner, suite):
        """Math question triggers both web_search and calculator."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "research_with_math")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.result == TestResult.PASSED
        tools = run_result.trace.tool_call_sequence
        assert "web_search" in tools
        assert "calculator" in tools

    def test_save_uses_save_answer(self, runner, suite):
        """Save request triggers save_answer tool."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "research_saves")
        run_result = runner.run_test(test_case, agent_fn)

        assert run_result.result == TestResult.PASSED
        assert "save_answer" in run_result.trace.tool_call_sequence

    def test_multi_step_llm_calls(self, runner, suite):
        """LangGraph ReAct loop makes multiple LLM calls."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "research_with_math")
        run_result = runner.run_test(test_case, agent_fn)

        # Math query needs: search -> search -> calculate -> answer = at least 3 LLM calls
        assert run_result.trace.total_llm_calls >= 3

    def test_golden_trace_diffing(self, runner, suite):
        """Golden trace diffing works for LangGraph agent."""
        agent_fn = runner._import_agent()
        test_case = next(t for t in suite.tests if t.name == "research_basic")

        if test_case.golden_trace and os.path.exists(test_case.golden_trace):
            run_result = runner.run_test(test_case, agent_fn)
            assert len(run_result.diffs) == 0


# ── Cross-Agent Tests ──────────────────────────────────────

# Maps agent dir to its module and mock activation functions
_AGENT_MOCKS = {
    "openai_agent": openai_agent_mod,
    "anthropic_agent": anthropic_agent_mod,
    "langgraph_example": langgraph_agent_mod,
}


class TestCrossAgent:
    """Tests that validate AgentCI works consistently across agent types."""

    def _run_agent_suite(self, example_dir: str):
        """Run an agent suite with proper mock isolation."""
        agent_mod = _AGENT_MOCKS[example_dir]
        agent_mod._activate_mock()
        try:
            config_path = str(PROJECT_ROOT / "examples" / example_dir / "agentci.yaml")
            suite = load_config(config_path)
            runner = TestRunner(suite)
            return runner.run_suite()
        finally:
            agent_mod._deactivate_mock()

    def test_all_agents_pass(self):
        """All three agent suites pass their tests."""
        for agent_dir in ["openai_agent", "anthropic_agent", "langgraph_example"]:
            result = self._run_agent_suite(agent_dir)
            assert result.total_errors == 0, f"{agent_dir} had errors"
            assert result.total_failed == 0, f"{agent_dir} had failures"

    def test_all_agents_compute_costs(self):
        """All agents produce non-zero cost estimates."""
        for agent_dir in ["openai_agent", "anthropic_agent", "langgraph_example"]:
            result = self._run_agent_suite(agent_dir)
            assert result.total_cost_usd > 0, f"{agent_dir} had zero cost"

    def test_all_agents_record_tool_calls(self):
        """All agents produce traces with tool calls."""
        for agent_dir in ["openai_agent", "anthropic_agent", "langgraph_example"]:
            result = self._run_agent_suite(agent_dir)
            for run_result in result.results:
                assert len(run_result.trace.tool_call_sequence) > 0, \
                    f"{agent_dir}/{run_result.test_name} had no tool calls"
