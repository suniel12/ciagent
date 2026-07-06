"""
Tests for the mock runner — synthetic trace generation from spec expectations.
"""
from ciagent.engine.mock_runner import mock_run, run_mock_spec
from ciagent.models import Trace


class TestMockRun:
    """Tests for mock_run()."""

    def test_returns_trace(self):
        trace = mock_run("Hello", {})
        assert isinstance(trace, Trace)
        assert trace.test_name == "Hello"

    def test_populates_expected_tools(self):
        spec = {"path": {"expected_tools": ["search_docs", "grade_answer"]}}
        trace = mock_run("What is X?", spec)
        tool_names = [tc.tool_name for tc in trace.spans[0].tool_calls]
        assert "search_docs" in tool_names
        assert "grade_answer" in tool_names

    def test_populates_expected_keywords_in_output(self):
        spec = {"correctness": {"expected_in_answer": ["$199", "Business plan"]}}
        trace = mock_run("What is the price?", spec)
        assert "$199" in trace.metadata["final_output"]
        assert "Business plan" in trace.metadata["final_output"]

    def test_respects_max_llm_calls(self):
        spec = {"cost": {"max_llm_calls": 5}}
        trace = mock_run("test", spec)
        # Should stay within budget (mock uses min(max, 2))
        assert len(trace.spans[0].llm_calls) <= 5

    def test_zero_cost(self):
        trace = mock_run("test", {})
        assert trace.total_cost_usd == 0.0

    def test_no_tools_for_out_of_scope(self):
        spec = {"path": {"max_tool_calls": 0}}
        trace = mock_run("What is the weather?", spec)
        assert len(trace.spans[0].tool_calls) == 0

    def test_populates_any_expected_keywords_in_output(self):
        spec = {"correctness": {"any_expected_in_answer": ["pip", "brew", "conda"]}}
        trace = mock_run("How do I install?", spec)
        output = trace.metadata["final_output"]
        assert any(kw in output for kw in ["pip", "brew", "conda"])

    def test_populates_both_expected_and_any_expected(self):
        spec = {
            "correctness": {
                "expected_in_answer": ["$199", "Business plan"],
                "any_expected_in_answer": ["monthly", "annual"],
            }
        }
        trace = mock_run("What is the price?", spec)
        output = trace.metadata["final_output"]
        # AND keywords must all be present
        assert "$199" in output
        assert "Business plan" in output
        # At least one OR keyword must be present
        assert any(kw in output for kw in ["monthly", "annual"])

    def test_no_keywords_shows_placeholder(self):
        trace = mock_run("Hello", {"correctness": {}})
        assert "Mock response" in trace.metadata["final_output"]


class TestRunMockSpec:
    """Tests for run_mock_spec()."""

    def test_generates_traces_for_all_queries(self):
        from ciagent.schema.spec_models import AgentCISpec, GoldenQuery

        spec = AgentCISpec(
            agent="test-agent",
            queries=[
                GoldenQuery(query="Hello"),
                GoldenQuery(query="What is X?"),
                GoldenQuery(query="Goodbye"),
            ],
        )
        traces = run_mock_spec(spec)
        assert len(traces) == 3
        assert "Hello" in traces
        assert "What is X?" in traces
        assert "Goodbye" in traces

    def test_all_traces_are_valid(self):
        from ciagent.schema.spec_models import AgentCISpec, GoldenQuery

        spec = AgentCISpec(
            agent="test-agent",
            queries=[GoldenQuery(query="test query")],
        )
        traces = run_mock_spec(spec)
        for trace in traces.values():
            assert isinstance(trace, Trace)
            assert len(trace.spans) > 0
