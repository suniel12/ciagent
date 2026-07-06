"""
Tests for the cost estimator — pre-execution cost estimates.
"""
from ciagent.engine.cost_estimator import estimate_cost, format_estimate


class TestEstimateCost:
    """Tests for estimate_cost()."""

    def test_returns_required_keys(self):
        est = estimate_cost(num_queries=10)
        assert "agent_cost" in est
        assert "judge_cost" in est
        assert "total_estimate" in est
        assert "total_low" in est
        assert "total_high" in est

    def test_more_queries_costs_more(self):
        est5 = estimate_cost(num_queries=5)
        est15 = estimate_cost(num_queries=15)
        assert est15["total_estimate"] > est5["total_estimate"]

    def test_no_judge_when_disabled(self):
        est = estimate_cost(num_queries=10, has_llm_judge=False)
        assert est["judge_cost"] == 0.0

    def test_with_judge_costs_more(self):
        no_judge = estimate_cost(num_queries=10, has_llm_judge=False)
        with_judge = estimate_cost(num_queries=10, has_llm_judge=True)
        assert with_judge["total_estimate"] > no_judge["total_estimate"]

    def test_range_brackets_estimate(self):
        est = estimate_cost(num_queries=10)
        assert est["total_low"] < est["total_estimate"]
        assert est["total_high"] > est["total_estimate"]

    def test_unknown_model_uses_default(self):
        est = estimate_cost(num_queries=10, agent_model="unknown-model-xyz")
        assert est["agent_cost"] > 0  # should use gpt-4o default

    def test_zero_queries(self):
        est = estimate_cost(num_queries=0)
        assert est["total_estimate"] == 0.0


class TestFormatEstimate:
    """Tests for format_estimate()."""

    def test_contains_query_count(self):
        est = estimate_cost(num_queries=15)
        text = format_estimate(est, 15)
        assert "15" in text

    def test_contains_dollar_amounts(self):
        est = estimate_cost(num_queries=10)
        text = format_estimate(est, 10)
        assert "$" in text

    def test_shows_judge_line_when_present(self):
        est = estimate_cost(num_queries=10, has_llm_judge=True)
        text = format_estimate(est, 10)
        assert "Judge" in text

    def test_no_judge_line_when_zero(self):
        est = estimate_cost(num_queries=10, has_llm_judge=False)
        text = format_estimate(est, 10)
        assert "Judge" not in text
