"""
Unit tests for the Cost Engine (Layer 3 — Soft Warning).

All tests are deterministic — builds Trace objects with specific metric values.
"""

from __future__ import annotations

import pytest

from agentci.engine.cost import evaluate_cost
from agentci.engine.results import LayerStatus
from agentci.models import LLMCall, Span, SpanKind, Trace
from agentci.schema.spec_models import CostSpec


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_trace(
    cost_usd: float = 0.0,
    tokens: int = 0,
    llm_calls: int = 0,
    duration_ms: float = 0.0,
) -> Trace:
    """Build a Trace with pre-set aggregate metrics."""
    t = Trace()
    t.total_cost_usd = cost_usd
    t.total_tokens = tokens
    t.total_llm_calls = llm_calls
    t.total_duration_ms = duration_ms
    return t


def cost(**kwargs) -> CostSpec:
    return CostSpec(**kwargs)


# ── max_cost_multiplier ───────────────────────────────────────────────────────


class TestMaxCostMultiplier:
    def test_under_multiplier_passes(self):
        trace = make_trace(cost_usd=0.002)
        baseline = make_trace(cost_usd=0.002)
        result = evaluate_cost(trace, cost(max_cost_multiplier=2.0), baseline)
        assert result.status == LayerStatus.PASS

    def test_over_multiplier_warns(self):
        trace = make_trace(cost_usd=0.006)
        baseline = make_trace(cost_usd=0.002)
        result = evaluate_cost(trace, cost(max_cost_multiplier=2.0), baseline)
        assert result.status == LayerStatus.WARN
        assert any("Cost" in m for m in result.messages)

    def test_exactly_at_multiplier_passes(self):
        trace = make_trace(cost_usd=0.004)
        baseline = make_trace(cost_usd=0.002)
        result = evaluate_cost(trace, cost(max_cost_multiplier=2.0), baseline)
        assert result.status == LayerStatus.PASS

    def test_no_baseline_skips_multiplier_check(self):
        trace = make_trace(cost_usd=999.0)
        result = evaluate_cost(trace, cost(max_cost_multiplier=1.0), baseline_trace=None)
        assert result.status == LayerStatus.PASS

    def test_zero_baseline_cost_skips_check(self):
        trace = make_trace(cost_usd=0.01)
        baseline = make_trace(cost_usd=0.0)
        result = evaluate_cost(trace, cost(max_cost_multiplier=2.0), baseline)
        # Division by zero avoided, check skipped
        assert result.status == LayerStatus.PASS


# ── max_total_tokens ──────────────────────────────────────────────────────────


class TestMaxTotalTokens:
    def test_under_limit_passes(self):
        trace = make_trace(tokens=400)
        result = evaluate_cost(trace, cost(max_total_tokens=500))
        assert result.status == LayerStatus.PASS

    def test_at_limit_passes(self):
        trace = make_trace(tokens=500)
        result = evaluate_cost(trace, cost(max_total_tokens=500))
        assert result.status == LayerStatus.PASS

    def test_over_limit_warns(self):
        trace = make_trace(tokens=600)
        result = evaluate_cost(trace, cost(max_total_tokens=500))
        assert result.status == LayerStatus.WARN
        assert any("Tokens" in m for m in result.messages)


# ── max_llm_calls ─────────────────────────────────────────────────────────────


class TestMaxLLMCalls:
    def test_under_limit_passes(self):
        trace = make_trace(llm_calls=1)
        result = evaluate_cost(trace, cost(max_llm_calls=3))
        assert result.status == LayerStatus.PASS

    def test_at_limit_passes(self):
        trace = make_trace(llm_calls=2)
        result = evaluate_cost(trace, cost(max_llm_calls=2))
        assert result.status == LayerStatus.PASS

    def test_over_limit_warns(self):
        trace = make_trace(llm_calls=5)
        result = evaluate_cost(trace, cost(max_llm_calls=2))
        assert result.status == LayerStatus.WARN
        assert any("LLM calls" in m for m in result.messages)


# ── max_latency_ms ────────────────────────────────────────────────────────────


class TestMaxLatencyMs:
    def test_under_limit_passes(self):
        trace = make_trace(duration_ms=800.0)
        result = evaluate_cost(trace, cost(max_latency_ms=1000))
        assert result.status == LayerStatus.PASS

    def test_at_limit_passes(self):
        trace = make_trace(duration_ms=1000.0)
        result = evaluate_cost(trace, cost(max_latency_ms=1000))
        assert result.status == LayerStatus.PASS

    def test_over_limit_warns(self):
        trace = make_trace(duration_ms=2000.0)
        result = evaluate_cost(trace, cost(max_latency_ms=1000))
        assert result.status == LayerStatus.WARN
        assert any("Latency" in m for m in result.messages)


# ── max_cost_usd ──────────────────────────────────────────────────────────────


class TestMaxCostUsd:
    def test_under_limit_passes(self):
        trace = make_trace(cost_usd=0.005)
        result = evaluate_cost(trace, cost(max_cost_usd=0.01))
        assert result.status == LayerStatus.PASS

    def test_at_limit_passes(self):
        trace = make_trace(cost_usd=0.01)
        result = evaluate_cost(trace, cost(max_cost_usd=0.01))
        assert result.status == LayerStatus.PASS

    def test_over_limit_warns(self):
        trace = make_trace(cost_usd=0.02)
        result = evaluate_cost(trace, cost(max_cost_usd=0.01))
        assert result.status == LayerStatus.WARN
        assert any("Cost" in m for m in result.messages)


# ── combined / edge cases ──────────────────────────────────────────────────────


class TestCombined:
    def test_multiple_warnings_all_reported(self):
        trace = make_trace(tokens=600, llm_calls=5)
        result = evaluate_cost(trace, cost(max_total_tokens=500, max_llm_calls=2))
        assert result.status == LayerStatus.WARN
        assert len(result.messages) == 2

    def test_empty_spec_passes(self):
        trace = make_trace(cost_usd=100.0, tokens=100_000)
        result = evaluate_cost(trace, cost())
        assert result.status == LayerStatus.PASS
        assert result.messages == ["Cost within bounds"]

    def test_details_always_contain_actual(self):
        trace = make_trace(cost_usd=0.001, tokens=100, llm_calls=1, duration_ms=500.0)
        result = evaluate_cost(trace, cost())
        assert "actual" in result.details
        actual = result.details["actual"]
        assert actual["cost_usd"] == pytest.approx(0.001)
        assert actual["total_tokens"] == 100
        assert actual["llm_calls"] == 1
        assert actual["latency_ms"] == 500.0


# ── descriptive PASS messages ────────────────────────────────────────────────


class TestDescriptivePassMessages:
    def test_llm_calls_describes_count(self):
        trace = make_trace(llm_calls=3)
        result = evaluate_cost(trace, cost(max_llm_calls=5))
        assert result.status == LayerStatus.PASS
        assert any("LLM calls: 3 ≤ max 5" in m for m in result.messages)

    def test_tokens_describes_count(self):
        trace = make_trace(tokens=1200)
        result = evaluate_cost(trace, cost(max_total_tokens=3000))
        assert result.status == LayerStatus.PASS
        assert any("Tokens: 1200 ≤ max 3000" in m for m in result.messages)

    def test_cost_usd_describes_amount(self):
        trace = make_trace(cost_usd=0.001)
        result = evaluate_cost(trace, cost(max_cost_usd=0.01))
        assert result.status == LayerStatus.PASS
        assert any("Cost: $" in m and "≤ max $" in m for m in result.messages)

    def test_latency_describes_ms(self):
        trace = make_trace(duration_ms=500.0)
        result = evaluate_cost(trace, cost(max_latency_ms=2000.0))
        assert result.status == LayerStatus.PASS
        assert any("Latency:" in m and "≤ max" in m for m in result.messages)
