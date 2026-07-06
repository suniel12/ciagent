"""
Tests for cost computation.
"""
from ciagent.cost import compute_cost

def test_compute_cost():
    cost = compute_cost("openai", "gpt-4o", 1000, 1000)
    assert cost > 0
