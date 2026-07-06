# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Cost estimator for AgentCI — estimates API cost before running tests.

Provides a pre-execution cost estimate based on query count, model pricing,
and typical token usage patterns. Shows the estimate to the user before
running live tests so there are no surprise bills.
"""

from __future__ import annotations

import os


# Per-million-token pricing (updated Feb 2026)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-6":      {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":       {"input": 0.80,  "output": 4.00},
    "claude-opus-4-6":        {"input": 15.00, "output": 75.00},
    # OpenAI
    "gpt-4o":                 {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":            {"input": 0.15,  "output": 0.60},
    "gpt-4.1":                {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini":           {"input": 0.40,  "output": 1.60},
}

# Estimated tokens per query (based on typical RAG agent patterns)
DEFAULT_TOKENS_PER_QUERY: dict[str, int] = {
    "input": 2000,        # system prompt + KB retrieval + user query
    "output": 500,        # agent response
    "judge_input": 1500,  # judge prompt + agent response
    "judge_output": 200,  # judge verdict
}


def _default_judge_model() -> str:
    """Pick judge model based on available API keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-6"
    return "gpt-4o"


def estimate_cost(
    num_queries: int,
    agent_model: str = "gpt-4o",
    judge_model: str | None = None,
    has_llm_judge: bool = True,
) -> dict[str, float]:
    """Estimate the cost of running ciagent test.

    Parameters
    ----------
    num_queries : int
        Number of test queries to run.
    agent_model : str
        Model used by the agent under test.
    judge_model : str
        Model used by the LLM judge (from judge_config).
    has_llm_judge : bool
        Whether any queries use llm_judge evaluation.

    Returns
    -------
    dict
        Keys: agent_cost, judge_cost, total_estimate, total_low, total_high.
    """
    effective_judge = judge_model or _default_judge_model()
    agent_price = MODEL_PRICING.get(agent_model, MODEL_PRICING["gpt-4o"])
    judge_price = MODEL_PRICING.get(effective_judge, MODEL_PRICING["gpt-4o"])

    agent_cost = num_queries * (
        DEFAULT_TOKENS_PER_QUERY["input"] * agent_price["input"] / 1_000_000
        + DEFAULT_TOKENS_PER_QUERY["output"] * agent_price["output"] / 1_000_000
    )

    judge_cost = 0.0
    if has_llm_judge:
        judge_cost = num_queries * (
            DEFAULT_TOKENS_PER_QUERY["judge_input"] * judge_price["input"] / 1_000_000
            + DEFAULT_TOKENS_PER_QUERY["judge_output"] * judge_price["output"] / 1_000_000
        )

    total = agent_cost + judge_cost
    return {
        "agent_cost": agent_cost,
        "judge_cost": judge_cost,
        "total_estimate": total,
        "total_low": total * 0.5,
        "total_high": total * 2.0,
    }


def format_estimate(estimate: dict[str, float], num_queries: int) -> str:
    """Format a cost estimate as a human-readable string.

    Parameters
    ----------
    estimate : dict
        Output of ``estimate_cost()``.
    num_queries : int
        Number of queries (for display).

    Returns
    -------
    str
        Multi-line formatted cost estimate.
    """
    lines = [
        f"Estimated cost for {num_queries} queries:",
        f"  Agent:  ~${estimate['agent_cost']:.4f}",
    ]
    if estimate["judge_cost"] > 0:
        lines.append(f"  Judge:  ~${estimate['judge_cost']:.4f}")
    lines.append(
        f"  Range:  ${estimate['total_low']:.4f} – ${estimate['total_high']:.4f}"
    )
    return "\n".join(lines)
