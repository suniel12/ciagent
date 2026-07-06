"""
Live integration tests for the LLM judge.

These tests make real API calls to Anthropic and are gated by:
    AGENTCI_LIVE_TESTS=1

Run with:
    AGENTCI_LIVE_TESTS=1 python -m pytest tests/integration/test_judge_live.py -v
"""

import os

import pytest

from ciagent.engine.judge import run_judge
from ciagent.schema.spec_models import JudgeRubric

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENTCI_LIVE_TESTS") != "1",
    reason="Set AGENTCI_LIVE_TESTS=1 to run live judge tests",
)


@pytest.fixture
def clear_rubric() -> JudgeRubric:
    return JudgeRubric(
        rule="The response clearly answers the question asked",
        scale=[
            "1: Does not address the question at all",
            "3: Partially answers the question",
            "5: Completely and clearly answers the question",
        ],
        threshold=0.5,
    )


def test_judge_returns_pass_for_good_answer(clear_rubric):
    result = run_judge(
        answer="To install AgentCI, run: pip install agentci",
        rubric=clear_rubric,
        config={"model": "claude-haiku-4-5-20251001", "temperature": 0},
    )
    assert "passed" in result
    assert "score" in result
    assert isinstance(result["score"], int)
    assert 1 <= result["score"] <= 5
    assert result["label"] in ("pass", "fail", "borderline")
    assert result["rationale"]


def test_judge_returns_fail_for_irrelevant_answer(clear_rubric):
    result = run_judge(
        answer="The sky is blue and clouds are white.",
        rubric=JudgeRubric(
            rule="The response explains how to install Python packages",
            threshold=0.6,
        ),
        config={"model": "claude-haiku-4-5-20251001", "temperature": 0},
    )
    assert "passed" in result
    # Irrelevant answer should likely fail this rubric
    assert result["score"] <= 3


def test_judge_structured_output_parseable(clear_rubric):
    result = run_judge(
        answer="This is a test response",
        rubric=clear_rubric,
        config={"model": "claude-haiku-4-5-20251001", "temperature": 0},
    )
    # All fields must be present and correctly typed
    assert isinstance(result["passed"], bool)
    assert isinstance(result["score"], int)
    assert result["label"] in ("pass", "fail", "borderline")
    assert isinstance(result["rationale"], str)
    assert len(result["rationale"]) > 0


def test_judge_with_context_grounds_evaluation():
    rubric = JudgeRubric(
        rule="All claims are grounded in the provided context; no fabricated facts",
        threshold=0.8,
    )
    context = "AgentCI is installed via pip. The command is: pip install agentci"
    result = run_judge(
        answer="To install AgentCI, run pip install agentci.",
        rubric=rubric,
        config={"model": "claude-haiku-4-5-20251001", "temperature": 0},
        context=context,
    )
    assert result["passed"] is True
