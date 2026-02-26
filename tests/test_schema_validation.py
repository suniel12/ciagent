"""
Tests for AgentCI v2 spec schema validation.

Every test either asserts a valid spec passes Pydantic validation or that
a specific invalid variant raises a ValidationError with a meaningful message.
"""

import pytest
from pydantic import ValidationError

from agentci.schema.spec_models import (
    AgentCISpec,
    CorrectnessSpec,
    CostSpec,
    GoldenQuery,
    JudgeRubric,
    MatchMode,
    PathSpec,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def minimal_query(**kwargs) -> dict:
    base = {"query": "How do I install AgentCI?"}
    base.update(kwargs)
    return base


def minimal_spec(**kwargs) -> dict:
    base = {"agent": "test-agent", "queries": [minimal_query()]}
    base.update(kwargs)
    return base


# ── Valid specs ──────────────────────────────────────────────────────────────


def test_minimal_spec_is_valid():
    spec = AgentCISpec(**minimal_spec())
    assert spec.agent == "test-agent"
    assert len(spec.queries) == 1
    assert spec.version == 1
    assert spec.baseline_dir == "./golden"


def test_full_spec_with_all_fields():
    spec = AgentCISpec(
        version=1,
        agent="rag-agent",
        baseline_dir="./baselines/rag",
        defaults={
            "correctness": {"hallucination_check": {"rule": "No fabricated facts", "threshold": 0.8}}
        },
        judge_config={"model": "claude-sonnet-4-6", "temperature": 0},
        queries=[
            {
                "query": "How do I install AgentCI?",
                "description": "Installation smoke test",
                "tags": ["smoke", "happy-path"],
                "correctness": {
                    "expected_in_answer": ["pip install", "agentci"],
                    "not_in_answer": ["error"],
                    "llm_judge": [{"rule": "Clear installation steps", "threshold": 0.7}],
                },
                "path": {
                    "expected_tools": ["retriever_tool"],
                    "max_tool_calls": 5,
                    "match_mode": "subset",
                    "min_tool_recall": 1.0,
                },
                "cost": {
                    "max_llm_calls": 3,
                    "max_total_tokens": 2000,
                    "max_cost_usd": 0.01,
                },
            }
        ],
    )
    assert spec.agent == "rag-agent"
    q = spec.queries[0]
    assert q.correctness is not None
    assert q.path is not None
    assert q.cost is not None
    assert q.path.match_mode == MatchMode.SUBSET


def test_multiple_queries_valid():
    spec = AgentCISpec(
        agent="router",
        queries=[
            {"query": "Cancel my subscription"},
            {"query": "My app is crashing"},
            {"query": "What is the meaning of life"},
        ],
    )
    assert len(spec.queries) == 3


def test_defaults_section_accepted():
    spec = AgentCISpec(
        **minimal_spec(defaults={"cost": {"max_cost_multiplier": 2.0}})
    )
    assert spec.defaults == {"cost": {"max_cost_multiplier": 2.0}}


def test_judge_config_section_accepted():
    spec = AgentCISpec(
        **minimal_spec(judge_config={"model": "claude-sonnet-4-6", "temperature": 0})
    )
    assert spec.judge_config["temperature"] == 0


def test_tags_accepted_as_list():
    spec = AgentCISpec(
        agent="agent",
        queries=[{"query": "Hello", "tags": ["smoke", "edge-case"]}],
    )
    assert spec.queries[0].tags == ["smoke", "edge-case"]


def test_json_schema_field_accepts_arbitrary_dict():
    q = GoldenQuery(
        query="Return structured data",
        correctness=CorrectnessSpec(
            json_schema={"type": "object", "properties": {"name": {"type": "string"}}}
        ),
    )
    assert q.correctness.json_schema is not None


def test_few_shot_examples_accepted():
    rubric = JudgeRubric(
        rule="Response is helpful",
        threshold=0.7,
        few_shot_examples=[
            {"input": "q", "output": "answer", "score": 4}
        ],
    )
    assert len(rubric.few_shot_examples) == 1


def test_match_mode_default_is_subset():
    path = PathSpec()
    assert path.match_mode == MatchMode.SUBSET


def test_all_match_modes_accepted():
    for mode in ("strict", "unordered", "subset", "superset"):
        path = PathSpec(match_mode=mode)
        assert path.match_mode.value == mode


def test_nested_correctness_path_cost_on_single_query():
    q = GoldenQuery(
        query="Test query",
        correctness=CorrectnessSpec(expected_in_answer=["yes"]),
        path=PathSpec(max_tool_calls=3, forbidden_tools=["dangerous_tool"]),
        cost=CostSpec(max_cost_usd=0.05, max_llm_calls=2),
    )
    assert q.correctness.expected_in_answer == ["yes"]
    assert q.path.forbidden_tools == ["dangerous_tool"]
    assert q.cost.max_cost_usd == 0.05


# ── Invalid specs — must raise ValidationError ───────────────────────────────


def test_empty_query_string_fails():
    with pytest.raises(ValidationError, match="query must not be empty"):
        GoldenQuery(query="   ")


def test_missing_agent_fails():
    with pytest.raises(ValidationError):
        AgentCISpec(queries=[minimal_query()])


def test_missing_queries_fails():
    with pytest.raises(ValidationError):
        AgentCISpec(agent="agent")


def test_empty_queries_list_fails():
    with pytest.raises(ValidationError):
        AgentCISpec(agent="agent", queries=[])


def test_invalid_match_mode_fails():
    with pytest.raises(ValidationError):
        PathSpec(match_mode="fuzzy")


def test_threshold_negative_fails():
    with pytest.raises(ValidationError):
        JudgeRubric(rule="Test", threshold=-0.1)


def test_threshold_over_one_fails():
    with pytest.raises(ValidationError):
        JudgeRubric(rule="Test", threshold=1.1)


def test_max_tool_calls_negative_fails():
    with pytest.raises(ValidationError):
        PathSpec(max_tool_calls=-1)


def test_max_cost_multiplier_zero_fails():
    with pytest.raises(ValidationError):
        CostSpec(max_cost_multiplier=0)


def test_max_cost_multiplier_negative_fails():
    with pytest.raises(ValidationError):
        CostSpec(max_cost_multiplier=-1.0)


def test_min_tool_recall_out_of_range_fails():
    with pytest.raises(ValidationError):
        PathSpec(min_tool_recall=1.5)


def test_max_loops_zero_fails():
    with pytest.raises(ValidationError):
        PathSpec(max_loops=0)
