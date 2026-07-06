"""
Tests for AgentCI v2 spec schema validation.

Every test either asserts a valid spec passes Pydantic validation or that
a specific invalid variant raises a ValidationError with a meaningful message.
"""

import pytest
from pydantic import ValidationError

from ciagent.schema.spec_models import (
    AgentCISpec,
    CorrectnessSpec,
    CostSpec,
    GoldenQuery,
    JudgeRubric,
    MatchMode,
    PathSpec,
    SpanAssert,
    SpanAssertionSpec,
    SpanAssertType,
    SpanKindSelector,
    SpanSelector,
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


def test_any_expected_in_answer_accepted_in_schema():
    q = GoldenQuery(
        query="Test query",
        correctness=CorrectnessSpec(any_expected_in_answer=["pip", "brew", "conda"]),
    )
    assert q.correctness.any_expected_in_answer == ["pip", "brew", "conda"]


def test_both_expected_and_any_expected_coexist():
    q = GoldenQuery(
        query="Test query",
        correctness=CorrectnessSpec(
            expected_in_answer=["3.10"],
            any_expected_in_answer=["pip", "brew"],
        ),
    )
    assert q.correctness.expected_in_answer == ["3.10"]
    assert q.correctness.any_expected_in_answer == ["pip", "brew"]


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


# ── max_loops default (1.1) ──────────────────────────────────────────────────


def test_max_loops_default_is_three():
    """PathSpec with no max_loops argument should default to 3."""
    path = PathSpec()
    assert path.max_loops == 3


def test_max_loops_per_query_override():
    """Per-query max_loops override (e.g. 10) should be respected."""
    path = PathSpec(max_loops=10)
    assert path.max_loops == 10


def test_max_loops_one_accepted():
    """min ge=1 constraint — value of 1 is valid."""
    path = PathSpec(max_loops=1)
    assert path.max_loops == 1


# ── SpanAssertionSpec (3.2) ──────────────────────────────────────────────────


class TestSpanAssertionSpec:
    def _make_span_assert(self, **kwargs) -> dict:
        base = {"type": "contains", "field": "attributes.tool.args.query", "value": "install"}
        base.update(kwargs)
        return base

    def _make_selector(self, **kwargs) -> dict:
        base = {"kind": "TOOL", "name": "retrieve_docs"}
        base.update(kwargs)
        return base

    def test_valid_span_assertion_spec(self):
        """A valid SpanAssertionSpec with one CONTAINS assert validates correctly."""
        spec = SpanAssertionSpec(
            selector=SpanSelector(kind=SpanKindSelector.TOOL, name="retrieve_docs"),
            asserts=[
                SpanAssert(
                    type=SpanAssertType.CONTAINS,
                    field="attributes.tool.args.query",
                    value="install",
                )
            ],
        )
        assert spec.selector.kind == SpanKindSelector.TOOL
        assert spec.selector.name == "retrieve_docs"
        assert len(spec.asserts) == 1

    def test_llm_judge_assert_type_accepted(self):
        """LLM_JUDGE assert type validates with a rule."""
        span_assert = SpanAssert(
            type=SpanAssertType.LLM_JUDGE,
            field="attributes.tool.args.query",
            rule="Query is relevant to the user's question",
            threshold=0.8,
        )
        assert span_assert.type == SpanAssertType.LLM_JUDGE
        assert span_assert.threshold == 0.8

    def test_all_assert_types_accepted(self):
        """All SpanAssertType values are valid."""
        for assert_type in SpanAssertType:
            sa = SpanAssert(type=assert_type, field="output_data", value="ok")
            assert sa.type == assert_type

    def test_all_span_kind_selectors_accepted(self):
        """All SpanKindSelector values are valid."""
        for kind in SpanKindSelector:
            sel = SpanSelector(kind=kind, name="test_span")
            assert sel.kind == kind

    def test_empty_asserts_list_fails(self):
        """SpanAssertionSpec.asserts must have at least one element."""
        with pytest.raises(ValidationError):
            SpanAssertionSpec(
                selector=SpanSelector(kind=SpanKindSelector.TOOL, name="tool"),
                asserts=[],
            )

    def test_missing_field_fails(self):
        """SpanAssert without field fails validation."""
        with pytest.raises(ValidationError):
            SpanAssert(type=SpanAssertType.CONTAINS, value="test")

    def test_invalid_type_fails(self):
        """Invalid SpanAssertType fails validation."""
        with pytest.raises(ValidationError):
            SpanAssert(type="fuzzy_match", field="output_data", value="test")

    def test_invalid_kind_fails(self):
        """Invalid SpanKindSelector fails validation."""
        with pytest.raises(ValidationError):
            SpanSelector(kind="INVALID_KIND", name="tool")

    def test_span_assertions_on_golden_query(self):
        """GoldenQuery accepts span_assertions list."""
        q = GoldenQuery(
            query="How do I install?",
            span_assertions=[
                SpanAssertionSpec(
                    selector=SpanSelector(kind=SpanKindSelector.TOOL, name="retrieve_docs"),
                    asserts=[
                        SpanAssert(
                            type=SpanAssertType.CONTAINS,
                            field="attributes.tool.args.query",
                            value="install",
                        )
                    ],
                )
            ],
        )
        assert len(q.span_assertions) == 1

    def test_span_assertions_default_empty_list(self):
        """GoldenQuery.span_assertions defaults to empty list."""
        q = GoldenQuery(query="What is 2+2?")
        assert q.span_assertions == []

    def test_threshold_out_of_range_fails(self):
        """SpanAssert.threshold must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            SpanAssert(
                type=SpanAssertType.LLM_JUDGE,
                field="output_data",
                rule="Test rule",
                threshold=1.5,
            )

    def test_multiple_assertions_on_same_span(self):
        """SpanAssertionSpec accepts multiple asserts (AND semantics)."""
        spec = SpanAssertionSpec(
            selector=SpanSelector(kind=SpanKindSelector.TOOL, name="retrieve_docs"),
            asserts=[
                SpanAssert(type=SpanAssertType.CONTAINS, field="output_data", value="AgentCI"),
                SpanAssert(type=SpanAssertType.NOT_CONTAINS, field="output_data", value="error"),
                SpanAssert(
                    type=SpanAssertType.LLM_JUDGE,
                    field="output_data",
                    rule="Result is relevant to the query",
                    threshold=0.7,
                ),
            ],
        )
        assert len(spec.asserts) == 3
