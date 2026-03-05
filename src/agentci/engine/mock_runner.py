# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Mock runner for AgentCI — generates synthetic traces from spec expectations.

Allows users to validate their agentci_spec.yaml structure without making
real API calls. Useful for:
- Spec validation: "Are my queries well-formed?"
- CI without API keys: run `agentci test --mock` in pipelines
- Quick iteration: see what a passing run looks like
"""

from __future__ import annotations

from ..models import Trace, Span, SpanKind, ToolCall, LLMCall


def mock_run(query: str, query_spec: dict) -> Trace:
    """Generate a synthetic trace that matches the spec's expectations.

    Parameters
    ----------
    query : str
        The test query string.
    query_spec : dict
        The query's spec dict (from GoldenQuery.model_dump()), containing
        path, correctness, and cost expectations.

    Returns
    -------
    Trace
        A synthetic trace with tool calls, LLM calls, and output that
        satisfy the spec's assertions.
    """
    trace = Trace(agent_name="mock-agent", test_name=query, metadata={"query": query})
    span = Span(kind=SpanKind.AGENT, name="mock-agent")

    # Build tool calls from expected_tools in path spec
    path_spec = query_spec.get("path") or {}
    expected_tools: list[str] = path_spec.get("expected_tools") or []
    for tool_name in expected_tools:
        span.tool_calls.append(
            ToolCall(
                tool_name=tool_name,
                arguments={"query": query},
                result="[mock result]",
            )
        )

    # Build expected answer from correctness spec (AND + OR keywords)
    correctness_spec = query_spec.get("correctness") or {}
    expected_keywords: list[str] = correctness_spec.get("expected_in_answer") or []
    any_expected_keywords: list[str] = correctness_spec.get("any_expected_in_answer") or []

    keywords_to_inject = list(expected_keywords)
    if any_expected_keywords:
        keywords_to_inject.append(any_expected_keywords[0])

    if keywords_to_inject:
        output_text = f"Based on our documentation: {', '.join(keywords_to_inject)}."
    else:
        output_text = "[Mock response — no expected keywords defined]"
    span.output_data = output_text
    trace.metadata["final_output"] = output_text

    # Set LLM calls within budget
    cost_spec = query_spec.get("cost") or {}
    max_llm_calls = cost_spec.get("max_llm_calls") or 10
    llm_call_count = min(max_llm_calls, 2)  # stay comfortably within budget
    for _ in range(llm_call_count):
        span.llm_calls.append(
            LLMCall(model="mock-model", tokens_in=100, tokens_out=50, cost_usd=0.0)
        )

    trace.spans.append(span)
    trace.compute_metrics()
    return trace


def run_mock_spec(spec, **_kwargs) -> dict[str, Trace]:
    """Run all queries in a spec using mock traces.

    Parameters
    ----------
    spec : AgentCISpec
        The loaded spec with queries.

    Returns
    -------
    dict[str, Trace]
        Mapping of query string to synthetic trace.
    """
    traces: dict[str, Trace] = {}
    for q in spec.queries:
        query_dict = q.model_dump() if hasattr(q, "model_dump") else {}
        traces[q.query] = mock_run(q.query, query_dict)
    return traces
