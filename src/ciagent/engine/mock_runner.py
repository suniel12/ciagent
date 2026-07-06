# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Mock runner for AgentCI — generates synthetic traces from spec expectations.

Allows users to validate their agentci_spec.yaml structure without making
real API calls. Useful for:
- Spec validation: "Are my queries well-formed?"
- CI without API keys: run `ciagent test --mock` in pipelines
- Quick iteration: see what a passing run looks like
"""

from __future__ import annotations

from ..models import Trace, Span, SpanKind, ToolCall, LLMCall


def mock_run(query: str, query_spec: dict, flaky_break: bool = False) -> Trace:
    """Generate a synthetic trace that matches the spec's expectations.

    Parameters
    ----------
    query : str
        The test query string.
    query_spec : dict
        The query's spec dict (from GoldenQuery.model_dump()), containing
        path, correctness, and cost expectations.
    flaky_break : bool
        When True, produce an answer that omits the expected keywords —
        simulates agent-variance for multi-run stability testing.

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

    if flaky_break:
        output_text = "[Mock flaky variant — simulated agent variance, keywords omitted]"
    elif keywords_to_inject:
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


def run_mock_spec(
    spec,
    run_index: int = 0,
    flaky: bool = False,
    flaky_style: str = "alternate",
    **_kwargs,
) -> dict[str, Trace]:
    """Run all queries in a spec using mock traces.

    Parameters
    ----------
    spec : AgentCISpec
        The loaded spec with queries.
    run_index : int
        Which run this is (0-based) in a multi-run stability session.
    flaky : bool
        Simulate deterministic pseudo-flakiness for stability testing
        (agent-variance). Judge-flake cannot be simulated here — identical
        answers cannot flip deterministic checks by construction; that path
        is covered by unit tests constructing results directly.
    flaky_style : str
        How simulated flakiness is distributed across runs:

        - ``"alternate"`` (default): even-indexed queries break on odd run
          indices. Half the suite flips at once, so the aggregate score
          visibly dips on flaky runs.
        - ``"spread"``: exactly one of the first three queries breaks per
          run (query ``run_index % 3``). The aggregate score stays constant
          across runs while individual verdicts flip — the "stable score,
          unstable system" shape the bundled demo exists to show.

    Returns
    -------
    dict[str, Trace]
        Mapping of query string to synthetic trace.
    """
    traces: dict[str, Trace] = {}
    for i, q in enumerate(spec.queries):
        query_dict = q.model_dump() if hasattr(q, "model_dump") else {}
        if flaky_style == "spread":
            flaky_break = flaky and i < 3 and run_index % 3 == i
        else:
            flaky_break = flaky and (i % 2 == 0) and (run_index % 2 == 1)
        traces[q.query] = mock_run(q.query, query_dict, flaky_break=flaky_break)
    return traces


def mock_conversation_runner(scenario):
    """Build a mock ConversationRunner for a scripted scenario.

    Each turn's synthetic trace satisfies the scenario's per_turn checks; the
    final scripted turn additionally satisfies the outcome checks — so a
    well-formed scenario passes end-to-end with zero API keys, mirroring what
    `ciagent test --mock` does for single-turn queries.
    """
    n_turns = min(len(scenario.turns or []), scenario.max_turns)

    def _merged_spec(is_last: bool) -> dict:
        blocks = [scenario.per_turn]
        if is_last:
            blocks.append(scenario.outcome)
        correctness: dict = {}
        path: dict = {}
        cost: dict = {}
        for b in blocks:
            if b is None:
                continue
            if b.correctness is not None:
                c = b.correctness
                if c.expected_in_answer:
                    correctness.setdefault("expected_in_answer", []).extend(c.expected_in_answer)
                if c.any_expected_in_answer:
                    correctness.setdefault("any_expected_in_answer", []).extend(c.any_expected_in_answer)
            if b.path is not None and b.path.expected_tools:
                seen = path.setdefault("expected_tools", [])
                seen.extend(t for t in b.path.expected_tools if t not in seen)
            if b.cost is not None and b.cost.max_llm_calls:
                cost["max_llm_calls"] = b.cost.max_llm_calls
        return {"correctness": correctness or None, "path": path or None, "cost": cost or None}

    def run(messages: list[dict]) -> Trace:
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        is_last = user_turns >= n_turns
        return mock_run(messages[-1]["content"], _merged_spec(is_last))

    return run
