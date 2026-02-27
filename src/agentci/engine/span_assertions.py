"""
Span Assertions Engine — sub-layer of Correctness (Hard Fail).

Evaluates span-level assertions against trace data. Lets tests assert on the
data *flowing between* nodes/tools (tool args, node inputs) to catch bugs
like "grader evaluated wrong text" that are invisible to answer-level checks.

Design:
  - Follows OTel attribute conventions: "tool.args.query", "llm.input", etc.
  - Hard-FAIL on any assertion failure (same severity as Correctness layer).
  - Multiple spans matched by a selector → ALL must pass (AND semantics).
  - No span matched → FAIL with a clear "No span matched" message.
  - LLM_JUDGE asserts call run_judge() (mocked in tests, real in integration).

Public API:
    evaluate_span_assertions(spec, trace, judge_config, spec_dir) → LayerResult
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from agentci.engine.results import LayerResult, LayerStatus
from agentci.models import SpanKind, Span
from agentci.schema.spec_models import (
    SpanAssert,
    SpanAssertionSpec,
    SpanAssertType,
    SpanKindSelector,
)

if TYPE_CHECKING:
    from agentci.models import Trace

# Map SpanKindSelector → SpanKind enum values
_KIND_MAP: dict[SpanKindSelector, SpanKind] = {
    SpanKindSelector.TOOL: SpanKind.TOOL_CALL,
    SpanKindSelector.NODE: SpanKind.AGENT,       # LangGraph nodes map to AGENT spans
    SpanKindSelector.HANDOFF: SpanKind.HANDOFF,
    SpanKindSelector.GUARDRAIL: SpanKind.GUARDRAIL,
    SpanKindSelector.LLM: SpanKind.LLM_CALL,
}


def evaluate_span_assertions(
    spec: list[SpanAssertionSpec],
    trace: "Trace",
    judge_config: Optional[dict[str, Any]] = None,
    spec_dir: Optional[str] = None,
) -> LayerResult:
    """Evaluate span-level assertions against trace data.

    Args:
        spec:         List of SpanAssertionSpec from the golden query.
        trace:        The agent's execution trace.
        judge_config: Global LLM judge config (model, temperature).
        spec_dir:     Spec file directory for context_file resolution.

    Returns:
        LayerResult with PASS or FAIL. Hard-fails on any assertion failure.
    """
    if not spec:
        return LayerResult(
            status=LayerStatus.PASS,
            details={},
            messages=["No span assertions defined"],
        )

    failures: list[str] = []
    details: dict[str, Any] = {"span_assertions": []}

    for assertion_spec in spec:
        selector = assertion_spec.selector
        matched_spans = _select_spans(trace, selector)

        spec_details: dict[str, Any] = {
            "selector": {"kind": selector.kind.value, "name": selector.name},
            "matched_spans": len(matched_spans),
            "results": [],
        }

        if not matched_spans:
            msg = f"No span matched selector {selector.kind.value}:{selector.name}"
            failures.append(msg)
            spec_details["error"] = msg
            details["span_assertions"].append(spec_details)
            continue

        for span in matched_spans:
            for span_assert in assertion_spec.asserts:
                result = _evaluate_single_assert(
                    span=span,
                    span_assert=span_assert,
                    judge_config=judge_config,
                    spec_dir=spec_dir,
                )
                spec_details["results"].append(result)
                if not result["passed"]:
                    failures.append(
                        f"Span {selector.kind.value}:{selector.name} "
                        f"field '{span_assert.field}' "
                        f"{span_assert.type.value} failed: {result.get('message', '')}"
                    )

        details["span_assertions"].append(spec_details)

    if failures:
        return LayerResult(status=LayerStatus.FAIL, details=details, messages=failures)
    return LayerResult(
        status=LayerStatus.PASS,
        details=details,
        messages=["All span assertions passed"],
    )


# ── Span Selection ─────────────────────────────────────────────────────────────


def _select_spans(trace: "Trace", selector) -> list["Span"]:
    """Return all spans matching the selector kind and name."""
    target_kind = _KIND_MAP.get(selector.kind)
    if target_kind is None:
        return []
        
    spans = [
        s for s in trace.spans
        if s.kind == target_kind and s.name == selector.name
    ]
    
    # Virtual fallback for Phase 1 ToolCalls (nested inside AGENT spans)
    if target_kind == SpanKind.TOOL_CALL:
        for agent_s in trace.spans:
            for tc in agent_s.tool_calls:
                if tc.tool_name == selector.name:
                    attr = {"tool.args": tc.arguments}
                    if isinstance(tc.arguments, dict):
                        for k, v in tc.arguments.items():
                            attr[f"tool.args.{k}"] = v
                            
                    spans.append(Span(
                        kind=SpanKind.TOOL_CALL,
                        name=tc.tool_name,
                        attributes=attr
                    ))
                    
    return spans


# ── Field Extraction ───────────────────────────────────────────────────────────


def _resolve_field(span: "Span", field_path: str) -> Any:
    """Resolve a dotted field path against a span.

    Supports:
        "output_data"                       → span.output_data
        "input_data"                        → span.input_data
        "attributes.tool.args.query"        → span.attributes["tool.args"]["query"]
                                              (first tries full key, then nested)
        "metadata.handoffs"                 → span.metadata["handoffs"]

    Returns None if the path cannot be resolved.
    """
    parts = field_path.split(".", 1)
    top = parts[0]

    # Direct span attributes
    if top in ("output_data", "input_data", "name", "stop_reason"):
        return getattr(span, top, None)

    if top == "attributes":
        return _resolve_nested(span.attributes, parts[1] if len(parts) > 1 else "")

    if top == "metadata":
        return _resolve_nested(span.metadata, parts[1] if len(parts) > 1 else "")

    # Fallback: try as a direct span field
    return getattr(span, top, None)


def _resolve_nested(container: dict, dotted_path: str) -> Any:
    """Resolve a dotted key path in a nested dict.

    Strategy:
        1. Try the full remaining path as a single dict key (e.g. "tool.args.query")
        2. Walk part-by-part into nested dicts.
    """
    if not dotted_path:
        return container

    # Try full key first (handles "tool.args" as a single key)
    if dotted_path in container:
        return container[dotted_path]

    parts = dotted_path.split(".", 1)
    key = parts[0]

    if key not in container:
        return None

    value = container[key]
    if len(parts) == 1:
        return value

    if isinstance(value, dict):
        return _resolve_nested(value, parts[1])

    return None


# ── Single Assert Evaluation ───────────────────────────────────────────────────


def _evaluate_single_assert(
    span: "Span",
    span_assert: SpanAssert,
    judge_config: Optional[dict[str, Any]],
    spec_dir: Optional[str],
) -> dict[str, Any]:
    """Evaluate one SpanAssert against a span. Returns a result dict."""
    field_value = _resolve_field(span, span_assert.field)

    if field_value is None:
        return {
            "passed": False,
            "field": span_assert.field,
            "type": span_assert.type.value,
            "message": f"Field '{span_assert.field}' not found in span",
        }

    str_value = str(field_value) if not isinstance(field_value, str) else field_value

    if span_assert.type == SpanAssertType.CONTAINS:
        passed = span_assert.value is not None and span_assert.value in str_value
        return {
            "passed": passed,
            "field": span_assert.field,
            "type": span_assert.type.value,
            "actual": str_value[:200],
            "expected_contains": span_assert.value,
            "message": f"'{span_assert.value}' not found in field value" if not passed else "",
        }

    if span_assert.type == SpanAssertType.NOT_CONTAINS:
        passed = span_assert.value is None or span_assert.value not in str_value
        return {
            "passed": passed,
            "field": span_assert.field,
            "type": span_assert.type.value,
            "actual": str_value[:200],
            "message": f"Forbidden value '{span_assert.value}' found in field" if not passed else "",
        }

    if span_assert.type == SpanAssertType.EQUALS:
        passed = str_value == span_assert.value
        return {
            "passed": passed,
            "field": span_assert.field,
            "type": span_assert.type.value,
            "actual": str_value[:200],
            "expected": span_assert.value,
            "message": f"Expected '{span_assert.value}', got '{str_value[:100]}'" if not passed else "",
        }

    if span_assert.type == SpanAssertType.REGEX:
        passed = span_assert.value is not None and bool(re.search(span_assert.value, str_value))
        return {
            "passed": passed,
            "field": span_assert.field,
            "type": span_assert.type.value,
            "pattern": span_assert.value,
            "message": f"Regex '{span_assert.value}' did not match" if not passed else "",
        }

    if span_assert.type == SpanAssertType.LLM_JUDGE:
        return _evaluate_llm_judge_assert(
            str_value, span_assert, judge_config, spec_dir
        )

    return {
        "passed": False,
        "field": span_assert.field,
        "type": span_assert.type.value,
        "message": f"Unknown assertion type: {span_assert.type}",
    }


def _evaluate_llm_judge_assert(
    field_value: str,
    span_assert: SpanAssert,
    judge_config: Optional[dict[str, Any]],
    spec_dir: Optional[str],
) -> dict[str, Any]:
    """Run an LLM judge against a span field value."""
    from agentci.engine.judge import run_judge
    from agentci.schema.spec_models import JudgeRubric

    if not span_assert.rule:
        return {
            "passed": False,
            "field": span_assert.field,
            "type": "llm_judge",
            "message": "llm_judge assertion requires a 'rule' field",
        }

    rubric = JudgeRubric(rule=span_assert.rule, threshold=span_assert.threshold)
    try:
        verdict = run_judge(
            answer=field_value,
            rubric=rubric,
            config=judge_config,
            spec_dir=spec_dir,
        )
        return {
            "passed": verdict["passed"],
            "field": span_assert.field,
            "type": "llm_judge",
            "score": verdict.get("score"),
            "rationale": verdict.get("rationale"),
            "message": f"LLM judge failed: {span_assert.rule[:80]}" if not verdict["passed"] else "",
        }
    except Exception as e:
        return {
            "passed": False,
            "field": span_assert.field,
            "type": "llm_judge",
            "message": f"Judge call failed: {e}",
        }
