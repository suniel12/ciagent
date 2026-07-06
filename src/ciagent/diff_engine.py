# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Golden Trace Diff Engine.

Compares a "current" trace against a "golden" (known-good) trace
and produces categorized diffs. The key insight: don't just say
"traces differ." Say exactly WHAT differs and WHY it matters.

Phase 1: Exact matching on tool names and arguments.
Phase 2: Add semantic matching via LLM-as-Judge for fuzzy comparison.
"""

from .models import Trace, DiffResult, DiffType
import os
import json

class DiffReport:
    """Wrapper around diff results for easy assertions.

    Attributes:
        diffs: List of DiffResult objects.
        has_regression: True if any diff has error severity.
        summary: Human-readable summary string.

    Example:
        >>> from ciagent import diff, load_baseline
        >>> report = diff(golden_trace, current_trace)
        >>> assert not report.has_regression, report.summary
    """
    def __init__(self, diffs: list[DiffResult]):
        self.diffs = diffs
        # Any error-severity diff is considered a regression
        self.has_regression = any(d.severity == "error" for d in diffs)
        self.summary = ", ".join([d.message for d in diffs]) if diffs else "No regressions"

def diff(golden: Trace, current: Trace) -> DiffReport:
    """Compare a golden trace against a current trace and return a DiffReport.

    Args:
        golden: The known-good baseline trace.
        current: The trace from the current test run.

    Returns:
        A DiffReport with has_regression and summary attributes.

    Example:
        >>> from ciagent import diff
        >>> report = diff(golden_trace, current_trace)
        >>> assert not report.has_regression, report.summary
    """
    from .diff_engine import diff_traces
    return DiffReport(diff_traces(current, golden))

def load_baseline(name: str) -> dict[str, Trace]:
    """Load a saved baseline collection of traces from golden/{name}.json.

    Args:
        name: The baseline name (without path prefix or extension).

    Returns:
        A dict mapping test_name -> Trace object.

    Raises:
        FileNotFoundError: If golden/{name}.json doesn't exist.

    Example:
        >>> from ciagent import load_baseline
        >>> baselines = load_baseline("v1-baseline")
        >>> golden_trace = baselines["test_billing_routing"]
    """
    path = f"golden/{name}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Baseline '{path}' not found.\n"
            f"  Fix: Run 'ciagent record <test_name> -o {path}' to create a baseline."
        )
        
    with open(path, "r") as f:
        data = json.load(f)
        
    resolved = {}
    for key, trace_data in data.items():
        resolved[key] = Trace(**trace_data)
        
    return resolved


def diff_traces(current: Trace, golden: Trace) -> list[DiffResult]:
    """Compare current trace against golden trace and return categorized diffs.

    Detects 11 categories of regression: TOOLS_CHANGED, ARGS_CHANGED,
    SEQUENCE_CHANGED, COST_SPIKE, STEPS_CHANGED, STOP_REASON_CHANGED,
    ROUTING_CHANGED, GUARDRAILS_CHANGED, AVAILABLE_HANDOFFS_CHANGED, etc.

    Args:
        current: The trace from the current test run.
        golden: The known-good baseline trace.

    Returns:
        A list of DiffResult objects, each with diff_type, severity, and message.

    Example:
        >>> from ciagent.diff_engine import diff_traces
        >>> diffs = diff_traces(current_trace, golden_trace)
        >>> errors = [d for d in diffs if d.severity == "error"]
        >>> assert not errors, [d.message for d in errors]
    """
    diffs: list[DiffResult] = []
    
    # 0. HANDOFF / ROUTING DIFF
    current_handoffs = [h.to_agent for h in current.get_handoffs()]
    golden_handoffs = [h.to_agent for h in golden.get_handoffs()]
    
    if current_handoffs != golden_handoffs:
        diffs.append(DiffResult(
            diff_type=DiffType.ROUTING_CHANGED,
            severity="error",
            message=f"Routing changed: {golden_handoffs} → {current_handoffs}",
            details={
                "golden_routing": golden_handoffs,
                "current_routing": current_handoffs,
            }
        ))
    
    # 0a. GUARDRAIL DIFF
    current_guardrails = current.guardrails_triggered
    golden_guardrails = golden.guardrails_triggered

    if set(current_guardrails) != set(golden_guardrails):
        diffs.append(DiffResult(
            diff_type=DiffType.GUARDRAILS_CHANGED,
            severity="error",
            message=f"Guardrails changed: {golden_guardrails} → {current_guardrails}",
            details={
                "golden_guardrails": golden_guardrails,
                "current_guardrails": current_guardrails,
            }
        ))

    # 0b. AVAILABLE HANDOFFS DIFF
    current_available = current.available_handoffs
    golden_available = golden.available_handoffs

    if current_available != golden_available:
        diffs.append(DiffResult(
            diff_type=DiffType.AVAILABLE_HANDOFFS_CHANGED,
            severity="warning",
            message=f"Available handoff options changed",
            details={
                "golden_available": golden_available,
                "current_available": current_available,
            }
        ))

    # 1. TOOL SEQUENCE DIFF
    current_tools = current.tool_call_sequence
    golden_tools = golden.tool_call_sequence
    
    if current_tools != golden_tools:
        if set(current_tools) != set(golden_tools):
            # Different tools called entirely
            added = set(current_tools) - set(golden_tools)
            removed = set(golden_tools) - set(current_tools)
            diffs.append(DiffResult(
                diff_type=DiffType.TOOLS_CHANGED,
                severity="error",
                message=f"Tool set changed: +{added or 'none'} -{removed or 'none'}",
                details={
                    "golden_tools": golden_tools,
                    "current_tools": current_tools,
                    "added": list(added),
                    "removed": list(removed),
                }
            ))
        else:
            # Same tools, different order
            diffs.append(DiffResult(
                diff_type=DiffType.SEQUENCE_CHANGED,
                severity="warning",
                message=f"Tool call order changed",
                details={
                    "golden_sequence": golden_tools,
                    "current_sequence": current_tools,
                }
            ))
    
    # 2. ARGUMENT DIFF (for tools that appear in both)
    current_calls = current.tool_call_details
    golden_calls = golden.tool_call_details
    
    paired_calls = _pair_tool_calls(current_calls, golden_calls)
    for current_tc, golden_tc in paired_calls:
        arg_diffs = _diff_arguments(current_tc.arguments, golden_tc.arguments)
        if arg_diffs:
            diffs.append(DiffResult(
                diff_type=DiffType.ARGS_CHANGED,
                severity="warning",
                message=f"Arguments changed for '{current_tc.tool_name}'",
                details={
                    "tool": current_tc.tool_name,
                    "changes": arg_diffs,
                }
            ))
    
    # 3. COST DIFF
    if golden.total_cost_usd > 0:
        cost_ratio = current.total_cost_usd / golden.total_cost_usd
        if cost_ratio > 1.5:  # 50% cost increase threshold (configurable)
            diffs.append(DiffResult(
                diff_type=DiffType.COST_SPIKE,
                severity="error" if cost_ratio > 2.0 else "warning",
                message=f"Cost increased {cost_ratio:.1f}x: "
                        f"${golden.total_cost_usd:.4f} → ${current.total_cost_usd:.4f}",
                details={
                    "golden_cost": golden.total_cost_usd,
                    "current_cost": current.total_cost_usd,
                    "ratio": cost_ratio,
                }
            ))
    
    # 4. STEPS DIFF (number of LLM calls)
    if golden.total_llm_calls > 0:
        step_ratio = current.total_llm_calls / golden.total_llm_calls
        if step_ratio > 1.5:
            diffs.append(DiffResult(
                diff_type=DiffType.STEPS_CHANGED,
                severity="warning",
                message=f"LLM calls increased: {golden.total_llm_calls} → {current.total_llm_calls}",
                details={
                    "golden_steps": golden.total_llm_calls,
                    "current_steps": current.total_llm_calls,
                }
            ))
            
    # 5. STOP REASON DIFF (Detecting silent failures like hitting max_tools/max_tokens)
    if golden.spans and current.spans:
        golden_reason = golden.spans[-1].stop_reason
        current_reason = current.spans[-1].stop_reason
        
        # We only flag it if the golden run had a known stop reason, and the current one diverges.
        if golden_reason and current_reason and golden_reason != current_reason:
            diffs.append(DiffResult(
                diff_type=DiffType.STOP_REASON_CHANGED,
                severity="error",  # A changed stop mechanism is almost always a bug/regression
                message=f"Agent stop reason changed: '{golden_reason}' → '{current_reason}' (Possible silent failure)",
                details={
                    "golden_reason": golden_reason,
                    "current_reason": current_reason,
                }
            ))
    
    return diffs


def _pair_tool_calls(current_calls, golden_calls):
    """
    Match tool calls between traces by name for comparison.
    Uses positional matching within each tool name group.
    """
    from collections import defaultdict
    
    current_by_name = defaultdict(list)
    golden_by_name = defaultdict(list)
    
    for tc in current_calls:
        current_by_name[tc.tool_name].append(tc)
    for tc in golden_calls:
        golden_by_name[tc.tool_name].append(tc)
    
    pairs = []
    for name in set(current_by_name) & set(golden_by_name):
        for c, g in zip(current_by_name[name], golden_by_name[name]):
            pairs.append((c, g))
    
    return pairs


def _diff_arguments(current_args: dict, golden_args: dict) -> list[dict]:
    """Produce a list of specific argument differences."""
    changes = []
    
    all_keys = set(current_args) | set(golden_args)
    for key in sorted(all_keys):
        current_val = current_args.get(key)
        golden_val = golden_args.get(key)
        
        if current_val != golden_val:
            changes.append({
                "field": key,
                "golden": golden_val,
                "current": current_val,
            })
    
    return changes
