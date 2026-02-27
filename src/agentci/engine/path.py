"""
Path Engine — Layer 2 (Soft Warning).

Evaluates the agent's tool-use trajectory. Most exceedances produce WARN
status (visible in annotations, does not block CI). The single exception is
forbidden_tools: if the agent used a tool it was explicitly prohibited from
using, the result escalates to FAIL.

Depends on:
    agentci.models.Trace  (tool_call_sequence, get_handoffs)
    agentci.engine.metrics (pure metric functions)
    agentci.schema.spec_models.PathSpec / MatchMode
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from agentci.engine.metrics import (
    compute_sequence_lcs,
    compute_tool_precision,
    compute_tool_recall,
    detect_loops,
)
from agentci.engine.results import LayerResult, LayerStatus
from agentci.schema.spec_models import MatchMode, PathSpec

if TYPE_CHECKING:
    from agentci.models import Trace


def evaluate_path(
    trace: "Trace",
    spec: PathSpec,
    baseline_trace: Optional["Trace"] = None,
) -> LayerResult:
    """Evaluate trajectory/path assertions.

    Args:
        trace:          Current execution trace.
        spec:           PathSpec assertions to evaluate.
        baseline_trace: Optional golden baseline for sequence comparison.

    Returns:
        LayerResult with PASS, WARN, or FAIL status.
        FAIL only when forbidden_tools are violated.
    """
    warnings: list[str] = []
    pass_messages: list[str] = []
    details: dict[str, Any] = {}
    hard_fail = False
    used_tools = trace.tool_call_sequence  # list[str], ordered

    # ── 1. max_tool_calls ────────────────────────────────────────────────────
    if spec.max_tool_calls is not None:
        actual = len(used_tools)
        details["tool_calls"] = {"actual": actual, "max": spec.max_tool_calls}
        if actual > spec.max_tool_calls:
            warnings.append(
                f"Tool calls: {actual} > max {spec.max_tool_calls}"
            )
        else:
            pass_messages.append(f"Tool calls: {actual} ≤ max {spec.max_tool_calls}")

    # ── 2. forbidden_tools (safety boundary → hard FAIL) ────────────────────
    if spec.forbidden_tools:
        used_set = set(used_tools)
        violations = used_set & set(spec.forbidden_tools)
        details["forbidden_tools"] = {"violations": sorted(violations)}
        if violations:
            warnings.append(f"Forbidden tools used: {sorted(violations)}")
            hard_fail = True
        else:
            pass_messages.append("No forbidden tools used")

    # ── 3. tool recall ───────────────────────────────────────────────────────
    if spec.expected_tools:
        expected_set = set(spec.expected_tools)
        used_set = set(used_tools)
        recall = compute_tool_recall(expected_set, used_set)
        details["tool_recall"] = round(recall, 3)
        if spec.min_tool_recall is not None and recall < spec.min_tool_recall:
            warnings.append(
                f"Tool recall: {recall:.3f} < min {spec.min_tool_recall}"
            )
        else:
            expected_str = ", ".join(spec.expected_tools)
            pass_messages.append(f"Tool recall: {recall:.3f} (expected: [{expected_str}])")

    # ── 4. tool precision ────────────────────────────────────────────────────
    if spec.expected_tools and used_tools:
        expected_set = set(spec.expected_tools)
        used_set = set(used_tools)
        precision = compute_tool_precision(expected_set, used_set)
        details["tool_precision"] = round(precision, 3)
        if spec.min_tool_precision is not None and precision < spec.min_tool_precision:
            warnings.append(
                f"Tool precision: {precision:.3f} < min {spec.min_tool_precision}"
            )
        else:
            pass_messages.append(f"Tool precision: {precision:.3f}")

    # ── 5. sequence similarity (LCS vs baseline) ─────────────────────────────
    if spec.min_sequence_similarity is not None and baseline_trace is not None:
        baseline_tools = baseline_trace.tool_call_sequence
        similarity = compute_sequence_lcs(used_tools, baseline_tools)
        details["sequence_similarity"] = round(similarity, 3)
        if similarity < spec.min_sequence_similarity:
            warnings.append(
                f"Sequence similarity: {similarity:.3f} < min {spec.min_sequence_similarity}"
            )
        else:
            pass_messages.append(f"Sequence similarity: {similarity:.3f} ≥ min {spec.min_sequence_similarity}")

    # ── 6. loop detection ────────────────────────────────────────────────────
    # max_loops always has a value (default=3); always run loop detection
    loops = detect_loops(used_tools)
    details["loops_detected"] = loops
    if loops > spec.max_loops:
        warnings.append(f"Loops: {loops} > max {spec.max_loops}")
    elif loops == 0:
        pass_messages.append("No loops detected")
    else:
        pass_messages.append(f"Loops: {loops} ≤ max {spec.max_loops}")

    # ── 7. match mode (vs baseline) ──────────────────────────────────────────
    if baseline_trace is not None:
        baseline_tools = baseline_trace.tool_call_sequence
        match_result = _evaluate_match_mode(used_tools, baseline_tools, spec.match_mode)
        details["match_mode"] = match_result
        if not match_result["matched"]:
            warnings.append(
                f"Match mode '{spec.match_mode.value}' failed: {match_result['reason']}"
            )

    # ── 8. handoff assertions ────────────────────────────────────────────────
    if spec.expected_handoff or spec.expected_handoffs_available or spec.max_handoff_count is not None:
        handoffs = trace.get_handoffs()
        actual_targets = [h.to_agent for h in handoffs if h.to_agent]
        details["handoffs"] = {"actual_targets": actual_targets, "count": len(handoffs)}

        if spec.expected_handoff:
            details["handoffs"]["expected"] = spec.expected_handoff
            if spec.expected_handoff not in actual_targets:
                warnings.append(
                    f"Expected handoff to '{spec.expected_handoff}', got {actual_targets}"
                )
            else:
                pass_messages.append(f"Handoff to '{spec.expected_handoff}' verified")

        if spec.expected_handoffs_available:
            available = {t for targets in trace.available_handoffs for t in targets}
            details["handoffs"]["expected_available"] = spec.expected_handoffs_available
            details["handoffs"]["available"] = sorted(available)
            missing_available = set(spec.expected_handoffs_available) - available
            if missing_available:
                warnings.append(
                    f"Expected handoff targets not available: {sorted(missing_available)}"
                )

        if spec.max_handoff_count is not None:
            count = len(handoffs)
            details["handoffs"]["max_count"] = spec.max_handoff_count
            if count > spec.max_handoff_count:
                warnings.append(
                    f"Handoff count: {count} > max {spec.max_handoff_count}"
                )
            else:
                pass_messages.append(f"Handoff count: {count} ≤ max {spec.max_handoff_count}")

    # ── Determine status ─────────────────────────────────────────────────────
    if hard_fail:
        status = LayerStatus.FAIL
    elif warnings:
        status = LayerStatus.WARN
    else:
        status = LayerStatus.PASS

    return LayerResult(
        status=status,
        details=details,
        messages=warnings if warnings else (pass_messages or ["Path OK"]),
    )


# ── Match Mode Evaluation ──────────────────────────────────────────────────────


def _evaluate_match_mode(
    used_tools: list[str],
    reference_tools: list[str],
    mode: MatchMode,
) -> dict[str, Any]:
    """Compare used tool sequence against reference using the specified match mode.

    Returns:
        {"matched": bool, "reason": str}
    """
    used_set = set(used_tools)
    ref_set = set(reference_tools)

    if mode == MatchMode.STRICT:
        matched = used_tools == reference_tools
        reason = "" if matched else f"Expected {reference_tools}, got {used_tools}"

    elif mode == MatchMode.UNORDERED:
        matched = used_set == ref_set
        reason = "" if matched else f"Expected set {sorted(ref_set)}, got {sorted(used_set)}"

    elif mode == MatchMode.SUBSET:
        # Reference tools must appear in used_tools (extras OK)
        missing = ref_set - used_set
        matched = not missing
        reason = "" if matched else f"Reference tools not found in used: {sorted(missing)}"

    elif mode == MatchMode.SUPERSET:
        # All used tools must be in the reference set
        unexpected = used_set - ref_set
        matched = not unexpected
        reason = "" if matched else f"Used tools not in reference: {sorted(unexpected)}"

    else:
        matched = False
        reason = f"Unknown match mode: {mode}"

    return {"matched": matched, "reason": reason}
