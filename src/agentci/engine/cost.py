"""
Cost Engine — Layer 3 (Soft Warning).

Evaluates efficiency metrics against the spec's budget thresholds.
All exceedances produce WARN status — they appear as GitHub annotations
but do NOT block the CI pipeline.

Maps to existing Trace fields:
    total_cost_usd     — for max_cost_usd / max_cost_multiplier
    total_tokens       — for max_total_tokens
    total_llm_calls    — for max_llm_calls
    total_duration_ms  — for max_latency_ms
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from agentci.engine.results import LayerResult, LayerStatus
from agentci.schema.spec_models import CostSpec

if TYPE_CHECKING:
    from agentci.models import Trace


def evaluate_cost(
    trace: "Trace",
    spec: CostSpec,
    baseline_trace: Optional["Trace"] = None,
) -> LayerResult:
    """Evaluate cost/efficiency assertions.

    Args:
        trace:          Current execution trace.
        spec:           CostSpec thresholds to check.
        baseline_trace: Optional golden baseline for multiplier comparison.

    Returns:
        LayerResult with PASS or WARN status (never FAIL).
    """
    warnings: list[str] = []
    pass_messages: list[str] = []
    details: dict[str, Any] = {
        "actual": {
            "cost_usd": round(trace.total_cost_usd, 6),
            "total_tokens": trace.total_tokens,
            "llm_calls": trace.total_llm_calls,
            "latency_ms": round(trace.total_duration_ms, 1),
        }
    }

    # ── 1. max_cost_multiplier (requires baseline) ───────────────────────────
    if spec.max_cost_multiplier is not None and baseline_trace is not None:
        baseline_cost = baseline_trace.total_cost_usd
        if baseline_cost > 0:
            multiplier = trace.total_cost_usd / baseline_cost
            details["cost_multiplier"] = round(multiplier, 3)
            if multiplier > spec.max_cost_multiplier:
                warnings.append(
                    f"Cost {multiplier:.2f}x baseline (max {spec.max_cost_multiplier}x): "
                    f"${trace.total_cost_usd:.6f} vs ${baseline_cost:.6f}"
                )
            else:
                pass_messages.append(f"Cost multiplier: {multiplier:.2f}x ≤ max {spec.max_cost_multiplier}x")
        else:
            details["cost_multiplier"] = None
            details["cost_multiplier_skipped"] = "baseline cost is 0"

    # ── 2. max_total_tokens ──────────────────────────────────────────────────
    if spec.max_total_tokens is not None:
        details["max_total_tokens"] = spec.max_total_tokens
        if trace.total_tokens > spec.max_total_tokens:
            warnings.append(
                f"Tokens: {trace.total_tokens} > max {spec.max_total_tokens}"
            )
        else:
            pass_messages.append(f"Tokens: {trace.total_tokens} ≤ max {spec.max_total_tokens}")

    # ── 3. max_llm_calls ────────────────────────────────────────────────────
    if spec.max_llm_calls is not None:
        details["max_llm_calls"] = spec.max_llm_calls
        if trace.total_llm_calls > spec.max_llm_calls:
            warnings.append(
                f"LLM calls: {trace.total_llm_calls} > max {spec.max_llm_calls}"
            )
        else:
            pass_messages.append(f"LLM calls: {trace.total_llm_calls} ≤ max {spec.max_llm_calls}")

    # ── 4. max_latency_ms ───────────────────────────────────────────────────
    if spec.max_latency_ms is not None:
        details["max_latency_ms"] = spec.max_latency_ms
        if trace.total_duration_ms > spec.max_latency_ms:
            warnings.append(
                f"Latency: {trace.total_duration_ms:.0f}ms > max {spec.max_latency_ms}ms"
            )
        else:
            pass_messages.append(f"Latency: {trace.total_duration_ms:.0f}ms ≤ max {spec.max_latency_ms}ms")

    # ── 5. max_cost_usd ─────────────────────────────────────────────────────
    if spec.max_cost_usd is not None:
        details["max_cost_usd"] = spec.max_cost_usd
        if trace.total_cost_usd > spec.max_cost_usd:
            warnings.append(
                f"Cost: ${trace.total_cost_usd:.6f} > max ${spec.max_cost_usd:.6f}"
            )
        else:
            pass_messages.append(f"Cost: ${trace.total_cost_usd:.6f} ≤ max ${spec.max_cost_usd:.6f}")

    return LayerResult(
        status=LayerStatus.WARN if warnings else LayerStatus.PASS,
        details=details,
        messages=warnings if warnings else (pass_messages or ["Cost within bounds"]),
    )
