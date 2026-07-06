# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
AgentCI v2 Diff Engine — Three-Tier Baseline Comparison.

Compares two versioned baseline trace files (produced by `ciagent save`)
and produces a structured DiffReport broken into three evaluation layers:

    Layer 1: Correctness  — did the output quality change?
    Layer 2: Path         — did tool usage / trajectory change?
    Layer 3: Cost         — did token/latency/cost change?

This replaces the flat v1 DiffReport (in diff_engine.py) for `ciagent diff`
comparisons. The v1 `diff_traces()` function is preserved for backward compat.

Usage:
    from ciagent.engine.diff import diff_baselines, DiffReport

    report = diff_baselines(baseline_data, compare_data, spec=spec)
    print(report.summary_console())
    exit(1 if report.has_regression else 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ciagent.conversation import ConversationEnvelope
    from ciagent.models import Trace
    from ciagent.schema.spec_models import AgentCISpec


# ── Public API ─────────────────────────────────────────────────────────────────


@dataclass
class MetricDelta:
    """Delta for a single numeric metric between two baseline versions."""
    label: str
    before: Any
    after: Any

    @property
    def pct_change(self) -> Optional[float]:
        """Percentage change from before → after, or None if before is 0/None."""
        if self.before is None or self.after is None:
            return None
        if isinstance(self.before, (int, float)) and self.before != 0:
            return ((self.after - self.before) / abs(self.before)) * 100.0
        return None

    @property
    def direction_arrow(self) -> str:
        """▲ for increase, ▼ for decrease, — for no change."""
        pct = self.pct_change
        if pct is None or abs(pct) < 0.1:
            return "—"
        return "▲" if pct > 0 else "▼"

    @property
    def pct_str(self) -> str:
        """Human-readable percentage string, e.g. '▼ 98.8%'."""
        pct = self.pct_change
        if pct is None:
            return ""
        return f"{self.direction_arrow} {abs(pct):.1f}%"


@dataclass
class DiffReport:
    """
    Three-tier diff between two versioned baseline traces.

    Attributes:
        agent:               Agent identifier.
        from_version:        Version tag of the baseline (e.g. 'v1-broken').
        to_version:          Version tag of the compare trace (e.g. 'v2-fixed').
        query:               The query text both traces correspond to.
        correctness_delta:   Layer 1 — output quality change.
        path_deltas:         Layer 2 — trajectory / tool usage deltas.
        cost_deltas:         Layer 3 — efficiency / cost deltas.
        legacy_diffs:        v1-compatible DiffResult list (auto-generated).

    Example:
        >>> report = diff_baselines(baseline_data, compare_data)
        >>> print(report.summary_console())
        >>> assert not report.has_regression
    """
    agent: str
    from_version: str
    to_version: str
    query: str = ""
    correctness_delta: dict[str, Any] = field(default_factory=dict)
    path_deltas: list[MetricDelta] = field(default_factory=list)
    cost_deltas: list[MetricDelta] = field(default_factory=list)
    legacy_diffs: list[Any] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        """True if correctness went from pass to fail, or a forbidden tool was added."""
        cd = self.correctness_delta
        if cd.get("before") == "pass" and cd.get("after") == "fail":
            return True
        # Path regressions: tool count increased significantly
        for delta in self.path_deltas:
            if delta.label == "tool_calls":
                if (
                    isinstance(delta.before, (int, float))
                    and isinstance(delta.after, (int, float))
                    and delta.after > delta.before * 2
                    and delta.after > delta.before + 3
                ):
                    return True
        return False

    @property
    def has_improvement(self) -> bool:
        """True if any metric improved without any regressions."""
        if self.has_regression:
            return False
        for delta in self.path_deltas + self.cost_deltas:
            pct = delta.pct_change
            if pct is not None and pct < -10:
                return True
        return False

    def summary_console(self) -> str:
        """Render the three-tier diff in a rich console box."""
        lines = [
            f"╔{'═' * 62}╗",
            f"║  AgentCI Diff: {self.agent} ({self.from_version} → {self.to_version}){' ' * max(0, 62 - 16 - len(self.agent) - len(self.from_version) - len(self.to_version))}║",
            f"╠{'═' * 62}╣",
        ]

        # Correctness
        cd = self.correctness_delta
        if cd:
            before_s = cd.get("before", "?")
            after_s = cd.get("after", "?")
            changed = cd.get("changed", False)
            icon = "✅" if not changed else ("❌" if after_s == "fail" else "⚠️ ")
            status = f"{icon}  Correctness: {before_s.upper()} → {after_s.upper()}"
            lines.append(f"║  {status:<60}║")
        else:
            lines.append(f"║  — Correctness: no spec provided{' ' * 30}║")

        lines.append(f"║{' ' * 62}║")

        # Path deltas
        if self.path_deltas:
            lines.append(f"║  📈 Path:{' ' * 53}║")
            for delta in self.path_deltas:
                val_str = _format_value(delta.before, delta.after, delta.label)
                row = f"     {delta.label:<24} {val_str}"
                lines.append(f"║  {row:<60}║")
            lines.append(f"║{' ' * 62}║")

        # Cost deltas
        if self.cost_deltas:
            lines.append(f"║  💰 Cost:{' ' * 53}║")
            for delta in self.cost_deltas:
                val_str = _format_value(delta.before, delta.after, delta.label)
                row = f"     {delta.label:<24} {val_str}"
                lines.append(f"║  {row:<60}║")

        lines.append(f"╚{'═' * 62}╝")
        return "\n".join(lines)

    def summary_json(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        return {
            "agent": self.agent,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "query": self.query,
            "has_regression": self.has_regression,
            "has_improvement": self.has_improvement,
            "correctness": self.correctness_delta,
            "path": [
                {
                    "metric": d.label,
                    "before": d.before,
                    "after": d.after,
                    "pct_change": d.pct_change,
                }
                for d in self.path_deltas
            ],
            "cost": [
                {
                    "metric": d.label,
                    "before": d.before,
                    "after": d.after,
                    "pct_change": d.pct_change,
                }
                for d in self.cost_deltas
            ],
        }


def diff_baselines(
    baseline_data: dict[str, Any],
    compare_data: dict[str, Any],
    spec: Optional["AgentCISpec"] = None,
) -> DiffReport:
    """
    Compare two versioned baseline trace files with three-tier analysis.

    Args:
        baseline_data: Parsed JSON from the 'before' baseline file.
        compare_data:  Parsed JSON from the 'after' baseline file.
        spec:          Optional AgentCISpec to run correctness layer evaluation.

    Returns:
        DiffReport with correctness_delta, path_deltas, cost_deltas, legacy_diffs.

    Example:
        >>> import json
        >>> from ciagent.baselines import load_baseline
        >>> from ciagent.engine.diff import diff_baselines
        >>> base = load_baseline("rag-agent", "v1-broken")
        >>> comp = load_baseline("rag-agent", "v2-fixed")
        >>> report = diff_baselines(base, comp)
        >>> print(report.summary_console())
    """
    from ciagent.models import Trace

    agent = baseline_data.get("agent", compare_data.get("agent", "unknown"))
    from_version = baseline_data.get("version", "baseline")
    to_version = compare_data.get("version", "compare")
    query_text = baseline_data.get("query", compare_data.get("query", ""))

    # Reconstruct Trace objects
    baseline_trace = Trace(**baseline_data["trace"]) if "trace" in baseline_data else None
    compare_trace = Trace(**compare_data["trace"]) if "trace" in compare_data else None

    report = DiffReport(
        agent=agent,
        from_version=from_version,
        to_version=to_version,
        query=query_text,
    )

    # ── Layer 1: Correctness ───────────────────────────────────────────────────
    if spec and query_text:
        report.correctness_delta = _compute_correctness_delta(
            baseline_trace, compare_trace, query_text, spec
        )
    else:
        # Heuristic: compare final answers for length/content changes
        before_ans = _extract_answer(baseline_trace) if baseline_trace else ""
        after_ans = _extract_answer(compare_trace) if compare_trace else ""
        changed = (len(before_ans) > 0) != (len(after_ans) > 0)
        report.correctness_delta = {
            "before": "pass" if before_ans else "unknown",
            "after": "pass" if after_ans else "unknown",
            "changed": changed,
            "note": "No spec provided — heuristic only",
        }

    # ── Layer 2: Path ─────────────────────────────────────────────────────────
    if baseline_trace and compare_trace:
        report.path_deltas = _compute_path_deltas(baseline_trace, compare_trace)

    # ── Layer 3: Cost ─────────────────────────────────────────────────────────
    if baseline_trace and compare_trace:
        report.cost_deltas = _compute_cost_deltas(baseline_trace, compare_trace)

    # ── Legacy compat: generate v1 DiffResult list ────────────────────────────
    if baseline_trace and compare_trace:
        report.legacy_diffs = _generate_legacy_diffs(baseline_trace, compare_trace)

    return report


# ── Internal helpers ───────────────────────────────────────────────────────────


def _compute_correctness_delta(
    baseline_trace: Optional["Trace"],
    compare_trace: Optional["Trace"],
    query_text: str,
    spec: "AgentCISpec",
) -> dict[str, Any]:
    """Evaluate both traces against the spec and compare correctness results."""
    from ciagent.engine.correctness import evaluate_correctness
    from ciagent.engine.results import LayerStatus
    from ciagent.loader import _deep_merge

    # Find matching query spec
    query_spec = None
    for q in spec.queries:
        if q.query == query_text:
            query_spec = q
            break

    if not query_spec or not query_spec.correctness:
        return {
            "before": "unknown",
            "after": "unknown",
            "changed": False,
            "note": "No correctness spec for this query",
        }

    before_status = "unknown"
    after_status = "unknown"

    if baseline_trace:
        before_ans = _extract_answer(baseline_trace)
        before_result = evaluate_correctness(
            answer=before_ans,
            spec=query_spec.correctness,
            trace=baseline_trace,
            judge_config=spec.judge_config,
        )
        before_status = before_result.status.value

    if compare_trace:
        after_ans = _extract_answer(compare_trace)
        after_result = evaluate_correctness(
            answer=after_ans,
            spec=query_spec.correctness,
            trace=compare_trace,
            judge_config=spec.judge_config,
        )
        after_status = after_result.status.value

    return {
        "before": before_status,
        "after": after_status,
        "changed": before_status != after_status,
    }


def _compute_path_deltas(baseline: "Trace", compare: "Trace") -> list[MetricDelta]:
    """Compute tool usage / trajectory deltas between two traces."""
    deltas = []

    before_tools = baseline.tool_call_sequence
    after_tools = compare.tool_call_sequence

    # Tool call count — only emit when counts differ
    before_count = len(before_tools)
    after_count = len(after_tools)
    if before_count != after_count:
        deltas.append(MetricDelta(
            label="tool_calls",
            before=before_count,
            after=after_count,
        ))

    # Unique tools — only emit when counts differ
    before_unique = len(set(before_tools))
    after_unique = len(set(after_tools))
    if before_unique != after_unique:
        deltas.append(MetricDelta(
            label="unique_tools",
            before=before_unique,
            after=after_unique,
        ))

    # Tool recall (if both have tools to compare)
    if before_tools or after_tools:
        from ciagent.engine.metrics import compute_tool_recall, compute_sequence_lcs
        before_set = set(before_tools)
        after_set = set(after_tools)

        # Tool set overlap
        recall = compute_tool_recall(before_set, after_set) if before_set else 1.0
        if abs(recall - 1.0) > 0.01:
            deltas.append(MetricDelta(
                label="tool_recall",
                before=round(recall, 3),
                after=1.0 if after_set else 0.0,
            ))

        # Sequence similarity — only emit if sequences differ
        if before_tools != after_tools:
            lcs_sim = compute_sequence_lcs(before_tools, after_tools)
            deltas.append(MetricDelta(
                label="sequence_similarity",
                before=1.0,  # baseline is the reference (perfect similarity with itself)
                after=round(lcs_sim, 3),
            ))

    # LLM calls
    before_llm = baseline.total_llm_calls
    after_llm = compare.total_llm_calls
    if before_llm != after_llm:
        deltas.append(MetricDelta(
            label="llm_calls",
            before=before_llm,
            after=after_llm,
        ))

    return deltas


def _compute_cost_deltas(baseline: "Trace", compare: "Trace") -> list[MetricDelta]:
    """Compute cost / efficiency deltas between two traces."""
    deltas = []

    pairs = [
        ("cost_usd", baseline.total_cost_usd, compare.total_cost_usd),
        ("total_tokens", baseline.total_tokens, compare.total_tokens),
        ("latency_ms", baseline.total_duration_ms, compare.total_duration_ms),
    ]

    for label, before, after in pairs:
        if before != after:
            deltas.append(MetricDelta(label=label, before=before, after=after))

    return deltas


def _generate_legacy_diffs(baseline: "Trace", compare: "Trace") -> list[Any]:
    """Generate v1 DiffResult list from the diff for backward compatibility."""
    from ciagent.diff_engine import diff_traces
    import warnings
    # Suppress the deprecation warning for this internal call
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return diff_traces(compare, baseline)


def _extract_answer(trace: "Trace") -> str:
    """Extract the agent's final text answer from the trace.

    Strategy:
        1. trace.metadata["final_output"] (explicitly set by runner).
        2. Last span's output_data (fallback for runners that don't set metadata).
        3. Empty string if nothing found.
    """
    meta_output = trace.metadata.get("final_output")
    if meta_output is not None:
        return str(meta_output)

    if trace.spans:
        last = trace.spans[-1]
        out = last.output_data
        if out is not None:
            return out if isinstance(out, str) else str(out)

    return ""


# ── Conversation-aware diff (F6 Phase 2) ──────────────────────────────────────


@dataclass
class TurnDiff:
    """Per-turn comparison between a golden conversation and a fresh run."""
    turn_index: int
    user_message: str = ""
    tools_before: list[str] = field(default_factory=list)
    tools_after: list[str] = field(default_factory=list)
    answer_before: str = ""
    answer_after: str = ""

    @property
    def tools_changed(self) -> bool:
        return self.tools_before != self.tools_after

    @property
    def answer_changed(self) -> bool:
        return self.answer_before != self.answer_after

    @property
    def changed(self) -> bool:
        return self.tools_changed or self.answer_changed


@dataclass
class ConversationDiff:
    """Diff of a replayed (or re-run) conversation against a golden envelope.

    Turns are paired by index — replay feeds the golden's user turns verbatim,
    so index i is the same user message on both sides unless the turn count
    changed (early stop_when exit, infra-error, agent now stopping earlier).
    """
    agent: str = ""
    scenario: str = ""
    turns_before: int = 0
    turns_after: int = 0
    turn_diffs: list[TurnDiff] = field(default_factory=list)

    @property
    def turn_count_changed(self) -> bool:
        return self.turns_before != self.turns_after

    @property
    def has_changes(self) -> bool:
        return self.turn_count_changed or any(t.changed for t in self.turn_diffs)

    @property
    def tools_changed(self) -> bool:
        return any(t.tools_changed for t in self.turn_diffs)

    def summary_json(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "scenario": self.scenario,
            "turns_before": self.turns_before,
            "turns_after": self.turns_after,
            "turn_count_changed": self.turn_count_changed,
            "has_changes": self.has_changes,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "user_message": t.user_message,
                    "tools_before": t.tools_before,
                    "tools_after": t.tools_after,
                    "tools_changed": t.tools_changed,
                    "answer_changed": t.answer_changed,
                    "answer_before": t.answer_before,
                    "answer_after": t.answer_after,
                }
                for t in self.turn_diffs
                if t.changed
            ],
        }


def diff_envelopes(
    golden: "ConversationEnvelope",
    current: "ConversationEnvelope",
) -> ConversationDiff:
    """Compare two conversation envelopes turn by turn.

    Reports turn-count changes and, for each index-paired turn, tool-sequence
    and answer changes. Purely observational: gating stays with the scenario's
    own checks — a changed answer is a signal to look, not a verdict.
    """
    diff = ConversationDiff(
        agent=golden.agent or current.agent,
        scenario=(golden.scenario or {}).get("name", ""),
        turns_before=len(golden.turns),
        turns_after=len(current.turns),
    )
    for i in range(max(len(golden.turns), len(current.turns))):
        g = golden.turns[i] if i < len(golden.turns) else None
        c = current.turns[i] if i < len(current.turns) else None
        present = g or c
        diff.turn_diffs.append(TurnDiff(
            turn_index=i,
            user_message=present.user_message if present else "",
            tools_before=list(g.trace.tool_call_sequence or []) if g else [],
            tools_after=list(c.trace.tool_call_sequence or []) if c else [],
            answer_before=_extract_answer(g.trace) if g else "",
            answer_after=_extract_answer(c.trace) if c else "",
        ))
    return diff


def _format_value(before: Any, after: Any, label: str) -> str:
    """Format a before→after value pair for console display."""
    if label == "cost_usd":
        b = f"${before:.4f}" if isinstance(before, float) else str(before)
        a = f"${after:.4f}" if isinstance(after, float) else str(after)
    elif label in ("sequence_similarity", "tool_recall"):
        b = f"{before:.3f}" if isinstance(before, float) else str(before)
        a = f"${after:.3f}" if isinstance(after, float) else str(after)
    else:
        b = str(before)
        a = str(after)

    delta = MetricDelta(label=label, before=before, after=after)
    pct = f"  ({delta.pct_str})" if delta.pct_change is not None else ""
    return f"{b} → {a}{pct}"
