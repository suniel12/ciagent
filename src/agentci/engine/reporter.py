# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
AgentCI v2 Reporter.

Generates output in multiple formats and returns the appropriate exit code.

Exit code contract:
    0  — All correctness layers pass (warnings are annotations, not failures)
    1  — Any correctness failure or forbidden tool violation
    2  — Runtime/infrastructure error (set by caller)

Format options:
    console     — Human-readable rich output (default)
    github      — GitHub Actions annotations (::error:: / ::warning::)
    json        — Machine-readable JSON for dashboards
    prometheus  — Prometheus exposition format for Grafana
    html        — Self-contained HTML report for sharing
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

from agentci.engine.results import LayerStatus, QueryResult

if TYPE_CHECKING:
    from agentci.engine.stability import StabilityReport

# GitHub limits visible inline annotations per job; exceeding this silently
# drops annotations. Warnings are budget-capped; errors are always emitted.
MAX_INLINE_ANNOTATIONS: int = 10


# ── Public API ─────────────────────────────────────────────────────────────────


def report_results(
    results: list[QueryResult],
    format: str = "console",
    spec_file: str = "agentci_spec.yaml",
    output_path: str | None = None,
    stability: Optional["StabilityReport"] = None,
) -> int:
    """Generate output and return the appropriate exit code.

    Args:
        results:    List of QueryResult from the evaluation engine.
        format:     Output format: 'console', 'github', 'json', 'prometheus', 'html'.
        spec_file:  Path to the spec file (used in GitHub annotation file references).
        output_path: File path for HTML output (default: agentci-report.html).
        stability:  Optional multi-run StabilityReport (from `--runs N`). When
                    provided, a stability section is added to the output and
                    failure means "failed in every run", not "failed in the
                    run being rendered".

    Returns:
        Exit code: 0 = pass, 1 = correctness fail (consistent across runs when
        a stability report is provided).
    """
    if stability is not None:
        has_hard_failures = bool(stability.consistent_failures)
    else:
        has_hard_failures = any(r.hard_fail for r in results)

    # Always emit annotations when running in GitHub Actions
    if format == "github" or _is_github_actions():
        _emit_github_annotations(results, spec_file)
        if stability is not None:
            _emit_stability_github(stability, spec_file)

    if format == "json":
        _emit_json(results, stability=stability)
    elif format == "prometheus":
        _emit_prometheus(results)
    elif format == "html":
        _emit_html(results, spec_file, output_path or "agentci-report.html", stability=stability)
    else:
        _emit_console(results)
        if stability is not None:
            emit_stability_console(stability)

    return 1 if has_hard_failures else 0


# ── GitHub Actions Annotations ─────────────────────────────────────────────────


def _is_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _emit_github_annotations(results: list[QueryResult], spec_file: str) -> None:
    """Emit GitHub Actions workflow commands for inline PR feedback.

    Format: ::level file=<path>::<message>
    - ::error  → red, blocks merge (no cap — correctness failures always visible)
    - ::warning → yellow, non-blocking (capped at MAX_INLINE_ANNOTATIONS)

    Overflow warnings beyond the cap are written to GITHUB_STEP_SUMMARY so
    they remain accessible without silently disappearing.
    """
    warning_count: int = 0
    overflow_warnings: list[str] = []

    for r in results:
        query_short = r.query[:60]

        # Hard fails → always emit as ::error (no cap)
        if r.correctness.status == LayerStatus.FAIL:
            for msg in r.correctness.messages:
                print(f"::error file={spec_file}::[CORRECTNESS] {query_short}: {msg}")

        if r.path.status == LayerStatus.FAIL:
            for msg in r.path.messages:
                print(f"::error file={spec_file}::[PATH] {query_short}: {msg}")

        # Soft warnings → budget-capped
        if r.path.status == LayerStatus.WARN:
            for msg in r.path.messages:
                annotation = f"[PATH] {query_short}: {msg}"
                if warning_count < MAX_INLINE_ANNOTATIONS:
                    print(f"::warning file={spec_file}::{annotation}")
                    warning_count += 1
                else:
                    overflow_warnings.append(annotation)

        if r.cost.status == LayerStatus.WARN:
            for msg in r.cost.messages:
                annotation = f"[COST] {query_short}: {msg}"
                if warning_count < MAX_INLINE_ANNOTATIONS:
                    print(f"::warning file={spec_file}::{annotation}")
                    warning_count += 1
                else:
                    overflow_warnings.append(annotation)

    if overflow_warnings:
        _write_step_summary(overflow_warnings)


def _write_step_summary(messages: list[str]) -> None:
    """Write overflow warning messages to the GitHub Actions step summary.

    Appends a markdown table to $GITHUB_STEP_SUMMARY so that warnings
    beyond MAX_INLINE_ANNOTATIONS remain accessible in the Actions UI.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("\n## AgentCI — Additional Warnings\n\n")
            fh.write("| # | Warning |\n")
            fh.write("|---|--------|\n")
            for i, msg in enumerate(messages, start=1):
                fh.write(f"| {i} | {msg} |\n")
    except OSError:
        pass  # Step summary write failure is non-fatal


# ── Console Output ─────────────────────────────────────────────────────────────


def emit_query_result(r: QueryResult) -> None:
    """Print a single query result to console. Used for streaming output."""
    print(f"\n{'=' * 60}")
    print(f"Query: {r.query}")

    # Always show the agent's answer right after the query
    if getattr(r, "trace", None):
        _print_answer_preview(r.trace)

    _print_layer("CORRECTNESS", r.correctness, fail_icon="❌", pass_icon="✅")
    _print_layer("PATH", r.path, fail_icon="⚠️", pass_icon="📈", warn_icon="⚠️")
    _print_layer("COST", r.cost, fail_icon="⚠️", pass_icon="💰", warn_icon="⚠️")

    if r.hard_fail and getattr(r, "trace", None):
        _print_trace_summary(r.trace)


def emit_summary(results: list[QueryResult]) -> None:
    """Print the final summary line."""
    total = len(results)
    passed = sum(1 for r in results if not r.hard_fail)
    warned = sum(1 for r in results if r.has_warnings and not r.hard_fail)
    failed = sum(1 for r in results if r.hard_fail)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed  |  {warned} warnings  |  {failed} failures")


def _emit_console(results: list[QueryResult]) -> None:
    """Rich console output with three-tier report per query."""
    for r in results:
        emit_query_result(r)
    emit_summary(results)


def emit_stability_console(report: "StabilityReport") -> None:
    """Print the multi-run stability section.

    Designed around one contrast: the suite score per run (stable) versus the
    queries whose verdicts flipped underneath it (not stable) — with every
    flip attributed to its source so it's actionable.
    """
    print(f"\n{'─' * 60}")
    print("Stability Report")
    print(f"{'─' * 60}")

    scores = "  /  ".join(f"{s:.0%}" for s in report.per_run_scores)
    print(f"Suite score across {report.runs} runs: {scores}")

    if report.duplicate_queries:
        print(f"\n⚠️  {len(report.duplicate_queries)} duplicate query text(s) in the spec "
              f"— their runs merge into one record each:")
        for text in report.duplicate_queries:
            print(f"   • {_short(text, 70)}")

    if report.is_stable:
        print(f"\n✅ STABLE — all {report.total_queries} queries returned the same "
              f"verdict in every run")
    else:
        flipped = report.flipped_queries
        print(f"\n⚠️  FLAKY — {len(flipped)}/{report.total_queries} queries flipped "
              f"verdicts across runs:")
        for q in flipped:
            label = q.flip_source.value if q.flip_source else "unknown"
            partial = f"  [partial: {q.runs}/{q.expected_runs} runs]" if q.partial else ""
            print(
                f"   {_short(q.query, 44):<46} {q.verdict_string}  "
                f"pass_rate={q.pass_rate:.2f}  "
                f"source: {label} ({q.flip_reason}){partial}"
            )
        from agentci.engine.stability import FlipSource
        agent_side = sum(1 for q in flipped if q.flip_source == FlipSource.AGENT_VARIANCE)
        judge_side = sum(1 for q in flipped if q.flip_source == FlipSource.JUDGE_FLAKE)
        infra = sum(1 for q in flipped if q.flip_source == FlipSource.INFRA_ERROR)
        mixed = len(flipped) - agent_side - judge_side - infra
        print(
            f"\n   Flip sources: {agent_side} agent-variance (fix the agent) │ "
            f"{judge_side} judge-flake (fix the eval) │ "
            f"{infra} infra-error (retry) │ {mixed} mixed"
        )

    stable_partials = [q for q in report.partial_queries if not q.flipped]
    if stable_partials:
        print(f"\n⚠️  {len(stable_partials)} query(ies) missing from some runs "
              f"(runner failures) — verdicts aggregate over fewer runs:")
        for q in stable_partials:
            print(f"   • {_short(q.query, 60)} [{q.runs}/{q.expected_runs} runs]")

    if report.consistent_failures:
        print(f"\n❌ {len(report.consistent_failures)} query(ies) failed in EVERY run "
              f"(consistent failures, not flakiness):")
        for q in report.consistent_failures:
            print(f"   • {_short(q.query, 70)}")

    print(f"\nStability verdict: {report.verdict}")


def _short(text: str, max_len: int) -> str:
    text = " ".join(text.split())
    return f'"{text[: max_len - 1]}…"' if len(text) > max_len else f'"{text}"'


def _emit_stability_github(report: "StabilityReport", spec_file: str) -> None:
    """Emit stability findings as GitHub annotations.

    Flipped queries are warnings (actionable, non-blocking); consistent
    failures already surface as ::error via the normal results path.
    """
    for q in report.flipped_queries:
        label = q.flip_source.value if q.flip_source else "unknown"
        print(
            f"::warning file={spec_file}::[STABILITY] {q.query[:60]}: verdict "
            f"flipped across {q.runs} runs ({q.verdict_string}) — source: "
            f"{label} ({q.flip_reason})"
        )

def _print_answer_preview(trace: Any, max_len: int = 500) -> None:
    """Show a truncated preview of the extracted answer."""
    from agentci.engine.runner import _extract_answer

    answer = _extract_answer(trace)

    if not answer:
        print("Answer: (no answer extracted from trace)")
        return

    # Collapse whitespace for compact display
    preview = " ".join(answer.split())
    if len(preview) > max_len:
        preview = preview[:max_len] + "..."
    print(f"Answer: {preview}")


def _print_trace_summary(trace: Any) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.text import Text

    console = Console()
    tree = Tree("[bold magenta]Trace Execution Summary[/]")
    for span in trace.spans:
        label = Text(f"{span.name} ", style="cyan bold")
        if span.kind == "tool_call":
            args = str(span.input_data)
            if len(args) > 80:
                args = args[:77] + "..."
            label.append(f"({args})", style="dim")
        else:
            label.append(f"[{span.kind}]", style="magenta dim")

        node = tree.add(label)
        if span.stop_reason == "error":
            node.add(Text(f"ERROR: {span.stop_reason}", style="bold red"))
        elif span.output_data:
            out = str(span.output_data)
            # Remove newlines for compact display
            out = " ".join(out.splitlines())
            if len(out) > 120:
                out = out[:117] + "..."
            node.add(Text(out, style="green"))

    console.print(Panel(tree, border_style="magenta", expand=False))

def _print_layer(
    name: str,
    layer_result,
    fail_icon: str = "❌",
    pass_icon: str = "✅",
    warn_icon: str = "⚠️",
) -> None:
    status = layer_result.status
    if status == LayerStatus.PASS:
        icon = pass_icon
    elif status == LayerStatus.FAIL:
        icon = fail_icon
    elif status == LayerStatus.WARN:
        icon = warn_icon
    else:
        icon = "—"

    # SKIP: inline reason on the status line, no bullet points
    if status == LayerStatus.SKIP:
        reason = layer_result.messages[0] if layer_result.messages else "not configured"
        print(f"  {icon}  {name}: SKIP ({reason})")
        return

    print(f"  {icon}  {name}: {status.value.upper()}")
    for msg in layer_result.messages:
        if status == LayerStatus.PASS:
            print(f"       ✓ {msg}")
        else:
            print(f"       • {msg}")


# ── Judge Audit Output ─────────────────────────────────────────────────────────


def emit_judge_audit_console(report: Any) -> None:
    """Print the judge audit report (JudgeAuditReport)."""
    print(f"\n{'─' * 60}")
    print("Judge Audit")
    print(f"{'─' * 60}")

    checkable = report.checkable_queries
    print(
        f"Judged queries: {len(report.judged)}  │  fact-checkable: {len(checkable)}"
        f"  │  judgment-only: {report.judgment_only_count}  │  repeats: {report.repeats}"
    )

    if report.all_judge_calls_errored:
        print("\n❌ Every judge call errored — the judge never actually ran.")
        first = report.judged[0].judge_rationales[0] if report.judged else ""
        if first:
            print(f"   First error: {first[:120]}")
        print("   Set ANTHROPIC_API_KEY or OPENAI_API_KEY (or judge_config.model) and retry.")
        print("\nJudge verdict: ERROR (no honest verdict possible)")
        return
    if report.total_judge_errors:
        print(f"\n⚠️  {report.total_judge_errors} judge call(s) errored and were "
              f"counted as fails — rates below include them.")

    # Mode 1 — disagreement matrix
    if checkable:
        agree = report.agreement_rate
        print(f"\nJudge vs. deterministic checks (agreement: {agree:.0%}):")
        both_pass = sum(1 for q in checkable if q.judge_verdict and q.det_verdict)
        both_fail = sum(1 for q in checkable if not q.judge_verdict and not q.det_verdict)
        print(f"   both pass: {both_pass}   both fail: {both_fail}   "
              f"judge-only fail: {len(report.false_alarms)}   "
              f"judge PASS / check FAIL: {len(report.false_passes)}")
        if report.false_passes:
            print("\n❌ Judge PASSED answers a deterministic fact-check FAILED:")
            for q in report.false_passes:
                print(f"   • {_short(q.query, 70)}")
                if q.judge_rationales and q.judge_rationales[0]:
                    print(f"     judge said: {q.judge_rationales[0][:100]}")
    else:
        print("\nNo queries have BOTH deterministic checks and judge rubrics —")
        print("Mode 1 (judge vs. checks) has nothing to compare. Add fact checks")
        print("to judged queries, or provide --labels for direct measurement.")

    # Mode 2 — retest stability
    if report.flip_rate is not None:
        flipped = [q for q in report.judged if q.judge_flipped]
        print(f"\nJudge retest stability across {report.repeats} repeats: "
              f"{report.flip_rate:.0%} of queries flipped")
        for q in flipped:
            verdicts = "".join("✅" if v else "❌" for v in q.judge_verdicts)
            print(f"   {_short(q.query, 56):<58} {verdicts}  (same answer every time)")

    # Mode 3 — hand labels
    if report.label_agreement is not None:
        kappa = report.cohens_kappa
        kappa_str = f", Cohen's κ = {kappa:.2f}" if kappa is not None else ""
        print(f"\nJudge vs. hand labels ({len(report.labeled_queries)} labeled): "
              f"agreement {report.label_agreement:.0%}{kappa_str}")
        if kappa is not None and kappa < 0.75:
            print("   κ < 0.75 — below the standard trust floor for judge adoption")

    print(f"\n{report.scope_note}")
    print(f"\nJudge verdict: {report.verdict}")


def _serialize_judge_audit(report: Any) -> dict[str, Any]:
    return {
        "verdict": report.verdict,
        "repeats": report.repeats,
        "judged": len(report.judged),
        "checkable": len(report.checkable_queries),
        "judgment_only": report.judgment_only_count,
        "agreement_rate": report.agreement_rate,
        "false_pass_rate": report.false_pass_rate,
        "flip_rate": report.flip_rate,
        "label_agreement": report.label_agreement,
        "cohens_kappa": report.cohens_kappa,
        "low_sample": report.low_sample,
        "judge_errors": report.total_judge_errors,
        "scope_note": report.scope_note,
        "queries": [
            {
                "query": q.query,
                "det_verdict": q.det_verdict,
                "judge_verdict": q.judge_verdict,
                "judge_verdicts": q.judge_verdicts,
                "judge_flipped": q.judge_flipped,
                "false_pass": q.false_pass,
                "label": q.label,
                "rationales": q.judge_rationales,
            }
            for q in report.queries
        ],
    }


def report_judge_audit(report: Any, format: str = "console") -> int:
    """Render a judge audit and return an exit code.

    0 = TRUSTWORTHY / NEEDS CALIBRATION, 1 = UNRELIABLE, 2 = judge never ran.
    """
    if format == "json":
        print(json.dumps(_serialize_judge_audit(report), indent=2))
    else:
        emit_judge_audit_console(report)
    if report.verdict == "ERROR":
        return 2
    return 1 if report.verdict == "UNRELIABLE" else 0


# ── JSON Output ────────────────────────────────────────────────────────────────


def _emit_json(
    results: list[QueryResult],
    stability: Optional["StabilityReport"] = None,
) -> None:
    """Structured JSON for dashboards and external tooling."""
    output: dict[str, Any] = {
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if not r.hard_fail),
            "failed": sum(1 for r in results if r.hard_fail),
            "warnings": sum(1 for r in results if r.has_warnings),
        },
        "results": [_serialize_result(r) for r in results],
    }
    if stability is not None:
        output["stability"] = _serialize_stability(stability)
    print(json.dumps(output, indent=2))


def _serialize_stability(report: "StabilityReport") -> dict[str, Any]:
    return {
        "runs": report.runs,
        "verdict": report.verdict,
        "per_run_scores": report.per_run_scores,
        "flipped": len(report.flipped_queries),
        "consistent_failures": len(report.consistent_failures),
        "duplicate_queries": report.duplicate_queries,
        # pass@k / pass^k are ESTIMATES computed from the observed pass rate with
        # k = observed runs — at small k they restate the pass rate, which is why
        # the console shows observed facts only and estimates live here, labeled.
        "estimate_note": "pass_at_k/pass_pow_k are estimates from observed pass_rate with k=runs",
        "queries": [
            {
                "query": q.query,
                "verdicts": q.verdicts,
                "runs": q.runs,
                "expected_runs": q.expected_runs,
                "partial": q.partial,
                "pass_rate": round(q.pass_rate, 3),
                "pass_at_k_estimate": q.pass_at_k,
                "pass_pow_k_estimate": q.pass_pow_k,
                "flipped": q.flipped,
                "flip_source": q.flip_source.value if q.flip_source else None,
                "flip_reason": q.flip_reason or None,
                "answer_similarity": q.answer_similarity,
                "cost_usd": q.cost_usd,
                "latency_ms": q.latency_ms,
            }
            for q in report.queries
        ],
    }


def _serialize_result(r: QueryResult) -> dict[str, Any]:
    # Answer text included so JSON consumers (coding agents, judge-audit
    # answer sources) can see what the agent said, not just the verdicts.
    answer = None
    if r.trace is not None:
        metadata = getattr(r.trace, "metadata", None) or {}
        answer = metadata.get("final_output")
    return {
        "query": r.query,
        "answer": answer,
        "hard_fail": r.hard_fail,
        "has_warnings": r.has_warnings,
        "correctness": {
            "status": r.correctness.status.value,
            "messages": r.correctness.messages,
            "details": r.correctness.details,
        },
        "path": {
            "status": r.path.status.value,
            "messages": r.path.messages,
            "details": r.path.details,
        },
        "cost": {
            "status": r.cost.status.value,
            "messages": r.cost.messages,
            "details": r.cost.details,
        },
    }


# ── Prometheus Output ──────────────────────────────────────────────────────────


def _emit_prometheus(results: list[QueryResult]) -> None:
    """Prometheus exposition format for Grafana dashboards."""
    print("# AgentCI evaluation metrics")
    for r in results:
        label = r.query[:40].replace('"', '\\"').replace("\n", " ")
        ql = f'query="{label}"'

        # Correctness as boolean gauge
        val = 1 if r.correctness.status == LayerStatus.PASS else 0
        print(f'agentci_correctness_pass{{{ql}}} {val}')

        if "tool_recall" in r.path.details:
            print(f'agentci_tool_recall{{{ql}}} {r.path.details["tool_recall"]}')
        if "tool_precision" in r.path.details:
            print(f'agentci_tool_precision{{{ql}}} {r.path.details["tool_precision"]}')
        if "sequence_similarity" in r.path.details:
            print(f'agentci_sequence_similarity{{{ql}}} {r.path.details["sequence_similarity"]}')

        if "actual" in r.cost.details:
            actual = r.cost.details["actual"]
            print(f'agentci_cost_usd{{{ql}}} {actual["cost_usd"]}')
            print(f'agentci_latency_ms{{{ql}}} {actual["latency_ms"]}')
            print(f'agentci_total_tokens{{{ql}}} {actual["total_tokens"]}')
            print(f'agentci_llm_calls{{{ql}}} {actual["llm_calls"]}')


# ── HTML Output ───────────────────────────────────────────────────────────────


def _emit_html(
    results: list[QueryResult],
    spec_file: str,
    output_path: str,
    stability: Optional["StabilityReport"] = None,
) -> None:
    """Render a self-contained HTML report and write it to *output_path*."""
    from datetime import datetime, timezone
    from pathlib import Path

    from jinja2 import Environment, FileSystemLoader

    template_dir = Path(__file__).resolve().parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("report.html.j2")

    # Compute summary stats
    total = len(results)
    passed = sum(1 for r in results if not r.hard_fail)
    warned = sum(1 for r in results if r.has_warnings and not r.hard_fail)
    failed = sum(1 for r in results if r.hard_fail)

    # Compute total cost from cost layer details
    total_cost = 0.0
    has_cost = False
    for r in results:
        if "actual" in r.cost.details:
            total_cost += r.cost.details["actual"].get("cost_usd", 0.0)
            has_cost = True

    # Build per-result view models with answer + flattened spans
    view_results = []
    for r in results:
        answer = _extract_answer_for_html(r)
        spans = _flatten_spans_for_html(r)
        view_results.append(_HTMLQueryView(r, answer, spans))

    # Get version
    try:
        from agentci import __version__ as version
    except ImportError:
        version = "dev"

    html = template.render(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        spec_file=spec_file,
        total=total,
        passed=passed,
        warned=warned,
        failed=failed,
        total_cost=total_cost if has_cost else None,
        results=view_results,
        version=version,
        stability=stability,
    )

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"HTML report written to {output_path}")


class _HTMLQueryView:
    """Lightweight view model for the Jinja2 template."""

    def __init__(
        self,
        result: QueryResult,
        answer: str,
        spans: list[dict[str, Any]],
    ) -> None:
        self.query = result.query
        self.hard_fail = result.hard_fail
        self.has_warnings = result.has_warnings
        self.correctness = result.correctness
        self.path = result.path
        self.cost = result.cost
        self.answer = answer
        self.spans = spans


def _extract_answer_for_html(r: QueryResult) -> str:
    """Extract the agent's answer text from a QueryResult."""
    trace = getattr(r, "trace", None)
    if not trace:
        return ""
    return trace.metadata.get("final_output", "")


def _flatten_spans_for_html(r: QueryResult) -> list[dict[str, Any]]:
    """Flatten trace spans into a list of dicts for the template."""
    trace = getattr(r, "trace", None)
    if not trace or not hasattr(trace, "spans"):
        return []

    flat: list[dict[str, Any]] = []
    for span in trace.spans:
        kind = getattr(span, "kind", "unknown")
        if hasattr(kind, "value"):
            kind = kind.value
        tool_names = ""
        if hasattr(span, "tool_calls") and span.tool_calls:
            names = [getattr(tc, "tool_name", "") for tc in span.tool_calls]
            tool_names = ", ".join(n for n in names if n)
        flat.append({
            "kind": kind,
            "name": getattr(span, "name", ""),
            "duration_ms": getattr(span, "duration_ms", None),
            "tool_names": tool_names,
            "depth": 0,  # flat for now; tree depth can be added later
        })
    return flat
