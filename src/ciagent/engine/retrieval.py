# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Retrieval Engine — Layer 2.5 (Soft Warning).

Evaluates deterministic assertions on the retriever tool's captured
``ToolCall.result``. This is the layer the Shopify study was missing: every
failure chain started with an empty (or wrong) retrieval that no eval layer
looked at — the agent answered anyway and the judge graded against the same
lost ground truth.

Result-interpretation contract (eng review 2026-07-05, binding):
``ToolCall.result`` is an untyped blob, so the layer has explicit reading
rules and fails CLOSED — it SKIPs whenever it cannot read, never guesses:

  - Empty means None, [], "", whitespace-only, or a literal (case-insensitive)
    match on the spec's ``empty_markers``. Anything else is non-empty; when
    in doubt, the layer does not warn.
  - ``result_format: list|json|text`` hints parsing; a result that does not
    parse as the hinted format SKIPs the layer with a message.
  - Uncaptured results (the adapter never populates ``result`` for any tool
    call) SKIP with an explicit "not captured by this adapter" message. When
    other tool calls in the trace carry results, a None on the retriever is
    a real empty return, not a capture gap.
  - ``facts_in_context`` is informational-only in v1: number-format variance
    ("4.5%" vs "0.045") makes substring matching guess-prone, and a
    determinism brand cannot ship a guessing warner. Reported, never WARNs.

Like path/cost, exceedances WARN — retrieval never hard-fails in v1; the
correctness layer owns hard failures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ciagent.engine.results import LayerResult, LayerStatus
from ciagent.schema.spec_models import CorrectnessSpec, RetrievalSpec

if TYPE_CHECKING:
    from ciagent.models import Trace, ToolCall

# Answer substrings that count as a refusal for forbid_empty — an agent that
# says "I don't know" on empty retrieval is behaving correctly. Deterministic
# case-insensitive substring match; spec `refusal_markers` overrides the list.
DEFAULT_REFUSAL_MARKERS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "couldn't find",
    "could not find",
    "cannot find",
    "can't find",
    "no information",
    "don't have information",
    "do not have information",
    "don't have that information",
    "unable to find",
    "unable to answer",
    "cannot answer",
    "can't answer",
    "no results",
)

# Dict keys treated as a document identifier when extracting a source set
# from retrieved items (for RETRIEVAL_CHANGED diffing). Deliberately narrow:
# text content is never a "source", and an unextractable source set means
# no diff — fail closed.
_SOURCE_KEYS: tuple[str, ...] = (
    "source", "doc_id", "document_id", "path", "file", "filename", "url",
)

# Reading outcomes for a trace's retriever calls
_READ_OK = "ok"
_READ_NOT_CALLED = "not-called"
_READ_UNCAPTURED = "uncaptured"
_READ_UNPARSEABLE = "unparseable"


@dataclass
class _Reading:
    """Parsed view of the retriever tool's captured results in one trace."""
    status: str
    message: str = ""
    parsed: list[Any] = field(default_factory=list)   # one entry per call
    all_empty: bool = False


def evaluate_retrieval(
    trace: "Trace",
    spec: RetrievalSpec,
    correctness: Optional[CorrectnessSpec] = None,
    answer: Optional[str] = None,
) -> LayerResult:
    """Evaluate retrieval assertions against the retriever's captured results.

    Args:
        trace:       Current execution trace.
        spec:        RetrievalSpec assertions to evaluate.
        correctness: The query's correctness spec — source of the fact terms
                     that ``facts_in_context`` cross-checks.
        answer:      The agent's final answer (for forbid_empty's refusal
                     detection). Extracted from the trace when omitted.

    Returns:
        LayerResult with PASS, WARN, or SKIP status — never FAIL.
    """
    if answer is None:
        from ciagent.engine.runner import _extract_answer

        answer = _extract_answer(trace)

    reading = _read_retrieval(trace, spec)
    if reading.status != _READ_OK:
        return LayerResult(
            status=LayerStatus.SKIP,
            details={"reason": reading.status, "tool": spec.tool},
            messages=[reading.message],
        )

    warnings: list[str] = []
    pass_messages: list[str] = []
    details: dict[str, Any] = {
        "tool": spec.tool,
        "calls": len(reading.parsed),
        "empty": reading.all_empty,
    }
    evaluated = False
    context_text = _serialize_parsed(reading.parsed)

    # ── 1. forbid_empty (the study's root-cause check) ───────────────────────
    if spec.forbid_empty:
        evaluated = True
        if not reading.all_empty:
            details["forbid_empty"] = {"empty": False, "violated": False}
            pass_messages.append("Retrieval non-empty")
        else:
            refused = _matches_refusal(answer, spec)
            substantive = bool(answer.strip()) and not refused
            details["forbid_empty"] = {
                "empty": True,
                "violated": substantive,
                "refusal_detected": refused,
            }
            if substantive:
                warnings.append(
                    "Ungrounded answer: retrieval was empty but the agent "
                    "produced a substantive answer"
                )
            elif refused:
                pass_messages.append(
                    "Retrieval was empty and the agent refused to answer — "
                    "correct behavior"
                )
            else:
                pass_messages.append(
                    "Retrieval was empty and the agent gave no answer"
                )

    # ── 2. min_results (count floor; countable lists only) ──────────────────
    if spec.min_results is not None:
        if all(isinstance(p, list) for p in reading.parsed):
            evaluated = True
            total = sum(len(p) for p in reading.parsed)
            details["min_results"] = {"actual": total, "min": spec.min_results}
            if total < spec.min_results:
                warnings.append(
                    f"Retrieved results: {total} < min {spec.min_results}"
                )
            else:
                pass_messages.append(
                    f"Retrieved results: {total} ≥ min {spec.min_results}"
                )
        else:
            details["min_results"] = {"skipped": "result is not a list"}
            pass_messages.append(
                "min_results: SKIP — result did not parse as a list, cannot "
                "count items (never guessed)"
            )

    # ── 3. expected_sources (source recall) ─────────────────────────────────
    if spec.expected_sources:
        evaluated = True
        haystack = context_text.lower()
        missing = [s for s in spec.expected_sources if s.lower() not in haystack]
        found = [s for s in spec.expected_sources if s.lower() in haystack]
        details["expected_sources"] = {"found": found, "missing": missing}
        if missing:
            warnings.append(f"Expected sources not retrieved: {missing}")
        else:
            pass_messages.append(
                f"All {len(found)} expected source(s) retrieved"
            )

    # ── 4. facts_in_context (informational-only in v1) ──────────────────────
    if spec.facts_in_context:
        terms = _fact_terms(correctness)
        if not terms:
            details["facts_in_context"] = {
                "skipped": "no correctness terms to cross-check"
            }
            pass_messages.append(
                "facts_in_context: SKIP — no expected_in_answer / "
                "any_expected_in_answer terms to cross-check"
            )
        else:
            evaluated = True
            haystack = context_text.lower()
            ungrounded = [t for t in terms if t.lower() not in haystack]
            details["facts_in_context"] = {
                "terms": terms,
                "ungrounded": ungrounded,
                "informational": True,
            }
            if ungrounded:
                # Reported, never WARNs (v1): right-for-the-wrong-reason
                # answers (parametric memory) are a signal to look, not a
                # verdict.
                pass_messages.append(
                    f"facts_in_context (informational): {len(ungrounded)} of "
                    f"{len(terms)} fact term(s) not found in retrieved "
                    f"context: {ungrounded} — answer may be right for the "
                    f"wrong reason (parametric memory)"
                )
            else:
                pass_messages.append(
                    f"facts_in_context (informational): all {len(terms)} "
                    f"fact term(s) grounded in retrieved context"
                )

    if warnings:
        return LayerResult(
            status=LayerStatus.WARN,
            details=details,
            messages=warnings + pass_messages,
        )
    if not evaluated:
        return LayerResult(
            status=LayerStatus.SKIP,
            details=details,
            messages=pass_messages or ["No evaluable retrieval assertions"],
        )
    return LayerResult(
        status=LayerStatus.PASS,
        details=details,
        messages=pass_messages,
    )


def is_empty_retrieval(trace: "Trace", spec: RetrievalSpec) -> Optional[bool]:
    """Whether the retriever returned nothing in this trace.

    Returns None when the question cannot be answered honestly — retriever
    not called, results not captured, or unparseable — so callers (judge
    audit) can report "unknown" instead of guessing.
    """
    reading = _read_retrieval(trace, spec)
    if reading.status != _READ_OK:
        return None
    return reading.all_empty


def retrieval_signature(trace: "Trace", tool: str) -> Optional[str]:
    """Stable serialization of the retriever's captured results in one trace.

    Used by the stability engine to detect a retrieved set that changed
    across runs. None when the retriever wasn't called or results weren't
    captured — attribution must then not blame the retriever (fail closed).
    """
    calls = _retriever_calls(trace, tool)
    if not calls:
        return None
    results = [c.result for c in calls]
    if all(r is None for r in results) and not _trace_captures_results(trace):
        return None
    try:
        return json.dumps(results, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(results)


def extract_source_set(trace: "Trace", tool: str) -> Optional[set[str]]:
    """Extract the set of document identifiers the retriever returned.

    Looks for source-like keys (`source`, `doc_id`, `path`, ...) on dict
    items in list-shaped results. None when no source set is extractable —
    the diff engine then emits no RETRIEVAL_CHANGED, fail closed.
    """
    reading = _read_retrieval_by_tool(trace, tool)
    if reading.status != _READ_OK:
        return None
    sources: set[str] = set()
    for parsed in reading.parsed:
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            if not isinstance(item, dict):
                continue
            for key in _SOURCE_KEYS:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    sources.add(value.strip())
    return sources or None


# ── Internal helpers ───────────────────────────────────────────────────────────


def _retriever_calls(trace: "Trace", tool: str) -> list["ToolCall"]:
    return [
        tc
        for span in trace.spans
        for tc in span.tool_calls
        if tc.tool_name == tool
    ]


def _trace_captures_results(trace: "Trace") -> bool:
    """True if ANY tool call in the trace carries a captured result — the
    adapter demonstrably captures results, so a None elsewhere is a real
    empty return rather than a capture gap."""
    return any(
        tc.result is not None
        for span in trace.spans
        for tc in span.tool_calls
    )


def _read_retrieval(trace: "Trace", spec: RetrievalSpec) -> _Reading:
    return _do_read(trace, spec.tool, spec.result_format, spec.empty_markers)


def _read_retrieval_by_tool(trace: "Trace", tool: str) -> _Reading:
    return _do_read(trace, tool, None, None)


def _do_read(
    trace: "Trace",
    tool: str,
    result_format: Optional[str],
    empty_markers: Optional[list[str]],
) -> _Reading:
    calls = _retriever_calls(trace, tool)
    if not calls:
        return _Reading(
            status=_READ_NOT_CALLED,
            message=f"Retriever tool '{tool}' was not called in this trace",
        )

    results = [c.result for c in calls]
    if all(r is None for r in results) and not _trace_captures_results(trace):
        return _Reading(
            status=_READ_UNCAPTURED,
            message=(
                f"Retriever output not captured by this adapter — no tool "
                f"call in the trace carries a result; retrieval checks "
                f"cannot run (never guessed)"
            ),
        )

    parsed: list[Any] = []
    for r in results:
        p, ok = _parse_result(r, result_format)
        if not ok:
            return _Reading(
                status=_READ_UNPARSEABLE,
                message=(
                    f"Retriever result did not parse as "
                    f"'{result_format}' — retrieval checks cannot run "
                    f"(never guessed)"
                ),
            )
        parsed.append(p)

    markers = [m.strip().lower() for m in (empty_markers or [])]
    all_empty = all(_is_empty(p, markers) for p in parsed)
    return _Reading(status=_READ_OK, parsed=parsed, all_empty=all_empty)


def _parse_result(result: Any, result_format: Optional[str]) -> tuple[Any, bool]:
    """Apply the reading rules to one captured result.

    Returns (parsed_value, ok). ok=False means the result violates the
    format hint — the caller SKIPs the layer.
    """
    if result_format == "list":
        if result is None or isinstance(result, list):
            return result, True
        maybe = _try_json(result)
        if isinstance(maybe, list):
            return maybe, True
        return result, False

    if result_format == "json":
        if result is None or isinstance(result, (list, dict)):
            return result, True
        maybe = _try_json(result)
        if isinstance(maybe, (list, dict)):
            return maybe, True
        return result, False

    if result_format == "text":
        if result is None or isinstance(result, str):
            return result, True
        return str(result), True

    # No hint: structures pass through; strings that parse as JSON
    # structures are read as such; everything else is text.
    if result is None or isinstance(result, (list, dict)):
        return result, True
    if isinstance(result, str):
        maybe = _try_json(result)
        if isinstance(maybe, (list, dict)):
            return maybe, True
        return result, True
    return str(result), True


def _try_json(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None


def _is_empty(parsed: Any, empty_markers: list[str]) -> bool:
    """The binding empty definition: None, [], "", whitespace-only, or a
    literal marker match. Anything else is non-empty."""
    if parsed is None:
        return True
    if isinstance(parsed, list):
        return len(parsed) == 0
    if isinstance(parsed, str):
        stripped = parsed.strip()
        if not stripped:
            return True
        return stripped.lower() in empty_markers
    return False


def _matches_refusal(answer: str, spec: RetrievalSpec) -> bool:
    markers = spec.refusal_markers or list(DEFAULT_REFUSAL_MARKERS)
    normalized = answer.replace("’", "'").lower()
    return any(m.lower() in normalized for m in markers)


def _fact_terms(correctness: Optional[CorrectnessSpec]) -> list[str]:
    """The correctness layer's fact terms — what facts_in_context grounds."""
    if correctness is None:
        return []
    terms: list[str] = []
    terms.extend(correctness.expected_in_answer or [])
    terms.extend(correctness.any_expected_in_answer or [])
    return terms


def _serialize_parsed(parsed: list[Any]) -> str:
    """Deterministic text form of the retrieved context for substring checks."""
    parts: list[str] = []
    for p in parsed:
        if p is None:
            continue
        if isinstance(p, str):
            parts.append(p)
        else:
            try:
                parts.append(json.dumps(p, sort_keys=True, default=str))
            except (TypeError, ValueError):
                parts.append(str(p))
    return "\n".join(parts)
