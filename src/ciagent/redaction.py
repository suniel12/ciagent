# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Capture-time redaction for staged conversation envelopes.

Design: Plan_docs/redaction_capture.md. Deterministic only — no LLM, no
entropy scanning. The redactor serializes the envelope, walks every string
value (key-aware for dicts), substitutes shape-preserving placeholders, and
reconstructs via Pydantic. Idempotent by construction: matchers skip any
region overlapping the placeholder grammar, so re-redacting redacted output
is a byte-level no-op.

Zero imports from promotion/CLI — reusable as-is by a future
`world record --redact`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

from pydantic import ValidationError

from .conversation import ConversationEnvelope

# ── Placeholder grammar (D3) ────────────────────────────────────────────────────
# Every matcher must skip regions overlapping these — idempotence by construction.

PLACEHOLDER_RE = re.compile(
    r"\[SECRET:[a-z]+#\d+\]|redacted-\d+@example\.com|\+1-555-01\d{2}"
)

# ── Parse-safety guards (D1/A3) ─────────────────────────────────────────────────
# Values under these keys are never pattern-matched. This is a PARSE-SAFETY
# list (rewriting these breaks Pydantic reconstruction or id semantics), not a
# redaction-scope list — a new field defaults to being walked.

STRUCTURAL_KEYS = frozenset({
    "schema_version", "mode", "kind", "span_id", "parent_span_id", "trace_id",
    "timestamp", "created_at", "captured_at", "staged_at", "run_id",
    "turn_index", "classification", "flip_source", "source", "scenario_id",
    "lifecycle", "promoted_at", "staged_run_id", "classification_at_promotion",
    "flipped_at", "model", "provider", "framework", "stop_reason",
})

# A timestamp cannot contain a secret; skip ISO-8601 strings entirely.
_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ][\d:.]+(Z|[+-]\d{2}:?\d{2})?)?$"
)

# Keys whose last dotted segment matches this get their whole value redacted
# (per-string patterns cannot see `"api_key": "abc123"` — two JSON strings).
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|apikey|secret|password|passwd|token|access[_-]?key|"
    r"authorization|auth|credentials?)$"
)

# Regex-valued check fields are patterns, not data: family 1 (prefixes) only.
_REGEX_KEY_RE = re.compile(r"(?i)regex")

# ── Pattern families (D2) ───────────────────────────────────────────────────────

# Family 1: known secret prefixes — the value alone is sufficient evidence.
_PREFIX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")),
    ("aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github", re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("slack", re.compile(r"\bxox[bpars]-[A-Za-z0-9\-]{10,}\b")),
    ("google", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe", re.compile(r"\b(?:[rs]k|pk)_live_[A-Za-z0-9]{16,}\b")),
]

# Family 2 (in-string context): the value after a sensitive key or Bearer.
_CONTEXT_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|authorization)\b"
    r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-.+/]{8,})[\"']?"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+([A-Za-z0-9_\-.=+/]{8,})")

# Family 3: PII. Constraints per ADR A6 — separators required for phones,
# digit-boundary isolation + Luhn for bare card runs.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_US_RE = re.compile(
    r"(?<![\dA-Za-z])(\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?![\dA-Za-z])"
)
_PHONE_INTL_RE = re.compile(r"(?<![\w.+])\+\d{1,3}([\s.\-]?\d{2,4}){2,4}(?!\d)")
_CARD_GROUPED_RE = re.compile(r"(?<!\d)\d{4}([ \-])\d{4}\1\d{4}\1\d{1,7}(?!\d)")
_CARD_BARE_RE = re.compile(r"(?<![\d.\-T:])\d{13,19}(?![\d.\-:])")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class _SubState:
    """Per-invocation substitution map: value → placeholder, plus counts.

    Local to one redact call, so concurrency can never race it, and counts
    are exact (from the map, not from scanning output — D6/A10)."""

    def __init__(self) -> None:
        self.by_value: dict[tuple[str, str], str] = {}
        self.counters: dict[str, int] = {}

    def placeholder(self, family: str, value: str, *, kind: str) -> str:
        key = (kind, value)
        if key in self.by_value:
            return self.by_value[key]
        n = self.counters.get(kind, 0) + 1
        self.counters[kind] = n
        if kind == "email":
            ph = f"redacted-{n}@example.com"
        elif kind == "phone":
            # Reserved fictional range holds 100 values; beyond that,
            # numbering reuses the last slot (accepted bound, ADR A12).
            ph = f"+1-555-01{min(n - 1, 99):02d}"
        else:
            ph = f"[SECRET:{family}#{n}]"
        self.by_value[key] = ph
        return ph

    def counts(self) -> dict[str, int]:
        return dict(self.counters)


class Redactor:
    """Deterministic capture-time redactor (ADR: redaction_capture.md).

    Satisfies the ``Callable[[ConversationEnvelope], ConversationEnvelope]``
    seam that ``StageStore`` takes; ``redact_with_counts`` additionally
    returns exact substitution counts and the degraded flag.
    """

    def __init__(self, *, extra_patterns: Sequence[str] = ()) -> None:
        self._custom = [re.compile(p) for p in extra_patterns]

    # -- public API --------------------------------------------------------------

    def __call__(self, env: ConversationEnvelope) -> ConversationEnvelope:
        return self.redact_with_counts(env)[0]

    def redact_with_counts(
        self, env: ConversationEnvelope
    ) -> tuple[ConversationEnvelope, dict[str, int], bool]:
        data = json.loads(env.model_dump_json())
        state = _SubState()
        walked = self._walk(data, state, key=None, prefix_only=False)
        try:
            return ConversationEnvelope.model_validate(walked), state.counts(), False
        except ValidationError:
            # Degraded fallback (A3): prefix family + key-aware rule only —
            # these cannot produce parse-breaking rewrites given the
            # structural-key guard. If even this fails, the error propagates
            # as the caller's best-effort staging warning.
            state = _SubState()
            walked = self._walk(data, state, key=None, prefix_only=True)
            return ConversationEnvelope.model_validate(walked), state.counts(), True

    def redact_text(self, text: str) -> str:
        return self._redact_string(text, _SubState(), prefix_only=False)

    # -- walk --------------------------------------------------------------------

    def _walk(self, node: Any, state: _SubState, *, key: Optional[str],
              prefix_only: bool, under_sensitive: bool = False) -> Any:
        if isinstance(node, dict):
            return {
                k: self._walk(
                    v, state, key=k, prefix_only=prefix_only,
                    under_sensitive=under_sensitive or self._is_sensitive_key(k),
                )
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [
                self._walk(v, state, key=key, prefix_only=prefix_only,
                           under_sensitive=under_sensitive)
                for v in node
            ]
        if not isinstance(node, str):
            return node
        return self._walk_string(node, state, key=key, prefix_only=prefix_only,
                                 under_sensitive=under_sensitive)

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        return bool(_SENSITIVE_KEY_RE.search(key.rsplit(".", 1)[-1]))

    def _walk_string(self, value: str, state: _SubState, *, key: Optional[str],
                     prefix_only: bool, under_sensitive: bool) -> str:
        last_segment = (key or "").rsplit(".", 1)[-1]
        if last_segment in STRUCTURAL_KEYS:
            return value
        if _ISO_RE.match(value):
            return value
        # Key-aware rule (A2): the whole value under (or anywhere below) a
        # sensitive key is a secret regardless of shape.
        if under_sensitive or (key is not None and _SENSITIVE_KEY_RE.search(last_segment)):
            if PLACEHOLDER_RE.fullmatch(value):
                return value
            return state.placeholder("context", value, kind="context")
        # Regex-valued check fields: patterns, not data — prefixes only.
        if key is not None and _REGEX_KEY_RE.search(last_segment):
            return self._redact_string(value, state, prefix_only=True)
        return self._redact_string(value, state, prefix_only=prefix_only)

    # -- string substitution -----------------------------------------------------

    def _redact_string(self, text: str, state: _SubState, *,
                       prefix_only: bool) -> str:
        protected = [(m.start(), m.end()) for m in PLACEHOLDER_RE.finditer(text)]

        def overlaps_protected(start: int, end: int) -> bool:
            return any(s < end and start < e for s, e in protected)

        spans: list[tuple[int, int, str]] = []  # (start, end, replacement)

        def claim(start: int, end: int, replacement: str) -> None:
            if overlaps_protected(start, end):
                return
            if any(s < end and start < e for s, e, _ in spans):
                return
            spans.append((start, end, replacement))

        for family, pat in _PREFIX_PATTERNS:
            for m in pat.finditer(text):
                claim(m.start(), m.end(),
                      state.placeholder(family, m.group(0), kind="secret"))

        if not prefix_only:
            for m in _CONTEXT_RE.finditer(text):
                claim(m.start(2), m.end(2),
                      state.placeholder("context", m.group(2), kind="context"))
            for m in _BEARER_RE.finditer(text):
                claim(m.start(1), m.end(1),
                      state.placeholder("context", m.group(1), kind="context"))
            for m in _EMAIL_RE.finditer(text):
                claim(m.start(), m.end(),
                      state.placeholder("email", m.group(0), kind="email"))
            for m in _PHONE_US_RE.finditer(text):
                claim(m.start(), m.end(),
                      state.placeholder("phone", m.group(0), kind="phone"))
            for m in _PHONE_INTL_RE.finditer(text):
                claim(m.start(), m.end(),
                      state.placeholder("phone", m.group(0), kind="phone"))
            for m in _CARD_GROUPED_RE.finditer(text):
                claim(m.start(), m.end(),
                      state.placeholder("card", m.group(0), kind="card"))
            for m in _CARD_BARE_RE.finditer(text):
                if _luhn_ok(m.group(0)):
                    claim(m.start(), m.end(),
                          state.placeholder("card", m.group(0), kind="card"))
            for pat in self._custom:
                for m in pat.finditer(text):
                    claim(m.start(), m.end(),
                          state.placeholder("custom", m.group(0), kind="custom"))

        if not spans:
            return text
        out: list[str] = []
        pos = 0
        for start, end, replacement in sorted(spans):
            out.append(text[pos:start])
            out.append(replacement)
            pos = end
        out.append(text[pos:])
        return "".join(out)


def contains_placeholder(text: str) -> bool:
    """True if the string carries any redaction placeholder — used by verify
    and promote to warn when check literals reference redacted values (D3)."""
    return bool(PLACEHOLDER_RE.search(text))
