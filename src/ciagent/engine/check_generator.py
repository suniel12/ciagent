# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Check Generator — KB-derived deterministic fact checks.

Mines the knowledge base and golden answers for hard facts and emits them as
deterministic assertions, reserving the LLM judge for answers with nothing
checkable. The generation step uses an LLM once, at authoring time; the
output is deterministic checks that run forever at zero cost.

A tool whose pitch is "stop your evals from lying to you" must not generate
brittle string checks that fail correct paraphrases. Three design rules:

  1. Only extract facts that don't paraphrase — prices, rates, SKUs, codes,
     version numbers, explicit policy numbers. Prose facts become variant
     sets (`any_expected_in_answer`) or regex alternations, never a single
     literal string.
  2. Every candidate is VALIDATED against known-good answers (golden
     baselines) before it is offered. A check that fails a known-good answer
     is rejected automatically — the gate, not the user, absorbs bad
     extractions.
  3. Nothing is written silently: candidates that survive the gate still go
     through interactive review (or an explicit --yes).

Merging never overwrites user-written assertions — only empty fields are
filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import yaml

from ciagent.schema.spec_models import AgentCISpec, CorrectnessSpec

LlmFn = Callable[[str], str]

# Fields generate-checks is allowed to propose
_ALLOWED_FIELDS = ("any_expected_in_answer", "not_in_answer", "regex_match")

# Bound the KB excerpt sent to the extraction model
_MAX_KB_CHARS = 24_000
_MAX_FILE_CHARS = 6_000


@dataclass
class CandidateCheck:
    """One proposed deterministic assertion for one query."""
    query: str
    field: str                      # any_expected_in_answer | not_in_answer | regex_match
    value: Any                      # list[str] for keyword fields, str for regex
    fact: str = ""                  # the KB fact this check pins, human-readable
    status: str = "pending"         # validated | unvalidated | rejected
    reason: str = ""                # why rejected / why unvalidated


@dataclass
class GenerationResult:
    candidates: list[CandidateCheck] = field(default_factory=list)

    @property
    def validated(self) -> list[CandidateCheck]:
        return [c for c in self.candidates if c.status == "validated"]

    @property
    def unvalidated(self) -> list[CandidateCheck]:
        return [c for c in self.candidates if c.status == "unvalidated"]

    @property
    def rejected(self) -> list[CandidateCheck]:
        return [c for c in self.candidates if c.status == "rejected"]


_EXTRACTION_PROMPT = """\
You extract HARD FACTS from a knowledge base to create deterministic test
assertions for an AI agent's answers. For each test query below, find facts
in the KB that a correct answer MUST contain (or must NOT contain).

STRICT RULES — the checks must survive correct paraphrasing:
1. ONLY extract facts that don't paraphrase: prices, rates, percentages,
   SKUs, product codes, version numbers, explicit quantities ("30 days",
   "3 business days"). A number is a number in any phrasing.
2. For a fact that can be phrased multiple ways, give 2-4 short variants as
   any_expected_in_answer (ANY one matching passes). Each variant must be a
   short substring likely to appear verbatim, not a sentence.
3. Use regex_match ONLY for genuinely patterned facts (e.g. an order-ID
   format). Keep patterns simple.
4. Use not_in_answer ONLY for facts the KB explicitly contradicts (e.g. a
   discontinued product the agent must not claim to sell).
5. If a query has NO checkable hard fact in the KB, return an empty list for
   it — do NOT invent soft checks. Judgment belongs to the judge.
6. Never propose a check for subjective qualities (tone, helpfulness).

KNOWLEDGE BASE:
{kb}

QUERIES:
{queries}

Respond ONLY with valid YAML — a list of objects:
- query: "<exact query text from the list>"
  checks:
    - field: any_expected_in_answer
      value: ["4.5%", "4.5 percent"]
      fact: "APR rate is 4.5%"
"""


def extract_candidates(
    spec: AgentCISpec,
    kb_text: str,
    llm_fn: LlmFn,
) -> GenerationResult:
    """Ask the LLM to mine candidate checks for every spec query.

    Defensive parse: anything malformed is dropped silently rather than
    crashing — the validation gate and review still stand between the model
    and the spec file.
    """
    queries_block = "\n".join(f"- {q.query}" for q in spec.queries)
    prompt = _EXTRACTION_PROMPT.format(kb=kb_text, queries=queries_block)
    raw = llm_fn(prompt)

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return GenerationResult()
    if not isinstance(parsed, list):
        return GenerationResult()

    known_queries = {q.query for q in spec.queries}
    result = GenerationResult()
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        query = entry.get("query")
        if query not in known_queries:
            continue
        for chk in entry.get("checks") or []:
            if not isinstance(chk, dict):
                continue
            fld = chk.get("field")
            value = chk.get("value")
            if fld not in _ALLOWED_FIELDS or value in (None, "", []):
                continue
            if fld in ("any_expected_in_answer", "not_in_answer"):
                if isinstance(value, str):
                    value = [value]
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    continue
                value = [v for v in (s.strip() for s in value) if v]
                if not value:
                    continue
            elif fld == "regex_match" and not isinstance(value, str):
                continue
            result.candidates.append(
                CandidateCheck(
                    query=str(query), field=str(fld), value=value,
                    fact=str(chk.get("fact") or ""),
                )
            )
    return result


def validate_candidates(
    result: GenerationResult,
    known_good_answers: dict[str, list[str]],
) -> None:
    """Gate every candidate against known-good answers (in place).

    - A check that FAILS any known-good answer is rejected: it would have
      failed a correct answer, which is exactly the brittleness we refuse
      to ship.
    - An invalid regex is rejected outright.
    - A candidate whose query has no recorded known-good answer cannot be
      gated — marked `unvalidated`, applied only with explicit user consent.

    The gate itself lives in engine/artifact_gate.py, shared with F6's
    conversation-golden gate and F7's import round-trip gate.
    """
    from ciagent.engine.artifact_gate import gate_candidate_check

    for cand in result.candidates:
        gate = gate_candidate_check(
            cand.field, cand.value, known_good_answers.get(cand.query) or [],
        )
        if gate.accepted:
            cand.status = "validated"
            cand.reason = ""
        else:
            # gate statuses map 1:1 onto candidate statuses
            cand.status = "rejected" if gate.rejected else "unvalidated"
            cand.reason = "; ".join(gate.reasons)


def merge_candidates(
    spec: AgentCISpec,
    accepted: list[CandidateCheck],
) -> tuple[AgentCISpec, list[str]]:
    """Apply accepted candidates to a copy of the spec.

    Only fills fields the user left empty — an existing user-written
    assertion is never overwritten (a skipped merge is reported).

    Returns (updated spec, human-readable change log).
    """
    changes: list[str] = []
    updated = spec.model_copy(deep=True)
    by_query = {q.query: q for q in updated.queries}

    for cand in accepted:
        gq = by_query.get(cand.query)
        if gq is None:
            continue
        if gq.correctness is None:
            gq.correctness = CorrectnessSpec()
        current = getattr(gq.correctness, cand.field, None)
        if cand.field in ("any_expected_in_answer", "not_in_answer"):
            existing = list(current or [])
            new_terms = [v for v in cand.value if v.lower() not in {e.lower() for e in existing}]
            if not new_terms:
                continue
            if existing:
                setattr(gq.correctness, cand.field, existing + new_terms)
                changes.append(
                    f"{cand.query[:50]!r}: appended {new_terms} to {cand.field}"
                )
            else:
                setattr(gq.correctness, cand.field, new_terms)
                changes.append(f"{cand.query[:50]!r}: set {cand.field} = {new_terms}")
        else:  # regex_match — scalar, never overwrite
            if current is not None:
                changes.append(
                    f"{cand.query[:50]!r}: SKIPPED regex_match (user already set one)"
                )
                continue
            gq.correctness.regex_match = cand.value
            changes.append(f"{cand.query[:50]!r}: set regex_match = {cand.value!r}")

    return updated, changes


def collect_kb_text(kb_dir: str) -> str:
    """Concatenate KB markdown/text files into a bounded excerpt."""
    from pathlib import Path

    chunks: list[str] = []
    total = 0
    for f in sorted(Path(kb_dir).rglob("*")):
        if f.suffix.lower() not in (".md", ".txt") or not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")[:_MAX_FILE_CHARS]
        except OSError:
            continue
        header = f"--- {f} ---\n"
        if total + len(header) + len(text) > _MAX_KB_CHARS:
            break
        chunks.append(header + text)
        total += len(header) + len(text)
    return "\n\n".join(chunks)


def default_llm(prompt: str) -> str:
    """One-shot completion using whichever provider key is available."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required: pip install ciagent[anthropic]"
            ) from e
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai package required: pip install ciagent[openai]") from e
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    raise RuntimeError(
        "generate-checks needs an LLM for the one-time extraction step. "
        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY. (The generated checks "
        "themselves run deterministically, forever, with no keys.)"
    )


