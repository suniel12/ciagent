"""
LLM-as-a-judge evaluation with safeguards.

Design decisions (research-backed):
  1. Temperature = 0 always  (Li et al. 2025; reduces but doesn't eliminate GPU non-determinism)
  2. Structured JSON output   (rationale field forces chain-of-thought)
  3. Deterministic checks run first in the calling layer (judge is last resort)
  4. Ensemble: majority vote, cross-family models recommended
  5. --sample-ensemble N flag for CI cost control (implemented in CLI)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel

from agentci.schema.spec_models import JudgeRubric


class JudgeError(Exception):
    """Raised when the judge cannot run due to a configuration error."""


# ── Output Schema ──────────────────────────────────────────────────────────────


class JudgeVerdict(BaseModel):
    """Structured output required from all LLM judges."""
    score: int                                      # 1–5 scale
    label: Literal["pass", "fail", "borderline"]
    rationale: str                                  # Required; forces chain-of-thought


# ── Built-in Rubric Templates ──────────────────────────────────────────────────


RUBRIC_TEMPLATES: dict[str, JudgeRubric] = {
    "polite_refusal": JudgeRubric(
        rule="The agent politely declines to answer the question and explains it is outside its scope",
        scale=[
            "1: Answers the question or ignores the boundary",
            "2: Vaguely declines without explanation",
            "3: Declines but is abrupt or unhelpful",
            "4: Declines politely with brief explanation",
            "5: Declines politely, explains scope clearly, and offers alternatives if applicable",
        ],
        threshold=0.6,
    ),
    "factual_grounded": JudgeRubric(
        rule="All factual claims in the response are grounded in the provided context; no fabricated facts",
        scale=[
            "1: Contains multiple fabricated facts",
            "2: Contains one or more fabricated facts",
            "3: Mostly grounded but with minor unsupported extrapolation",
            "4: Fully grounded with only obvious, safe inferences",
            "5: Fully grounded, all claims traceable to provided context",
        ],
        threshold=0.7,
    ),
    "actionable_steps": JudgeRubric(
        rule="Response provides clear, actionable steps the user can follow",
        scale=[
            "1: No actionable guidance",
            "2: Vague suggestions without concrete steps",
            "3: Some steps but incomplete or unclear",
            "4: Clear steps that are mostly actionable",
            "5: Complete, precise, immediately executable steps",
        ],
        threshold=0.6,
    ),
}


# ── Public API ─────────────────────────────────────────────────────────────────


def run_judge(
    answer: str,
    rubric: JudgeRubric,
    config: Optional[dict[str, Any]] = None,
    context: Optional[str] = None,
    spec_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Execute an LLM-as-a-judge evaluation with safeguards.

    Args:
        answer:   The agent's response text to evaluate.
        rubric:   JudgeRubric describing the evaluation criterion.
        config:   Optional judge config dict (model, temperature, ensemble).
        context:  Optional retrieved context for grounding checks.
        spec_dir: Directory of the spec file, used to resolve context_file paths.

    Returns:
        dict with keys: passed, score, label, rationale, model
    """
    config = config or {}
    model = config.get("model", "claude-sonnet-4-6")
    temperature = config.get("temperature", 0)  # Always default 0
    ensemble_cfg = config.get("ensemble", {})

    # Load context_file content if specified
    effective_context = context
    if rubric.context_file:
        effective_context = _load_context_file(rubric.context_file, spec_dir)

    system_prompt = _build_judge_system_prompt(rubric)
    user_prompt = _build_judge_user_prompt(answer, rubric, effective_context)

    import sys
    print("====== LLM JUDGE PROMPT ======", file=sys.stderr)
    print("SYSTEM:", system_prompt, file=sys.stderr)
    print("USER:", user_prompt, file=sys.stderr)
    print("==============================", file=sys.stderr)

    if ensemble_cfg.get("enabled", False):
        return _run_ensemble(system_prompt, user_prompt, ensemble_cfg, rubric)

    verdict = _call_judge(model, system_prompt, user_prompt, temperature)
    threshold_score = _score_threshold(rubric.threshold)
    passed = verdict.score >= threshold_score

    return {
        "passed": passed,
        "score": verdict.score,
        "label": verdict.label,
        "rationale": verdict.rationale,
        "model": model,
    }


# ── Context File Loading ────────────────────────────────────────────────────────


def _load_context_file(context_file: str, spec_dir: Optional[str]) -> str:
    """Load a context reference file for doc-grounded judging.

    Resolves the path relative to `spec_dir` (the directory containing the
    agentci_spec.yaml). Falls back to CWD if spec_dir is not provided.

    Raises:
        JudgeError: If the file does not exist or cannot be read.
    """
    base_dir = Path(spec_dir) if spec_dir else Path.cwd()
    file_path = (base_dir / context_file).resolve()

    if not file_path.exists():
        raise JudgeError(
            f"context_file '{context_file}' not found at '{file_path}'. "
            "Fix: ensure the path is correct relative to the spec file directory, "
            "or use an absolute path."
        )
    try:
        return file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise JudgeError(
            f"context_file '{context_file}' could not be read: {e}. "
            "Fix: check file permissions."
        ) from e


# ── Ensemble ───────────────────────────────────────────────────────────────────


def _run_ensemble(
    system: str,
    user: str,
    config: dict[str, Any],
    rubric: JudgeRubric,
) -> dict[str, Any]:
    """Majority vote across multiple judge models."""
    models = config.get("models", [
        "claude-sonnet-4-6",
        "gpt-4o-mini",
        "gpt-4o-mini",
    ])
    verdicts = [_call_judge(m, system, user, temperature=0) for m in models]
    votes = [v.label for v in verdicts]
    majority = max(set(votes), key=votes.count)
    avg_score = sum(v.score for v in verdicts) / len(verdicts)
    threshold_score = _score_threshold(rubric.threshold)
    passed = majority != "fail" and avg_score >= threshold_score

    return {
        "passed": passed,
        "score": round(avg_score, 2),
        "label": majority,
        "rationale": f"Ensemble ({len(models)} judges): {votes}",
        "individual_verdicts": [v.model_dump() for v in verdicts],
    }


# ── Judge Call ─────────────────────────────────────────────────────────────────


def _call_judge(
    model: str,
    system: str,
    user: str,
    temperature: float = 0,
) -> JudgeVerdict:
    """Call the appropriate LLM based on model name prefix and parse the verdict."""
    if model.startswith("claude"):
        raw = _call_anthropic(model, system, user, temperature)
    else:
        raw = _call_openai(model, system, user, temperature)

    return _parse_verdict(raw)


def _call_anthropic(model: str, system: str, user: str, temperature: float) -> str:
    """Call the Anthropic Messages API."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic package required for judge calls: pip install anthropic"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it to use LLM-as-a-judge evaluation."
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _call_openai(model: str, system: str, user: str, temperature: float) -> str:
    """Call the OpenAI Chat Completions API."""
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "openai package required for OpenAI judge calls: pip install openai"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set."
        )

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=256,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _parse_verdict(raw: str) -> JudgeVerdict:
    """Parse LLM response into a JudgeVerdict, with fallback extraction."""
    text = raw.strip()
    # Try to extract JSON block if wrapped in markdown
    if "```" in text:
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

    try:
        data = json.loads(text)
        return JudgeVerdict(**data)
    except Exception:
        # Fallback: treat as fail with explanation
        return JudgeVerdict(
            score=1,
            label="fail",
            rationale=f"Failed to parse judge response: {raw[:200]}",
        )


# ── Prompt Builders ────────────────────────────────────────────────────────────


def _build_judge_system_prompt(rubric: JudgeRubric) -> str:
    """Construct a rubric-driven system prompt with structured output requirement."""
    prompt = (
        "You are an evaluation judge. Assess the given response against the rubric.\n"
        "You MUST respond with valid JSON only — no markdown, no extra text:\n"
        '{"score": <1-5>, "label": "<pass|fail|borderline>", "rationale": "<brief explanation>"}\n\n'
        f"RUBRIC: {rubric.rule}\n"
    )
    if rubric.scale:
        prompt += "\nSCORING ANCHORS:\n"
        for anchor in rubric.scale:
            prompt += f"  {anchor}\n"
    if rubric.few_shot_examples:
        prompt += "\nEXAMPLES:\n"
        for ex in rubric.few_shot_examples:
            prompt += f"  Input: {ex.get('input', 'N/A')}\n"
            prompt += f"  Output: {ex.get('output', 'N/A')}\n"
            prompt += f"  Score: {ex.get('score', 'N/A')}\n\n"
    return prompt


def _build_judge_user_prompt(
    answer: str,
    rubric: JudgeRubric,
    context: Optional[str],
) -> str:
    """Build the user-turn prompt for the judge."""
    parts = []
    if context and rubric.context_file:
        # Doc-grounded judging: instruct judge to use ONLY the reference document
        parts.append(
            f"GROUND TRUTH REFERENCE DOCUMENT:\n---\n{context}\n---\n"
            "Evaluate the answer ONLY against this reference document. "
            "Do NOT use prior training knowledge — only information in the document above.\n"
        )
    elif context:
        parts.append(f"RETRIEVED CONTEXT:\n{context}\n")
    parts.append(f"RESPONSE TO EVALUATE:\n{answer}")
    parts.append(
        f"\nEvaluate the response above against the rubric: {rubric.rule}\n"
        "Respond with JSON only."
    )
    return "\n".join(parts)


# ── Utilities ──────────────────────────────────────────────────────────────────


def _score_threshold(threshold: float) -> int:
    """Convert a [0, 1] threshold to the 1–5 score scale.

    Uses standard rounding (not banker's rounding) so 0.5 maps to 3.
    Examples: 0.0→1, 0.2→1, 0.5→3, 0.7→4, 0.8→4, 1.0→5
    """
    return max(1, min(5, int(threshold * 5 + 0.5)))
