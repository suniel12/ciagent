"""
Unit tests for agentci.engine.judge.

All LLM API calls are mocked — no real network requests made here.
See tests/integration/test_judge_live.py for real API tests.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agentci.engine.judge import (
    JudgeError,
    JudgeVerdict,
    _build_judge_system_prompt,
    _build_judge_user_prompt,
    _load_context_file,
    _parse_verdict,
    _run_ensemble,
    _score_threshold,
    run_judge,
)
from agentci.schema.spec_models import JudgeRubric


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_rubric(**kwargs) -> JudgeRubric:
    base = {"rule": "Response is helpful and accurate", "threshold": 0.6}
    base.update(kwargs)
    return JudgeRubric(**base)


def make_verdict(score: int = 4, label: str = "pass", rationale: str = "Good") -> JudgeVerdict:
    return JudgeVerdict(score=score, label=label, rationale=rationale)


# ── _score_threshold ──────────────────────────────────────────────────────────


class TestScoreThreshold:
    def test_zero_maps_to_one(self):
        assert _score_threshold(0.0) == 1

    def test_half_maps_to_three(self):
        assert _score_threshold(0.5) == 3

    def test_point_eight_maps_to_four(self):
        assert _score_threshold(0.8) == 4

    def test_one_maps_to_five(self):
        assert _score_threshold(1.0) == 5

    def test_point_two_maps_to_one(self):
        assert _score_threshold(0.2) == 1

    def test_point_six_maps_to_three(self):
        assert _score_threshold(0.6) == 3

    def test_point_seven_maps_to_four(self):
        assert _score_threshold(0.7) == 4


# ── _build_judge_system_prompt ────────────────────────────────────────────────


class TestBuildSystemPrompt:
    def test_contains_rubric_rule(self):
        rubric = make_rubric(rule="Check for accuracy")
        prompt = _build_judge_system_prompt(rubric)
        assert "Check for accuracy" in prompt

    def test_requires_json_output(self):
        rubric = make_rubric()
        prompt = _build_judge_system_prompt(rubric)
        assert "score" in prompt and "label" in prompt and "rationale" in prompt

    def test_includes_scale_anchors(self):
        rubric = make_rubric(scale=["1: Bad", "5: Perfect"])
        prompt = _build_judge_system_prompt(rubric)
        assert "1: Bad" in prompt
        assert "5: Perfect" in prompt

    def test_includes_few_shot_examples(self):
        rubric = make_rubric(
            few_shot_examples=[{"input": "q", "output": "a", "score": 4}]
        )
        prompt = _build_judge_system_prompt(rubric)
        assert "EXAMPLES" in prompt
        assert "score: 4" in prompt or "4" in prompt

    def test_no_scale_uses_default_anchors(self):
        rubric = make_rubric()
        prompt = _build_judge_system_prompt(rubric)
        assert "SCORING ANCHORS" in prompt
        assert "Completely fails to address" in prompt
        assert "Perfectly addresses all rubric" in prompt
        assert "EXAMPLES" not in prompt


# ── _build_judge_user_prompt ──────────────────────────────────────────────────


class TestBuildUserPrompt:
    def test_contains_answer(self):
        rubric = make_rubric()
        prompt = _build_judge_user_prompt("My answer here", rubric, context=None)
        assert "My answer here" in prompt

    def test_contains_context_when_provided(self):
        rubric = make_rubric()
        prompt = _build_judge_user_prompt("answer", rubric, context="Retrieved doc")
        assert "Retrieved doc" in prompt
        assert "RETRIEVED CONTEXT" in prompt

    def test_no_context_section_when_none(self):
        rubric = make_rubric()
        prompt = _build_judge_user_prompt("answer", rubric, context=None)
        assert "RETRIEVED CONTEXT" not in prompt

    def test_includes_query_when_provided(self):
        rubric = make_rubric()
        prompt = _build_judge_user_prompt("answer", rubric, context=None, query="What is AgentCI?")
        assert "USER QUERY:" in prompt
        assert "What is AgentCI?" in prompt

    def test_no_query_section_when_none(self):
        rubric = make_rubric()
        prompt = _build_judge_user_prompt("answer", rubric, context=None, query=None)
        assert "USER QUERY" not in prompt


# ── _parse_verdict ────────────────────────────────────────────────────────────


class TestParseVerdict:
    def test_parses_valid_json(self):
        raw = '{"score": 4, "label": "pass", "rationale": "Good response"}'
        verdict = _parse_verdict(raw)
        assert verdict.score == 4
        assert verdict.label == "pass"
        assert verdict.rationale == "Good response"

    def test_parses_json_in_markdown_block(self):
        raw = '```json\n{"score": 3, "label": "borderline", "rationale": "OK"}\n```'
        verdict = _parse_verdict(raw)
        assert verdict.score == 3

    def test_parses_json_with_preamble_text(self):
        """Strategy 3: JSON object with preamble text before it."""
        raw = 'Here is my evaluation:\n{"score": 5, "label": "pass", "rationale": "Excellent"}'
        verdict = _parse_verdict(raw)
        assert verdict.score == 5
        assert verdict.label == "pass"

    def test_parses_truncated_json_via_regex(self):
        """Strategy 4: Regex extraction from truncated/malformed JSON."""
        raw = '{"score": 4, "label": "pass", "rationale": "Good answer but'
        verdict = _parse_verdict(raw)
        assert verdict.score == 4
        assert verdict.label == "pass"

    def test_regex_fallback_infers_label_from_score(self):
        """Strategy 4: When label is missing, infer from score."""
        raw = '{"score": 2, "rationale": "Poor response'
        verdict = _parse_verdict(raw)
        assert verdict.score == 2
        assert verdict.label == "fail"  # score < 3 → fail

    def test_regex_fallback_infers_pass_from_high_score(self):
        """Strategy 4: High score without label infers pass."""
        raw = '{"score": 4, "rationale": "Solid answer'
        verdict = _parse_verdict(raw)
        assert verdict.score == 4
        assert verdict.label == "pass"  # score >= 3 → pass

    def test_fallback_on_invalid_json(self):
        verdict = _parse_verdict("This is not JSON at all")
        assert verdict.label == "fail"
        assert verdict.score == 1
        assert "Failed to parse" in verdict.rationale


# ── run_judge ─────────────────────────────────────────────────────────────────


class TestRunJudge:
    def _mock_call_judge(self, verdict: JudgeVerdict):
        """Return a context manager that patches _call_judge."""
        return patch(
            "agentci.engine.judge._call_judge",
            return_value=verdict,
        )

    def test_passes_when_score_meets_threshold(self):
        rubric = make_rubric(threshold=0.6)  # threshold score = 3
        verdict = make_verdict(score=4, label="pass")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is True
        assert result["score"] == 4

    def test_fails_when_score_below_threshold(self):
        rubric = make_rubric(threshold=0.8)  # threshold score = 4
        verdict = make_verdict(score=2, label="fail")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is False

    def test_fails_when_score_equals_threshold_minus_one(self):
        rubric = make_rubric(threshold=0.6)  # threshold = 3
        verdict = make_verdict(score=2, label="fail")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is False

    def test_passes_when_score_exactly_at_threshold(self):
        rubric = make_rubric(threshold=0.6)  # threshold = 3
        verdict = make_verdict(score=3, label="pass")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is True

    def test_result_contains_expected_keys(self):
        rubric = make_rubric()
        verdict = make_verdict(score=4)
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert "passed" in result
        assert "score" in result
        assert "label" in result
        assert "rationale" in result
        assert "model" in result

    def test_uses_default_model(self):
        rubric = make_rubric()
        verdict = make_verdict(score=4)
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["model"] == "claude-sonnet-4-6"

    def test_uses_custom_model_from_config(self):
        rubric = make_rubric()
        verdict = make_verdict(score=4)
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric, config={"model": "gpt-4o-mini"})
        assert result["model"] == "gpt-4o-mini"

    def test_ensemble_not_triggered_by_default(self):
        rubric = make_rubric()
        verdict = make_verdict(score=4)
        with self._mock_call_judge(verdict) as mock:
            run_judge("answer", rubric)
        mock.assert_called_once()  # Only one call, no ensemble

    def test_label_pass_overrides_low_score(self):
        """Judge says 'pass' but score is below threshold — label wins."""
        rubric = make_rubric(threshold=0.8)  # threshold score = 4
        verdict = make_verdict(score=3, label="pass")  # score < threshold
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is True  # label "pass" overrides score

    def test_label_fail_with_high_score_still_fails(self):
        """Judge says 'fail' with high score — label fail + score below threshold = fail."""
        rubric = make_rubric(threshold=0.8)  # threshold score = 4
        verdict = make_verdict(score=2, label="fail")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is False

    def test_borderline_label_relies_on_score(self):
        """Borderline label — pass/fail depends on score vs threshold."""
        rubric = make_rubric(threshold=0.6)  # threshold score = 3
        verdict = make_verdict(score=4, label="borderline")
        with self._mock_call_judge(verdict):
            result = run_judge("answer", rubric)
        assert result["passed"] is True  # score 4 >= threshold 3


# ── _run_ensemble ─────────────────────────────────────────────────────────────


class TestRunEnsemble:
    def _ensemble_config(self, models=None):
        return {
            "enabled": True,
            "models": models or ["model-a", "model-b", "model-c"],
            "strategy": "majority_vote",
        }

    def test_majority_pass(self):
        rubric = make_rubric(threshold=0.6)  # threshold score = 3
        verdicts = [
            make_verdict(score=4, label="pass"),
            make_verdict(score=4, label="pass"),
            make_verdict(score=2, label="fail"),
        ]
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts):
            result = _run_ensemble("sys", "user", self._ensemble_config(), rubric)
        assert result["passed"] is True
        assert result["label"] == "pass"

    def test_majority_fail(self):
        rubric = make_rubric(threshold=0.6)
        verdicts = [
            make_verdict(score=2, label="fail"),
            make_verdict(score=2, label="fail"),
            make_verdict(score=4, label="pass"),
        ]
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts):
            result = _run_ensemble("sys", "user", self._ensemble_config(), rubric)
        assert result["passed"] is False
        assert result["label"] == "fail"

    def test_ensemble_returns_avg_score(self):
        rubric = make_rubric(threshold=0.4)
        verdicts = [
            make_verdict(score=4, label="pass"),
            make_verdict(score=2, label="fail"),
            make_verdict(score=3, label="borderline"),
        ]
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts):
            result = _run_ensemble("sys", "user", self._ensemble_config(), rubric)
        assert result["score"] == pytest.approx(3.0)

    def test_ensemble_result_contains_individual_verdicts(self):
        rubric = make_rubric()
        verdicts = [make_verdict(score=4), make_verdict(score=3), make_verdict(score=4)]
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts):
            result = _run_ensemble("sys", "user", self._ensemble_config(), rubric)
        assert "individual_verdicts" in result
        assert len(result["individual_verdicts"]) == 3

    def test_ensemble_calls_all_models(self):
        rubric = make_rubric()
        models = ["model-x", "model-y", "model-z"]
        verdicts = [make_verdict(score=4)] * 3
        with patch("agentci.engine.judge._call_judge", side_effect=verdicts) as mock:
            _run_ensemble("sys", "user", self._ensemble_config(models), rubric)
        assert mock.call_count == 3


# ── context_file (Milestone 3.4) ──────────────────────────────────────────────


class TestContextFile:
    """Tests for context_file loading and doc-grounded prompt injection."""

    def _make_rubric_with_context(self, context_file: str) -> JudgeRubric:
        return JudgeRubric(
            rule="Answer matches the reference document",
            threshold=0.8,
            context_file=context_file,
        )

    def test_context_file_content_injected_into_prompt(self):
        """When context_file is set, its content appears in the judge prompt."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("The SLA uptime guarantee is 99.9%.\n")
            tmp_path = f.name

        try:
            rubric = self._make_rubric_with_context(tmp_path)
            verdict = make_verdict(score=4, label="pass")
            with patch("agentci.engine.judge._call_judge", return_value=verdict) as mock_call:
                run_judge("The uptime is 99.9%.", rubric, spec_dir=None)

            # Inspect the user prompt passed to _call_judge
            call_args = mock_call.call_args
            user_prompt = call_args[0][2]  # positional: model, system, user
            assert "99.9%" in user_prompt
            assert "GROUND TRUTH REFERENCE DOCUMENT" in user_prompt
        finally:
            os.unlink(tmp_path)

    def test_context_file_none_no_change_to_behavior(self):
        """When context_file is None, existing behavior is unchanged."""
        rubric = make_rubric()  # no context_file
        verdict = make_verdict(score=4)
        with patch("agentci.engine.judge._call_judge", return_value=verdict) as mock_call:
            run_judge("Some answer", rubric)
        user_prompt = mock_call.call_args[0][2]
        assert "GROUND TRUTH REFERENCE DOCUMENT" not in user_prompt

    def test_context_file_missing_raises_judge_error(self):
        """When context_file does not exist, JudgeError is raised."""
        rubric = self._make_rubric_with_context("/nonexistent/path/file.md")
        with pytest.raises(JudgeError, match="not found"):
            run_judge("Any answer", rubric, spec_dir=None)

    def test_context_file_resolved_relative_to_spec_dir(self):
        """context_file path is resolved relative to spec_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_file = os.path.join(tmpdir, "sla.md")
            with open(ref_file, "w") as f:
                f.write("SLA content here.")

            rubric = self._make_rubric_with_context("sla.md")
            verdict = make_verdict(score=4, label="pass")
            with patch("agentci.engine.judge._call_judge", return_value=verdict) as mock_call:
                run_judge("SLA content here.", rubric, spec_dir=tmpdir)
            user_prompt = mock_call.call_args[0][2]
            assert "SLA content here." in user_prompt

    def test_load_context_file_directly_raises_judge_error(self):
        """_load_context_file raises JudgeError with actionable message on missing file."""
        with pytest.raises(JudgeError) as exc_info:
            _load_context_file("does_not_exist.md", spec_dir=None)
        assert "not found" in str(exc_info.value)
        assert "Fix:" in str(exc_info.value)

    def test_doc_grounded_prompt_excludes_prior_knowledge_instruction(self):
        """Doc-grounded prompt includes instruction to ignore prior knowledge."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("Pricing: $29/month for Pro plan.")
            tmp_path = f.name

        try:
            rubric = self._make_rubric_with_context(tmp_path)
            verdict = make_verdict(score=4)
            with patch("agentci.engine.judge._call_judge", return_value=verdict) as mock_call:
                run_judge("Pro plan costs $29/month.", rubric, spec_dir=None)
            user_prompt = mock_call.call_args[0][2]
            assert "prior training knowledge" in user_prompt or "prior knowledge" in user_prompt
        finally:
            os.unlink(tmp_path)
