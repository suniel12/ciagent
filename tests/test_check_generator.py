# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the KB-derived check generator.

The extraction LLM is injected as a fake, so every path — parsing, the
brittleness validation gate, and the never-overwrite merge — runs with zero
API calls. The gate is the feature's real defense: a candidate that fails a
known-good answer must die before the user ever sees it.
"""

from __future__ import annotations

import yaml

from agentci.engine.check_generator import (
    CandidateCheck,
    GenerationResult,
    collect_kb_text,
    extract_candidates,
    merge_candidates,
    validate_candidates,
)
from agentci.schema.spec_models import AgentCISpec, CorrectnessSpec, GoldenQuery


def make_spec(*queries: GoldenQuery) -> AgentCISpec:
    return AgentCISpec(agent="gen-test", queries=list(queries))


def fake_llm(payload):
    """Build an llm_fn returning the given payload as YAML."""
    text = yaml.safe_dump(payload)
    return lambda prompt: text


# ── Extraction parsing ─────────────────────────────────────────────────────────


class TestExtraction:
    def test_parses_valid_candidates(self):
        spec = make_spec(GoldenQuery(query="what rate?"))
        llm = fake_llm([{
            "query": "what rate?",
            "checks": [
                {"field": "any_expected_in_answer", "value": ["4.5%", "4.5 percent"],
                 "fact": "APR is 4.5%"},
            ],
        }])
        result = extract_candidates(spec, "kb text", llm)
        assert len(result.candidates) == 1
        c = result.candidates[0]
        assert c.field == "any_expected_in_answer"
        assert c.value == ["4.5%", "4.5 percent"]
        assert c.fact == "APR is 4.5%"

    def test_string_value_coerced_to_list_for_keyword_fields(self):
        spec = make_spec(GoldenQuery(query="q"))
        llm = fake_llm([{"query": "q", "checks": [
            {"field": "not_in_answer", "value": "discontinued"},
        ]}])
        result = extract_candidates(spec, "kb", llm)
        assert result.candidates[0].value == ["discontinued"]

    def test_unknown_query_and_field_dropped(self):
        spec = make_spec(GoldenQuery(query="real query"))
        llm = fake_llm([
            {"query": "hallucinated query", "checks": [
                {"field": "any_expected_in_answer", "value": ["x"]}]},
            {"query": "real query", "checks": [
                {"field": "expected_in_answer", "value": ["x"]},  # not allowed (AND logic too brittle)
                {"field": "llm_judge", "value": ["x"]},
            ]},
        ])
        result = extract_candidates(spec, "kb", llm)
        assert result.candidates == []

    def test_garbage_yaml_returns_empty(self):
        spec = make_spec(GoldenQuery(query="q"))
        result = extract_candidates(spec, "kb", lambda p: "not: [valid: yaml")
        assert result.candidates == []

    def test_code_fences_stripped(self):
        spec = make_spec(GoldenQuery(query="q"))
        payload = yaml.safe_dump([{"query": "q", "checks": [
            {"field": "regex_match", "value": r"\d+ days"},
        ]}])
        result = extract_candidates(spec, "kb", lambda p: f"```yaml\n{payload}\n```")
        assert len(result.candidates) == 1


# ── The validation gate ────────────────────────────────────────────────────────


class TestValidationGate:
    def test_check_failing_known_good_answer_is_rejected(self):
        # The golden answer paraphrases ("4.5 percent") — a literal "4.5%"
        # check would fail correct output. The gate must catch it.
        result = GenerationResult(candidates=[
            CandidateCheck(query="q", field="any_expected_in_answer", value=["4.5%"]),
        ])
        validate_candidates(result, {"q": ["Our APR is 4.5 percent for all loans."]})
        assert result.candidates[0].status == "rejected"
        assert "known-good" in result.candidates[0].reason

    def test_check_passing_known_good_answer_is_validated(self):
        result = GenerationResult(candidates=[
            CandidateCheck(query="q", field="any_expected_in_answer",
                           value=["4.5%", "4.5 percent"]),
        ])
        validate_candidates(result, {"q": ["Our APR is 4.5 percent for all loans."]})
        assert result.candidates[0].status == "validated"

    def test_not_in_answer_found_in_good_answer_is_rejected(self):
        # The golden answer legitimately mentions the term → forbidding it
        # would fail correct output.
        result = GenerationResult(candidates=[
            CandidateCheck(query="q", field="not_in_answer", value=["gift cards"]),
        ])
        validate_candidates(result, {"q": ["We stopped selling gift cards in 2024."]})
        assert result.candidates[0].status == "rejected"

    def test_invalid_regex_rejected(self):
        result = GenerationResult(candidates=[
            CandidateCheck(query="q", field="regex_match", value="([unclosed"),
        ])
        validate_candidates(result, {"q": ["anything"]})
        assert result.candidates[0].status == "rejected"
        assert "invalid regex" in result.candidates[0].reason

    def test_no_known_good_answer_is_unvalidated_not_validated(self):
        result = GenerationResult(candidates=[
            CandidateCheck(query="q", field="any_expected_in_answer", value=["x"]),
        ])
        validate_candidates(result, {})
        assert result.candidates[0].status == "unvalidated"

    def test_gate_checks_every_answer(self):
        # Passes the first known-good answer, fails the second → rejected
        result = GenerationResult(candidates=[
            CandidateCheck(query="q", field="any_expected_in_answer", value=["30 days"]),
        ])
        validate_candidates(result, {"q": [
            "Returns accepted within 30 days.",
            "You have one month (thirty days) to return items.",
        ]})
        assert result.candidates[0].status == "rejected"


# ── Merge: never overwrite the user ────────────────────────────────────────────


class TestMerge:
    def test_fills_empty_field(self):
        spec = make_spec(GoldenQuery(query="q"))
        updated, changes = merge_candidates(spec, [
            CandidateCheck(query="q", field="any_expected_in_answer", value=["4.5%"]),
        ])
        assert updated.queries[0].correctness.any_expected_in_answer == ["4.5%"]
        assert len(changes) == 1
        # original untouched
        assert spec.queries[0].correctness is None

    def test_appends_to_existing_keyword_list_without_dupes(self):
        spec = make_spec(GoldenQuery(
            query="q",
            correctness=CorrectnessSpec(any_expected_in_answer=["4.5%"]),
        ))
        updated, changes = merge_candidates(spec, [
            CandidateCheck(query="q", field="any_expected_in_answer",
                           value=["4.5%", "4.5 percent"]),
        ])
        assert updated.queries[0].correctness.any_expected_in_answer == ["4.5%", "4.5 percent"]
        assert "appended" in changes[0]

    def test_never_overwrites_user_regex(self):
        spec = make_spec(GoldenQuery(
            query="q", correctness=CorrectnessSpec(regex_match=r"user pattern"),
        ))
        updated, changes = merge_candidates(spec, [
            CandidateCheck(query="q", field="regex_match", value=r"\d+"),
        ])
        assert updated.queries[0].correctness.regex_match == "user pattern"
        assert "SKIPPED" in changes[0]

    def test_duplicate_terms_produce_no_change(self):
        spec = make_spec(GoldenQuery(
            query="q", correctness=CorrectnessSpec(any_expected_in_answer=["4.5%"]),
        ))
        _, changes = merge_candidates(spec, [
            CandidateCheck(query="q", field="any_expected_in_answer", value=["4.5%"]),
        ])
        assert changes == []


# ── KB collection ──────────────────────────────────────────────────────────────


class TestKBCollection:
    def test_collects_md_and_txt_only(self, tmp_path):
        (tmp_path / "a.md").write_text("rate is 4.5%")
        (tmp_path / "b.txt").write_text("30 day returns")
        (tmp_path / "c.py").write_text("code = True")
        text = collect_kb_text(str(tmp_path))
        assert "4.5%" in text and "30 day" in text and "code = True" not in text

    def test_empty_dir_returns_empty(self, tmp_path):
        assert collect_kb_text(str(tmp_path)) == ""


# ── CLI integration ────────────────────────────────────────────────────────────


class TestCLIGenerateChecks:
    @staticmethod
    def _project(tmp_path, golden_answer="Our APR is 4.5 percent."):
        import json

        spec = tmp_path / "agentci_spec.yaml"
        spec.write_text(
            """
agent: gen-cli-test
baseline_dir: ./golden
queries:
  - query: "what rate do you charge?"
"""
        )
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "rates.md").write_text("Our standard APR is 4.5% (4.5 percent).")
        golden = tmp_path / "golden"
        golden.mkdir()
        (golden / "rate.golden.json").write_text(json.dumps({
            "spans": [{"kind": "agent", "name": "a", "output_data": ""}],
            "test_name": "what rate do you charge?",
            "metadata": {"final_output": golden_answer},
        }))
        return tmp_path

    @staticmethod
    def _fake_extraction(monkeypatch, checks):
        payload = yaml.safe_dump([{"query": "what rate do you charge?", "checks": checks}])
        monkeypatch.setattr(
            "agentci.engine.check_generator.default_llm", lambda prompt: payload,
        )

    def test_yes_applies_validated_checks(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from agentci.cli import cli

        project = self._project(tmp_path)
        self._fake_extraction(monkeypatch, [
            {"field": "any_expected_in_answer", "value": ["4.5%", "4.5 percent"],
             "fact": "APR is 4.5%"},
        ])
        monkeypatch.chdir(project)
        result = CliRunner().invoke(
            cli, ["generate-checks", "--kb", str(project / "kb"), "--yes"],
        )
        assert result.exit_code == 0, result.output
        assert "gate passed" in result.output
        updated = yaml.safe_load((project / "agentci_spec.yaml").read_text())
        assert updated["queries"][0]["correctness"]["any_expected_in_answer"] == [
            "4.5%", "4.5 percent",
        ]
        assert (project / "agentci_spec.yaml.bak").exists()

    def test_brittle_check_rejected_and_not_written(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from agentci.cli import cli

        # Golden answer says "4.5 percent" — a literal-only "4.5%" check is brittle
        project = self._project(tmp_path, golden_answer="Our APR is 4.5 percent.")
        self._fake_extraction(monkeypatch, [
            {"field": "any_expected_in_answer", "value": ["4.5%"], "fact": "APR"},
        ])
        monkeypatch.chdir(project)
        result = CliRunner().invoke(
            cli, ["generate-checks", "--kb", str(project / "kb"), "--yes"],
        )
        assert result.exit_code == 0, result.output
        assert "rejected by the validation gate" in result.output
        updated = yaml.safe_load((project / "agentci_spec.yaml").read_text())
        assert "correctness" not in (updated["queries"][0] or {})

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        from agentci.cli import cli

        project = self._project(tmp_path)
        self._fake_extraction(monkeypatch, [
            {"field": "any_expected_in_answer", "value": ["4.5 percent"]},
        ])
        monkeypatch.chdir(project)
        before = (project / "agentci_spec.yaml").read_text()
        result = CliRunner().invoke(
            cli, ["generate-checks", "--kb", str(project / "kb"), "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output
        assert (project / "agentci_spec.yaml").read_text() == before

    def test_yes_never_applies_unvalidated(self, tmp_path, monkeypatch):
        import shutil

        from click.testing import CliRunner

        from agentci.cli import cli

        project = self._project(tmp_path)
        shutil.rmtree(project / "golden")  # no known-good answers → ungated
        self._fake_extraction(monkeypatch, [
            {"field": "any_expected_in_answer", "value": ["4.5 percent"]},
        ])
        monkeypatch.chdir(project)
        result = CliRunner().invoke(
            cli, ["generate-checks", "--kb", str(project / "kb"), "--yes"],
        )
        assert result.exit_code == 0, result.output
        assert "unvalidated" in result.output
        updated = yaml.safe_load((project / "agentci_spec.yaml").read_text())
        assert "correctness" not in (updated["queries"][0] or {})
