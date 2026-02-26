"""
Unit tests for the AgentCI v2 YAML loader.

Uses pytest's tmp_path fixture for file system operations — no real file system
pollution.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentci.exceptions import ConfigError
from agentci.loader import filter_by_tags, load_spec


# ── Helpers ────────────────────────────────────────────────────────────────────


def write_spec(tmp_path: Path, data: dict) -> Path:
    """Write a YAML spec to a temp file and return its path."""
    p = tmp_path / "agentci_spec.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def minimal_spec_data(**kwargs) -> dict:
    base = {"agent": "test-agent", "queries": [{"query": "Hello world"}]}
    base.update(kwargs)
    return base


# ── load_spec ─────────────────────────────────────────────────────────────────


class TestLoadSpec:
    def test_loads_valid_yaml(self, tmp_path):
        p = write_spec(tmp_path, minimal_spec_data())
        spec = load_spec(p)
        assert spec.agent == "test-agent"
        assert len(spec.queries) == 1

    def test_file_not_found_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_spec(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises_config_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(": invalid: yaml: {", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_spec(p)

    def test_non_mapping_yaml_raises_config_error(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_spec(p)

    def test_missing_agent_field_raises_config_error(self, tmp_path):
        p = write_spec(tmp_path, {"queries": [{"query": "Hello"}]})
        with pytest.raises(ConfigError, match="validation failed"):
            load_spec(p)

    def test_default_baseline_dir(self, tmp_path):
        p = write_spec(tmp_path, minimal_spec_data())
        spec = load_spec(p)
        assert spec.baseline_dir == "./golden"

    def test_custom_baseline_dir(self, tmp_path):
        p = write_spec(tmp_path, minimal_spec_data(baseline_dir="./custom/dir"))
        spec = load_spec(p)
        assert spec.baseline_dir == "./custom/dir"

    def test_accepts_string_path(self, tmp_path):
        p = write_spec(tmp_path, minimal_spec_data())
        spec = load_spec(str(p))
        assert spec.agent == "test-agent"


# ── defaults merging ──────────────────────────────────────────────────────────


class TestDefaultsMerging:
    def test_defaults_applied_to_query_without_cost_spec(self, tmp_path):
        data = minimal_spec_data(
            defaults={"cost": {"max_llm_calls": 3}},
            queries=[{"query": "Simple query"}],
        )
        p = write_spec(tmp_path, data)
        spec = load_spec(p)
        q = spec.queries[0]
        assert q.cost is not None
        assert q.cost.max_llm_calls == 3

    def test_query_cost_overrides_defaults(self, tmp_path):
        data = minimal_spec_data(
            defaults={"cost": {"max_llm_calls": 3}},
            queries=[{"query": "Q", "cost": {"max_llm_calls": 10}}],
        )
        p = write_spec(tmp_path, data)
        spec = load_spec(p)
        assert spec.queries[0].cost.max_llm_calls == 10

    def test_deep_nested_merge(self, tmp_path):
        # defaults has hallucination_check; query adds expected_in_answer
        # Both should be present after merge
        data = minimal_spec_data(
            defaults={
                "correctness": {
                    "hallucination_check": {"rule": "No hallucinations", "threshold": 0.8}
                }
            },
            queries=[
                {"query": "Q", "correctness": {"expected_in_answer": ["pip"]}}
            ],
        )
        p = write_spec(tmp_path, data)
        spec = load_spec(p)
        q = spec.queries[0]
        assert q.correctness is not None
        assert q.correctness.expected_in_answer == ["pip"]
        assert q.correctness.hallucination_check is not None
        assert q.correctness.hallucination_check.rule == "No hallucinations"

    def test_no_defaults_section_leaves_queries_unchanged(self, tmp_path):
        data = minimal_spec_data(queries=[{"query": "Q"}])
        p = write_spec(tmp_path, data)
        spec = load_spec(p)
        assert spec.queries[0].cost is None
        assert spec.queries[0].correctness is None

    def test_defaults_applied_to_all_queries(self, tmp_path):
        data = minimal_spec_data(
            defaults={"cost": {"max_cost_usd": 0.01}},
            queries=[{"query": "Q1"}, {"query": "Q2"}, {"query": "Q3"}],
        )
        p = write_spec(tmp_path, data)
        spec = load_spec(p)
        for q in spec.queries:
            assert q.cost is not None
            assert q.cost.max_cost_usd == 0.01


# ── filter_by_tags ────────────────────────────────────────────────────────────


class TestFilterByTags:
    def _make_spec(self, tmp_path):
        data = minimal_spec_data(
            queries=[
                {"query": "Q1", "tags": ["smoke", "happy-path"]},
                {"query": "Q2", "tags": ["edge-case"]},
                {"query": "Q3", "tags": ["smoke", "edge-case"]},
                {"query": "Q4"},  # no tags
            ],
        )
        return load_spec(write_spec(tmp_path, data))

    def test_filter_by_single_tag(self, tmp_path):
        spec = self._make_spec(tmp_path)
        filtered = filter_by_tags(spec, ["smoke"])
        queries = [q.query for q in filtered.queries]
        assert "Q1" in queries
        assert "Q3" in queries
        assert "Q2" not in queries
        assert "Q4" not in queries

    def test_filter_by_multiple_tags_returns_union(self, tmp_path):
        spec = self._make_spec(tmp_path)
        filtered = filter_by_tags(spec, ["smoke", "edge-case"])
        assert len(filtered.queries) == 3  # Q1, Q2, Q3

    def test_empty_tag_list_returns_all(self, tmp_path):
        spec = self._make_spec(tmp_path)
        filtered = filter_by_tags(spec, [])
        assert len(filtered.queries) == 4

    def test_no_matching_tags_returns_empty(self, tmp_path):
        spec = self._make_spec(tmp_path)
        filtered = filter_by_tags(spec, ["nonexistent"])
        assert len(filtered.queries) == 0

    def test_filtered_spec_preserves_other_fields(self, tmp_path):
        spec = self._make_spec(tmp_path)
        filtered = filter_by_tags(spec, ["smoke"])
        assert filtered.agent == spec.agent
        assert filtered.baseline_dir == spec.baseline_dir
