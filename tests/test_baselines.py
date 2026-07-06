"""
Unit tests for the AgentCI v2 Baseline Manager.

Uses tmp_path for file system operations. Mocks evaluate_correctness
for precheck tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ciagent.baselines import list_baselines, load_baseline, save_baseline
from ciagent.engine.results import LayerResult, LayerStatus
from ciagent.exceptions import BaselineError
from ciagent.models import Span, SpanKind, Trace
from ciagent.schema.spec_models import AgentCISpec, CorrectnessSpec, GoldenQuery

# Force runner module import before any patches.  Without this,
# patch("ciagent.engine.correctness.evaluate_correctness") can cause
# runner.py's first import to bind the Mock instead of the real function.
import ciagent.engine.runner  # noqa: F401


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_trace(output: str = "pip install agentci") -> Trace:
    span = Span(kind=SpanKind.AGENT, output_data=output)
    t = Trace(spans=[span])
    t.compute_metrics()
    return t


def make_spec(query: str = "How do I install AgentCI?") -> AgentCISpec:
    return AgentCISpec(
        agent="rag-agent",
        queries=[GoldenQuery(query=query)],
    )


def pass_layer() -> LayerResult:
    return LayerResult(status=LayerStatus.PASS, details={}, messages=["OK"])


def fail_layer() -> LayerResult:
    return LayerResult(status=LayerStatus.FAIL, details={}, messages=["Failed"])


# ── save_baseline ─────────────────────────────────────────────────────────────


class TestSaveBaseline:
    def test_creates_file_in_agent_subdir(self, tmp_path):
        trace = make_trace()
        spec = make_spec()
        path = save_baseline(
            trace=trace,
            agent="rag-agent",
            version="v1",
            spec=spec,
            baseline_dir=str(tmp_path),
            force=True,
        )
        assert path.exists()
        assert path.parent.name == "rag-agent"
        assert path.name == "v1.json"

    def test_baseline_json_contains_required_fields(self, tmp_path):
        trace = make_trace()
        spec = make_spec()
        path = save_baseline(
            trace=trace, agent="rag-agent", version="v1",
            spec=spec, baseline_dir=str(tmp_path), force=True,
        )
        import json
        data = json.loads(path.read_text())
        assert data["version"] == "v1"
        assert data["agent"] == "rag-agent"
        assert "captured_at" in data
        assert "metadata" in data
        assert "trace" in data

    def test_metadata_contains_spec_hash(self, tmp_path):
        trace = make_trace()
        spec = make_spec()
        path = save_baseline(
            trace=trace, agent="rag-agent", version="v1",
            spec=spec, baseline_dir=str(tmp_path), force=True,
        )
        import json
        data = json.loads(path.read_text())
        assert data["metadata"]["spec_hash"].startswith("sha256:")

    def test_force_true_skips_precheck(self, tmp_path):
        trace = make_trace()
        spec = make_spec(query="Q")
        spec_with_correctness = AgentCISpec(
            agent="rag-agent",
            queries=[GoldenQuery(
                query="Q",
                correctness=CorrectnessSpec(expected_in_answer=["must-find-this"]),
            )],
        )
        # Without force, precheck would fail because output doesn't contain "must-find-this"
        # With force=True, it should save without checking
        path = save_baseline(
            trace=trace, agent="rag-agent", version="v1",
            spec=spec_with_correctness, query_text="Q",
            baseline_dir=str(tmp_path), force=True,
        )
        assert path.exists()

    def test_precheck_passes_saves_baseline(self, tmp_path):
        trace = make_trace()
        spec = make_spec(query="Q")
        with patch(
            "ciagent.engine.correctness.evaluate_correctness",
            return_value=pass_layer(),
        ):
            path = save_baseline(
                trace=trace, agent="rag-agent", version="v1",
                spec=spec, query_text="Q",
                baseline_dir=str(tmp_path), force=False,
            )
        assert path.exists()

    def test_precheck_fails_raises_value_error(self, tmp_path):
        trace = make_trace()
        spec_with_correctness = AgentCISpec(
            agent="rag-agent",
            queries=[GoldenQuery(
                query="Q",
                correctness=CorrectnessSpec(expected_in_answer=["must-find-this"]),
            )],
        )
        with patch(
            "ciagent.engine.correctness.evaluate_correctness",
            return_value=fail_layer(),
        ):
            with pytest.raises(ValueError, match="Precheck failed"):
                save_baseline(
                    trace=trace, agent="rag-agent", version="v1",
                    spec=spec_with_correctness, query_text="Q",
                    baseline_dir=str(tmp_path), force=False,
                )

    def test_no_query_text_skips_precheck(self, tmp_path):
        """When query_text is empty, precheck is always skipped."""
        trace = make_trace()
        spec = make_spec()
        path = save_baseline(
            trace=trace, agent="rag-agent", version="v1",
            spec=spec, query_text="",  # no query → no precheck
            baseline_dir=str(tmp_path), force=False,
        )
        assert path.exists()

    def test_stores_query_text(self, tmp_path):
        import json
        trace = make_trace()
        spec = make_spec(query="How do I install AgentCI?")
        path = save_baseline(
            trace=trace, agent="rag-agent", version="v1",
            spec=spec, query_text="How do I install AgentCI?",
            baseline_dir=str(tmp_path), force=True,
        )
        data = json.loads(path.read_text())
        assert data["query"] == "How do I install AgentCI?"


# ── load_baseline ─────────────────────────────────────────────────────────────


class TestLoadBaseline:
    def test_load_saved_baseline(self, tmp_path):
        trace = make_trace()
        spec = make_spec()
        save_baseline(
            trace=trace, agent="rag-agent", version="v1",
            spec=spec, baseline_dir=str(tmp_path), force=True,
        )
        data = load_baseline("rag-agent", "v1", str(tmp_path))
        assert data["version"] == "v1"
        assert "trace" in data

    def test_load_nonexistent_raises_baseline_error(self, tmp_path):
        with pytest.raises(BaselineError, match="not found"):
            load_baseline("rag-agent", "nonexistent", str(tmp_path))

    def test_round_trip_preserves_version(self, tmp_path):
        trace = make_trace()
        spec = make_spec()
        save_baseline(
            trace=trace, agent="agent-x", version="v2-fixed",
            spec=spec, baseline_dir=str(tmp_path), force=True,
        )
        data = load_baseline("agent-x", "v2-fixed", str(tmp_path))
        assert data["version"] == "v2-fixed"


# ── list_baselines ────────────────────────────────────────────────────────────


class TestListBaselines:
    def test_lists_saved_versions(self, tmp_path):
        trace = make_trace()
        spec = make_spec()
        for v in ["v1", "v2", "v3"]:
            save_baseline(
                trace=trace, agent="rag-agent", version=v,
                spec=spec, baseline_dir=str(tmp_path), force=True,
            )
        versions = list_baselines("rag-agent", str(tmp_path))
        assert len(versions) == 3

    def test_empty_dir_returns_empty_list(self, tmp_path):
        result = list_baselines("rag-agent", str(tmp_path))
        assert result == []

    def test_nonexistent_agent_dir_returns_empty(self, tmp_path):
        result = list_baselines("unknown-agent", str(tmp_path))
        assert result == []

    def test_listed_entries_have_version_field(self, tmp_path):
        trace = make_trace()
        spec = make_spec()
        save_baseline(
            trace=trace, agent="rag-agent", version="v1",
            spec=spec, baseline_dir=str(tmp_path), force=True,
        )
        entries = list_baselines("rag-agent", str(tmp_path))
        assert entries[0]["version"] == "v1"

    def test_sorted_by_filename(self, tmp_path):
        trace = make_trace()
        spec = make_spec()
        for v in ["v3", "v1", "v2"]:
            save_baseline(
                trace=trace, agent="rag-agent", version=v,
                spec=spec, baseline_dir=str(tmp_path), force=True,
            )
        entries = list_baselines("rag-agent", str(tmp_path))
        names = [e["version"] for e in entries]
        assert names == sorted(names)
