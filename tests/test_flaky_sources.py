# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Source-aware flaky gating (Plan_docs/flip_attribution_deepening.md, F1-F6).
The wedge: gate on agent-variance, tolerate judge-flake — something no
single-run tool can do.
"""

from __future__ import annotations

import pytest

from ciagent.engine.stability import (
    FlipSource,
    flip_source_counts,
    parse_flaky_sources,
)


class TestParseFlakySources:
    def test_enum_values(self):
        assert parse_flaky_sources("agent-variance") == {FlipSource.AGENT_VARIANCE}

    def test_multiple(self):
        got = parse_flaky_sources("agent-variance,judge-flake")
        assert got == {FlipSource.AGENT_VARIANCE, FlipSource.JUDGE_FLAKE}

    def test_alias_agent_matches_flaky_agent_class(self):
        # F3: `agent`/`real` mirror promotion.FLIP_SOURCE_TO_CLASS FLAKY_AGENT
        from ciagent.promotion import Classification, FLIP_SOURCE_TO_CLASS

        flaky_agent = {s for s, c in FLIP_SOURCE_TO_CLASS.items()
                       if c == Classification.FLAKY_AGENT}
        assert parse_flaky_sources("agent") == flaky_agent
        assert parse_flaky_sources("real") == flaky_agent

    def test_alias_judge_infra_sim(self):
        assert parse_flaky_sources("judge") == {FlipSource.JUDGE_FLAKE}
        assert parse_flaky_sources("infra") == {FlipSource.INFRA_ERROR}
        assert parse_flaky_sources("sim") == {FlipSource.SIMULATION_VARIANCE}

    def test_world_miss_not_in_agent_alias(self):
        assert FlipSource.WORLD_MISS not in parse_flaky_sources("agent")
        # but gate-able when named explicitly
        assert parse_flaky_sources("world-miss") == {FlipSource.WORLD_MISS}

    def test_unknown_token_errors_with_valid_set(self):
        with pytest.raises(ValueError, match="unknown flip source"):
            parse_flaky_sources("nonsense")

    def test_empty_is_empty_set(self):
        assert parse_flaky_sources("") == set()
        assert parse_flaky_sources(" , ") == set()


class TestFlipSourceCounts:
    def test_counts_all_sources(self):
        class Q:
            def __init__(self, src):
                self.flip_source = src
        flipped = [Q(FlipSource.AGENT_VARIANCE), Q(FlipSource.AGENT_VARIANCE),
                   Q(FlipSource.RETRIEVAL_VARIANCE), Q(None)]
        counts = flip_source_counts(flipped)
        assert counts == {"agent-variance": 2, "retrieval-variance": 1,
                          "unknown": 1}
        # sums to the flip count (no source lost to a catch-all)
        assert sum(counts.values()) == len(flipped)


# ── Integration: source-aware gating over the real CLI ──────────────────────────

import json
import sys as _sys
from pathlib import Path

from click.testing import CliRunner

from ciagent.cli import cli

# A spec + runner where the mock-flaky harness produces a verdict flip; the
# gate decision depends only on the ATTRIBUTED source, which we assert from
# the JSON rather than guessing the harness's attribution.
QA_SPEC = """
agent: qa-test
runner: "toy_qa:run"
queries:
  - query: "what is the refund policy?"
    correctness:
      expected_in_answer: ["refund"]
"""

TOY = '''
from ciagent.models import Span, SpanKind, Trace
def run(query):
    span = Span(kind=SpanKind.AGENT, name="agent")
    span.output_data = "refunds take 5-7 days"
    t = Trace(agent_name="qa-test", test_name=query, spans=[span])
    t.metadata["final_output"] = "refunds take 5-7 days"
    t.compute_metrics()
    return t
'''


def _run(args, monkeypatch=None):
    r = CliRunner()
    with r.isolated_filesystem():
        Path("ciagent_spec.yaml").write_text(QA_SPEC)
        Path("toy_qa.py").write_text(TOY)
        _sys.path.insert(0, ".")
        try:
            return r.invoke(cli, args)
        finally:
            _sys.path.remove(".")
            _sys.modules.pop("toy_qa", None)


class TestFlakySourcesCLI:
    def test_unknown_source_exits_2(self):
        res = _run(["test", "--mock", "--flaky-sources", "bogus"])
        assert res.exit_code == 2
        assert "unknown flip source" in res.output

    def test_json_carries_flip_sources_and_gated_by(self, monkeypatch):
        monkeypatch.setenv("CIAGENT_MOCK_FLAKY", "1")
        res = _run(["test", "--mock", "--runs", "3", "--format", "json",
                    "--flaky-sources", "agent-variance"])
        payload = json.loads(res.stdout)
        stab = payload["stability"]
        assert "flip_sources" in stab
        assert stab["gated_by"] == ["agent-variance"]

    def test_source_gate_is_selective(self, monkeypatch):
        # Whatever the harness attributes the flip to, gating on a DIFFERENT
        # source must not exit 1 on the flip, while gating on the ACTUAL
        # source must. Derive the actual source from the JSON, then assert
        # both directions — proving the selectivity, harness-agnostic.
        monkeypatch.setenv("CIAGENT_MOCK_FLAKY", "1")
        probe = _run(["test", "--mock", "--runs", "3", "--format", "json"])
        stab = json.loads(probe.stdout)["stability"]
        if not stab["flip_sources"]:
            pytest.skip("mock harness produced no flip in this run")
        actual = max(stab["flip_sources"], key=stab["flip_sources"].get)
        # consistent failures would also gate; skip if any present
        if stab["consistent_failures"]:
            pytest.skip("consistent failure present — flip gate not isolable")

        other = ("judge-flake" if actual != "judge-flake" else "infra-error")
        res_other = _run(["test", "--mock", "--runs", "3",
                          "--flaky-sources", other], monkeypatch)
        res_actual = _run(["test", "--mock", "--runs", "3",
                           "--flaky-sources", actual], monkeypatch)
        assert res_other.exit_code == 0, res_other.output
        assert res_actual.exit_code == 1, res_actual.output

    def test_bare_fail_on_flaky_still_gates_any(self, monkeypatch):
        monkeypatch.setenv("CIAGENT_MOCK_FLAKY", "1")
        probe = _run(["test", "--mock", "--runs", "3", "--format", "json"])
        stab = json.loads(probe.stdout)["stability"]
        if not stab["flipped"] or stab["consistent_failures"]:
            pytest.skip("no isolable flip in this run")
        res = _run(["test", "--mock", "--runs", "3", "--fail-on-flaky"],
                   monkeypatch)
        assert res.exit_code == 1
