"""
Tests for engine/parallel.py — parallel execution + retry + run_spec API.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agentci.engine.parallel import resolve_runner, run_spec, run_spec_parallel
from agentci.models import Span, Trace
from agentci.schema.spec_models import AgentCISpec, GoldenQuery


# ── Fixtures ────────────────────────────────────────────────────────────────────


def _make_trace(query: str = "test") -> Trace:
    """Minimal Trace fixture."""
    span = Span(name="test")
    t = Trace(spans=[span], input_query=query)
    t.compute_metrics()
    return t


def _make_spec(queries: list[str], runner: str | None = None) -> AgentCISpec:
    return AgentCISpec(
        agent="test-agent",
        queries=[GoldenQuery(query=q) for q in queries],
        runner=runner,
    )


def _sync_runner(query: str) -> Trace:
    """Simple deterministic runner for testing."""
    return _make_trace(query)


# ── run_spec_parallel tests ─────────────────────────────────────────────────────


class TestRunSpecParallel:
    def test_returns_trace_for_each_query(self):
        spec = _make_spec(["query A", "query B", "query C"])
        result = run_spec_parallel(spec, _sync_runner, max_workers=2)

        assert set(result.keys()) == {"query A", "query B", "query C"}
        for trace in result.values():
            assert isinstance(trace, Trace)

    def test_single_worker_sequential(self):
        """max_workers=1 must still handle all queries."""
        spec = _make_spec(["q1", "q2"])
        result = run_spec_parallel(spec, _sync_runner, max_workers=1)
        assert len(result) == 2

    def test_empty_spec_returns_empty(self):
        """No queries → empty dict (spec validation catches this before us)."""
        spec = _make_spec(["placeholder"])  # can't make truly empty (pydantic min_length=1)
        result = run_spec_parallel(spec, lambda q: None, max_workers=1)
        # runner returns None → excluded from results
        assert len(result) == 0

    def test_runner_returning_none_excluded(self):
        """A runner that returns None should be excluded, not crash."""
        spec = _make_spec(["good", "bad"])

        def flaky_runner(q: str) -> Trace | None:
            return _make_trace(q) if q == "good" else None

        result = run_spec_parallel(spec, flaky_runner, max_workers=1)
        assert "good" in result
        assert "bad" not in result

    def test_query_indices_filter(self):
        """query_indices selects a subset of spec.queries."""
        spec = _make_spec(["q0", "q1", "q2", "q3"])
        result = run_spec_parallel(spec, _sync_runner, query_indices=[0, 2])
        assert set(result.keys()) == {"q0", "q2"}

    def test_non_retryable_exception_excluded(self):
        """Non-transient errors should not crash the whole run."""
        spec = _make_spec(["good", "bad"])

        def fail_runner(q: str) -> Trace:
            if q == "bad":
                raise ValueError("logic error")
            return _make_trace(q)

        result = run_spec_parallel(spec, fail_runner, max_workers=1)
        assert "good" in result
        assert "bad" not in result

    def test_retry_on_timeout_error(self):
        """Transient TimeoutError should retry and succeed on 3rd attempt."""
        calls = {"count": 0}

        def flaky_runner(q: str) -> Trace:
            calls["count"] += 1
            if calls["count"] < 3:
                raise TimeoutError("timeout")
            return _make_trace(q)

        spec = _make_spec(["q"])
        # retry_count=2 → up to 3 attempts total
        result = run_spec_parallel(
            spec, flaky_runner, max_workers=1,
            retry_count=2, retry_backoff=0.0,  # zero backoff for fast tests
        )
        assert "q" in result
        assert calls["count"] == 3

    def test_exhaust_retries_excludes_query(self):
        """If all retries fail, query is excluded from results."""
        def always_fails(q: str) -> Trace:
            raise TimeoutError("always")

        spec = _make_spec(["q"])
        result = run_spec_parallel(
            spec, always_fails, max_workers=1,
            retry_count=1, retry_backoff=0.0,
        )
        assert len(result) == 0

    def test_four_worker_parallel_completion(self):
        """All 8 queries complete correctly with 4 workers."""
        queries = [f"query_{i}" for i in range(8)]
        spec = _make_spec(queries)
        result = run_spec_parallel(spec, _sync_runner, max_workers=4)
        assert set(result.keys()) == set(queries)


# ── run_spec tests ──────────────────────────────────────────────────────────────


class TestRunSpec:
    def test_returns_query_results(self):
        spec = _make_spec(["hello"])
        results = run_spec(spec, _sync_runner)
        assert len(results) == 1
        result = results[0]
        assert result.query == "hello"
        # No correctness/path/cost spec → all SKIP
        assert result.correctness.status.value == "skip"
        assert result.path.status.value == "skip"
        assert result.cost.status.value == "skip"

    def test_end_to_end_with_query_indices(self):
        spec = _make_spec(["a", "b", "c"])
        results = run_spec(spec, _sync_runner, query_indices=[1])
        assert len(results) == 1
        assert results[0].query == "b"

    def test_hard_fail_propagates(self):
        """A correctness failure should appear in the result."""
        from agentci.schema.spec_models import CorrectnessSpec, GoldenQuery, AgentCISpec

        spec = AgentCISpec(
            agent="test-agent",
            queries=[
                GoldenQuery(
                    query="test",
                    correctness=CorrectnessSpec(
                        expected_in_answer=["MUST_NOT_BE_PRESENT"]
                    ),
                )
            ],
        )
        results = run_spec(spec, _sync_runner)
        assert len(results) == 1
        assert results[0].hard_fail  # runner returns empty output → term not found → FAIL

    def test_multiple_queries_all_evaluated(self):
        queries = ["a", "b", "c", "d"]
        spec = _make_spec(queries)
        results = run_spec(spec, _sync_runner, max_workers=4)
        assert len(results) == len(queries)


# ── resolve_runner tests ────────────────────────────────────────────────────────


class TestResolveRunner:
    def test_resolves_builtin_callable(self):
        # Use a builtin module:function that exists
        fn = resolve_runner("os.path:join")
        assert callable(fn)
        assert fn("a", "b") == "a/b"

    def test_missing_colon_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid runner path"):
            resolve_runner("mymodule.function")

    def test_missing_module_raises_import_error(self):
        with pytest.raises(ImportError, match="Cannot import runner module"):
            resolve_runner("nonexistent.module:func")

    def test_missing_function_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="not found in module"):
            resolve_runner("os.path:nonexistent_function")

    def test_non_callable_raises_value_error(self):
        with pytest.raises(ValueError, match="not callable"):
            resolve_runner("os:sep")  # os.sep is a string, not callable
