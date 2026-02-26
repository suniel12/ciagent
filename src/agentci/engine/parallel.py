"""
AgentCI v2 Parallel Execution Engine.

Runs spec queries concurrently using ThreadPoolExecutor (sync-friendly for
most agent code) with exponential backoff retry on transient infra errors.

Public API:
    run_spec_parallel(spec, runner_fn, max_workers, retry_count) → dict[str, Trace]
    run_spec(spec, runner_fn, max_workers, query_indices)        → list[QueryResult]
"""

from __future__ import annotations

import importlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from agentci.engine.results import QueryResult
    from agentci.models import Trace
    from agentci.schema.spec_models import AgentCISpec

logger = logging.getLogger(__name__)

# Errors that warrant retry (transient infra issues, not logic errors)
_RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,
)

# Try to include API-specific rate limit errors if available
try:
    import anthropic
    _RETRYABLE_EXCEPTIONS = _RETRYABLE_EXCEPTIONS + (anthropic.RateLimitError,)
except ImportError:
    pass

try:
    import openai
    _RETRYABLE_EXCEPTIONS = _RETRYABLE_EXCEPTIONS + (openai.RateLimitError,)
except ImportError:
    pass


def run_spec_parallel(
    spec: "AgentCISpec",
    runner_fn: Callable[[str], "Trace"],
    max_workers: int = 4,
    retry_count: int = 2,
    retry_backoff: float = 1.0,
    query_indices: Optional[list[int]] = None,
) -> dict[str, "Trace"]:
    """Execute spec queries in parallel with retry and exponential backoff.

    Args:
        spec:           Loaded AgentCISpec.
        runner_fn:      Callable (query: str) → Trace. Must be thread-safe.
        max_workers:    Max concurrent threads. Default 4 (conservative rate-limit budget).
        retry_count:    Max retries per query on transient infra errors.
        retry_backoff:  Base backoff seconds; doubles each retry (1s, 2s, 4s...).
        query_indices:  Optional subset of spec.queries to run (0-based).

    Returns:
        dict mapping query_text → Trace. Queries that fail after all retries
        are excluded (an error is logged, not raised).
    """
    queries = spec.queries
    if query_indices is not None:
        queries = [spec.queries[i] for i in query_indices if i < len(spec.queries)]

    results: dict[str, "Trace"] = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for gq in queries:
            fut = executor.submit(
                _run_with_retry,
                runner_fn,
                gq.query,
                retry_count,
                retry_backoff,
            )
            futures[fut] = gq.query

        for fut in as_completed(futures):
            query_text = futures[fut]
            try:
                trace = fut.result()
                if trace is not None:
                    results[query_text] = trace
                else:
                    logger.error("[AgentCI] Runner returned None for query: %s", query_text[:60])
            except Exception as exc:
                logger.error(
                    "[AgentCI] Query failed after retries: %s — %s",
                    query_text[:60],
                    exc,
                )

    return results


def run_spec(
    spec: "AgentCISpec",
    runner_fn: Callable[[str], "Trace"],
    max_workers: int = 4,
    query_indices: Optional[list[int]] = None,
    baseline_traces: Optional[dict[str, "Trace"]] = None,
) -> list["QueryResult"]:
    """High-level API: run agent against spec and return evaluated results.

    This is the pytest-native entry point:

        from agentci import load_spec, run_spec
        results = run_spec(spec, my_agent_fn)
        assert not results[0].hard_fail

    Args:
        spec:             Loaded AgentCISpec.
        runner_fn:        Callable (query: str) → Trace.
        max_workers:      Parallel workers for query execution.
        query_indices:    Run only these query indices (0-based). None = all.
        baseline_traces:  Optional golden baselines for cost/path comparison.

    Returns:
        List of QueryResult in spec.queries order (missing traces are excluded).
    """
    from agentci.engine.runner import evaluate_spec

    traces = run_spec_parallel(
        spec=spec,
        runner_fn=runner_fn,
        max_workers=max_workers,
        query_indices=query_indices,
    )

    return evaluate_spec(
        spec=spec,
        traces=traces,
        baselines=baseline_traces,
    )


def resolve_runner(runner_path: str) -> Callable[[str], "Trace"]:
    """Dynamically import a runner callable from a dotted path.

    Format: 'module.submodule:function_name'
    Example: 'myagent.run:run_agent'

    Args:
        runner_path: Dotted Python import path with ':' separator.

    Returns:
        The callable.

    Raises:
        ValueError:   If the path format is invalid.
        ImportError:  If the module cannot be imported.
        AttributeError: If the function name is not found in the module.
    """
    if ":" not in runner_path:
        raise ValueError(
            f"Invalid runner path '{runner_path}'. "
            "Expected format: 'module.submodule:function_name'"
        )

    module_path, func_name = runner_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import runner module '{module_path}': {exc}. "
            "Make sure the module is installed and your PYTHONPATH is correct."
        ) from exc

    try:
        fn = getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Function '{func_name}' not found in module '{module_path}'. "
            f"Available names: {[n for n in dir(module) if not n.startswith('_')]}"
        ) from exc

    if not callable(fn):
        raise ValueError(f"'{runner_path}' is not callable (got {type(fn).__name__})")

    return fn


# ── Internal helpers ────────────────────────────────────────────────────────────


def _run_with_retry(
    runner_fn: Callable[[str], "Trace"],
    query: str,
    retry_count: int,
    backoff: float,
) -> Optional["Trace"]:
    """Call runner_fn with exponential-backoff retry on transient errors."""
    last_exc: Optional[Exception] = None

    for attempt in range(retry_count + 1):
        try:
            return runner_fn(query)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < retry_count:
                wait = backoff * (2 ** attempt)
                logger.warning(
                    "[AgentCI][INFRA] Query '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                    query[:40],
                    attempt + 1,
                    retry_count + 1,
                    wait,
                    exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "[AgentCI][INFRA] Query '%s' failed after %d retries: %s",
                    query[:40],
                    retry_count + 1,
                    exc,
                )
        except Exception as exc:
            # Non-retryable error — re-raise immediately
            raise exc

    raise last_exc  # type: ignore[misc]
