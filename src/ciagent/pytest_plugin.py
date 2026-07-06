# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
pytest integration plugin.

Provides fixtures and hooks for running AgentCI tests via pytest.
"""
from __future__ import annotations

import pytest
import functools
from typing import Any, Callable
from .models import TestCase, Assertion
from .capture import TraceContext
from .cost import compute_cost
from pathlib import Path


def pytest_collect_file(parent, file_path):
    if file_path.name == "agentci_spec.yaml":
        return AgentCIFile.from_parent(parent, path=file_path)

class AgentCIFile(pytest.File):
    def collect(self):
        from .loader import load_spec
        from .exceptions import ConfigError
        try:
            spec = load_spec(self.path)
            for i, query in enumerate(spec.queries):
                if query.query:
                    # Give it a nice clean name
                    short_name = query.query[:60] + ('...' if len(query.query) > 60 else '')
                    name = f"{i+1:02d}_{short_name}"
                else:
                    name = f"query_{i+1:02d}"
                yield AgentCIItem.from_parent(self, name=name, spec=spec, query=query)
        except ConfigError as e:
            raise pytest.UsageError(f"Error loading AgentCI spec: {e}")

class AgentCIItem(pytest.Item):
    @classmethod
    def from_parent(cls, parent, *, name, spec, query):
        obj = super().from_parent(parent, name=name)
        obj.spec = spec
        obj.query = query
        return obj

    def runtest(self):
        from .engine.parallel import resolve_runner
        from .engine.correctness import evaluate_correctness
        from .baselines import load_baseline
        from .engine.runner import _extract_answer
        from .engine.cost import evaluate_cost
        from .engine.path import evaluate_path
        from .engine.results import LayerResult, LayerStatus
        
        # 1. Resolve Runner
        if not self.spec.runner:
            raise pytest.UsageError(f"No runner defined in {self.spec.agent} spec. Cannot execute interactive test.")
        
        try:
            runner_fn = resolve_runner(self.spec.runner)
        except Exception as e:
            raise AgentCITestFailure(self, f"Runner resolution failed: {e}")
            
        # 2. Run agent
        try:
            trace = runner_fn(self.query.query)
        except Exception as e:
            pytest.fail(f"Agent execution failed: {e}")

        answer = _extract_answer(trace)
        
        # 3. Load Baseline (Optional)
        baseline_trace = None
        if self.spec.baseline_dir:
            import glob
            from pathlib import Path
            import json
            baseline_path = Path(self.spec.baseline_dir)
            if baseline_path.exists():
                for f in glob.glob(str(baseline_path / "*.json")):
                    try:
                        b = load_baseline(f)
                        if "trace" in b and "query" in b.get("trace", {}):
                            if b["trace"]["query"] == self.query.query:
                                from .models import Trace
                                baseline_trace = Trace.from_dict(b["trace"])
                                break
                    except Exception:
                        pass
        
        # 4. Correctness Eval
        if self.query.correctness:
            result = evaluate_correctness(answer, self.query.correctness, trace, getattr(self.spec, 'judge_config', None))
            if result.status == LayerStatus.FAIL:
                messages = "\\n".join(result.messages)
                raise AgentCITestFailure(self, f"Correctness FAIL:\\n{messages}")
            elif result.status == LayerStatus.WARN:
                # We can't log as warning in pytest natively without raising or using warnings
                import warnings
                warnings.warn(f"Correctness warning for {self.name}: " + "; ".join(result.messages))
                
        # 5. Path Eval
        if self.query.path:
            p_result = evaluate_path(trace, self.query.path, baseline_trace)
            if p_result.status == LayerStatus.FAIL:
                messages = "\\n".join(p_result.messages)
                raise AgentCITestFailure(self, f"Path FAIL:\\n{messages}")
            elif p_result.status == LayerStatus.WARN:
                 import warnings
                 warnings.warn(f"Path warning for {self.name}: " + "; ".join(p_result.messages))
                 
        # 6. Cost Eval
        if self.query.cost:
            c_result = evaluate_cost(trace, self.query.cost, baseline_trace)
            if c_result.status == LayerStatus.FAIL:
                messages = "\\n".join(c_result.messages)
                raise AgentCITestFailure(self, f"Cost FAIL:\\n{messages}")
            elif c_result.status == LayerStatus.WARN:
                 import warnings
                 warnings.warn(f"Cost warning for {self.name}: " + "; ".join(c_result.messages))

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, AgentCITestFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.fspath, 0, f"agentci: {self.name}"

class AgentCITestFailure(Exception):
    def __init__(self, item, message):
        self.item = item
        self.message = message
    def __str__(self):
        return self.message

# Keep existing fixture components logic
def test(
    input: Any = None,
    assertions: list[Assertion | dict] | None = None,
    max_cost_usd: float | None = None,
    max_steps: int | None = None,
    golden_trace: str | None = None,
):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.pytestmark = [pytest.mark.agentci]
        wrapper._agentci_config = {
            "input": input,
            "assertions": assertions,
            "max_cost_usd": max_cost_usd,
            "max_steps": max_steps,
            "golden_trace": golden_trace,
        }
        return wrapper
    return decorator


@pytest.fixture
def agentci_trace(request):
    config = getattr(request.function, "_agentci_config", {})
    test_name = request.node.name
    
    with TraceContext(test_name=test_name) as ctx:
        yield ctx.trace
        
    trace = ctx.trace
    
    if config.get("max_cost_usd") is not None:
        limit = config["max_cost_usd"]
        if trace.total_cost_usd > limit:
            pytest.fail(f"Cost ${trace.total_cost_usd:.4f} exceeded budget ${limit:.4f}")

    from .assertions import evaluate_assertion
    if config.get("assertions"):
        for a in config["assertions"]:
            if isinstance(a, dict):
                a = Assertion(**a)
            passed, msg = evaluate_assertion(a, trace)
            if not passed:
                pytest.fail(msg)

    if config.get("golden_trace"):
        pass


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "agentci: mark test as an AgentCI agent test"
    )

