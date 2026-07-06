# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Test execution engine.

Loads a TestSuite, executes each TestCase by importing the agent function,
wrapping it in TraceContext, and collecting results.
"""
from __future__ import annotations

from .models import TestSuite, SuiteResult

import importlib
import time
from typing import Any, Callable
from .models import TestSuite, SuiteResult, RunResult, TestResult, Trace
from .capture import TraceContext
from .assertions import evaluate_assertion

class TestRunner:
    def __init__(self, suite: TestSuite):
        self.suite = suite
        self._agent_fn: Callable | None = None

    def _import_agent(self) -> Callable:
        """Dynamically import the agent function from a string path."""
        if self._agent_fn:
            return self._agent_fn
            
        if not self.suite.agent:
            from .exceptions import ConfigError
            raise ConfigError(
                "No agent import path provided in test suite.",
                fix="Set 'agent: myapp.agent:run_agent' in your agentci.yaml file."
            )

        try:
            module_path, fn_name = self.suite.agent.split(":")
            module = importlib.import_module(module_path)
            self._agent_fn = getattr(module, fn_name)
            return self._agent_fn
        except (ImportError, AttributeError, ValueError) as e:
            from .exceptions import ImportError_
            raise ImportError_(
                f"Could not import agent function '{self.suite.agent}': {e}",
                fix=f"Ensure the import path follows 'module.path:function_name' format. "
                    f"Example: 'myapp.agent:run_agent'. Check that the module is installed "
                    f"and the function exists."
            )

    def run_suite(self, runs: int = 1) -> SuiteResult:
        """Execute all tests in the suite."""
        agent_fn = self._import_agent()
        
        # Load mocks if configured
        mock_toolkit = None
        if self.suite.mocks:
            from .mocks import MockToolkit
            # Assuming relative path from config location? 
            # ideally config loading resolves absolute paths, but here we assume CWD or absolute
            try:
                mock_toolkit = MockToolkit.from_yaml(self.suite.mocks)
            except FileNotFoundError:
                from .exceptions import MockError
                raise MockError(
                    f"Mock file not found: {self.suite.mocks}",
                    fix=f"Create the mock file at '{self.suite.mocks}' or remove the 'mocks' "
                        f"key from ciagent.yaml. To record mocks from a live run: "
                        f"'ciagent record <test_name>'"
                )
            except Exception as e:
                from .exceptions import MockError
                raise MockError(
                    f"Failed to load mocks from {self.suite.mocks}: {e}",
                    fix="Check that the YAML file is valid and follows the expected format."
                )

        suite_result = SuiteResult(suite_name=self.suite.name)
        start_time = time.perf_counter()
        
        for test in self.suite.tests:
            for _ in range(runs):
                run_result = self.run_test(test, agent_fn, mock_toolkit)
                suite_result.results.append(run_result)
                
                if run_result.result == TestResult.PASSED:
                    suite_result.total_passed += 1
                elif run_result.result == TestResult.FAILED:
                    suite_result.total_failed += 1
                else:
                    suite_result.total_errors += 1
                
        suite_result.duration_ms = (time.perf_counter() - start_time) * 1000
        suite_result.total_cost_usd = sum(r.trace.total_cost_usd for r in suite_result.results)
        
        return suite_result

    def run_test(self, test: Any, agent_fn: Callable, mock_toolkit: Any = None) -> RunResult:
        """Run a single test case."""
        start_time = time.perf_counter()
        
        with TraceContext(agent_name=self.suite.agent, test_name=test.name) as ctx:
            try:
                # Prepare arguments
                import inspect
                sig = inspect.signature(agent_fn)
                kwargs = {}
                
                # If mocks are active, try to inject them
                if mock_toolkit:
                    # Reset mocks for this test run
                    mock_toolkit.reset_all()
                    
                    # If user defines a scenario tag for this test, we could set it?
                    # For now, let's look for "scenario" in test metadata or assertions?
                    # Let's simple pass the toolkit if the agent accepts 'tools' or 'toolkit'
                    if "tools" in sig.parameters:
                        # Pass dictionary of tools or the toolkit object? 
                        # Let's pass the toolkit object so they can .get("name")
                        # Or maybe a dict of callables? 
                        # The MockToolkit has .tools which is dict[str, MockTool]
                        kwargs["tools"] = mock_toolkit.tools
                    elif "toolkit" in sig.parameters:
                        kwargs["toolkit"] = mock_toolkit

                # Execute the agent
                if test.input_data is not None:
                     result = agent_fn(test.input_data, **kwargs)
                else:
                    # Try calling without args if input is None
                    try:
                        result = agent_fn(**kwargs)
                    except TypeError:
                         # Fallback: maybe it requires one argument (input)?
                         result = agent_fn("", **kwargs)
                
                # Update span with result
                ctx.trace.spans[0].output_data = result
                ctx.trace.compute_metrics()
                
                # Evaluate assertions
                assertion_results = []
                all_passed = True
                for assertion in test.assertions:
                    passed, message = evaluate_assertion(assertion, ctx.trace)
                    assertion_results.append({"passed": passed, "message": message})
                    if not passed:
                        all_passed = False
                
                # Check budgets (max_cost, max_steps)
                if test.max_cost_usd and ctx.trace.total_cost_usd > test.max_cost_usd:
                    all_passed = False
                    assertion_results.append({
                        "passed": False, 
                        "message": f"✗ Cost ${ctx.trace.total_cost_usd:.4f} exceeds budget ${test.max_cost_usd:.4f}"
                    })

                # Golden Trace Diffing
                diffs = []
                if test.golden_trace:
                    import json
                    from .diff_engine import diff_traces
                    try:
                        with open(test.golden_trace, 'r') as f:
                            golden = Trace.model_validate(json.load(f))
                            diffs = diff_traces(ctx.trace, golden)
                    except FileNotFoundError:
                        assertion_results.append({
                            "passed": True,
                            "message": f"⚠ Golden trace not found: {test.golden_trace}. "
                                       f"Fix: Run 'ciagent record {test.name} -o {test.golden_trace}' to create a baseline."
                        })
                    except Exception as e:
                        assertion_results.append({
                            "passed": False, 
                            "message": f"⚠ Failed to load golden trace: {e}"
                        })

                return RunResult(
                    test_name=test.name,
                    result=TestResult.PASSED if all_passed else TestResult.FAILED,
                    trace=ctx.trace,
                    assertion_results=assertion_results,
                    diffs=diffs,
                    duration_ms=(time.perf_counter() - start_time) * 1000
                )
                
            except Exception as e:
                return RunResult(
                    test_name=test.name,
                    result=TestResult.ERROR,
                    trace=ctx.trace,
                    error_message=str(e),
                    duration_ms=(time.perf_counter() - start_time) * 1000
                )
