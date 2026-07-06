import pytest
import ciagent
from ciagent.models import Trace

def mock_agent_function(input_text):
    return "Processed: " + input_text

@ciagent.test(
    max_cost_usd=0.01,
    assertions=[{"type": "cost_under", "threshold": 0.01}]
)
def test_decorated_agent(agentci_trace):
    """Verify that the decorator works and injects the trace."""
    assert isinstance(agentci_trace, Trace)
    result = mock_agent_function("test input")
    assert result == "Processed: test input"
    # Manual span creation to verify trace is active
    from ciagent.models import Span
    agentci_trace.spans.append(Span(name="manual_span"))

@ciagent.test()
def test_simple_decorator(agentci_trace):
    """Verify decorator works without arguments."""
    assert agentci_trace is not None

pytest_plugins = ["pytester"]

def test_pytest_plugin_collects_spec(pytester):
    """Verify that pytest automatically collects and parses agentci_spec.yaml."""
    pytester.makefile(".yaml", agentci_spec="""
agent: test-agent
runner: dummy:run
version: 1.0
queries:
  - query: "Hello World"
    description: "Basic greeting"
    """)
    
    # Write a dummy python runner so it doesn't fail import
    pytester.makepyfile(dummy="""
from ciagent.models import Trace
def run(query: str):
    return Trace(agent_name="dummy", test_name="t1")
    """)
    
    result = pytester.runpytest("--collect-only")
    result.stdout.fnmatch_lines([
        "*<AgentCIFile agentci_spec.yaml>*",
        "*<AgentCIItem 01_Hello World>*",
    ])
