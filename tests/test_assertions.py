"""
Tests for assertions.
"""
from ciagent.assertions import evaluate_assertion
from ciagent.models import Assertion, Trace

def test_basic_assertion():
    t = Trace()
    a = Assertion(type="tool_called", tool="test_tool")
    passed, msg = evaluate_assertion(a, t)
    assert not passed
