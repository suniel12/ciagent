# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
LangGraph Adapter.
"""
from .base import BaseAdapter
from ciagent.models import Trace, Span, ToolCall
import time
from typing import Any

class LangGraphAdapter(BaseAdapter):
    """
    Adapter for LangGraph agents.
    Expects the agent to return the final State dictionary.
    """
    
    def run(self, agent: Any, input_data: Any) -> Trace:
        start_time = time.monotonic()
        
        # Determine if the agent is async
        import inspect
        
        # Run the agent
        # Note: In a real test, the runner might handle async event loops
        # and just pass the result to the adapter to parse. For now
        # we assume a synchronous or already-awaited result.
        
        if callable(agent):
            try:
                # Need to handle async vs sync
                import asyncio
                if inspect.iscoroutinefunction(agent):
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()
                    final_state = asyncio.run(agent(input_data))
                else:
                    final_state = agent(input_data)
            except Exception as e:
                # Create a failed trace
                trace = Trace(
                    test_name="langgraph_test",
                    framework="langgraph",
                    total_duration_ms=(time.monotonic() - start_time) * 1000,
                )
                span = Span(name="execution_failed", error=str(e))
                trace.spans.append(span)
                trace.compute_metrics()
                return trace
        else:
            final_state = agent
            
        # Parse the LangGraph state
        trace = self.parse_state(final_state)
        trace.total_duration_ms = (time.monotonic() - start_time) * 1000
        trace.compute_metrics()
        
        return trace
        
    def parse_state(self, state: dict[str, Any]) -> Trace:
        """
        Parses a standard LangChain/LangGraph `messages` state list into a Trace.
        """
        trace = Trace(framework="langgraph", graph_state=state)

        messages = state.get("messages", [])

        # Extract final output from last AI message
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "ai":
                content = getattr(msg, "content", "")
                if content:
                    trace.metadata["final_output"] = str(content)
                    break

        current_span = Span(name="langgraph_execution")
        
        for msg in messages:
            # Check for tool calls (AIMessage)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_args = tc.get("args", {})
                    tool_call = ToolCall(
                        tool_name=tc.get("name", ""),
                        arguments=tool_args,
                        success=True
                    )
                    # Propagate tool args into span attributes for span assertions
                    current_span.attributes[f"tool.args.{tc.get('name', 'unknown')}"] = tool_args

                    # We try to pair it with the subsequent ToolMessage
                    # In a real parser we'd look ahead or map by tool_call_id
                    current_span.tool_calls.append(tool_call)
                    
            # Check for token usage
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                usage = msg.usage_metadata
                current_span.total_tokens_in += usage.get("input_tokens", 0)
                current_span.total_tokens_out += usage.get("output_tokens", 0)
                
            # Log all text bits as LLM Calls
            if hasattr(msg, "content") and msg.content and getattr(msg, "type", "") == "ai":
                current_span.llm_calls.append({"role": "ai", "content": str(msg.content)})
                
        trace.spans.append(current_span)
        return trace
