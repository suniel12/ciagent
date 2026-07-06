import time
from agentci.capture import _active_span
from agentci.models import ToolCall

def run_agent(input_text: str):
    """
    A specific "mock" agent that simulates tool calls based on input keywords.
    This allows us to test the CIAgent framework without a real LLM.
    """
    span = _active_span.get()
    
    # Check "Moon" first (handle_no_results)
    if "Moon" in input_text:
        # Simulate search only
        if span:
            span.tool_calls.append(ToolCall(
                tool_name="search_flights",
                arguments={"origin": "SFO", "destination": "MOON"}
            ))
        return "I'm sorry, I couldn't find any flights to the Moon."

    # Check "Book" and "flight" (book_flight_basic)
    elif "Book" in input_text and "flight" in input_text:
        # Simulate search
        if span:
            span.tool_calls.append(ToolCall(
                tool_name="search_flights",
                arguments={"origin": "SFO", "destination": "JFK", "date": "2023-03-15"}
            ))
        
        # Simulate thinking
        time.sleep(0.1)
        
        # Simulate booking
        if span:
            span.tool_calls.append(ToolCall(
                tool_name="book_flight",
                arguments={"flight_id": "FL123"}
            ))
            
        return "Booked flight FL123 for you."
    
    # Check "Find" (cost_guardrail)
    elif "Find" in input_text:
         # Simulate search for complex query
        if span:
            span.tool_calls.append(ToolCall(
                tool_name="search_flights",
                arguments={"origin": "SFO", "destination": "Tokyo", "stops": "Hawaii"}
            ))
        return "Here are the flights..."
        
    return "I don't understand."
