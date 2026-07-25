"""
LangGraph ReAct Research Agent.

A research assistant built with LangGraph's StateGraph that uses a
ReAct (Reason + Act) loop to answer questions by searching and calculating.

Mock mode (default): Uses a deterministic "fake LLM" that picks tools
based on input keywords, while still routing through the OpenAI client
path so capture.py can intercept calls.

Live mode: Set CIAGENT_LIVE=1 and OPENAI_API_KEY to use real ChatOpenAI.
"""

import os
import json
import openai
from typing import Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END, START

LIVE_MODE = os.environ.get("CIAGENT_LIVE", "0") == "1"


# --- State definition ---

class AgentState(TypedDict):
    """State passed between graph nodes."""
    input: str
    messages: list[dict[str, Any]]
    tool_calls_pending: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    final_answer: str
    iteration: int


# --- Tool implementations ---

MOCK_SEARCH_RESULTS = {
    "population of france": "The population of France is approximately 68 million people (2025 estimate).",
    "gdp of us": "The GDP of the United States is approximately $28.78 trillion (2025).",
    "population of us": "The population of the United States is approximately 335 million people (2025).",
    "capital of japan": "The capital of Japan is Tokyo.",
    "capital of france": "The capital of France is Paris.",
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_answer",
            "description": "Save the final answer for the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final answer to save"},
                },
                "required": ["answer"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return result string."""
    if name == "web_search":
        query = arguments.get("query", "").lower()
        for key, value in MOCK_SEARCH_RESULTS.items():
            if key in query:
                return value
        return f"No results found for: {arguments.get('query', '')}"

    if name == "calculator":
        expression = arguments.get("expression", "")
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

    if name == "save_answer":
        answer = arguments.get("answer", "")
        return f"Answer saved: {answer}"

    return f"Unknown tool: {name}"


# --- Mock OpenAI response objects (same shape as openai_agent) ---

class _MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _MockToolCall:
    def __init__(self, function: _MockFunction, call_id: str = "call_001"):
        self.id = call_id
        self.type = "function"
        self.function = function


class _MockMessage:
    def __init__(self, content=None, tool_calls=None, role="assistant"):
        self.content = content
        self.tool_calls = tool_calls
        self.role = role


class _MockChoice:
    def __init__(self, message: _MockMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason
        self.index = 0


class _MockUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _MockResponse:
    def __init__(self, choices, usage, model="gpt-4o-mini"):
        self.choices = choices
        self.usage = usage
        self.model = model
        self.id = "chatcmpl-mock-lg"


def _build_mock_response(messages: list) -> _MockResponse:
    """Build mock OpenAI response for the LangGraph agent based on conversation."""
    # Extract user input from messages
    user_input = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_input = msg.get("content", "")

    lower = user_input.lower()
    last_msg = messages[-1] if messages else {}
    last_role = last_msg.get("role", "")

    # Count how many tool results we've already gotten
    tool_result_count = sum(1 for m in messages if m.get("role") == "tool")

    # If we just got a tool result, decide what's next
    if last_role == "tool":
        tool_content = last_msg.get("content", "")

        # If "save" is in the query and we haven't saved yet
        if "save" in lower and "Answer saved" not in tool_content:
            # Check if we've done enough research to save
            if tool_result_count >= 1 and "Answer saved" not in str(messages):
                return _MockResponse(
                    choices=[_MockChoice(
                        _MockMessage(tool_calls=[_MockToolCall(
                            _MockFunction("save_answer", json.dumps({"answer": tool_content})),
                            call_id="call_save",
                        )]),
                        finish_reason="tool_calls",
                    )],
                    usage=_MockUsage(prompt_tokens=300, completion_tokens=40),
                )

        # If "divided by" or math is involved and we have search results but no calc yet
        if ("divided by" in lower or "multiply" in lower or "plus" in lower) and tool_result_count == 1:
            # Need to search for the second piece of data
            if "gdp" in lower and "population" in lower:
                if "population" not in tool_content.lower() or "gdp" not in tool_content.lower():
                    # Search for the missing piece
                    if "gdp" in tool_content.lower():
                        return _MockResponse(
                            choices=[_MockChoice(
                                _MockMessage(tool_calls=[_MockToolCall(
                                    _MockFunction("web_search", json.dumps({"query": "population of US"})),
                                    call_id="call_search2",
                                )]),
                                finish_reason="tool_calls",
                            )],
                            usage=_MockUsage(prompt_tokens=350, completion_tokens=35),
                        )
                    else:
                        return _MockResponse(
                            choices=[_MockChoice(
                                _MockMessage(tool_calls=[_MockToolCall(
                                    _MockFunction("web_search", json.dumps({"query": "GDP of US"})),
                                    call_id="call_search2",
                                )]),
                                finish_reason="tool_calls",
                            )],
                            usage=_MockUsage(prompt_tokens=350, completion_tokens=35),
                        )

        # If we have enough data for calculation
        if ("divided by" in lower or "multiply" in lower) and tool_result_count >= 2:
            # Check if we already did the calculation
            calc_done = any(
                "calculator" in str(m.get("tool_calls", ""))
                for m in messages if m.get("role") == "assistant"
            )
            if not calc_done:
                return _MockResponse(
                    choices=[_MockChoice(
                        _MockMessage(tool_calls=[_MockToolCall(
                            _MockFunction("calculator", json.dumps({"expression": "28780000000000 / 335000000"})),
                            call_id="call_calc",
                        )]),
                        finish_reason="tool_calls",
                    )],
                    usage=_MockUsage(prompt_tokens=400, completion_tokens=30),
                )

        # Default: return final answer
        return _MockResponse(
            choices=[_MockChoice(
                _MockMessage(content=f"Based on my research: {tool_content}"),
                finish_reason="stop",
            )],
            usage=_MockUsage(prompt_tokens=250, completion_tokens=50),
        )

    # First call: decide which tool to use
    if "population" in lower:
        country = "france"
        if "us" in lower or "united states" in lower or "america" in lower:
            country = "us"
        return _MockResponse(
            choices=[_MockChoice(
                _MockMessage(tool_calls=[_MockToolCall(
                    _MockFunction("web_search", json.dumps({"query": f"population of {country}"})),
                    call_id="call_search1",
                )]),
                finish_reason="tool_calls",
            )],
            usage=_MockUsage(prompt_tokens=200, completion_tokens=30),
        )

    if "gdp" in lower:
        return _MockResponse(
            choices=[_MockChoice(
                _MockMessage(tool_calls=[_MockToolCall(
                    _MockFunction("web_search", json.dumps({"query": "GDP of US"})),
                    call_id="call_search1",
                )]),
                finish_reason="tool_calls",
            )],
            usage=_MockUsage(prompt_tokens=200, completion_tokens=30),
        )

    if "capital" in lower:
        country = "japan"
        if "france" in lower:
            country = "france"
        return _MockResponse(
            choices=[_MockChoice(
                _MockMessage(tool_calls=[_MockToolCall(
                    _MockFunction("web_search", json.dumps({"query": f"capital of {country}"})),
                    call_id="call_search1",
                )]),
                finish_reason="tool_calls",
            )],
            usage=_MockUsage(prompt_tokens=200, completion_tokens=30),
        )

    # Fallback
    return _MockResponse(
        choices=[_MockChoice(
            _MockMessage(content="I can help research topics. Try asking me a question!"),
            finish_reason="stop",
        )],
        usage=_MockUsage(prompt_tokens=100, completion_tokens=20),
    )


# --- Mock mode: module-level patching ---

_original_openai_create = openai.resources.chat.completions.Completions.create


def _mock_openai_create(self_client, *args, **kwargs):
    """Mock create for LangGraph agent."""
    messages = kwargs.get("messages", [])
    return _build_mock_response(messages)


if not LIVE_MODE:
    openai.resources.chat.completions.Completions.create = _mock_openai_create


def _activate_mock():
    """Re-apply this agent's mock (call before running tests for this agent)."""
    if not LIVE_MODE:
        openai.resources.chat.completions.Completions.create = _mock_openai_create


def _deactivate_mock():
    """Restore original create (call after tests for cross-agent isolation)."""
    openai.resources.chat.completions.Completions.create = _original_openai_create


# --- Graph nodes ---

def reasoning_node(state: AgentState) -> AgentState:
    """Call the LLM to decide what to do next."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=state["messages"],
        tools=TOOLS_SCHEMA,
    )

    choice = response.choices[0]
    assistant_msg = choice.message

    if assistant_msg.tool_calls:
        tool_calls_pending = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            for tc in assistant_msg.tool_calls
        ]
        # Add assistant message to conversation
        state["messages"].append({
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_msg.tool_calls
            ],
        })
        state["tool_calls_pending"] = tool_calls_pending
    else:
        state["final_answer"] = assistant_msg.content or ""
        state["tool_calls_pending"] = []

    state["iteration"] = state.get("iteration", 0) + 1
    return state


def tool_execution_node(state: AgentState) -> AgentState:
    """Execute pending tool calls and add results to messages."""
    results = []
    for tc in state["tool_calls_pending"]:
        result = execute_tool(tc["name"], tc["arguments"])
        results.append({"tool_id": tc["id"], "result": result})
        state["messages"].append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })

    state["tool_results"] = results
    state["tool_calls_pending"] = []
    return state


def should_continue(state: AgentState) -> str:
    """Decide whether to continue the loop or finish."""
    if state.get("final_answer"):
        return "end"
    if state.get("iteration", 0) >= 5:
        return "end"
    if state.get("tool_calls_pending"):
        return "tools"
    return "end"


# --- Build the graph ---

def _build_graph() -> StateGraph:
    """Build the LangGraph ReAct agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("reason", reasoning_node)
    graph.add_node("tools", tool_execution_node)

    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", should_continue, {
        "tools": "tools",
        "end": END,
    })
    graph.add_edge("tools", "reason")

    return graph.compile()


# --- Agent entry point ---

_compiled_graph = _build_graph()


def run_agent(input_text: str) -> str:
    """
    Research agent using LangGraph StateGraph.

    Uses a ReAct loop: reason (LLM call) -> act (tool execution) -> reason -> ...
    """
    initial_state: AgentState = {
        "input": input_text,
        "messages": [
            {"role": "system", "content": "You are a research assistant. Use tools to find information and answer questions accurately."},
            {"role": "user", "content": input_text},
        ],
        "tool_calls_pending": [],
        "tool_results": [],
        "final_answer": "",
        "iteration": 0,
    }

    result = _compiled_graph.invoke(initial_state)
    return result.get("final_answer", "No answer generated.")
