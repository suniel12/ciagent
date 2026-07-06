"""
OpenAI Function-Calling Weather Agent.

A weather assistant that uses OpenAI's function-calling API to look up
weather data. Supports both mock mode (default) and live API mode.

Mock mode (default): Patches the OpenAI client's create method to return
canned responses. The call still flows through the OpenAI client path,
so CIAgent's capture.py can intercept and record tool calls.

Live mode: Set AGENTCI_LIVE=1 and OPENAI_API_KEY to use real OpenAI API.
"""

import os
import json
import openai

LIVE_MODE = os.environ.get("AGENTCI_LIVE", "0") == "1"

# --- Tool implementations (used in both modes) ---

MOCK_WEATHER_DATA = {
    "San Francisco": {"temperature": 62, "conditions": "Foggy", "humidity": 78},
    "New York": {"temperature": 45, "conditions": "Cloudy", "humidity": 55},
    "NYC": {"temperature": 45, "conditions": "Cloudy", "humidity": 55},
    "London": {"temperature": 50, "conditions": "Rainy", "humidity": 85},
}

MOCK_FORECAST_DATA = {
    "San Francisco": [
        {"day": 1, "high": 64, "low": 52, "conditions": "Foggy"},
        {"day": 2, "high": 68, "low": 54, "conditions": "Partly Cloudy"},
        {"day": 3, "high": 66, "low": 53, "conditions": "Sunny"},
    ],
    "New York": [
        {"day": 1, "high": 48, "low": 35, "conditions": "Cloudy"},
        {"day": 2, "high": 52, "low": 38, "conditions": "Rainy"},
        {"day": 3, "high": 55, "low": 40, "conditions": "Sunny"},
    ],
    "NYC": [
        {"day": 1, "high": 48, "low": 35, "conditions": "Cloudy"},
        {"day": 2, "high": 52, "low": 38, "conditions": "Rainy"},
        {"day": 3, "high": 55, "low": 40, "conditions": "Sunny"},
    ],
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get multi-day weather forecast for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Number of forecast days"},
                },
                "required": ["city", "days"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool call and return JSON result."""
    if name == "get_weather":
        city = arguments.get("city", "")
        data = MOCK_WEATHER_DATA.get(city)
        if data:
            return json.dumps(data)
        return json.dumps({"error": f"No weather data available for '{city}'"})

    if name == "get_forecast":
        city = arguments.get("city", "")
        days = arguments.get("days", 3)
        forecast = MOCK_FORECAST_DATA.get(city)
        if forecast:
            return json.dumps(forecast[:days])
        return json.dumps({"error": f"No forecast data available for '{city}'"})

    return json.dumps({"error": f"Unknown tool: {name}"})


# --- Mock OpenAI-like response objects ---
# These mimic the shape of openai.types.chat.ChatCompletion so
# capture.py can extract tool calls and usage data from them.

class _MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _MockToolCall:
    def __init__(self, function: _MockFunction):
        self.id = "call_mock_001"
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


class _MockChatCompletion:
    def __init__(self, choices, usage, model="gpt-4o-mini"):
        self.choices = choices
        self.usage = usage
        self.model = model
        self.id = "chatcmpl-mock"


def _build_mock_response(user_input: str, messages: list) -> _MockChatCompletion:
    """Build a mock OpenAI response based on conversation state."""
    last_msg = messages[-1] if messages else {}
    role = last_msg.get("role", "")

    # If the last message is a tool result, generate final answer
    if role == "tool":
        tool_result = last_msg.get("content", "")
        return _MockChatCompletion(
            choices=[_MockChoice(
                _MockMessage(content=f"Based on the data: {tool_result}"),
                finish_reason="stop",
            )],
            usage=_MockUsage(prompt_tokens=150, completion_tokens=60),
        )

    # First call: decide which tool to use based on input
    lower = user_input.lower()

    if "forecast" in lower:
        city = "New York"
        days = 3
        if "san francisco" in lower or "sf" in lower:
            city = "San Francisco"
        if "nyc" in lower or "new york" in lower:
            city = "NYC"
        for word in lower.split():
            if word.isdigit():
                days = int(word)

        return _MockChatCompletion(
            choices=[_MockChoice(
                _MockMessage(tool_calls=[_MockToolCall(
                    _MockFunction("get_forecast", json.dumps({"city": city, "days": days}))
                )]),
                finish_reason="tool_calls",
            )],
            usage=_MockUsage(prompt_tokens=200, completion_tokens=30),
        )

    if "weather" in lower or "temperature" in lower:
        city = "San Francisco"
        if "nyc" in lower or "new york" in lower:
            city = "New York"
        if "london" in lower:
            city = "London"
        if "mars" in lower:
            city = "Mars"

        return _MockChatCompletion(
            choices=[_MockChoice(
                _MockMessage(tool_calls=[_MockToolCall(
                    _MockFunction("get_weather", json.dumps({"city": city}))
                )]),
                finish_reason="tool_calls",
            )],
            usage=_MockUsage(prompt_tokens=180, completion_tokens=25),
        )

    # Fallback: no tool call
    return _MockChatCompletion(
        choices=[_MockChoice(
            _MockMessage(content="I can help with weather lookups. Try asking about weather in a city!"),
            finish_reason="stop",
        )],
        usage=_MockUsage(prompt_tokens=100, completion_tokens=20),
    )


# --- Mock mode: module-level patching ---
# Applied at import time so it's in place BEFORE capture.py's TraceContext
# patches. TraceContext wraps this mock, calls it as "original_create",
# and records the tool calls / usage from the mock response.

_original_openai_create = openai.resources.chat.completions.Completions.create


def _mock_openai_create(self_client, *args, **kwargs):
    """Mock create that returns canned responses based on message content."""
    messages = kwargs.get("messages", [])
    user_input = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_input = msg.get("content", "")
    return _build_mock_response(user_input, messages)


if not LIVE_MODE:
    openai.resources.chat.completions.Completions.create = _mock_openai_create


def _activate_mock():
    """Re-apply this agent's mock (call before running tests for this agent)."""
    if not LIVE_MODE:
        openai.resources.chat.completions.Completions.create = _mock_openai_create


def _deactivate_mock():
    """Restore original create (call after tests for cross-agent isolation)."""
    openai.resources.chat.completions.Completions.create = _original_openai_create


# --- Agent loop ---

def run_agent(input_text: str) -> str:
    """
    Weather agent that uses OpenAI function-calling.

    Runs a tool-calling loop:
    send message -> get tool call -> execute tool -> send result -> get answer.
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))

    messages = [
        {"role": "system", "content": "You are a helpful weather assistant. Use the provided tools to look up weather data."},
        {"role": "user", "content": input_text},
    ]

    max_iterations = 5
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        # If no tool calls, we're done
        if not assistant_msg.tool_calls:
            return assistant_msg.content or ""

        # Process each tool call
        messages.append({
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

        for tc in assistant_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "I wasn't able to complete the request within the allowed steps."
