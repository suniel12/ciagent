"""
Lightweight mock tools for zero-API-key testing.

Developers define mock responses in YAML or Python.
The demo agent ships with these pre-configured.
"""

from typing import Any, Callable
from .models import ToolCall


class MockTool:
    """
    A fake tool that returns predefined responses.
    
    Usage:
        search = MockTool(
            name="search_flights",
            responses={
                "default": {"flights": [{"id": 1, "price": 350}]},
                "no_results": {"flights": []},
            }
        )
        
        # In agent code, replace real tool with mock:
        result = search.call(origin="SFO", destination="JFK")
    """
    
    def __init__(
        self, 
        name: str, 
        responses: dict[str, Any] | None = None,
        handler: Callable[..., Any] | None = None,
        stateful: bool = False,
    ):
        self.name = name
        self.responses = responses or {"default": {}}
        self.handler = handler
        self.stateful = stateful
        self._state: dict[str, Any] = {}
        self._call_history: list[dict[str, Any]] = []
        self._scenario: str = "default"
    
    def set_scenario(self, scenario: str) -> None:
        """Switch to a named response scenario."""
        self._scenario = scenario
    
    def call(self, **kwargs) -> Any:
        """Execute the mock tool, recording the call."""
        self._call_history.append({"arguments": kwargs})
        
        if self.handler:
            assert self.handler is not None
            return self.handler(**kwargs, _state=self._state)
        
        return self.responses.get(self._scenario, self.responses["default"])
    
    @property
    def call_count(self) -> int:
        return len(self._call_history)
    
    def reset(self) -> None:
        self._call_history.clear()
        self._state.clear()
        self._scenario = "default"


class MockToolkit:
    """
    A collection of mock tools loaded from YAML.
    
    mocks.yaml:
        search_flights:
            default:
                flights:
                    - id: 1
                      price: 350
                      airline: "United"
            no_results:
                flights: []
        
        book_flight:
            default:
                confirmation: "ABC123"
                status: "confirmed"
    """
    
    def __init__(self):
        self.tools: dict[str, MockTool] = {}
    
    @classmethod
    def from_yaml(cls, path: str) -> "MockToolkit":
        # yaml is a dependency, so we import it at top level or here
        import yaml
        toolkit = cls()
        with open(path) as f:
            config = yaml.safe_load(f)
        
        for tool_name, responses in config.items():
            toolkit.tools[tool_name] = MockTool(
                name=tool_name,
                responses=responses,
            )
        
        return toolkit
    
    def get(self, name: str) -> MockTool:
        if name not in self.tools:
            raise KeyError(f"Mock tool '{name}' not found. Available: {list(self.tools)}")
        return self.tools[name]
    
    def set_all_scenarios(self, scenario: str) -> None:
        for tool in self.tools.values():
            tool.set_scenario(scenario)
    
    def reset_all(self) -> None:
        for tool in self.tools.values():
            tool.reset()


class AnthropicMocker:
    """
    Simulates a multi-turn Anthropic Claude agent loop.
    
    Instead of developers writing a 150-line fake client that parses 
    `messages` and yields `stop_reason="tool_use"`, this mocker
    takes a predetermined sequence of tool calls and automatically
    advances the simulation step-by-step.
    
    Usage:
        client = AnthropicMocker(
            mock_responses=[
                # Turn 1: Claude decides to call search()
                {"tool": "search_flights", "input": {"origin": "SFO"}},
                
                # Turn 2: Claude decides to book
                {"tool": "book_flight", "input": {"id": 123}},
                
                # Turn 3: Claude finishes and replies
                {"text": "I have booked your flight! Confirmation ABC."}
            ]
        )
        my_agent.client = client
    """
    
    def __init__(self, mock_responses: list[dict[str, Any]]):
        self.mock_responses = mock_responses
        self.turn_index = 0
        
        # Build the mock client structure that resembles anthropic.AsyncAnthropic
        from unittest.mock import AsyncMock, MagicMock
        
        self.client = AsyncMock()
        self.client.messages.create = AsyncMock(side_effect=self._mock_create)
        
    async def _mock_create(self, **kwargs) -> Any:
        from unittest.mock import MagicMock
        import json
        import uuid
        
        if self.turn_index >= len(self.mock_responses):
            # Fallback if the agent keeps calling
            response = MagicMock()
            response.stop_reason = "end_turn"
            
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Agent stopped because mock sequence ended."
            text_block.model_dump.return_value = {"type": "text", "text": text_block.text}
            
            response.content = [text_block]
            response.usage.input_tokens = 10
            response.usage.output_tokens = 10
            return response
            
        current_step = self.mock_responses[self.turn_index]
        self.turn_index += 1
        
        response = MagicMock()
        response.usage.input_tokens = current_step.get("input_tokens", 100)
        response.usage.output_tokens = current_step.get("output_tokens", 50)
        
        if "tool" in current_step:
            response.stop_reason = "tool_use"
            
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.id = f"toolu_{uuid.uuid4().hex[:16]}"
            tool_block.name = current_step["tool"]
            tool_block.input = current_step.get("input", {})
            
            tool_block.model_dump.return_value = {
                "type": "tool_use", 
                "id": tool_block.id, 
                "name": tool_block.name, 
                "input": tool_block.input
            }
            
            response.content = [tool_block]
            
        elif "text" in current_step:
            response.stop_reason = "end_turn"
            
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = current_step["text"]
            text_block.model_dump.return_value = {"type": "text", "text": text_block.text}
            
            response.content = [text_block]
            
        else:
            raise ValueError(f"Invalid mock response format at step {self.turn_index}: {current_step}")
            
        return response
