"""
Anthropic Tool-Use Article Summarizer Agent.

An article summarizer that uses Claude's tool_use API to fetch and
summarize articles. Supports both mock mode (default) and live API mode.

Mock mode (default): Patches the Anthropic client's create method to return
canned responses with tool_use content blocks. The call still flows through
the Anthropic client path, so CIAgent's capture.py can intercept and record.

Live mode: Set AGENTCI_LIVE=1 and ANTHROPIC_API_KEY to use real Claude API.
"""

import os
import json
import anthropic

LIVE_MODE = os.environ.get("AGENTCI_LIVE", "0") == "1"

# --- Tool implementations ---

MOCK_ARTICLES = {
    "https://example.com/ai": {
        "title": "The Rise of AI Agents",
        "text": "AI agents are autonomous software programs that can perceive their "
                "environment, make decisions, and take actions to achieve goals. "
                "They represent a significant shift from traditional software that "
                "follows pre-defined rules. Modern agents leverage large language "
                "models to reason about tasks, call tools, and adapt their behavior "
                "based on context. Key applications include customer service, "
                "software development, and scientific research.",
    },
    "https://example.com/climate": {
        "title": "Climate Change Report 2026",
        "text": "Global temperatures have risen by 1.2 degrees Celsius since "
                "pre-industrial times. The report highlights the urgent need for "
                "carbon emission reductions and investment in renewable energy. "
                "Key findings include accelerating ice sheet loss and rising sea "
                "levels threatening coastal communities worldwide.",
    },
}

TOOLS_SCHEMA = [
    {
        "name": "fetch_article",
        "description": "Fetch the full text of an article from a URL",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The article URL to fetch"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "save_summary",
        "description": "Save a summary of an article",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Article title"},
                "summary": {"type": "string", "description": "The summary text"},
            },
            "required": ["title", "summary"],
        },
    },
]

_saved_summaries: list[dict] = []


def execute_tool(name: str, tool_input: dict) -> str:
    """Execute a tool call and return result."""
    if name == "fetch_article":
        url = tool_input.get("url", "")
        article = MOCK_ARTICLES.get(url)
        if article:
            return json.dumps(article)
        return json.dumps({"error": f"Article not found at '{url}'"})

    if name == "save_summary":
        title = tool_input.get("title", "")
        summary = tool_input.get("summary", "")
        _saved_summaries.append({"title": title, "summary": summary})
        return json.dumps({"status": "saved", "title": title})

    return json.dumps({"error": f"Unknown tool: {name}"})


# --- Mock Anthropic-like response objects ---
# These mimic anthropic.types.Message so capture.py can extract
# tool_use blocks and usage data.

class _MockTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _MockToolUseBlock:
    def __init__(self, name: str, tool_input: dict):
        self.type = "tool_use"
        self.id = f"toolu_mock_{name}"
        self.name = name
        self.input = tool_input


class _MockUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockAnthropicMessage:
    def __init__(self, content, usage, model="claude-haiku-4-5-20251001",
                 stop_reason="end_turn", role="assistant"):
        self.content = content
        self.usage = usage
        self.model = model
        self.stop_reason = stop_reason
        self.role = role
        self.id = "msg_mock_001"
        self.type = "message"


def _build_mock_response(user_input: str, messages: list) -> _MockAnthropicMessage:
    """Build a mock Anthropic response based on conversation state."""
    last_msg = messages[-1] if messages else {}
    last_role = last_msg.get("role", "")

    # If last message is a tool_result, decide next action
    if last_role == "user" and isinstance(last_msg.get("content"), list):
        content_blocks = last_msg["content"]
        tool_results = [b for b in content_blocks if b.get("type") == "tool_result"]

        if tool_results:
            last_tool_result = tool_results[-1]
            result_content = last_tool_result.get("content", "")

            # If we just fetched an article and user wants it saved, save it
            try:
                article_data = json.loads(result_content)
                if "title" in article_data and "text" in article_data:
                    # Article was fetched — should we save?
                    lower = user_input.lower()
                    if "don't save" in lower or "no save" in lower or "just" in lower:
                        # User doesn't want to save, return summary directly
                        summary = article_data["text"][:150] + "..."
                        return _MockAnthropicMessage(
                            content=[_MockTextBlock(
                                f"Here's a summary of \"{article_data['title']}\": {summary}"
                            )],
                            usage=_MockUsage(input_tokens=300, output_tokens=80),
                            stop_reason="end_turn",
                        )
                    else:
                        # Save the summary
                        summary = article_data["text"][:150] + "..."
                        return _MockAnthropicMessage(
                            content=[_MockToolUseBlock(
                                "save_summary",
                                {"title": article_data["title"], "summary": summary},
                            )],
                            usage=_MockUsage(input_tokens=350, output_tokens=60),
                            stop_reason="tool_use",
                        )
            except (json.JSONDecodeError, KeyError):
                pass

            # After saving, return final message
            return _MockAnthropicMessage(
                content=[_MockTextBlock(
                    "I've fetched and saved the article summary for you."
                )],
                usage=_MockUsage(input_tokens=200, output_tokens=30),
                stop_reason="end_turn",
            )

    # First call: extract URL and fetch the article
    lower = user_input.lower()
    url = None
    for known_url in MOCK_ARTICLES:
        if known_url in user_input:
            url = known_url
            break

    if url:
        return _MockAnthropicMessage(
            content=[_MockToolUseBlock("fetch_article", {"url": url})],
            usage=_MockUsage(input_tokens=250, output_tokens=40),
            stop_reason="tool_use",
        )

    # No URL found
    return _MockAnthropicMessage(
        content=[_MockTextBlock(
            "I'd be happy to summarize an article for you. Please provide a URL."
        )],
        usage=_MockUsage(input_tokens=100, output_tokens=25),
        stop_reason="end_turn",
    )


# --- Mock mode: module-level patching ---

_original_anthropic_create = anthropic.resources.messages.Messages.create


def _mock_anthropic_create(self_client, *args, **kwargs):
    """Mock create that returns canned Anthropic responses."""
    messages = kwargs.get("messages", [])
    user_input = ""
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            user_input = msg["content"]
    return _build_mock_response(user_input, messages)


if not LIVE_MODE:
    anthropic.resources.messages.Messages.create = _mock_anthropic_create


def _activate_mock():
    """Re-apply this agent's mock (call before running tests for this agent)."""
    if not LIVE_MODE:
        anthropic.resources.messages.Messages.create = _mock_anthropic_create


def _deactivate_mock():
    """Restore original create (call after tests for cross-agent isolation)."""
    anthropic.resources.messages.Messages.create = _original_anthropic_create


# --- Agent loop ---

def run_agent(input_text: str) -> str:
    """
    Article summarizer agent using Anthropic's tool_use API.

    Runs a tool-calling loop:
    send message -> get tool_use block -> execute tool -> send result -> repeat.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "mock-key"))

    messages = [
        {"role": "user", "content": input_text},
    ]

    max_iterations = 5
    for _ in range(max_iterations):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system="You are a helpful article summarizer. Use the provided tools to fetch articles and save summaries.",
            messages=messages,
            tools=TOOLS_SCHEMA,
        )

        # Check if there are tool_use blocks
        tool_use_blocks = [b for b in response.content if getattr(b, "type", "") == "tool_use"]

        if not tool_use_blocks:
            # No tool calls — extract text and return
            text_parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
            return " ".join(text_parts)

        # Process tool calls
        tool_results = []
        for block in tool_use_blocks:
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        # Add assistant response and tool results to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "I wasn't able to complete the request within the allowed steps."
