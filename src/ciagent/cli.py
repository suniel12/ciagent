# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
CIAgent Command Line Interface.

Commands:
  ciagent init          Scaffold a new test suite
  ciagent run           Execute test suite
  ciagent run --runs N  Statistical mode (run N times)
  ciagent record        Run agent live, save golden trace
  ciagent diff          Compare latest run against golden
  ciagent report        Generate HTML report from last run
"""
from __future__ import annotations

import os
import sys
import shutil
import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

from .config import load_config
from .runner import TestRunner
from .models import TestResult

console = Console()


def _route_chrome(fmt: str) -> None:
    """In --format json, stdout must carry exactly one JSON document (#39):
    all rich chrome (banner, mode line, progress, warnings) moves to stderr.
    Resolved dynamically at write time, so test runners that swap sys.stderr
    still capture it."""
    console.stderr = fmt == "json"


def _print_error_panel(e):
    from rich.panel import Panel
    from rich.text import Text
    raw_msg = str(e)
    fix_idx = raw_msg.find("\n  Fix: ")
    if fix_idx != -1:
        raw_msg = raw_msg[:fix_idx]
    text = Text(raw_msg)
    if getattr(e, "fix", None):
        text.append("\n\n💡 Fix: ", style="bold green")
        text.append(e.fix)
    console.print(Panel(text, title=f"[bold red]{e.__class__.__name__}[/]", border_style="red"))

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("ciagent")
except Exception:
    __version__ = "0.0.0"


# ── ciagent init --generate helpers ───────────────────────────────────────────

_AGENT_KEYWORDS = [
    "@tool", "def retrieve", "def run", "SystemMessage",
    "bind_tools", "add_node", "add_edge", "ChatOpenAI", "ChatAnthropic",
    # Broader framework detection
    "anthropic.Anthropic", "openai.OpenAI", "client.messages.create",
    "client.chat.completions.create", "tool_use", "function_call",
    "create_sdk_mcp_server", "ToolNode",
    "CrewBase", "crew_ai", "Agent(", "Task(",
]
_KB_DIR_NAMES = {"knowledge_base", "kb", "docs", "data", "knowledge"}
_SKIP_DIRS = {"__pycache__", ".venv", "venv", ".git", "node_modules", "dist"}
_MAX_CONTEXT_CHARS = 48_000  # ~12k tokens at ~4 chars/token
_MAX_KB_FILE_CHARS = 2000    # deep-read per KB file

# Agent-type detection keywords (used in guided init interview)
_RAG_KEYWORDS = {"knowledge", "document", "retriev", "rag", "faq", "support", "qa", "question"}
_TOOL_KEYWORDS = {"tool", "function", "api", "action", "booking", "search", "plugin"}


def _scan_project(project_dir, kb_override: str | None = None) -> dict:
    """Scan project directory and return context for LLM test generation.

    Parameters
    ----------
    project_dir : path-like
        Root directory of the agent project.
    kb_override : str | None
        If provided, use this path as the knowledge base directory instead
        of auto-detecting from ``_KB_DIR_NAMES``.
    """
    from pathlib import Path

    project_dir = Path(project_dir)
    context: dict = {
        "agent_files": [],
        "knowledge_base": [],
        "existing_tests": [],
    }

    # 1. Agent code
    for py_file in project_dir.rglob("*.py"):
        parts = py_file.parts
        if any(skip in parts for skip in _SKIP_DIRS):
            continue
        if "tests" in parts or "test" in parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(kw in content for kw in _AGENT_KEYWORDS):
            lines = content.splitlines()[:150]
            context["agent_files"].append({
                "path": str(py_file.relative_to(project_dir)),
                "content": "\n".join(lines),
            })

    # 2. Knowledge base — deep sampling (up to _MAX_KB_FILE_CHARS per file)
    if kb_override is not None:
        kb_dirs_to_check = [Path(kb_override)]
    else:
        kb_dirs_to_check = [project_dir / d for d in _KB_DIR_NAMES]
    for kb_dir in kb_dirs_to_check:
        if not kb_dir.is_dir():
            continue
        # Sort by size ascending — smaller, focused files are more useful per char
        kb_files = sorted(
            [f for f in kb_dir.rglob("*") if f.suffix.lower() in {".md", ".txt"}],
            key=lambda f: f.stat().st_size,
        )
        for kb_file in kb_files:
            try:
                content = kb_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if len(content) > _MAX_KB_FILE_CHARS:
                snippet = content[:1000] + "\n...\n" + content[-500:]
            else:
                snippet = content
            context["knowledge_base"].append({
                "path": str(kb_file.relative_to(project_dir)),
                "snippet": snippet,
            })

    # 3. Existing tests
    for test_dir_name in ("tests", "test"):
        test_dir = project_dir / test_dir_name
        if not test_dir.is_dir():
            continue
        for test_file in test_dir.rglob("*.py"):
            try:
                lines = test_file.read_text(encoding="utf-8", errors="ignore").splitlines()[:100]
            except OSError:
                continue
            context["existing_tests"].append({
                "path": str(test_file.relative_to(project_dir)),
                "content": "\n".join(lines),
            })
        break  # only first matching test dir

    # Enforce total context limit (truncate longest pieces first)
    def _total_chars(ctx: dict) -> int:
        total = 0
        for f in ctx["agent_files"]:
            total += len(f["content"])
        for f in ctx["knowledge_base"]:
            total += len(f["snippet"])
        for f in ctx["existing_tests"]:
            total += len(f["content"])
        return total

    while _total_chars(context) > _MAX_CONTEXT_CHARS:
        # Find largest item and truncate it by 20%
        candidates = []
        for group, key in [("agent_files", "content"), ("knowledge_base", "snippet"), ("existing_tests", "content")]:
            for i, item in enumerate(context[group]):
                candidates.append((len(item[key]), group, i, key))
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, group, idx, key = candidates[0]
        current = context[group][idx][key]
        context[group][idx][key] = current[:int(len(current) * 0.8)]

    return context


def _detect_agent_type(description: str) -> str:
    """Infer agent type from user description. Returns 'rag', 'tool', or 'conversational'."""
    desc_lower = description.lower()
    if any(kw in desc_lower for kw in _RAG_KEYWORDS):
        return "rag"
    if any(kw in desc_lower for kw in _TOOL_KEYWORDS):
        return "tool"
    return "conversational"


_RAG_CODE_KEYWORDS = {"retriev", "vector", "embedding", "search_docs", "rag", "knowledge_base"}


def _detect_agent_type_from_code(
    context: dict, detected_tools: list[str], detected_kb: str | None,
) -> str:
    """Infer agent type from actual code analysis.

    Priority:
    1. KB directory exists → "rag"
    2. Agent code contains retrieval-related keywords → "rag"
    3. Tools detected → "tool"
    4. Fallback → "conversational"
    """
    if detected_kb is not None:
        return "rag"

    # Check agent file contents for retrieval patterns
    for agent_file in context.get("agent_files", []):
        content_lower = agent_file.get("content", "").lower()
        if any(kw in content_lower for kw in _RAG_CODE_KEYWORDS):
            return "rag"

    if detected_tools:
        return "tool"

    return "conversational"


def _detect_tools_from_code(project_dir) -> list[str]:
    """Scan agent code for tool/function definitions and return their names."""
    import re
    from pathlib import Path

    project_dir = Path(project_dir)
    tools: list[str] = []

    # Pattern 1: @tool decorator (existing)
    tool_decorator_pattern = re.compile(r'@tool\s*(?:\(.*?\))?\s*\ndef\s+(\w+)', re.DOTALL)

    # Pattern 2: .bind_tools([...]) (existing)
    bind_pattern = re.compile(r'\.bind_tools\s*\(\s*\[([^\]]+)\]', re.DOTALL)

    # Pattern 3: Anthropic/OpenAI tool schema dicts — "name": "tool_name"
    tool_schema_pattern = re.compile(r'["\']name["\']\s*:\s*["\'](\w+)["\']')

    # Pattern 4: ToolNode([func1, func2]) — LangGraph
    tool_node_pattern = re.compile(r'ToolNode\s*\(\s*\[([^\]]+)\]', re.DOTALL)

    # Pattern 5: @tool("name", ...) — first string arg IS the tool name
    tool_name_arg_pattern = re.compile(r'@tool\s*\(\s*["\'](\w+)["\']', re.DOTALL)

    for py_file in project_dir.rglob("*.py"):
        parts = py_file.parts
        if any(skip in parts for skip in _SKIP_DIRS):
            continue
        if "tests" in parts or "test" in parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Pattern 1: @tool decorated functions
        tools.extend(tool_decorator_pattern.findall(content))

        # Pattern 2: bind_tools([tool1, tool2])
        for match in bind_pattern.findall(content):
            if "{" in match:
                name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"')
                tools.extend(name_pattern.findall(match))
            else:
                tools.extend(name.strip().strip("'\"") for name in match.split(","))

        # Pattern 3: Tool schema dicts with "name" key
        # Only extract if this looks like a tools definition context
        if any(kw in content for kw in ("tools", "input_schema", "parameters", "function")):
            tools.extend(tool_schema_pattern.findall(content))

        # Pattern 4: ToolNode([func1, func2])
        for match in tool_node_pattern.findall(content):
            tools.extend(name.strip().strip("'\"") for name in match.split(","))

        # Pattern 5: @tool("name", ...) — name as first arg
        tools.extend(tool_name_arg_pattern.findall(content))

    # Deduplicate, preserve order, filter out common false positives
    _FALSE_POSITIVES = {"name", "type", "string", "object", "description", "text", "input", "output"}
    seen: set[str] = set()
    unique: list[str] = []
    for t in tools:
        if t and t not in seen and t.lower() not in _FALSE_POSITIVES:
            seen.add(t)
            unique.append(t)
    return unique


def _detect_kb_dir(project_dir) -> str | None:
    """Return the first matching knowledge base directory path, or None."""
    from pathlib import Path
    project_dir = Path(project_dir)
    for dir_name in _KB_DIR_NAMES:
        kb_dir = project_dir / dir_name
        if kb_dir.is_dir():
            kb_files = [f for f in kb_dir.rglob("*") if f.suffix.lower() in {".md", ".txt"}]
            if kb_files:
                return str(kb_dir)
    return None


def _load_golden_pairs(path: str) -> list[dict]:
    """Load golden Q&A pairs from a JSON or CSV file.

    Expected formats:
    - JSON: list of {"question": "...", "answer": "..."}
    - CSV: columns named 'question' and 'answer'
    """
    import json
    from pathlib import Path

    path_obj = Path(path)
    if not path_obj.exists():
        return []

    if path_obj.suffix.lower() == ".json":
        with open(path_obj) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [p for p in data if isinstance(p, dict) and "question" in p and "answer" in p]
        return []

    if path_obj.suffix.lower() == ".csv":
        import csv
        pairs: list[dict] = []
        with open(path_obj, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "question" in row and "answer" in row:
                    pairs.append({"question": row["question"], "answer": row["answer"]})
        return pairs

    return []


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "that",
    "this", "these", "those", "with", "from", "into", "about", "for",
    "and", "but", "not", "you", "your", "our", "their", "its", "also",
    "just", "only", "very", "more", "most", "some", "any", "each",
    "which", "who", "whom", "what", "when", "where", "how", "than",
    "then", "there", "here", "other", "such", "like", "well", "back",
})


def _extract_keywords_from_answer(answer: str, max_keywords: int = 5) -> list[str]:
    """Extract distinctive keywords from a golden-file answer.

    Returns up to *max_keywords* words (>3 chars, not stopwords) that can
    serve as ``any_expected_in_answer`` assertions.  Users are expected to
    refine these after generation.
    """
    import re
    words = re.findall(r"[A-Za-z0-9_.$/-]{4,}", answer)
    seen: set[str] = set()
    keywords: list[str] = []
    for w in words:
        low = w.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        keywords.append(w)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _build_golden_queries(pairs: list[dict]) -> list[dict]:
    """Convert golden Q&A pairs into query dicts with keyword assertions."""
    queries: list[dict] = []
    for p in pairs:
        query_dict: dict = {
            "query": p["question"],
            "description": f"Golden: {p['question'][:50]}",
        }
        if p.get("answer"):
            keywords = _extract_keywords_from_answer(p["answer"])
            if keywords:
                query_dict["correctness"] = {"any_expected_in_answer": keywords}
        queries.append(query_dict)
    return queries


_RUNNER_FN_NAMES = {"run_for_agentci", "run_agent", "run_for_agent", "run"}
_RUNNER_BODY_HINTS = (
    # Existing: explicit CIAgent trace usage
    "ctx.trace", "-> Trace", "TraceContext", "langgraph_trace",
    # Functions that call LLM APIs (auto-wrapped by TraceContext in parallel.py)
    "client.messages.create", "client.chat.completions.create",
    "anthropic.Anthropic", "openai.OpenAI",
    # Framework entry points
    "graph.invoke", "crew.kickoff", ".run(",
)


def _detect_runner(project_dir) -> str | None:
    """Scan project files and return a best-guess 'module:function' runner path."""
    import re
    from pathlib import Path

    project_dir = Path(project_dir)
    fn_pattern = re.compile(r"^def (\w+)\s*\(", re.MULTILINE)

    best: tuple[int, str] | None = None  # (priority, "module:fn")

    for py_file in project_dir.rglob("*.py"):
        parts = py_file.parts
        if any(skip in parts for skip in _SKIP_DIRS):
            continue
        if "tests" in parts or "test" in parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Only consider files that look like runner files
        if not any(hint in content for hint in _RUNNER_BODY_HINTS):
            continue

        # Convert file path to dotted module name
        try:
            rel = py_file.relative_to(project_dir)
        except ValueError:
            continue
        module = ".".join(rel.with_suffix("").parts)

        for fn_name in fn_pattern.findall(content):
            priority = (
                0 if fn_name == "run_for_agentci" else
                1 if fn_name == "run_agent" else
                2 if fn_name in _RUNNER_FN_NAMES else
                3
            )
            if best is None or priority < best[0]:
                best = (priority, f"{module}:{fn_name}")

    return best[1] if best else None


def _generate_tool_schemas(tools: list[str], sdk: str) -> str:
    """Generate tool schema definitions as Python code."""
    if not tools:
        return 'TOOLS = []  # No tools detected — add your tool schemas here'

    if sdk == "anthropic":
        schemas = []
        for tool in tools:
            schemas.append(
                '    {\n'
                f'        "name": "{tool}",\n'
                f'        "description": "TODO: Describe what {tool} does",\n'
                '        "input_schema": {\n'
                '            "type": "object",\n'
                '            "properties": {\n'
                '                "input": {\n'
                '                    "type": "string",\n'
                '                    "description": "TODO: Describe the input"\n'
                '                }\n'
                '            },\n'
                '            "required": ["input"],\n'
                '        },\n'
                '    }'
            )
        return "TOOLS = [\n" + ",\n".join(schemas) + "\n]"

    if sdk == "openai":
        schemas = []
        for tool in tools:
            schemas.append(
                '    {\n'
                '        "type": "function",\n'
                '        "function": {\n'
                f'            "name": "{tool}",\n'
                f'            "description": "TODO: Describe what {tool} does",\n'
                '            "parameters": {\n'
                '                "type": "object",\n'
                '                "properties": {\n'
                '                    "input": {"type": "string", "description": "TODO"}\n'
                '                },\n'
                '                "required": ["input"],\n'
                '            },\n'
                '        },\n'
                '    }'
            )
        return "TOOLS = [\n" + ",\n".join(schemas) + "\n]"

    return 'TOOLS = []'


def _generate_tool_implementations(tools: list[str]) -> str:
    """Generate placeholder tool implementations."""
    if not tools:
        return '    return f"Unknown tool: {name}"'

    lines = []
    for tool in tools:
        lines.append(f'    if name == "{tool}":')
        lines.append(f'        # TODO: Implement {tool} logic')
        lines.append(f'        return f"Result from {tool}: {{arguments}}"')
        lines.append('')
    lines.append('    return f"Unknown tool: {name}"')
    return "\n".join(lines)


def _generate_runner_file(
    project_dir,
    context: dict,
    detected_tools: list[str],
    agent_type: str,
) -> str | None:
    """Auto-generate a runner.py file based on detected agent code.

    Scans agent files to determine which SDK is used (anthropic, openai, langgraph)
    and generates a runner that:
    1. Imports the relevant SDK
    2. Defines tool schemas matching detected tools
    3. Implements a ReAct-style agent loop
    4. Returns a str (TraceContext auto-wrapping handles the rest)

    Returns the module:function path (e.g., "agentci_runner:run_agent") or None.
    """
    import re
    from pathlib import Path

    project_dir = Path(project_dir)
    runner_path = project_dir / "agentci_runner.py"

    # Detect which SDK the project uses
    sdk = None
    model = None

    for agent_file in context.get("agent_files", []):
        content = agent_file.get("content", "")

        if "anthropic" in content or "claude" in content.lower():
            sdk = "anthropic"
            model_match = re.search(r'model\s*=\s*["\']([^"\']+)["\']', content)
            if model_match:
                model = model_match.group(1)
        elif "openai" in content or "ChatOpenAI" in content:
            sdk = "openai"
            model_match = re.search(r'model\s*=\s*["\']([^"\']+)["\']', content)
            if model_match:
                model = model_match.group(1)

    if not sdk:
        return None

    # Default models
    if not model:
        model = "claude-sonnet-4-20250514" if sdk == "anthropic" else "gpt-4o"

    # Generate tool schemas
    tool_schemas_code = _generate_tool_schemas(detected_tools, sdk)
    tool_impl_code = _generate_tool_implementations(detected_tools)

    # Generate the runner file
    if sdk == "anthropic":
        runner_code = (
            '"""Auto-generated CIAgent runner.\n'
            'Created by: ciagent init --generate\n'
            '\n'
            'This runner wraps your agent\'s tools using the Anthropic SDK.\n'
            'CIAgent automatically captures all LLM calls and tool invocations\n'
            'via monkey-patching — just return the final answer as a string.\n'
            '\n'
            'Edit the tool implementations in execute_tool() to match your agent\'s logic.\n'
            '"""\n'
            'import os\n'
            'from typing import Any\n'
            'import anthropic\n'
            '\n'
            f'MODEL = os.environ.get("ANTHROPIC_MODEL", "{model}")\n'
            'MAX_ITERATIONS = 10\n'
            '\n'
            'SYSTEM_PROMPT = (\n'
            '    "You are a helpful assistant. Use the provided tools when appropriate. "\n'
            '    "After getting tool results, provide a clear final answer."\n'
            ')\n'
            '\n'
            f'{tool_schemas_code}\n'
            '\n'
            'def execute_tool(name: str, arguments: dict[str, Any]) -> str:\n'
            '    """Execute a tool by name. TODO: Implement your tool logic here."""\n'
            f'{tool_impl_code}\n'
            '\n'
            'def run_agent(query: str) -> str:\n'
            '    """CIAgent runner entry point. Returns the agent\'s final answer."""\n'
            '    client = anthropic.Anthropic()\n'
            '    messages = [{"role": "user", "content": query}]\n'
            '\n'
            '    for _ in range(MAX_ITERATIONS):\n'
            '        response = client.messages.create(\n'
            '            model=MODEL,\n'
            '            max_tokens=1024,\n'
            '            system=SYSTEM_PROMPT,\n'
            '            tools=TOOLS,\n'
            '            messages=messages,\n'
            '        )\n'
            '\n'
            '        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]\n'
            '\n'
            '        if not tool_use_blocks:\n'
            '            text_blocks = [b for b in response.content if b.type == "text"]\n'
            '            return " ".join(b.text for b in text_blocks) if text_blocks else ""\n'
            '\n'
            '        messages.append({"role": "assistant", "content": response.content})\n'
            '\n'
            '        tool_results = []\n'
            '        for tool_block in tool_use_blocks:\n'
            '            result = execute_tool(tool_block.name, tool_block.input)\n'
            '            tool_results.append({\n'
            '                "type": "tool_result",\n'
            '                "tool_use_id": tool_block.id,\n'
            '                "content": result,\n'
            '            })\n'
            '\n'
            '        messages.append({"role": "user", "content": tool_results})\n'
            '\n'
            '    return "Max iterations reached."\n'
        )
    elif sdk == "openai":
        runner_code = (
            '"""Auto-generated CIAgent runner.\n'
            'Created by: ciagent init --generate\n'
            '"""\n'
            'import os\n'
            'import json\n'
            'from typing import Any\n'
            'import openai\n'
            '\n'
            f'MODEL = os.environ.get("OPENAI_MODEL", "{model}")\n'
            'MAX_ITERATIONS = 10\n'
            '\n'
            f'{tool_schemas_code}\n'
            '\n'
            'def execute_tool(name: str, arguments: dict[str, Any]) -> str:\n'
            '    """Execute a tool by name. TODO: Implement your tool logic here."""\n'
            f'{tool_impl_code}\n'
            '\n'
            'def run_agent(query: str) -> str:\n'
            '    """CIAgent runner entry point. Returns the agent\'s final answer."""\n'
            '    client = openai.OpenAI()\n'
            '    messages = [{"role": "user", "content": query}]\n'
            '\n'
            '    for _ in range(MAX_ITERATIONS):\n'
            '        response = client.chat.completions.create(\n'
            '            model=MODEL,\n'
            '            messages=messages,\n'
            '            tools=TOOLS,\n'
            '        )\n'
            '        msg = response.choices[0].message\n'
            '\n'
            '        if not msg.tool_calls:\n'
            '            return msg.content or ""\n'
            '\n'
            '        messages.append(msg)\n'
            '\n'
            '        for tc in msg.tool_calls:\n'
            '            args = json.loads(tc.function.arguments)\n'
            '            result = execute_tool(tc.function.name, args)\n'
            '            messages.append({\n'
            '                "role": "tool",\n'
            '                "tool_call_id": tc.id,\n'
            '                "content": result,\n'
            '            })\n'
            '\n'
            '    return "Max iterations reached."\n'
        )
    else:
        return None

    runner_path.write_text(runner_code, encoding="utf-8")
    return "agentci_runner:run_agent"


def _prompt_for_queries_interactive() -> list[str]:
    """Prompt user to type test queries one-by-one for mock mode.

    Returns a list of query strings. Empty list if user enters nothing.
    """
    from rich.prompt import Prompt

    queries: list[str] = []
    console.print("\n[dim]Type test queries one per line. Enter 'done' or empty line to finish.[/]")
    while True:
        q = Prompt.ask(f"  Query {len(queries) + 1}", default="done")
        if q.lower() == "done" or not q.strip():
            break
        queries.append(q.strip())
    return queries


def _generate_skeleton_spec(
    agent_type: str,
    detected_tools: list[str],
    runner_path: str,
) -> str:
    """Generate a skeleton agentci_spec.yaml with TODO placeholders.

    Used when mock mode is selected and no queries are provided (golden file
    or interactive). Creates template queries based on detected agent type.
    """
    queries_yaml = ""

    if agent_type == "rag":
        queries_yaml = """
  - query: "TODO: Replace with an in-scope question your KB can answer"
    description: "Happy-path KB retrieval"
    correctness:
      any_expected_in_answer: ["TODO: keyword1", "TODO: keyword2"]
    cost:
      max_llm_calls: 10

  - query: "What is the weather today?"
    description: "Out-of-scope question — agent should decline"
    path:
      max_tool_calls: 0
    cost:
      max_llm_calls: 2

  - query: "Hello"
    description: "Boundary — greeting / off-topic"
    cost:
      max_llm_calls: 2
"""
    elif agent_type == "tool":
        for tool in list(detected_tools)[:5]:
            queries_yaml += f"""
  - query: "TODO: Replace with a query that triggers {tool}"
    description: "Tool usage test — {tool}"
    path:
      expected_tools: [{tool}]
    cost:
      max_llm_calls: 10
"""
        queries_yaml += """
  - query: "What is the weather today?"
    description: "Out-of-scope — should not use tools"
    path:
      max_tool_calls: 0
    cost:
      max_llm_calls: 2
"""
    else:  # conversational
        queries_yaml += """
  - query: "TODO: Replace with a topic your agent handles"
    description: "In-scope conversational test"
    correctness:
      any_expected_in_answer: ["TODO: keyword1", "TODO: keyword2"]
    cost:
      max_llm_calls: 10

  - query: "TODO: Replace with a topic your agent should decline"
    description: "Out-of-scope — agent should refuse"
    cost:
      max_llm_calls: 2
"""

    return f"""agent: my-agent
runner: "{runner_path}"
version: 1.0

judge_config:
  model: {_get_judge_model_for_spec()}
  temperature: 0

baseline_dir: ./baselines

queries:{queries_yaml}"""


def _build_next_steps(run_mode: str, created_workflow: bool, has_queries: bool) -> list[str]:
    """Build context-aware 'Next Steps' lines.

    Parameters
    ----------
    run_mode : str
        "live" or "mock"
    created_workflow : bool
        Whether .github/workflows/agentci.yml was created
    has_queries : bool
        Whether real queries were generated (vs skeleton template)
    """
    steps: list[str] = []

    if not has_queries:
        steps.append("1. Fill in the TODO queries in [cyan]agentci_spec.yaml[/]")

    if run_mode == "mock":
        n = len(steps) + 1
        steps.append(f"{n}. Run [cyan]ciagent test --mock[/] to validate your spec")
    else:
        n = len(steps) + 1
        steps.append(f"{n}. Run [cyan]ciagent test[/] to execute live tests")

    n = len(steps) + 1
    steps.append(
        f"{n}. Run [cyan]ciagent generate-checks[/] to mine deterministic "
        f"fact checks from your knowledge base"
    )

    if created_workflow:
        n = len(steps) + 1
        steps.append(f"{n}. Commit: [cyan]git add .github/ agentci_spec.yaml[/]")
        n += 1
        steps.append(f"{n}. Add your API key to GitHub repository secrets")
        n += 1
        steps.append(f"{n}. Push: [cyan]git push[/]")

    return steps


def _get_judge_model_for_spec() -> str:
    """Pick the best judge model for generated specs based on available API keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-6"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o"
    return "claude-sonnet-4-6"


def _generate_queries(context: dict, runner_path: str, interview: dict | None = None) -> list[dict]:
    """Call an LLM (Anthropic or OpenAI) to generate test queries from project context.

    Uses ANTHROPIC_API_KEY if set, falls back to OPENAI_API_KEY.

    Parameters
    ----------
    context : dict
        Output of ``_scan_project()`` with agent_files, knowledge_base, existing_tests.
    runner_path : str
        The ``module:function`` runner import path.
    interview : dict | None
        Guided interview answers: agent_description, agent_type, kb_path, tools,
        handle_topics, decline_topics, golden_pairs.
    """
    import yaml

    agent_files_text = "\n\n".join(
        f"# {f['path']}\n{f['content']}" for f in context["agent_files"]
    ) or "(none found)"

    kb_text = "\n\n".join(
        f"## {f['path']}\n{f['snippet']}" for f in context["knowledge_base"]
    ) or "(none found)"

    tests_text = "\n\n".join(
        f"# {f['path']}\n{f['content']}" for f in context["existing_tests"]
    ) or "(none found)"

    # Build interview context section
    interview_section = ""
    query_count = "10-15"
    query_guidance = ""
    if interview:
        parts: list[str] = []
        if interview.get("agent_description"):
            parts.append(f"AGENT DESCRIPTION:\n{interview['agent_description']}")
        if interview.get("agent_type"):
            parts.append(f"AGENT TYPE: {interview['agent_type']}")
        if interview.get("tools"):
            parts.append(f"TOOLS AVAILABLE: {', '.join(interview['tools'])}")
        if interview.get("handle_topics"):
            parts.append(f"TOPICS TO HANDLE: {interview['handle_topics']}")
        if interview.get("decline_topics"):
            parts.append(f"TOPICS TO DECLINE: {interview['decline_topics']}")
        if interview.get("golden_pairs"):
            pairs_text = "\n".join(
                f"  Q: {p['question']}\n  A: {p['answer']}" for p in interview["golden_pairs"]
            )
            parts.append(f"GOLDEN Q&A PAIRS (use these exact expected answers):\n{pairs_text}")
        if interview.get("_smoke_context"):
            parts.append(interview["_smoke_context"])
        interview_section = "\n\n".join(parts)
        query_count = interview.get("_query_count", "10-15")
        query_guidance = interview.get("_query_guidance", "")

    if query_guidance:
        query_instruction = query_guidance
    else:
        query_instruction = (
            f"Generate {query_count} test queries covering:\n"
            "1. Happy path (in-scope questions the agent should retrieve and answer)\n"
            "2. Out-of-scope (questions the agent must decline, max_tool_calls: 0)\n"
            "3. Edge cases (mixed intent, compound questions, unanswerable)\n"
            "4. At least 2 boundary cases (greeting, completely off-topic)"
        )

    prompt = f"""You are an expert AI agent test engineer. Given the following agent project context,
generate a diverse set of test queries for CIAgent's agentci_spec.yaml.

{interview_section}

AGENT CODE:
{agent_files_text}

KNOWLEDGE BASE CONTENT:
{kb_text}

EXISTING TEST COVERAGE:
{tests_text}

{query_instruction}

For each query, produce a YAML block with:
- query: the question string
- description: one sentence explaining what this tests
- tags: list of tags (smoke, in-scope, out-of-scope, edge-case, etc.)
- path: expected_tools list OR max_tool_calls: 0 for decline cases
- correctness: use any_expected_in_answer (OR logic) for list-type answers,
  expected_in_answer (AND logic) only when ALL terms are essential, or llm_judge rule.
  For judge rules on in-scope queries, include context_file pointing to the KB file
  that contains the answer so the judge can verify against the actual documentation.
- cost: max_llm_calls budget (default 10 for in-scope queries)

JUDGE RULE GUIDELINES:
Write judge rules that evaluate whether the response is HELPFUL and ACCURATE,
not whether it follows a rigid script. The agent is a documentation assistant,
not an exam candidate.

1. THRESHOLD SELECTION:
   - Use 0.7 for happy-path in-scope queries (agent should answer well)
   - Use 0.8 for out-of-scope decline queries (agent must clearly decline)
   - Use 0.7 for edge cases, compound, and mixed-intent queries (more tolerance)

2. PARTIAL KNOWLEDGE:
   For compound queries where one sub-question IS answerable from the KB and
   another is NOT, the judge rule must accept:
   - Answering the KB-covered part accurately
   - Gracefully declining or saying "I don't have that information" for the rest
   Do NOT require the agent to "address both parts" — require it to be accurate
   on what it can answer and honest about what it cannot.

3. RULE TONE:
   - Use "should" instead of "must" for non-critical criteria
   - Focus on what makes a GOOD response, not a checklist of requirements
   - Avoid rules that penalize the agent for being honest about knowledge gaps

4. KEYWORD CHECKS (expected_in_answer vs any_expected_in_answer):
   All keyword values MUST come directly from the knowledge base content or
   golden Q&A pairs above. Do NOT invent or hallucinate facts.
   If a fact is not in the provided content, use llm_judge instead.

   - Use `any_expected_in_answer` (OR logic) when the query expects a LIST or
     enumeration (e.g., "What tools are available?" → any ONE tool name suffices).
     This is the PREFERRED default for most keyword checks.
   - Use `expected_in_answer` (AND logic) ONLY when ALL terms are essential to a
     correct answer (e.g., "What is the install command?" → both "pip" and "install"
     must appear).
   - When in doubt, prefer `any_expected_in_answer` — it is more resilient to
     agent paraphrasing and partial answers.

5. COST BUDGET:
   Set max_llm_calls to 10 for in-scope queries (RAG agents typically use 4-10
   LLM calls per query). Use 2-3 for out-of-scope/greeting queries.

EXAMPLES OF GOOD vs BAD JUDGE RULES:

BAD (too strict, penalizes honest agents):
  rule: "The agent must address both the refund question and the support
        contact question with specific information from the documentation."
  threshold: 0.8

GOOD (accepts partial knowledge):
  rule: "The agent should answer based on retrieved documentation. If the KB
        covers the topic, the answer should be accurate. If the KB does not
        cover a sub-question, the agent should honestly say it doesn't have
        that information rather than inventing an answer."
  threshold: 0.7

BAD (requires specific phrasing):
  rule: "The agent must state it is a CIAgent documentation assistant and
        list its exact capabilities."
  threshold: 0.85

GOOD (evaluates helpfulness):
  rule: "The agent should respond with a friendly greeting and indicate
        readiness to help with CIAgent questions. It should not fabricate
        any CIAgent facts unprompted."
  threshold: 0.7

6. DOC-GROUNDED JUDGING (context_file):
   For in-scope queries where the agent retrieves from a KB, ALWAYS include
   context_file pointing to the KB file that contains the answer. This lets
   the judge verify the answer against the actual document instead of guessing.
   The file paths are listed in the KNOWLEDGE BASE CONTENT section above.

   Example:
     llm_judge:
       - rule: "The answer should accurately describe the feature based on
               retrieved documentation."
         threshold: 0.7
         context_file: knowledge_base/features.md

   Do NOT use context_file for out-of-scope or greeting queries (no KB to check).
   Pick the MOST RELEVANT KB file for each query — the one most likely to
   contain the answer.

Respond ONLY with valid YAML (a list of query objects, no surrounding text).
"""

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for AI generation. "
                "Install it with: pip install ciagent[anthropic]"
            )
        client = _anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
    else:
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "openai package required for AI generation. "
                "Install it with: pip install ciagent[openai]"
            )
        client = _openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    parsed = yaml.safe_load(raw)
    # Filter out entries with missing or empty query strings
    return [
        q for q in parsed
        if isinstance(q, dict) and isinstance(q.get("query"), str) and q["query"].strip()
    ]


def _generate_smoke_queries(context: dict, runner_path: str, interview: dict | None = None) -> list[dict]:
    """Generate a small batch of 3 smoke-test queries for quick validation.

    Uses the same LLM as ``_generate_queries`` but with a tighter prompt
    requesting only 3 diverse queries (1 happy path, 1 out-of-scope, 1 boundary).
    """
    # Temporarily override the prompt count via a patched interview
    smoke_interview = dict(interview or {})
    smoke_interview["_query_count"] = "exactly 3"
    smoke_interview["_query_guidance"] = (
        "Generate exactly 3 queries:\n"
        "1. One happy-path in-scope question the agent should answer well\n"
        "2. One out-of-scope question the agent should decline\n"
        "3. One boundary case (greeting or off-topic)\n"
    )
    return _generate_queries(context, runner_path, interview=smoke_interview)[:3]


def _generate_full_queries(
    context: dict,
    runner_path: str,
    interview: dict | None = None,
    smoke_results: list[dict] | None = None,
) -> list[dict]:
    """Generate a full batch of 10-12 additional queries, informed by smoke results.

    Parameters
    ----------
    smoke_results : list[dict] | None
        Summaries of smoke test results so the LLM can avoid similar mistakes.
        Each dict has keys: query, passed (bool), issues (str).
    """
    full_interview = dict(interview or {})
    if smoke_results:
        results_text = "\n".join(
            f"  - \"{r['query']}\": {'PASSED' if r['passed'] else 'FAILED'}"
            + (f" ({r['issues']})" if r.get("issues") else "")
            for r in smoke_results
        )
        full_interview["_smoke_context"] = (
            f"Previous smoke test results (avoid repeating mistakes):\n{results_text}"
        )
    full_interview["_query_count"] = "10-12"
    return _generate_queries(context, runner_path, interview=full_interview)


def _calibrate_spec_from_traces(
    queries: list[dict],
    runner_fn,
    console,
) -> list[dict]:
    """Run 1-2 sample queries against the real agent to calibrate max_llm_calls.

    Only called in live mode. Picks up to 2 in-scope queries, runs them,
    measures actual LLM call counts, and sets max_llm_calls with 50% headroom.
    Falls back gracefully if the runner fails.
    """
    candidates = [
        q for q in queries
        if "out-of-scope" not in (q.get("tags") or [])
        and "greeting" not in (q.get("tags") or [])
    ]
    sample = candidates[:2] if len(candidates) >= 2 else candidates[:1]
    if not sample:
        return queries

    console.print("\n[bold cyan]Calibrating cost budgets...[/]")
    observed_llm_calls: list[int] = []

    for q_spec in sample:
        try:
            trace = runner_fn(q_spec["query"])
            llm_count = sum(len(s.llm_calls) for s in trace.spans)
            observed_llm_calls.append(llm_count)
            query_preview = q_spec["query"][:50]
            console.print(f"  [dim]{query_preview}...[/] → {llm_count} LLM calls")
        except Exception:
            continue

    if not observed_llm_calls:
        console.print("[yellow]Calibration skipped:[/] no traces captured")
        return queries

    max_observed = max(observed_llm_calls)
    calibrated_max = max(8, int(max_observed * 1.5))

    for q in queries:
        tags = q.get("tags") or []
        cost = q.get("cost") or {}
        if "out-of-scope" in tags or "greeting" in tags:
            cost["max_llm_calls"] = max(2, min(observed_llm_calls))
        else:
            cost["max_llm_calls"] = calibrated_max
        q["cost"] = cost

    console.print(
        f"[green]✓[/] Calibrated: max_llm_calls = {calibrated_max} "
        f"(observed max: {max_observed})"
    )
    return queries


@click.group()
@click.version_option(version=__version__, prog_name="ciagent")
def cli():
    """CIAgent — Continuous Integration for AI Agents"""
    # Reset the chrome route each invocation: `console` is module-global, and a
    # prior in-process `--format json` run (tests) would otherwise leak stderr
    # routing into commands that never call _route_chrome.
    console.stderr = False

    # Suppress noisy Python warnings (Pydantic serializer, deprecations, etc.)
    import warnings
    warnings.filterwarnings("ignore")

    # Load .env variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Ensure current directory is in sys.path so we can import agents
    import sys
    import os
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    pass


def _scaffold_staging_gitignore(gitignore_path: str = ".gitignore") -> bool:
    """Create/append `.ciagent/staged/` to .gitignore. Idempotent.

    Returns True if a line was written, False if it was already ignored.
    """
    from pathlib import Path

    line = ".ciagent/staged/"
    p = Path(gitignore_path)
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    if any(ln.strip() == line for ln in existing.splitlines()):
        return False
    block = ("" if not existing or existing.endswith("\n") else "\n") + (
        "\n# CIAgent auto-staged failing conversations (may contain raw text)\n"
        f"{line}\n"
    )
    with open(p, "a", encoding="utf-8") as f:
        f.write(block)
    console.print(f"[green]✓ Ignored[/] {line} in {gitignore_path}")
    return True


@cli.command()
@click.option('--hook', is_flag=True, help='Also install a .git/hooks/pre-push script')
@click.option('--force', is_flag=True, help='Overwrite existing files')
@click.option('--example', is_flag=True, help='Generate a pre-populated agentci_spec.yaml with RAG example')
@click.option('--generate', is_flag=True,
              help='Scan project code + knowledge base and auto-generate agentci_spec.yaml using AI')
@click.option('--agent-description', default=None,
              help='DEPRECATED: Agent type is now auto-detected from code. Kept for backwards compatibility.')
@click.option('--kb-path', default=None, type=click.Path(exists=True),
              help='Path to knowledge base directory (non-interactive mode)')
@click.option('--mode', 'run_mode', default=None,
              type=click.Choice(['live', 'mock']),
              help='Test run mode: live (real API) or mock (synthetic traces)')
@click.option('--golden-file', default=None, type=click.Path(exists=True),
              help='Path to golden Q&A pairs (JSON/CSV) for mock mode')
@click.option('--runner', default=None,
              help='Runner import path (e.g. myagent.run:run_agent). Skips prompt.')
def init(hook, force, example, generate, agent_description, kb_path, run_mode, golden_file, runner):
    """Scaffold a new CIAgent test suite and CI/CD pipeline."""
    import jinja2
    from rich.prompt import Prompt
    from pathlib import Path

    if agent_description:
        console.print(
            "[yellow]Warning:[/] --agent-description is deprecated. "
            "Agent type is now auto-detected from code."
        )

    # Non-interactive when --mode is provided (signals CI/pipeline usage)
    non_interactive = run_mode is not None

    # Track whether we generated real queries vs skeleton
    has_queries = True
    interview: dict = {}

    # 0. Generate agentci_spec.yaml
    spec_dest = Path("agentci_spec.yaml")
    if spec_dest.exists() and not force:
        console.print(f"[yellow]Skipped:[/] {spec_dest} already exists. Use --force to overwrite.")
    else:
        console.print("\n[bold green]CIAgent Setup[/]")

        has_queries = False

        if generate:
            import yaml as _yaml

            # ── Step 1: Auto-scan project FIRST (no questions) ───────
            console.print("\nScanning project...")

            detected_kb = kb_path or _detect_kb_dir(Path("."))
            detected_tools = _detect_tools_from_code(Path("."))

            # Scan with KB override if known
            context = _scan_project(Path("."), kb_override=detected_kb)

            agent_type = _detect_agent_type_from_code(context, detected_tools, detected_kb)
            interview["agent_type"] = agent_type

            # Show scan summary
            console.print(
                f"Found [cyan]{len(context['agent_files'])}[/] agent files, "
                f"[cyan]{len(context['knowledge_base'])}[/] KB documents, "
                f"[cyan]{len(context['existing_tests'])}[/] test files"
            )
            if detected_tools:
                console.print(f"Detected tools: [cyan]{', '.join(detected_tools)}[/]")
            if detected_kb:
                console.print(f"Knowledge base: [cyan]{detected_kb}/[/]")
            console.print(f"Agent type: [cyan]{agent_type}[/]")

            # ── Step 2: Q1 — Run mode (was Q2) ──────────────────────
            if run_mode:
                interview["run_mode"] = run_mode
            else:
                mode_choice = Prompt.ask(
                    "\n[bold]Q1.[/] How do you want to run tests?",
                    choices=["live", "mock"],
                    default="live",
                )
                interview["run_mode"] = mode_choice

            # ── Step 3: API key guard — only for live mode ───────────
            if interview["run_mode"] == "live":
                if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
                    console.print(
                        "\n[bold red]Error:[/] Live mode requires ANTHROPIC_API_KEY or OPENAI_API_KEY.\n"
                        "Options:\n"
                        "  1. Set an API key and retry\n"
                        "  2. Use [cyan]ciagent init --generate --mode mock[/]\n"
                        "  3. Use [cyan]ciagent init --example[/] for a static template"
                    )
                    sys.exit(1)

            # ── Step 4: Q2 — Conditional follow-up ───────────────────
            if agent_type == "rag":
                # Confirm KB path
                if kb_path:
                    interview["kb_path"] = kb_path
                elif detected_kb and non_interactive:
                    interview["kb_path"] = detected_kb
                elif detected_kb:
                    from pathlib import Path as _P
                    kb_file_count = len([f for f in _P(detected_kb).rglob("*") if f.suffix.lower() in {".md", ".txt"}])
                    confirmed_kb = Prompt.ask(
                        f"\n[bold]Q2.[/] Confirm knowledge base directory\n"
                        f"    [dim]Auto-detected: {detected_kb}/ ({kb_file_count} files)[/]",
                        default=detected_kb,
                    )
                    interview["kb_path"] = confirmed_kb
                    # Re-scan if user changed path
                    if confirmed_kb != detected_kb:
                        context = _scan_project(Path("."), kb_override=confirmed_kb)
                elif non_interactive:
                    interview["kb_path"] = "./knowledge_base"
                    context = _scan_project(Path("."), kb_override=interview["kb_path"])
                else:
                    interview["kb_path"] = Prompt.ask(
                        "\n[bold]Q2.[/] Where is your knowledge base directory?",
                        default="./knowledge_base",
                    )
                    context = _scan_project(Path("."), kb_override=interview["kb_path"])

                # Golden pairs: --golden-file flag or interactive prompt
                if golden_file:
                    interview["golden_pairs"] = _load_golden_pairs(golden_file)
                    if interview["golden_pairs"]:
                        console.print(f"Loaded [cyan]{len(interview['golden_pairs'])}[/] golden pairs from file")
                elif not non_interactive:
                    golden_path = Prompt.ask(
                        "\n    Golden Q&A pairs? (JSON/CSV path, or Enter to skip)",
                        default="",
                    )
                    if golden_path:
                        interview["golden_pairs"] = _load_golden_pairs(golden_path)

            elif agent_type == "tool":
                if non_interactive:
                    if detected_tools:
                        interview["tools"] = detected_tools
                elif detected_tools:
                    tools_str = Prompt.ask(
                        f"\n[bold]Q2.[/] Confirm detected tools\n"
                        f"    [dim]{', '.join(detected_tools)}[/]\n"
                        f"    [dim](Enter to confirm, or type comma-separated list)[/]",
                        default=", ".join(detected_tools),
                    )
                    if tools_str:
                        interview["tools"] = [t.strip() for t in tools_str.split(",") if t.strip()]
                else:
                    tools_str = Prompt.ask(
                        "\n[bold]Q2.[/] What tools does your agent use? (comma-separated)",
                        default="",
                    )
                    if tools_str:
                        interview["tools"] = [t.strip() for t in tools_str.split(",") if t.strip()]

            elif not non_interactive:  # conversational — skip in non-interactive
                handle = Prompt.ask(
                    "\n[bold]Q2a.[/] Topics to handle? (comma-separated, optional)",
                    default="",
                )
                decline = Prompt.ask(
                    "[bold]Q2b.[/] Topics to decline? (comma-separated, optional)",
                    default="",
                )
                if handle:
                    interview["handle_topics"] = handle
                if decline:
                    interview["decline_topics"] = decline

            # ── Step 5: Runner detection ─────────────────────────────
            detected_runner = _detect_runner(Path("."))

            # If no runner found, generate one automatically
            if not detected_runner and detected_tools and interview.get("run_mode") == "live":
                console.print("\n[dim]No runner found — generating agentci_runner.py...[/]")
                generated_runner = _generate_runner_file(
                    Path("."), context, detected_tools, agent_type,
                )
                if generated_runner:
                    detected_runner = generated_runner
                    console.print(f"[green]\u2713[/] Generated runner: [cyan]{generated_runner}[/]")
                    console.print(
                        "[dim]  Edit agentci_runner.py to implement your tool logic "
                        "(look for TODO comments)[/]"
                    )

            runner_default = detected_runner or "myagent.run:run_agent"
            if runner:
                runner_path = runner
            elif non_interactive:
                runner_path = runner_default
                if detected_runner:
                    console.print(f"Auto-detected runner: [cyan]{detected_runner}[/]")
            else:
                if detected_runner:
                    console.print(f"\nDetected runner: [cyan]{detected_runner}[/]")
                runner_path = Prompt.ask(
                    "Runner import path",
                    default=runner_default,
                )

            # ── Step 6: Query generation (branched by mode) ──────────
            queries = None

            if interview["run_mode"] == "mock":
                # ── Mock path: zero API keys ─────────────────────────
                console.print("\n[bold]Mock Mode[/] — zero API cost")

                if golden_file:
                    # Option 1: Golden file provided via flag
                    pairs = _load_golden_pairs(golden_file)
                    if pairs:
                        queries = _build_golden_queries(pairs)
                        has_queries = True
                        console.print(f"Using [cyan]{len(queries)}[/] queries from golden file")
                elif interview.get("golden_pairs"):
                    # Option 1b: Golden pairs from interactive prompt
                    pairs = interview["golden_pairs"]
                    queries = _build_golden_queries(pairs)
                    has_queries = True
                    console.print(f"Using [cyan]{len(queries)}[/] golden Q&A queries")

                if not queries and not non_interactive:
                    # Option 2: Interactive query typing
                    if Confirm.ask("\nEnter test queries interactively? [Y/n]", default=True):
                        query_strings = _prompt_for_queries_interactive()
                        if query_strings:
                            queries = [
                                {"query": q, "description": f"User query: {q[:50]}"}
                                for q in query_strings
                            ]
                            has_queries = True
                            console.print(f"Added [cyan]{len(queries)}[/] queries")

                if not queries:
                    # Option 3: Skeleton template
                    console.print("\nGenerating skeleton template with TODO placeholders...")
                    spec_content = _generate_skeleton_spec(
                        agent_type, detected_tools, runner_path,
                    )
                    has_queries = False

            else:
                # ── Live path: API key required ──────────────────────
                console.print("\nGenerating initial smoke-test queries...")
                try:
                    smoke_queries = _generate_smoke_queries(context, runner_path, interview=interview)
                except Exception as e:
                    console.print(f"[yellow]Warning:[/] AI generation failed ({e}).")
                    smoke_queries = None

                if smoke_queries:
                    console.print(f"Generated [cyan]{len(smoke_queries)}[/] smoke-test queries:")
                    for i, q in enumerate(smoke_queries, 1):
                        tags = ", ".join(q.get("tags", [])) if q.get("tags") else ""
                        console.print(f"  {i}. \"{q['query']}\" [dim]{tags}[/]")

                    if Confirm.ask("\nGenerate more queries? [Y/n]", default=True):
                        console.print("Generating full test suite...")
                        try:
                            full_queries = _generate_full_queries(
                                context, runner_path, interview=interview,
                            )
                            queries = smoke_queries + full_queries
                            console.print(f"Generated [cyan]{len(queries)}[/] total test queries")
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/] Full generation failed ({e}). Using smoke queries only.")
                            queries = smoke_queries
                    else:
                        queries = smoke_queries
                        console.print(f"Using [cyan]{len(queries)}[/] smoke-test queries")
                else:
                    # Fallback: try full generation directly
                    try:
                        queries = _generate_queries(context, runner_path, interview=interview)
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/] AI generation failed ({e}).")

                if queries:
                    has_queries = True
                else:
                    console.print("Falling back to skeleton template...")
                    spec_content = _generate_skeleton_spec(
                        agent_type, detected_tools, runner_path,
                    )
                    has_queries = False

            # ── Calibrate cost budgets (live mode only) ─────────────
            if (
                queries
                and isinstance(queries, list)
                and interview.get("run_mode") == "live"
            ):
                try:
                    from .engine.parallel import resolve_runner
                    runner_fn = resolve_runner(runner_path)
                    queries = _calibrate_spec_from_traces(queries, runner_fn, console)
                except Exception as e:
                    console.print(f"[yellow]Calibration skipped:[/] {e}")

            # ── Write spec ───────────────────────────────────────────
            if queries and isinstance(queries, list):
                console.print(f"[green]✓[/] {len(queries)} test queries ready")
                spec_dict = {
                    "agent": "my-agent",
                    "runner": runner_path,
                    "version": 1.0,
                    "judge_config": {"model": _get_judge_model_for_spec(), "temperature": 0},
                    "baseline_dir": "./baselines",
                    "queries": queries,
                }
                spec_content = _yaml.dump(spec_dict, sort_keys=False, allow_unicode=True)

        else:
            # ── Non-generate modes (--example or bare init) ──────────
            runner_path = Prompt.ask(
                "What is the import path for your agent runner function?",
                default="myagent.run:run_agent"
            )

            if example:
                spec_content = f'''agent: rag-agent
runner: "{runner_path}"
version: 1.0

judge_config:
  model: {_get_judge_model_for_spec()}
  temperature: 0

baseline_dir: ./baselines

queries:
  - query: "How do I reset my password?"
    description: "Documentation retrieval test"
    expected_tools:
      - search_docs
    max_tool_calls: 3
    max_total_tokens: 2000
    llm_judge: "Response must provide clear, step-by-step instructions to reset a password."

  - query: "What is the weather in Tokyo?"
    description: "Out of scope query test"
    max_tool_calls: 0
    not_in_answer: ["degrees", "celsius", "fahrenheit", "cloudy", "sunny"]
'''
            else:
                spec_content = f'''agent: my-agent
runner: "{runner_path}"
version: 1.0

judge_config:
  model: {_get_judge_model_for_spec()}
  temperature: 0

baseline_dir: ./baselines

queries:
  # Add your test queries here
'''
            has_queries = False

        with open(spec_dest, "w") as f:
            f.write(spec_content)
        console.print(f"[green]✓ Created[/] {spec_dest}")
        console.print()

    # Auto-detect project characteristics
    dependency_file = "requirements.txt"
    if os.path.exists("pyproject.toml"):
        dependency_file = "pyproject.toml"

    test_path = "tests/"
    if not os.path.exists("tests") and os.path.exists("test"):
        test_path = "test/"

    # Python version (defaulting to current running version)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    template_data = {
        "python_version": python_version,
        "dependency_file": dependency_file,
        "test_path": test_path
    }

    # Set up Jinja environment pointing to the templates package
    from pathlib import Path
    template_dir = Path(__file__).parent / "templates"

    github_action_dest = Path(".github/workflows/agentci.yml")
    pre_push_dest = Path(".git/hooks/pre-push")
    created_workflow = False

    # 1. Create GitHub Actions Workflow
    github_action_dest.parent.mkdir(parents=True, exist_ok=True)
    if github_action_dest.exists() and not force:
        console.print(f"[yellow]Skipped:[/] {github_action_dest} already exists. Use --force to overwrite.")
    else:
        template_path = template_dir / "github_action.yml.j2"
        try:
            with open(template_path, "r") as f:
                template_str = f.read()
            import jinja2
            template = jinja2.Template(template_str)
            content = template.render(**template_data)
        except Exception as e:
            console.print(f"[yellow]Warning:[/] Could not load Jinja template ({e}). Using fallback.")
            content = f"# Scaffolded by CIAgent\n# Test Path: {test_path}\n# Deps: {dependency_file}\n"

        with open(github_action_dest, "w") as f:
            f.write(content)
        console.print(f"[green]✓ Created[/] {github_action_dest}")
        created_workflow = True

    # 1b. Gitignore the staging area — staged files can contain raw
    # conversation text (possibly PII), so they must never be committed.
    _scaffold_staging_gitignore()

    # 2. Create Pre-push Hook (if requested)
    if hook:
        if not Path(".git").exists():
            console.print("[red]Error:[/] Not a git repository. Cannot install pre-push hook.")
        else:
            pre_push_dest.parent.mkdir(parents=True, exist_ok=True)
            if pre_push_dest.exists() and not force:
                console.print(f"[yellow]Skipped:[/] {pre_push_dest} already exists.")
            else:
                template_path = template_dir / "pre_push_hook.sh.j2"
                try:
                    with open(template_path, "r") as f:
                        template_str = f.read()
                    template = jinja2.Template(template_str)
                    content = template.render(**template_data)
                except Exception as e:
                    content = f"#!/bin/sh\npytest {test_path} -m 'not live'"

                with open(pre_push_dest, "w") as f:
                    f.write(content)
                os.chmod(pre_push_dest, 0o755)  # Make executable
                console.print(f"[green]✓ Installed[/] {pre_push_dest}")

    # Context-aware Next Steps
    _run_mode = interview.get("run_mode", run_mode or "live")
    try:
        _has_queries = has_queries
    except NameError:
        _has_queries = True
    next_steps = _build_next_steps(_run_mode, created_workflow, _has_queries)
    console.print("\n[bold green]CIAgent Initialization Complete![/]")
    console.print("\n[bold]Next Steps:[/]")
    for step in next_steps:
        console.print(step)


@cli.command()
@click.option('--queries', type=click.Path(exists=True), help='Text file with one query per line (optional, falls back to interactive)')
@click.option('--agent', default='my-agent', help='Agent name for the spec')
@click.option('--runner', required=True, help='Import path for runner function, e.g. myagent.run:run')
@click.option('--output', default='agentci_spec.yaml', help='Output spec file')
@click.option('--baseline-dir', default='./baselines', help='Directory for baselines')
@click.option('--yes', '-y', is_flag=True,
              help='Accept every captured trace as golden without prompting '
                   '(for CI and coding agents; requires --queries)')
def bootstrap(queries, agent, runner, output, baseline_dir, yes):
    """Zero-to-Golden bootstrapper: run queries, record golden baselines, write a spec.

    The runner may return an ciagent.models.Trace or a plain string — string
    returns get automatic LLM/tool capture, same as `ciagent test`.
    """
    import yaml
    import json
    import re
    from datetime import datetime, timezone
    from pathlib import Path
    from rich.prompt import Prompt, Confirm
    from .engine.parallel import resolve_runner, _run_with_retry

    if yes and not queries:
        console.print("[bold red]--yes requires --queries[/] (nothing to confirm interactively)")
        sys.exit(2)

    try:
        runner_fn = resolve_runner(runner)
    except Exception as e:
        console.print(f"[bold red]Failed to load runner:[/] {e}")
        sys.exit(1)

    query_list = []
    if queries:
        with open(queries) as f:
            query_list = [q.strip() for q in f if q.strip()]
    else:
        console.print("\n[bold green]Zero-to-Golden Bootstrapper[/]")
        console.print("Enter test queries one by one (empty line to finish):")
        while True:
            q = Prompt.ask("Query")
            if not q.strip():
                break
            query_list.append(q.strip())

    if not query_list:
        console.print("[yellow]No queries provided. Exiting.[/]")
        sys.exit(0)

    spec_dict = {
        "agent": agent,
        "runner": runner,
        "version": 1.0,
        "judge_config": {
            "model": _get_judge_model_for_spec(),
            "temperature": 0,
        },
        "baseline_dir": baseline_dir,
        "queries": []
    }

    baseline_path = Path(baseline_dir) / agent
    baseline_path.mkdir(parents=True, exist_ok=True)

    for i, q in enumerate(query_list):
        console.print(f"\n[bold cyan]Running Query {i+1}/{len(query_list)}:[/] {q}")
        try:
            # Same execution path as `ciagent test`: TraceContext capture plus
            # coercion, so string-returning runners work here too.
            trace = _run_with_retry(runner_fn, q, retry_count=0, backoff=1.0, agent_name=agent)
            if trace is None:
                console.print("  [yellow]Runner returned nothing — skipping this query.[/]")
                continue

            # Print Tier 1 summary
            console.print(f"  Duration:   {trace.total_duration_ms:.1f}ms")
            console.print(f"  Cost:       ${trace.total_cost_usd:.4f}")
            console.print(f"  Tokens:     {trace.total_tokens}")
            console.print(f"  Tool Calls: {len(trace.tool_call_sequence)}")
            if trace.tool_call_sequence:
                console.print(f"  Path:       {' → '.join(trace.tool_call_sequence)}")
            answer_preview = (trace.metadata.get("final_output") or "")[:120]
            if answer_preview:
                console.print(f"  Answer:     {answer_preview}")

            if yes or Confirm.ask("Accept this trace as golden? [Y/n]", default=True):
                # Generates assertions
                expected_tools = trace.tool_call_sequence
                max_tool_calls = len(trace.tool_call_sequence) + 1
                max_total_tokens = int(trace.total_tokens * 1.5) or 2000
                
                query_entry = {
                    "query": q,
                    "description": f"Auto-generated test {i+1}",
                    "path": {
                        "expected_tools": expected_tools,
                        "max_tool_calls": max_tool_calls,
                    },
                    "cost": {
                        "max_total_tokens": max_total_tokens,
                    }
                }
                spec_dict["queries"].append(query_entry)
                
                # Save baseline
                baseline_data = {
                    "schema_version": 1,
                    "version": "v1",
                    "agent": agent,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "query": q,
                    "metadata": {
                        "model": "auto",
                        "precheck_passed": False
                    },
                    "trace": json.loads(trace.model_dump_json())
                }
                
                slug = re.sub(r'[^a-z0-9]+', '-', q.lower())[:30].strip('-') or f"query_{i}"
                b_file = baseline_path / f"v1-{slug}.json"
                with open(b_file, "w") as bf:
                    json.dump(baseline_data, bf, indent=2)
                console.print(f"[green]Saved baseline to {b_file}[/]")
        except Exception as e:
            console.print(f"[bold red]Error running agent:[/] {e}")

    if spec_dict["queries"]:
        with open(output, "w") as f:
            yaml.dump(spec_dict, f, sort_keys=False)
        console.print(f"\n[bold green]Bootstrap complete![/] Spec saved to {output}")
        console.print("Run [cyan]ciagent test[/] to verify.")


from collections import defaultdict
@cli.command()
@click.option('--suite', '-s', default='agentci.yaml', help='Path to test suite YAML')
@click.option('--runs', '-n', default=1, help='Number of runs for statistical mode')
@click.option('--tag', '-t', multiple=True, help='Only run tests with these tags')
@click.option('--diff/--no-diff', default=True, help='Compare against golden traces')
@click.option('--html', type=click.Path(), help='Generate HTML report at this path')
@click.option('--fail-on-cost', type=float, help='Fail if total cost exceeds threshold')
@click.option('--ci', is_flag=True, help='CI mode: exit code 1 on any failure')
@click.option('--json', 'output_json', is_flag=True, help='Output results as JSON (for agent consumption)')
def run(suite, runs, tag, diff, html, fail_on_cost, ci, output_json):
    """Execute the test suite. (DEPRECATED — use 'ciagent test')"""
    if not output_json:
        console.print(
            "[yellow]DEPRECATED:[/] 'ciagent run' is the legacy suite runner and will "
            "be removed in 0.9.0. Use [cyan]ciagent test[/] "
            "(and [cyan]ciagent test --runs N[/] for stability) instead.\n"
        )
        console.print(f"[bold blue]Agent CI[/] Running suite: [cyan]{suite}[/]")

    try:
        config = load_config(suite)
        runner = TestRunner(config)

        # Filter tests by tag if provided
        if tag:
            config.tests = [t for t in config.tests if any(tg in t.tags for tg in tag)]
            if not output_json:
                console.print(f"Filtered to [yellow]{len(config.tests)}[/] tests with tags: {tag}")

        suite_result = runner.run_suite(runs=runs)

        # JSON output mode — structured, machine-readable
        if output_json:
            import json as json_mod
            click.echo(suite_result.model_dump_json(indent=2))

            # Exit with appropriate code
            if fail_on_cost and suite_result.total_cost_usd > fail_on_cost:
                sys.exit(1)
            if ci and (suite_result.total_failed > 0 or suite_result.total_errors > 0):
                sys.exit(1)
            return

        # Display Results Table (human-readable)
        table = Table(title=f"Results: {suite_result.suite_name}")

        if runs > 1:
            # Statistical Display
            table.add_column("Test Case", style="cyan")
            table.add_column("Pass Rate", justify="center")
            table.add_column("Mean Cost", justify="right")
            table.add_column("Mean Duration", justify="right")
            table.add_column("Status")

            # Group by test name
            from collections import defaultdict
            grouped_results = defaultdict(list)
            for res in suite_result.results:
                grouped_results[res.test_name].append(res)

            for test_name, results in grouped_results.items():
                passed_count = sum(1 for r in results if r.result == TestResult.PASSED)
                pass_rate = (passed_count / len(results)) * 100
                mean_cost = sum(r.trace.total_cost_usd for r in results) / len(results)
                mean_duration = sum(r.duration_ms for r in results) / len(results)

                status_style = "green" if passed_count == len(results) else "yellow" if passed_count > 0 else "red"
                status_str = "STABLE" if passed_count == len(results) else "FLAKY" if passed_count > 0 else "FAILING"

                table.add_row(
                    test_name,
                    f"{passed_count}/{len(results)} ({pass_rate:.0f}%)",
                    f"${mean_cost:.4f}",
                    f"{mean_duration:.1f}ms",
                    f"[{status_style}]{status_str}[/]"
                )
        else:
            # Single Run Display (Existing logic)
            table.add_column("Test Case", style="cyan")
            table.add_column("Result", justify="center")
            table.add_column("Cost (USD)", justify="right")
            table.add_column("Duration (ms)", justify="right")
            table.add_column("Diffs/Details")

            for res in suite_result.results:
                result_str = "[green]PASSED[/]" if res.result == TestResult.PASSED else \
                             "[red]FAILED[/]" if res.result == TestResult.FAILED else \
                             "[bold red]ERROR[/]"

                details = []
                if res.error_message:
                    details.append(f"[red]{res.error_message}[/]")

                if res.assertion_results:
                    for r in res.assertion_results:
                        if not r['passed']:
                            details.append(r['message'])

                # Add Diff Details
                if res.diffs:
                    for d in res.diffs:
                        color = "red" if d.severity == "error" else "yellow"
                        details.append(f"[{color}]{d.message}[/]")

                table.add_row(
                    res.test_name,
                    result_str,
                    f"${res.trace.total_cost_usd:.4f}",
                    f"{res.duration_ms:.1f}",
                    "\n".join(details)
                )

        console.print(table)

        # Summary
        console.print(f"\n[bold]Summary:[/] [green]{suite_result.total_passed} Passed[/], "
                      f"[red]{suite_result.total_failed} Failed[/], "
                      f"[bold red]{suite_result.total_errors} Errors[/]")
        console.print(f"Total Cost: [bold]${suite_result.total_cost_usd:.4f}[/]")
        console.print(f"Total Duration: [bold]{suite_result.duration_ms:.1f}ms[/]")

        # Check fail-on-cost
        if fail_on_cost and suite_result.total_cost_usd > fail_on_cost:
             console.print(f"[bold red]FAILURE:[/] Total cost ${suite_result.total_cost_usd:.4f} "
                           f"exceeds limit ${fail_on_cost:.4f}")
             if ci:
                 sys.exit(1)

        if ci and (suite_result.total_failed > 0 or suite_result.total_errors > 0):
            sys.exit(1)

    except Exception as e:
        if output_json:
            import json as json_mod
            click.echo(json_mod.dumps({"error": str(e)}))
        else:
            console.print(f"[bold red]Error:[/] {e}")
        if ci:
            sys.exit(1)


@cli.command()
@click.argument('test_name')
@click.option('--suite', '-s', default='agentci.yaml')
@click.option('--output', '-o', help='Output path for golden trace')
@click.option('--json', 'output_json', is_flag=True, help='Output trace as JSON (for agent consumption)')
def record(test_name, suite, output, output_json):
    """Run agent live and save the trace as a golden baseline."""
    try:
        config = load_config(suite)
        runner = TestRunner(config)
        agent_fn = runner._import_agent()

        # Find the specific test
        test = next((t for t in config.tests if t.name == test_name), None)
        if not test:
            if output_json:
                import json as json_mod
                click.echo(json_mod.dumps({"error": f"Test '{test_name}' not found in {suite}"}))
            else:
                console.print(f"[bold red]Error:[/] Test '{test_name}' not found in {suite}")
            return

        if not output_json:
            console.print(f"Recording trace for [cyan]{test_name}[/]...")

        # Run the test
        result = runner.run_test(test, agent_fn)

        # Attach the final output text to the trace
        if result.trace and result.final_output:
            result.trace.metadata["final_output"] = str(result.final_output)
            if result.trace.spans:
                result.trace.spans[-1].output_data = str(result.final_output)

        # Determine output path
        if output:
            save_path = output
        elif test.golden_trace:
            save_path = test.golden_trace
        else:
            save_path = f"golden/{test_name}.golden.json"

        # JSON output mode — structured, machine-readable
        if output_json:
            import json as json_mod
            output_data = {
                "test_name": test_name,
                "save_path": save_path,
                "duration_ms": result.duration_ms,
                "cost_usd": result.trace.total_cost_usd,
                "tool_calls": result.trace.tool_call_sequence,
                "error": result.error_message,
            }
            # Auto-save in JSON mode (no interactive prompt)
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(result.trace.model_dump_json(indent=2))
            output_data["saved"] = True
            click.echo(json_mod.dumps(output_data, indent=2))
            return

        # Show summary (human-readable)
        console.print(f"Duration: {result.duration_ms:.1f}ms")
        console.print(f"Cost: ${result.trace.total_cost_usd:.4f}")
        console.print(f"Tool Calls: {len(result.trace.tool_call_sequence)}")

        if result.error_message:
             console.print(f"[bold red]Error during run:[/] {result.error_message}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # Prompt user
        from rich.prompt import Confirm
        if Confirm.ask(f"Save golden trace to [yellow]{save_path}[/]? [Y/n]", default=True):
            with open(save_path, 'w') as f:
                f.write(result.trace.model_dump_json(indent=2))
            console.print(f"[green]Saved![/]")
        else:
            console.print("[yellow]Cancelled.[/]")

    except Exception as e:
        if output_json:
            import json as json_mod
            click.echo(json_mod.dumps({"error": str(e)}))
        else:
            console.print(f"[bold red]Error:[/] {e}")


@cli.command(name="diff")
@click.option('--baseline', '-b', required=True, help="Baseline version tag (e.g. 'v1-broken')")
@click.option('--compare', '-c', required=True, help="Compare version tag (e.g. 'v2-fixed')")
@click.option('--agent', '-a', required=True, help="Agent identifier (matches baseline file naming)")
@click.option('--config', 'spec_path', default=None, type=click.Path(exists=True),
              help='Path to agentci_spec.yaml for correctness evaluation (optional)')
@click.option('--baseline-dir', default='./baselines', show_default=True,
              help='Directory containing versioned baseline JSON files')
@click.option('--format', 'fmt', type=click.Choice(['console', 'github', 'json']),
              default='console', show_default=True, help='Output format')
@click.option('--query', 'query_filter', default=None,
              help='Only compare baselines for this specific query (partial match)')
def diff_cmd(baseline, compare, agent, spec_path, baseline_dir, fmt, query_filter):
    """Compare two versioned baselines with three-tier (Correctness/Path/Cost) analysis.

    Loads the AGENT.BASELINE.json and AGENT.COMPARE.json files from BASELINE_DIR
    and renders a structured diff report.

    Example:
        ciagent diff --baseline v1-broken --compare v2-fixed --agent rag-agent

    Exit codes:
        0  No regressions detected
        1  Correctness regression (pass → fail)
        2  Error loading baselines
    """
    _route_chrome(fmt)
    import json as json_mod
    from pathlib import Path
    from .engine.diff import diff_baselines

    baseline_dir_path = Path(baseline_dir)

    # Collect matching baseline files
    baseline_pattern = f"{agent}.{baseline}.json"
    compare_pattern = f"{agent}.{compare}.json"

    baseline_file = baseline_dir_path / baseline_pattern
    compare_file = baseline_dir_path / compare_pattern

    if not baseline_file.exists():
        # Try glob — maybe multiple queries are stored as separate files
        baseline_files = sorted(baseline_dir_path.glob(f"{agent}.{baseline}.*.json"))
        compare_files = sorted(baseline_dir_path.glob(f"{agent}.{compare}.*.json"))
    else:
        baseline_files = [baseline_file]
        compare_files = [compare_file]

    if not baseline_files:
        console.print(f"[red]Error:[/] No baseline files found for '{agent}.{baseline}' in {baseline_dir}")
        console.print(f"[dim]Looked for: {baseline_dir_path / baseline_pattern}[/]")
        sys.exit(2)

    if not compare_files:
        console.print(f"[red]Error:[/] No compare files found for '{agent}.{compare}' in {baseline_dir}")
        sys.exit(2)

    # Pair up files by query index
    pairs = list(zip(baseline_files, compare_files))

    # Apply query filter
    if query_filter:
        pairs = [(b, c) for b, c in pairs if query_filter.lower() in b.stem.lower()]

    # Load optional spec
    spec = None
    if spec_path:
        from .loader import load_spec
        try:
            spec = load_spec(spec_path)
        except Exception as e:
            console.print(f"[yellow]Warning:[/] Could not load spec ({e}) — correctness comparison disabled")

    # Run diff for each pair
    reports = []
    for b_file, c_file in pairs:
        try:
            with open(b_file) as f:
                b_data = json_mod.load(f)
            with open(c_file) as f:
                c_data = json_mod.load(f)
        except Exception as e:
            console.print(f"[red]Error loading baselines:[/] {e}")
            sys.exit(2)

        report = diff_baselines(b_data, c_data, spec=spec)
        reports.append(report)

    if not reports:
        console.print("[yellow]No matching baseline pairs found.[/]")
        sys.exit(0)

    # Render output
    any_regression = False
    for report in reports:
        if report.has_regression:
            any_regression = True

        if fmt == "console":
            console.print(report.summary_console())
            if len(pairs) > 1:
                console.print()  # blank line between multiple reports

        elif fmt == "json":
            import json as json_mod
            click.echo(json_mod.dumps(report.summary_json(), indent=2))

        elif fmt == "github":
            j = report.summary_json()
            prefix = "error" if report.has_regression else "notice"
            title = f"CIAgent Diff: {report.agent} ({report.from_version} → {report.to_version})"
            body_parts = []
            for p in j.get("path", []):
                pct = f" ({p['pct_change']:+.1f}%)" if p.get("pct_change") is not None else ""
                body_parts.append(f"{p['metric']}: {p['before']} → {p['after']}{pct}")
            for c in j.get("cost", []):
                pct = f" ({c['pct_change']:+.1f}%)" if c.get("pct_change") is not None else ""
                body_parts.append(f"{c['metric']}: {c['before']} → {c['after']}{pct}")
            body = " | ".join(body_parts) if body_parts else "No metric changes"
            click.echo(f"::{prefix} title={title}::{body}")

    sys.exit(1 if any_regression else 0)


@cli.command()
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True,
              help='Path to a JSON results file (from --format json)')
@click.option('--output', '-o', 'output_path', type=click.Path(), default='agentci-report.html',
              show_default=True, help='Output HTML file path')
def report(input_path, output_path):
    """Generate an HTML report from a JSON results file.

    Convert previously saved JSON output (from `ciagent test --format json`)
    into a self-contained HTML report.

    \b
    Example:
        ciagent test -c spec.yaml --format json > results.json
        ciagent report -i results.json -o report.html
    """
    import json as _json
    from .engine.results import LayerResult, LayerStatus, QueryResult
    from .engine.reporter import _emit_html

    try:
        with open(input_path, encoding="utf-8") as f:
            data = _json.load(f)
    except (_json.JSONDecodeError, OSError) as e:
        console.print(f"[bold red]Error reading JSON:[/] {e}")
        sys.exit(2)

    # Reconstruct QueryResult objects from serialized JSON
    raw_results = data.get("results", [])
    results: list[QueryResult] = []
    for r in raw_results:
        def _to_layer(d: dict) -> LayerResult:
            status_map = {s.value: s for s in LayerStatus}
            return LayerResult(
                status=status_map.get(d.get("status", "skip"), LayerStatus.SKIP),
                messages=d.get("messages", []),
                details=d.get("details", {}),
            )

        results.append(QueryResult(
            query=r.get("query", ""),
            correctness=_to_layer(r.get("correctness", {})),
            path=_to_layer(r.get("path", {})),
            cost=_to_layer(r.get("cost", {})),
            retrieval=_to_layer(r.get("retrieval", {})),
        ))

    _emit_html(results, spec_file="(from JSON)", output_path=output_path)
    console.print(f"[green]Report generated:[/] {output_path}")


@cli.command(name="calibrate")
@click.option('--spec', '-s', 'config', default='agentci_spec.yaml',
              type=click.Path(exists=True),
              help='Path to spec file', show_default=True)
@click.option('--samples', '-n', default=2, type=int,
              help='Number of sample queries to run', show_default=True)
@click.option('--dry-run', is_flag=True,
              help='Show suggested budgets without updating the spec')
@click.option('--yes', '-y', is_flag=True,
              help='Auto-confirm spec updates without prompting')
def calibrate_cmd(config, samples, dry_run, yes):
    """Calibrate cost/path budgets by running sample queries.

    Runs N sample queries against your agent to measure actual resource
    usage, then suggests budget values with headroom.

    \b
    Examples:
        ciagent calibrate
        ciagent calibrate --samples 3
        ciagent calibrate --dry-run
    """
    from pathlib import Path
    import yaml
    from rich.panel import Panel
    from .loader import load_spec
    from .engine.parallel import resolve_runner

    console.print(Panel(f"[bold cyan]CIAgent Calibration[/] v{__version__}"))

    spec_path = Path(config)
    spec = load_spec(str(spec_path))

    if not spec.runner:
        console.print("[red]Error:[/] No runner configured in spec. Add a 'runner' field.")
        sys.exit(1)

    try:
        runner_fn = resolve_runner(spec.runner)
    except Exception as e:
        console.print(f"[red]Error resolving runner:[/] {e}")
        sys.exit(1)

    # Select in-scope sample queries
    candidates = [
        q for q in spec.queries
        if not (q.tags and any(t in ["out-of-scope", "greeting"] for t in q.tags))
    ]
    if not candidates:
        console.print("[yellow]No eligible queries for calibration[/]")
        sys.exit(0)

    sample_queries = candidates[:min(samples, len(candidates))]
    console.print(f"\n[bold]Running {len(sample_queries)} sample queries...[/]\n")

    # Run queries and collect metrics
    observations = []
    for q in sample_queries:
        query_text = q.query
        console.print(f"  {query_text[:60]}...")
        try:
            trace = runner_fn(query_text)
            obs = {
                "query": query_text,
                "llm_calls": trace.total_llm_calls,
                "tool_calls": trace.total_tool_calls,
                "tokens": trace.total_tokens,
                "cost_usd": trace.total_cost_usd,
            }
            observations.append(obs)
            console.print(
                f"    [dim]LLM: {obs['llm_calls']}  Tools: {obs['tool_calls']}  "
                f"Tokens: {obs['tokens']:,}  Cost: ${obs['cost_usd']:.4f}[/]"
            )
        except Exception as e:
            console.print(f"    [red]Failed:[/] {e}")

    if not observations:
        console.print("\n[red]No successful runs. Cannot calibrate.[/]")
        sys.exit(1)

    # Compute suggested budgets with headroom
    max_llm = max(o["llm_calls"] for o in observations)
    max_tools = max(o["tool_calls"] for o in observations)
    max_tokens = max(o["tokens"] for o in observations)
    max_cost = max(o["cost_usd"] for o in observations)

    suggested = {
        "max_llm_calls": max(10, int(max_llm * 1.5)),
        "max_tool_calls": max(5, int(max_tools * 1.5)),
        "max_total_tokens": int(max_tokens * 2.0) if max_tokens else None,
        "max_cost_usd": round(max_cost * 2.0, 4) if max_cost else None,
    }
    # Drop None values
    suggested = {k: v for k, v in suggested.items() if v is not None}

    # Show comparison table
    table = Table(title="\nCalibration Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Observed Max", justify="right", style="yellow")
    table.add_column("Suggested Budget", justify="right", style="green")
    table.add_column("Headroom", justify="right", style="dim")

    table.add_row("LLM Calls", str(max_llm), str(suggested["max_llm_calls"]), "+50%")
    table.add_row("Tool Calls", str(max_tools), str(suggested["max_tool_calls"]), "+50%")
    if max_tokens:
        table.add_row("Total Tokens", f"{max_tokens:,}", f"{suggested['max_total_tokens']:,}", "+100%")
    if max_cost:
        table.add_row("Cost (USD)", f"${max_cost:.4f}", f"${suggested['max_cost_usd']:.4f}", "+100%")

    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry-run mode:[/] No changes made to spec file")
        return

    if not yes:
        confirmed = Confirm.ask(
            f"\nUpdate {spec_path.name} with these budgets? [Y/n]",
            default=True,
        )
        if not confirmed:
            console.print("[yellow]Calibration cancelled[/]")
            return

    # Update spec file
    with spec_path.open() as f:
        spec_data = yaml.safe_load(f)

    # Apply to defaults if present, otherwise per-query
    if "defaults" in spec_data:
        if "cost" not in spec_data["defaults"]:
            spec_data["defaults"]["cost"] = {}
        spec_data["defaults"]["cost"].update(suggested)
    else:
        for q in spec_data.get("queries", []):
            if "cost" not in q:
                q["cost"] = {}
            q["cost"].update(suggested)

    with spec_path.open("w") as f:
        yaml.dump(spec_data, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]✓[/] Updated {spec_path.name}")
    console.print("[dim]Run 'ciagent test' to validate the new budgets[/]")


# ── v2 Commands ────────────────────────────────────────────────────────────────


@cli.command()
@click.argument('spec_path', type=click.Path(exists=True))
def validate(spec_path):
    """Validate an agentci_spec.yaml file against the schema.

    Exits 0 on success, 1 on validation failure.
    """
    from .loader import load_spec
    from .exceptions import ConfigError
    from pydantic import ValidationError

    try:
        spec = load_spec(spec_path)
        todo_count = sum(1 for q in spec.queries if q.query.strip().upper().startswith("TODO"))
        real_count = len(spec.queries) - todo_count
        console.print(
            f"[green]✅ Valid:[/] {len(spec.queries)} queries, agent='{spec.agent}'"
        )
        if todo_count:
            console.print(
                f"[yellow]⚠ {todo_count} query(ies) still have TODO placeholders — "
                f"edit them before running tests[/]"
            )
        sys.exit(0)
    except (ConfigError, ValidationError) as e:
        console.print(f"[bold red]❌ Validation failed:[/]\n{e}")
        sys.exit(1)


@cli.command(name="doctor")
@click.option('--config', '-c', default='agentci_spec.yaml',
              help='Path to agentci_spec.yaml', show_default=True)
def doctor_cmd(config):
    """Check your CIAgent setup for common issues.

    Validates the spec, runner, API keys, knowledge base, dependencies,
    and CI configuration. Prints a pass/warn/fail summary with fix hints.
    """
    from pathlib import Path

    checks: list[tuple[str, str, str]] = []  # (status, message, fix_hint)

    # 1. Spec file exists and validates
    spec = None
    if not Path(config).exists():
        checks.append(("fail", f"{config} not found", "Run: ciagent init --generate"))
    else:
        try:
            from .loader import load_spec
            spec = load_spec(config)
            checks.append(("pass", f"{config} valid ({len(spec.queries)} queries)", ""))
        except Exception as e:
            checks.append(("fail", f"{config} invalid: {e}", "Run: ciagent validate"))

    # 2. Runner imports successfully
    if spec and spec.runner:
        try:
            from .engine.parallel import resolve_runner
            resolve_runner(spec.runner)
            checks.append(("pass", f"Runner '{spec.runner}' imports successfully", ""))
        except Exception as e:
            checks.append(("fail", f"Runner '{spec.runner}' failed: {e}",
                           "Check the module:function path in your spec"))
    elif spec:
        checks.append(("warn", "No runner declared in spec",
                        "Add runner: \"module:function\" to your spec, or use --mock"))

    # 3. API keys — at least one is required for LLM judge & query generation
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if has_anthropic:
        checks.append(("pass", "ANTHROPIC_API_KEY is set", ""))
    if has_openai:
        checks.append(("pass", "OPENAI_API_KEY is set", ""))

    if not has_anthropic and not has_openai:
        checks.append(("fail", "No API key set",
                        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY for LLM judge & init --generate"))
    elif not has_anthropic and not has_openai:
        pass  # unreachable, but explicit
    else:
        # At least one key is set — no warning needed
        pass

    # 4. Knowledge base directory
    detected_kb = _detect_kb_dir(Path("."))
    if detected_kb:
        from pathlib import Path as _P
        kb_count = len([f for f in _P(detected_kb).rglob("*") if f.suffix.lower() in {".md", ".txt"}])
        checks.append(("pass", f"Knowledge base: {detected_kb}/ ({kb_count} files)", ""))
    else:
        checks.append(("warn", "No knowledge base directory found",
                        "Create knowledge_base/, kb/, or docs/ with .md/.txt files"))

    # 5. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        checks.append(("pass", f"Python {py_ver} (>=3.10 required)", ""))
    else:
        checks.append(("fail", f"Python {py_ver} (>=3.10 required)",
                        "Upgrade Python to 3.10+"))

    # 6. Key dependencies
    for pkg in ["pydantic", "click", "rich", "pyyaml"]:
        import_name = "yaml" if pkg == "pyyaml" else pkg
        try:
            __import__(import_name)
            checks.append(("pass", f"{pkg} is installed", ""))
        except ImportError:
            checks.append(("fail", f"{pkg} is not installed", f"pip install {pkg}"))

    # 7. GitHub Actions workflow
    if Path(".github/workflows/agentci.yml").exists():
        checks.append(("pass", ".github/workflows/agentci.yml exists", ""))
    else:
        checks.append(("warn", "No GitHub Actions workflow found",
                        "Run: ciagent init"))

    # 8. requirements.txt includes ciagent
    for req_file in ("requirements.txt", "pyproject.toml"):
        if Path(req_file).exists():
            content = Path(req_file).read_text()
            if "ciagent" in content:
                checks.append(("pass", f"{req_file} includes ciagent", ""))
            else:
                checks.append(("warn", f"{req_file} does not mention ciagent",
                                f"Add ciagent to {req_file}"))
            break

    # ── Print results ─────────────────────────────────────────────────────
    icons = {"pass": "[green]✓[/]", "warn": "[yellow]![/]", "fail": "[red]✗[/]"}
    console.print("\n[bold blue]CIAgent Doctor[/]\n")
    for status, msg, fix in checks:
        line = f"  {icons[status]} {msg}"
        if fix and status != "pass":
            line += f"  [dim]→ {fix}[/]"
        console.print(line)

    passed = sum(1 for s, _, _ in checks if s == "pass")
    warned = sum(1 for s, _, _ in checks if s == "warn")
    failed = sum(1 for s, _, _ in checks if s == "fail")
    console.print(f"\n  [bold]{passed} passed[/], {warned} warnings, {failed} failures\n")

    sys.exit(1 if failed > 0 else 0)


@cli.command(name="generate-checks")
@click.option('--config', '-c', default='agentci_spec.yaml',
              help='Path to agentci_spec.yaml', show_default=True)
@click.option('--kb', 'kb_path', default=None, type=click.Path(exists=True),
              help='Knowledge base directory (default: auto-detect)')
@click.option('--baseline-dir', default=None,
              help='Golden baselines used to VALIDATE generated checks '
                   '(default: spec baseline_dir)')
@click.option('--dry-run', is_flag=True, help='Show candidates, write nothing')
@click.option('--yes', '-y', is_flag=True,
              help='Accept all gate-validated checks without prompting '
                   '(unvalidated ones are still skipped)')
def generate_checks_cmd(config, kb_path, baseline_dir, dry_run, yes):
    """Mine the knowledge base for deterministic fact checks.

    Extracts hard facts (prices, rates, SKUs, versions, policy numbers) from
    your KB and proposes them as deterministic assertions on existing spec
    queries — reserving the LLM judge for answers with nothing checkable.

    \b
    Brittleness gate: every candidate is validated against your recorded
    golden answers first. A check that would FAIL a known-good answer is
    rejected automatically — you only review checks that survived. Queries
    with no recorded golden answer can't be gated; those candidates are
    flagged and require explicit confirmation.

    Extraction uses an LLM once, at authoring time. The generated checks run
    deterministically forever, at zero cost.

    User-written assertions are never overwritten — only empty fields fill.
    """
    from pathlib import Path

    import yaml

    from .engine.check_generator import (
        collect_kb_text,
        default_llm,
        extract_candidates,
        merge_candidates,
        validate_candidates,
    )
    from .engine.judge_audit import load_answers_from_baselines
    from .exceptions import ConfigError
    from .loader import load_spec

    try:
        spec = load_spec(config)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        sys.exit(2)

    effective_kb = kb_path or _detect_kb_dir(Path("."))
    if not effective_kb:
        console.print(
            "[bold red]No knowledge base found.[/] Pass --kb or create "
            "knowledge_base/, kb/, or docs/ with .md/.txt files."
        )
        sys.exit(2)
    kb_text = collect_kb_text(effective_kb)
    if not kb_text:
        console.print(f"[bold red]No .md/.txt files found in '{effective_kb}'.[/]")
        sys.exit(2)

    effective_baseline_dir = baseline_dir or spec.baseline_dir
    answers = load_answers_from_baselines(effective_baseline_dir)
    known_good = {q: [a] for q, a in answers.items()}
    if not answers:
        console.print(
            f"[yellow]Warning:[/] no golden baselines in '{effective_baseline_dir}' — "
            "generated checks cannot be validated against known-good answers.\n"
            "Record baselines first ([cyan]ciagent record[/]) for the full gate.\n"
        )

    console.print(
        f"[bold blue]CIAgent v{__version__}[/] │ generate-checks │ "
        f"kb: [cyan]{effective_kb}[/] │ queries: [cyan]{len(spec.queries)}[/] │ "
        f"golden answers: [cyan]{len(answers)}[/]"
    )
    console.print("[dim]Extracting hard facts (one-time LLM call)...[/]\n")

    try:
        result = extract_candidates(spec, kb_text, default_llm)
    except (RuntimeError, ImportError) as e:
        console.print(f"[bold red]Extraction error:[/] {e}")
        sys.exit(2)
    if not result.candidates:
        console.print(
            "No checkable hard facts found for these queries. That's a valid "
            "outcome — judgment-only queries belong to the judge."
        )
        sys.exit(0)

    validate_candidates(result, known_good)

    # ── Present the gate results ───────────────────────────────────────────
    if result.rejected:
        console.print(f"[red]✗ {len(result.rejected)} candidate(s) rejected by the "
                      f"validation gate[/] (would fail a known-good answer):")
        for c in result.rejected:
            console.print(f"   • {c.query[:50]!r} {c.field}={c.value!r} — [dim]{c.reason}[/]")
        console.print()

    accepted = []
    for c in result.validated:
        console.print(f"[green]✓ gate passed[/] {c.query[:60]!r}")
        console.print(f"   {c.field}: {c.value!r}" + (f"  [dim]({c.fact})[/]" if c.fact else ""))
        if dry_run:
            continue
        if yes:
            accepted.append(c)
        else:
            from rich.prompt import Confirm
            if Confirm.ask("   Add this check?", default=True):
                accepted.append(c)

    for c in result.unvalidated:
        console.print(f"[yellow]⚠ unvalidated[/] {c.query[:60]!r} — {c.reason}")
        console.print(f"   {c.field}: {c.value!r}" + (f"  [dim]({c.fact})[/]" if c.fact else ""))
        if dry_run or yes:
            continue  # --yes never auto-accepts ungated checks
        from rich.prompt import Confirm
        if Confirm.ask("   Add anyway (no known-good answer to gate it)?", default=False):
            accepted.append(c)

    if dry_run:
        console.print(f"\n[dim]Dry run — nothing written. "
                      f"{len(result.validated)} validated, "
                      f"{len(result.unvalidated)} unvalidated, "
                      f"{len(result.rejected)} rejected.[/]")
        sys.exit(0)

    if not accepted:
        console.print("\nNo checks accepted — spec unchanged.")
        sys.exit(0)

    updated, changes = merge_candidates(spec, accepted)
    if not changes:
        console.print("\nAll accepted checks already present — spec unchanged.")
        sys.exit(0)

    backup = Path(config).with_suffix(Path(config).suffix + ".bak")
    backup.write_text(Path(config).read_text(encoding="utf-8"), encoding="utf-8")
    Path(config).write_text(
        yaml.safe_dump(
            updated.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True,
        ),
        encoding="utf-8",
    )
    console.print(f"\n[green]Updated {config}[/] (backup: {backup}; note: YAML comments are not preserved)")
    for change in changes:
        console.print(f"   • {change}")
    console.print("\nVerify: [cyan]ciagent test --mock --yes[/]")
    sys.exit(0)


@cli.command(name="judge-audit")
@click.option('--config', '-c', default='agentci_spec.yaml',
              help='Path to agentci_spec.yaml', show_default=True)
@click.option('--baseline-dir', default=None,
              help='Directory of recorded golden baselines (default: spec baseline_dir)')
@click.option('--repeats', '-r', default=3, show_default=True, type=int,
              help='Judge calls per query — measures retest flip rate')
@click.option('--labels', 'labels_path', default=None, type=click.Path(exists=True),
              help='Hand-labels file (YAML/JSON: query → pass|fail) for agreement + Cohen\'s κ')
@click.option('--sample', default=None, type=int,
              help='Audit only the first N judged queries (cost control)')
@click.option('--live', is_flag=True,
              help='Re-run the agent for FRESH answers before scoring — breaks '
                   'the baseline circularity (needs `runner:` in the spec)')
@click.option('--answers', 'answers_path', default=None, type=click.Path(exists=True),
              help='Score answers from a `ciagent test --format json` results file '
                   'instead of golden baselines')
@click.option('--format', 'fmt', type=click.Choice(['console', 'json']),
              default='console', show_default=True)
@click.option('--yes', '-y', is_flag=True, help='Skip cost confirmation')
def judge_audit_cmd(config, baseline_dir, repeats, labels_path, sample, live,
                    answers_path, fmt, yes):
    """Audit your LLM judge against ground truth you already have.

    Answer sources, in order of rigor:

    \b
      --live           re-run the agent, score FRESH answers (most honest)
      --answers FILE   score a `ciagent test --format json` results file
      (default)        score recorded golden baselines

    Auditing against goldens has a known blind spot: `generate-checks`
    guarantees its checks pass those same baselines, so on generated checks
    "judge PASS / check FAIL" can never fire and agreement inflates by
    construction. Audit on fresh answers; gate on goldens.

    \b
    Three measurements:
      1. Judge vs. deterministic checks — the disagreement matrix. The row
         that matters: answers the judge PASSED that a fact-check FAILED.
      2. Retest stability — same answer judged --repeats times; flips on
         identical input are the judge's own noise floor.
      3. Judge vs. hand labels (--labels) — agreement + Cohen's κ.

    A judge that fails where you CAN check it should not be trusted where
    you can't. The reverse does not hold: passing this audit is a smoke
    test, not a guarantee, for judgment-only queries.

    \b
    Exit codes:
        0 — verdict TRUSTWORTHY or NEEDS CALIBRATION
        1 — verdict UNRELIABLE
        2 — configuration / infrastructure error
    """
    _route_chrome(fmt)
    from .engine.judge_audit import (
        collect_live_answers,
        load_answers_from_baselines,
        load_answers_from_results_json,
        load_labels_file,
        load_retrieval_flags_from_baselines,
        run_judge_audit,
    )
    from .engine.reporter import report_judge_audit
    from .exceptions import ConfigError
    from .loader import load_spec

    if live and answers_path:
        console.print("[bold red]--live and --answers are mutually exclusive:[/] "
                      "pick one answer source.")
        sys.exit(2)

    try:
        spec = load_spec(config)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        sys.exit(2)

    effective_baseline_dir = baseline_dir or spec.baseline_dir

    # ── Answer source: --live (fresh) > --answers (results file) > baselines ──
    if answers_path:
        try:
            answers = load_answers_from_results_json(answers_path)
        except (ValueError, OSError) as e:
            console.print(f"[bold red]Answers file error:[/] {e}")
            sys.exit(2)
        source_label = f"answers: {answers_path}"
    elif live:
        answers = {}  # collected below, after the judged set is known
        source_label = "answers: live (fresh agent runs)"
    else:
        answers = load_answers_from_baselines(effective_baseline_dir)
        if not answers:
            console.print(
                f"[bold red]No recorded answers found in '{effective_baseline_dir}'.[/]\n"
                "Give the audit answers to score:\n"
                "  [cyan]ciagent judge-audit --live[/]  (fresh agent runs — most honest)\n"
                "  [cyan]ciagent test --format json > results.json[/] then "
                "[cyan]--answers results.json[/]\n"
                "  or record baselines: [cyan]ciagent record <test>[/]"
            )
            sys.exit(2)
        source_label = "answers: golden baselines"

    labels = None
    if labels_path:
        try:
            labels = load_labels_file(labels_path)
        except (ValueError, OSError) as e:
            console.print(f"[bold red]Labels file error:[/] {e}")
            sys.exit(2)

    # Which queries will actually be judged? Rubrics required; an answer is
    # required too unless --live produces them on demand.
    from .engine.judge_audit import _judge_rubrics
    judged_queries = []
    for q in spec.queries:
        if q.correctness is None:
            continue
        if not live and q.query not in answers:
            continue
        if _judge_rubrics(q.correctness):
            judged_queries.append(q)
    if sample is not None:
        judged_queries = judged_queries[:sample]
    if not judged_queries:
        console.print(
            "[bold red]No auditable queries:[/] none of the spec's queries have "
            "judge rubrics" + (" with a recorded answer." if not live else ".")
        )
        sys.exit(2)

    n_calls = sum(
        len(_judge_rubrics(q.correctness)) for q in judged_queries
    ) * max(1, repeats)
    live_note = f" │ agent runs: [cyan]{len(judged_queries)}[/]" if live else ""
    console.print(
        f"[bold blue]CIAgent v{__version__}[/] │ judge-audit │ agent: [cyan]{spec.agent}[/] │ "
        f"queries: [cyan]{len(judged_queries)}[/] │ repeats: [cyan]{repeats}[/] │ "
        f"judge calls: [cyan]~{n_calls}[/]{live_note} │ {source_label}"
    )
    if not yes and fmt == "console":
        from rich.prompt import Confirm
        prompt = (
            f"Proceed with {len(judged_queries)} agent run(s) + ~{n_calls} judge calls? [Y/n]"
            if live else f"Proceed with ~{n_calls} judge calls? [Y/n]"
        )
        if not Confirm.ask(prompt, default=True):
            console.print("[yellow]Aborted.[/]")
            sys.exit(0)

    def _progress(q: str) -> None:
        if fmt == "console":
            console.print(f"  [dim]audited:[/] {q[:70]}")

    if live:
        def _run_progress(q: str) -> None:
            if fmt == "console":
                console.print(f"  [dim]agent ran:[/] {q[:70]}")

        try:
            answers = collect_live_answers(
                spec, queries=judged_queries, progress=_run_progress,
            )
        except (ValueError, ImportError, AttributeError) as e:
            console.print(f"[bold red]--live runner error:[/] {e}")
            sys.exit(2)
        if not answers:
            console.print(
                "[bold red]--live produced no answers:[/] every agent run "
                "failed or returned empty output."
            )
            sys.exit(2)

    # F4: was each judged query's recorded retrieval empty? (reported row).
    # Only meaningful for the baseline source — for --live / --answers the
    # scored answers' retrieval isn't what the baselines recorded.
    retrieval_flags = (
        load_retrieval_flags_from_baselines(effective_baseline_dir, spec)
        if not live and not answers_path else None
    )

    try:
        report = run_judge_audit(
            spec, answers, repeats=repeats, labels=labels,
            sample=sample, progress=_progress,
            retrieval_flags=retrieval_flags,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Audit error:[/] {e}")
        sys.exit(2)

    sys.exit(report_judge_audit(report, format=fmt))


@cli.command(name="import")
@click.argument('trace_file', type=click.Path(exists=True))
@click.option('--config', '-c', default='agentci_spec.yaml',
              help='Path to agentci_spec.yaml', show_default=True)
@click.option('--version', 'version_tag', default=None,
              help='Baseline version tag (default: imported-<n>)')
@click.option('--dry-run', is_flag=True,
              help='Map + gate the trace and show what would be written, write nothing')
@click.option('--force-save', is_flag=True,
              help="Import a trace that FAILS its query's own correctness assertions "
                   "— the found failure itself becomes the golden. Off by default so a "
                   "bad capture isn't planted silently; the round-trip completeness "
                   "gate still applies either way.")
def import_cmd(trace_file, config, version_tag, dry_run, force_save):
    """Convert an exported production trace into a spec query + golden baseline.

    TRACE_FILE formats are auto-detected:

    \b
      - OTel GenAI spans (what OTel instrumentation / Langfuse emit):
        OTLP/JSON envelope, {"spans": [...]} wrapper, or a flat span list
      - LangSmith run exports (langsmith run export / Client.list_runs):
        JSON or JSONL run objects, flat or nested RunTree

    That production failure from last Tuesday becomes a CI test.

    \b
    Round-trip gate (always on): the import must produce a golden that loads
    and evaluates cleanly BEFORE anything is written. Partial traces — no
    user input, no final output, no spans — are rejected with the missing
    fields named. A golden that can never pass is a permanent false
    regression; this command refuses to plant one.

    The spec gains a minimal query (tagged `imported`) if the query text is
    new; existing queries are never modified — only the golden is written.

    \b
    If the query already carries correctness assertions the trace FAILS, the
    save prechecks and stops (exit 1) — pass --force-save to keep the failing
    trace as the golden (the found failure becomes the regression test).

    \b
    Exit codes:
        0 — imported (or --dry-run gate passed)
        1 — trace rejected by the round-trip gate, or fails its query's
            assertions without --force-save
        2 — file/config error
    """
    from pathlib import Path

    import yaml

    from .baselines import save_baseline
    from .engine.artifact_gate import gate_imported_golden
    from .engine.runner import _extract_answer
    from .exceptions import ConfigError
    from .importers import import_trace_file
    from .loader import load_spec
    from .schema.spec_models import GoldenQuery

    try:
        trace, query, source_format = import_trace_file(trace_file)
    except ValueError as e:  # any importer's read error
        console.print(f"[bold red]Import error:[/] {e}")
        sys.exit(2)

    gate = gate_imported_golden(trace, query)
    if not gate.accepted:
        console.print(
            f"[bold red]Rejected by the round-trip gate[/] — nothing written:"
        )
        for reason in gate.reasons:
            console.print(f"  • {reason}")
        console.print(
            "[dim]Partial traces are never silently imported: a golden that "
            "can never pass is a permanent false regression in your CI.[/]"
        )
        sys.exit(1)

    query = str(query)  # gate acceptance guarantees it is present
    answer = _extract_answer(trace)
    console.print(
        f"[bold blue]CIAgent v{__version__}[/] │ import │ "
        f"format: [cyan]{source_format}[/] │ "
        f"tool calls: [cyan]{len(trace.tool_call_sequence)}[/] │ "
        f"llm calls: [cyan]{trace.total_llm_calls}[/]"
    )
    console.print(f"  query:  {query[:100]}")
    console.print(f"  answer: {answer[:100]}")

    if dry_run:
        console.print("[green]Gate passed.[/] --dry-run: nothing written.")
        sys.exit(0)

    try:
        spec = load_spec(config)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        sys.exit(2)

    # ── Spec: add a minimal imported query; never touch existing ones ────────
    if any(q.query == query for q in spec.queries):
        console.print("  spec:   query already present — spec unchanged")
    else:
        updated = spec.model_copy(deep=True)
        updated.queries.append(GoldenQuery(query=str(query), tags=["imported"]))
        backup = Path(config).with_suffix(".yaml.bak")
        backup.write_text(Path(config).read_text(encoding="utf-8"), encoding="utf-8")
        Path(config).write_text(
            yaml.safe_dump(
                # mode="json" so enum fields (e.g. path.match_mode) serialize as
                # their plain-string values — yaml.safe_dump cannot represent an
                # Enum object, which otherwise crashes the rewrite of any spec
                # that carries a path assertion.
                updated.model_dump(mode="json", exclude_none=True), sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        spec = updated
        console.print(f"  spec:   query added (tagged 'imported'; backup: {backup.name})")

    # ── Golden: versioned baseline the suite can gate on ─────────────────────
    if version_tag is None:
        existing = list(
            (Path(spec.baseline_dir) / spec.agent).glob("imported-*.json")
        ) if (Path(spec.baseline_dir) / spec.agent).exists() else []
        version_tag = f"imported-{len(existing) + 1}"
    try:
        out_path = save_baseline(
            trace,
            agent=spec.agent,
            version=version_tag,
            spec=spec,
            query_text=str(query),
            baseline_dir=spec.baseline_dir,
            force=force_save,
        )
    except ValueError as e:
        # The matching query already carries correctness assertions the trace
        # fails — importing a FOUND FAILURE. That's the point of F7; surface
        # the fix rather than a bare stack trace.
        console.print(f"[bold red]Precheck failed[/] — the trace fails its query's "
                      f"own correctness assertions:")
        console.print(f"  {e}")
        console.print(
            "[dim]This is expected when the imported trace IS the failure. "
            "Re-run with [/][cyan]--force-save[/][dim] to keep it as the golden "
            "(the found failure becomes the regression test).[/]"
        )
        sys.exit(1)
    console.print(f"  golden: [green]{out_path}[/]")
    console.print(
        "\nNext: add assertions to the imported query (or run "
        "[cyan]ciagent generate-checks[/]), then gate with [cyan]ciagent test[/]."
    )
    sys.exit(0)


def _finish_stability_session(fmt, config, output, run_results, stability, fail_on_flaky):
    """Render a multi-run stability session and exit with the right code.

    Console/github: per-query detail only for consistent failures — flips
    belong to the stability section, not the per-query noise. Other formats
    render the last run's results with the stability block attached.

    Exit: 1 if any query failed in EVERY run, or flipped with --fail-on-flaky.
    """
    from .engine.reporter import (
        emit_query_result,
        emit_stability_console,
        report_results,
    )

    if fmt in ("console", "github"):
        failed_queries = {q.query for q in stability.consistent_failures}
        for r in run_results[-1]:
            if r.query in failed_queries:
                emit_query_result(r)
        emit_stability_console(stability)
        if os.environ.get("GITHUB_ACTIONS") == "true" or fmt == "github":
            from .engine.reporter import _emit_github_annotations, _emit_stability_github
            _emit_github_annotations(
                [r for r in run_results[-1] if r.query in failed_queries], config,
            )
            _emit_stability_github(stability, config)
        exit_code = 1 if stability.consistent_failures else 0
    else:
        exit_code = report_results(
            run_results[-1], format=fmt, spec_file=config,
            output_path=output, stability=stability,
        )
    if fail_on_flaky and stability.flipped_queries:
        exit_code = max(exit_code, 1)
    sys.exit(exit_code)


@cli.command(name="test")
@click.option('--config', '-c', default='agentci_spec.yaml',
              help='Path to agentci_spec.yaml', show_default=True)
@click.option('--tags', '-t', multiple=True, help='Only evaluate queries with these tags')
@click.option('--format', 'fmt',
              type=click.Choice(['console', 'github', 'json', 'prometheus', 'html']),
              default='console', show_default=True, help='Output format')
@click.option('--output', '-o', default=None, type=click.Path(),
              help='Output file path (used with --format html, default: agentci-report.html)')
@click.option('--baseline-dir', default=None,
              help='Override baseline directory from spec')
@click.option('--workers', '-w', default=4, show_default=True, type=int,
              help='Max parallel workers for query execution')
@click.option('--sample-ensemble', default=None, type=float,
              help='Fraction of queries to use ensemble judging (0.0-1.0, e.g. 0.2)')
@click.option('--mock', is_flag=True,
              help='Run with synthetic traces (no API calls). Validates spec structure.')
@click.option('--yes', '-y', is_flag=True,
              help='Skip cost estimate confirmation (for CI)')
@click.option('--runs', '-n', 'runs', default=1, show_default=True, type=int,
              help='Run every query N times and report verdict stability with '
                   'flip-source attribution (agent-variance vs judge-flake)')
@click.option('--fail-on-flaky', is_flag=True,
              help='With --runs > 1: exit 1 if any query flips verdicts across runs')
def test_cmd(config, tags, fmt, output, baseline_dir, workers, sample_ensemble, mock, yes,
             runs, fail_on_flaky):
    """Run CIAgent v2 evaluation against a spec file.

    Loads agentci_spec.yaml, runs the agent for each query (capturing traces),
    evaluates all three layers (Correctness / Path / Cost), and reports results.

    Use --mock to validate your spec with synthetic traces (zero API cost).
    With no agentci_spec.yaml present, --mock runs a bundled demo spec —
    try `ciagent test --mock --runs 3` in an empty directory (zero API keys).

    Use --runs N to run the whole suite N times: a stable aggregate score can
    hide per-query verdict flips. Each flip is attributed to its source —
    agent-variance (the answer changed; fix the agent) or judge-flake (same
    answer, the LLM judge changed its mind; fix the eval).

    \b
    Requires the spec to declare a runner (unless --mock is used):
        runner: "myagent.run:run_agent"

    The runner function must accept (query: str) and return an ciagent.models.Trace.

    \b
    Exit codes:
        0 — all correctness checks pass (warnings emitted as annotations)
        1 — one or more correctness failures (with --runs: failed in EVERY run,
            or flipped and --fail-on-flaky is set)
        2 — runtime / infrastructure error
    """
    _route_chrome(fmt)
    import json as _json
    from pathlib import Path

    from .loader import load_spec, filter_by_tags
    from .engine.reporter import report_results
    from .engine.runner import evaluate_spec
    from .exceptions import ConfigError

    # ── Zero-key demo fallback ────────────────────────────────────────────────
    # Only when the user did not pass --config themselves: an explicitly named
    # spec that is missing must stay an error, never silently become the demo.
    demo_mode = False
    config_source = click.get_current_context().get_parameter_source("config")
    if config_source == click.core.ParameterSource.DEFAULT and not Path(config).exists():
        if mock:
            from importlib.resources import files

            config = str(files("ciagent").joinpath("examples", "demo_spec.yaml"))
            demo_mode = True
        else:
            console.print(
                "[bold red]No agentci_spec.yaml found in this directory.[/]\n\n"
                "  [cyan]ciagent init[/]                     scaffold a spec for your own agent\n"
                "  [cyan]ciagent test --mock --runs 3[/]     try the bundled demo (synthetic traces, zero API keys)"
            )
            sys.exit(2)

    try:
        spec = load_spec(config)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        sys.exit(2)

    if tags:
        spec = filter_by_tags(spec, list(tags))
        if not spec.queries:
            console.print(f"[yellow]No queries match tags: {tags}[/]")
            sys.exit(0)

    effective_baseline_dir = baseline_dir or spec.baseline_dir

    # ── Warn about TODO placeholder queries ───────────────────────────────────
    todo_queries = [q for q in spec.queries if q.query.strip().upper().startswith("TODO")]
    if todo_queries:
        console.print(
            f"[yellow]Warning:[/] {len(todo_queries)} query(ies) still have TODO placeholders — skipping them:"
        )
        for tq in todo_queries:
            console.print(f"  [dim]• {tq.query[:80]}[/]")
        spec.queries = [q for q in spec.queries if q not in todo_queries]
        if not spec.queries:
            console.print("[bold red]Error:[/] All queries are TODO placeholders. Edit agentci_spec.yaml first.")
            sys.exit(1)
        console.print()

    if runs < 1:
        console.print("[bold red]--runs must be at least 1[/]")
        sys.exit(2)

    # ── Mock mode ─────────────────────────────────────────────────────────────
    if mock:
        from .engine.mock_runner import run_mock_spec

        console.print(
            f"[bold blue]CIAgent v{__version__}[/] │ agent: [cyan]{spec.agent}[/] │ "
            f"queries: [cyan]{len(spec.queries)}[/] │ mode: [cyan]mock[/]"
            + (f" │ runs: [cyan]{runs}[/]" if runs > 1 else "")
        )
        console.print("[dim]Running with synthetic traces — zero API cost[/]\n")

        if demo_mode:
            console.print(
                "[yellow]Demo mode:[/] no agentci_spec.yaml found — using the bundled "
                "demo spec ([bold]synthetic data[/], simulated agent).\n"
                "[dim]Test your own agent instead: ciagent init[/]\n"
            )

        # AGENTCI_MOCK_FLAKY=1 simulates agent-variance across runs so the
        # stability report (and its tests) can be exercised without API keys.
        # Demo mode defaults the simulated flakiness ON for multi-run sessions —
        # the whole point of the demo is showing a stable score hiding verdict
        # flips — but an explicit env value always wins.
        flaky_env = os.environ.get("AGENTCI_MOCK_FLAKY")
        if flaky_env is None:
            mock_flaky = demo_mode and runs > 1
        else:
            mock_flaky = flaky_env.lower() in ("1", "true", "yes")

        try:
            run_results = []
            for run_index in range(runs):
                traces = run_mock_spec(
                    spec,
                    run_index=run_index,
                    flaky=mock_flaky,
                    flaky_style="spread" if demo_mode else "alternate",
                )
                results = evaluate_spec(spec, traces, None)
                run_results.append(results)
                if runs > 1 and fmt in ("console", "github"):
                    passed = sum(1 for r in results if not r.hard_fail)
                    console.print(f"Run {run_index + 1}/{runs}: {passed}/{len(results)} passed")
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]Evaluation error:[/] {e}")
            sys.exit(2)

        if runs > 1:
            from .engine.stability import build_stability_report

            stability = build_stability_report(spec, run_results)
            _finish_stability_session(
                fmt, config, output, run_results, stability, fail_on_flaky,
            )

        exit_code = report_results(run_results[0], format=fmt, spec_file=config)
        if demo_mode and fmt == "console":
            console.print(
                "\n[dim]Tip: ciagent test --mock --runs 3 runs the demo suite three times "
                "and shows the stability report with flip-source attribution.[/]"
            )
        sys.exit(exit_code)

    # ── Check for runner ──────────────────────────────────────────────────────
    if not spec.runner:
        console.print(
            f"[bold blue]CIAgent v{__version__}[/] spec has [cyan]{len(spec.queries)}[/] "
            f"queries for agent '[cyan]{spec.agent}[/]'\n"
        )
        console.print(
            "[yellow]ℹ[/] No [bold]runner[/] declared in spec. Add one to run live:\n\n"
            "  [cyan]runner: \"myagent.run:run_agent\"[/]\n\n"
            "The function must accept [bold](query: str) → Trace[/].\n\n"
            "Or use [cyan]ciagent test --mock[/] to validate your spec without API calls.\n\n"
            "Or use the Python API in your test suite:\n"
            "  [cyan]from ciagent import load_spec, run_spec[/]\n"
            "  [cyan]results = run_spec(spec, my_agent_fn)[/]"
        )
        sys.exit(0)

    # ── Resolve runner ────────────────────────────────────────────────────────
    from .engine.parallel import run_spec_parallel, resolve_runner

    try:
        runner_fn = resolve_runner(spec.runner)
    except (ImportError, AttributeError, ValueError) as e:
        console.print(f"[bold red]Runner error:[/] {e}")
        sys.exit(2)

    # ── Inject sample-ensemble into judge_config ──────────────────────────────
    if sample_ensemble is not None:
        if not (0.0 <= sample_ensemble <= 1.0):
            console.print("[bold red]--sample-ensemble must be between 0.0 and 1.0[/]")
            sys.exit(2)
        spec.judge_config = spec.judge_config or {}
        spec.judge_config["sample_ensemble"] = sample_ensemble

    # ── Cost estimate ─────────────────────────────────────────────────────────
    if not yes and fmt == "console":
        from .engine.cost_estimator import estimate_cost, format_estimate

        # Detect if any queries use llm_judge
        has_judge = any(
            (q.correctness and q.correctness.llm_judge)
            for q in spec.queries
            if q.correctness
        )
        judge_model = (spec.judge_config or {}).get("model", "claude-sonnet-4-6")
        # Strip provider prefix (e.g. "openai:gpt-4o-mini" -> "gpt-4o-mini")
        if ":" in judge_model:
            judge_model = judge_model.split(":", 1)[1]

        est = estimate_cost(
            num_queries=len(spec.queries) * runs,
            judge_model=judge_model,
            has_llm_judge=has_judge,
        )
        console.print(f"\n[dim]{format_estimate(est, len(spec.queries) * runs)}[/]")

        from rich.prompt import Confirm
        if not Confirm.ask("Proceed? [Y/n]", default=True):
            console.print("[yellow]Aborted.[/]")
            sys.exit(0)

    # ── Run queries in parallel ───────────────────────────────────────────────
    console.print(
        f"[bold blue]CIAgent v{__version__}[/] │ agent: [cyan]{spec.agent}[/] │ "
        f"queries: [cyan]{len(spec.queries)}[/] │ workers: [cyan]{workers}[/]"
        + (f" │ runs: [cyan]{runs}[/]" if runs > 1 else "")
    )
    if fmt in ("console", "github"):
        console.print("")

    # ── Load baselines (optional) — needed before streaming eval ───────────
    baselines: dict = {}
    baseline_path = Path(effective_baseline_dir)
    if baseline_path.exists() and baseline_path.is_dir():
        import glob
        from .baselines import load_baseline
        for f in glob.glob(str(baseline_path / "*.json")):
            try:
                b = load_baseline(f)
                if "trace" in b and "query" in b.get("trace", {}):
                    q = b["trace"]["query"]
                    from .models import Trace
                    baselines[q] = Trace.from_dict(b["trace"])
            except Exception:  # noqa: BLE001
                pass  # Skip malformed baseline files

    # ── Multi-run stability mode: N sequential passes, then attribution ────
    if runs > 1:
        from .engine.stability import build_stability_report

        spec_dir = str(Path(config).parent) if config else None
        run_results = []
        for run_index in range(runs):
            try:
                traces = run_spec_parallel(spec, runner_fn, max_workers=workers)
            except Exception as e:  # noqa: BLE001
                console.print(f"[bold red]Infrastructure error:[/] {e}")
                sys.exit(2)
            if not traces:
                console.print("[bold red]Error:[/] No traces captured — runner may have failed for all queries.")
                sys.exit(1)
            try:
                results = evaluate_spec(spec, traces, baselines, spec_dir=spec_dir)
            except Exception as e:  # noqa: BLE001
                console.print(f"[bold red]Evaluation error:[/] {e}")
                sys.exit(2)
            run_results.append(results)
            if fmt in ("console", "github"):
                passed = sum(1 for r in results if not r.hard_fail)
                console.print(f"Run {run_index + 1}/{runs}: {passed}/{len(results)} passed")

        stability = build_stability_report(spec, run_results)
        _finish_stability_session(
            fmt, config, output, run_results, stability, fail_on_flaky,
        )

    # ── Streaming evaluation: print each result as its trace arrives ───────
    import threading
    from .engine.runner import evaluate_query
    from .engine.reporter import emit_query_result, emit_summary

    query_lookup = {q.query: q for q in spec.queries}
    streaming_results: list = []
    _print_lock = threading.Lock()
    stream_console = fmt in ("console", "github") and fmt != "html"

    def _on_trace(query_text: str, trace):
        gq = query_lookup.get(query_text)
        if gq is None:
            return
        try:
            result = evaluate_query(
                query=gq,
                trace=trace,
                baseline_trace=baselines.get(query_text),
                judge_config=spec.judge_config,
                spec_dir=str(Path(config).parent) if config else None,
            )
            with _print_lock:
                streaming_results.append(result)
                if stream_console:
                    emit_query_result(result)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Evaluation warning:[/] '{query_text[:40]}': {exc}")

    try:
        traces = run_spec_parallel(
            spec, runner_fn, max_workers=workers, on_trace=_on_trace,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Infrastructure error:[/] {e}")
        sys.exit(2)

    if not traces:
        console.print("[bold red]Error:[/] No traces captured — runner may have failed for all queries.")
        sys.exit(1)

    # ── Report summary / non-console formats ──────────────────────────────
    results = streaming_results
    if stream_console:
        # Individual results already printed; just emit summary + annotations
        emit_summary(results)
        if os.environ.get("GITHUB_ACTIONS") == "true" or fmt == "github":
            from .engine.reporter import _emit_github_annotations
            _emit_github_annotations(results, config)
        exit_code = 1 if any(r.hard_fail for r in results) else 0
    else:
        exit_code = report_results(results, format=fmt, spec_file=config, output_path=output)
    sys.exit(exit_code)

def _collect_replay_paths(path, agent):
    """Resolve --replay PATH to golden envelope files.

    A file replays as-is. A directory prefers the recorded-golden layout
    (<dir>/<agent>/scenarios/*.json — what --record writes into baseline_dir),
    falling back to the .json files directly inside it.
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        scenarios_dir = path / agent / "scenarios"
        if scenarios_dir.is_dir():
            return sorted(scenarios_dir.glob("*.json"))
        return sorted(p for p in path.glob("*.json") if p.is_file())
    return []


def _checks_use_judge(checks) -> bool:
    if checks is None or checks.correctness is None:
        return False
    c = checks.correctness
    return bool(
        getattr(c, "llm_judge", None)
        or getattr(c, "safety_check", None)
        or getattr(c, "hallucination_check", None)
    )


def _simulation_turn_plan(scenarios) -> tuple[int, int, int]:
    """Planned (agent_turns, persona_turns, judged_turns) for ONE run.

    Scripted scenarios run min(len(turns), max_turns) turns; generative ones
    are budgeted at max_turns — the ceiling is exactly what the estimate is
    for. Judged turns count per-turn judge checks per turn plus one for a
    judged outcome.
    """
    agent_turns = persona_turns = judged_turns = 0
    for s in scenarios:
        planned = min(len(s.turns), s.max_turns) if s.turns else s.max_turns
        agent_turns += planned
        if not s.turns:
            persona_turns += planned
        if _checks_use_judge(s.per_turn):
            judged_turns += planned
        if _checks_use_judge(s.outcome):
            judged_turns += 1
    return agent_turns, persona_turns, judged_turns


def _resolve_redactor(spec):
    """Redactor per spec staging config: the real one unless `redact: false`.

    StageStore's constructor default stays `_identity` (storage is policy-free,
    ADR A8) — policy is resolved here, at the CLI layer, in the two places
    that construct stores.
    """
    from .promotion import _identity
    from .redaction import Redactor

    staging = getattr(spec, "staging", None)
    if staging is not None and not getattr(staging, "redact", True):
        return _identity
    patterns = list(getattr(staging, "redact_patterns", []) or [])
    return Redactor(extra_patterns=patterns)


def _warn_redacted_check_literals(env) -> None:
    """ADR A4: a `not_in_answer` check whose literal was redacted is vacuous,
    and an `expected_in_answer` on a redacted literal can never match a live
    answer. This never self-surfaces, so verify/promote must warn."""
    from .redaction import contains_placeholder

    def _strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from _strings(v)
        elif isinstance(node, list):
            for v in node:
                yield from _strings(v)

    spec_dict = (env.scenario or {}).get("spec") or {}
    if any(contains_placeholder(s) for s in _strings(spec_dict)):
        console.print(
            "[yellow]warning:[/] this scenario's checks reference redacted "
            "values — verify semantics are degraded. A `not_in_answer` check "
            "on a redacted literal is vacuous; express leak-gates as regex "
            "checks (regex values are not rewritten by redaction)."
        )


def _auto_stage_failures(results, stability, spec, staged_dir, source):
    """Stage every failing scenario result (best-effort).

    Returns the count staged. Any StageStore error prints a warning and is
    swallowed so staging can never change the run's exit code.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from .engine.simulate import scenario_slug
    from .promotion import (
        DEFAULT_STAGED_DIR,
        StageStore,
        TriageClassifier,
        build_staging_block,
    )

    try:
        if not getattr(spec.staging, "redact", True):
            console.print(
                "[yellow]staging warning:[/] redact is disabled — staged files "
                "will contain RAW conversation text (possibly PII/secrets)."
            )
        root = Path(staged_dir or DEFAULT_STAGED_DIR)
        store = StageStore(
            root,
            cap=getattr(spec.staging, "cap", 10),
            max_age_days=getattr(spec.staging, "max_age_days", 30),
            redactor=_resolve_redactor(spec),
        )
        run_id = datetime.now(timezone.utc).strftime("sim-%Y%m%dT%H%M%S")
        # stability is aligned to results by index when runs > 1
        by_index = stability or []
        staged = 0
        for i, r in enumerate(results):
            if not r.hard_fail:
                continue
            stab = by_index[i] if i < len(by_index) else None
            runs = stab.runs if stab is not None else 1
            klass = TriageClassifier.classify(stab, runs=runs)
            name = r.scenario.display_name()
            summary = "; ".join(
                (r.outcome.correctness.messages[:1] if r.outcome else [])
            ) or (r.error or "scenario failed its checks")
            block = build_staging_block(
                run_id=run_id,
                scenario_id=scenario_slug(name),
                source=source,
                classification=klass,
                stability=stab,
                runs_observed=runs,
                failure_summary=summary,
            )
            try:
                store.stage(r.to_envelope(agent=spec.agent, mode=r.mode), staging_block=block)
                staged += 1
            except Exception as e:  # noqa: BLE001 — best-effort, never fail the run
                console.print(f"[yellow]staging warning:[/] {name}: {e}")
        if staged:
            # Existing repos never re-run `init`, so protect them here: the
            # scaffold is idempotent and staged files must not be committed
            # (redaction reduces blast radius, it does not make them public).
            try:
                _scaffold_staging_gitignore()
            except OSError:
                pass
            console.print(
                f"[dim]staged {staged} failing conversation(s) → "
                f"[cyan]ciagent stage list[/] / [cyan]ciagent promote[/][/]"
            )
        return staged
    except Exception as e:  # noqa: BLE001 — staging is never load-bearing
        console.print(f"[yellow]staging warning:[/] {e}")
        return 0


def _print_conversation_diff(diff):
    """Console block for a replay's turn-by-turn diff vs its golden."""
    if not diff.has_changes:
        console.print("   [dim]diff vs golden: no changes[/]")
        return
    if diff.turn_count_changed:
        console.print(
            f"   [yellow]diff: turn count changed[/] {diff.turns_before} → {diff.turns_after}"
        )
    for t in diff.turn_diffs:
        if not t.changed:
            continue
        if t.tools_changed:
            console.print(
                f"   [yellow]diff turn {t.turn_index + 1}: tools[/] "
                f"{t.tools_before} → {t.tools_after}"
            )
        if t.answer_changed:
            console.print(
                f"   [yellow]diff turn {t.turn_index + 1}: answer[/] "
                f"{t.answer_before[:60]!r} → {t.answer_after[:60]!r}"
            )


@cli.command(name="simulate")
@click.option('--config', '-c', default='agentci_spec.yaml',
              help='Path to agentci_spec.yaml', show_default=True)
@click.option('--mock', is_flag=True,
              help='Synthetic conversations, zero API calls: each turn satisfies '
                   'the scenario checks. Validates scenario structure.')
@click.option('--record', is_flag=True,
              help='Save each scenario\'s conversation as a golden envelope under '
                   '<baseline_dir>/<agent>/scenarios/. Failed scenarios record too — '
                   'that is the found-bug → regression-test conversion.')
@click.option('--record-dir', default=None, type=click.Path(),
              help='Record into this directory instead of the spec\'s baseline_dir '
                   '(implies --record).')
@click.option('--replay', 'replay_path', default=None, type=click.Path(),
              help='Replay recorded conversation envelope(s) — a golden .json file '
                   'or a directory — feeding the recorded user turns back to the '
                   'agent verbatim. Spec scenarios are ignored.')
@click.option('--runs', default=1, show_default=True, type=int,
              help='Run every scenario N times and report per-scenario stability '
                   'with flip attribution (agent-variance vs simulation-variance '
                   'vs judge-flake).')
@click.option('--workers', '-w', default=4, show_default=True, type=int,
              help='Max parallel scenarios; turns within a conversation stay '
                   'sequential.')
@click.option('--max-cost', default=None, type=float,
              help='Session cost ceiling in USD. When breached, the session '
                   'hard-aborts mid-conversation; partial results are reported '
                   'and clearly marked.')
@click.option('--yes', '-y', is_flag=True, help='Skip the live-run confirmation')
@click.option('--stage/--no-stage', 'stage_flag', default=None,
              help='Auto-stage failing conversations for later promotion '
                   '(overrides spec staging.enabled; default ON — staged '
                   'files are redacted at capture time).')
@click.option('--staged-dir', default=None, type=click.Path(),
              help='Staging root (default: .ciagent/staged)')
@click.option('--format', 'fmt', type=click.Choice(['console', 'json']),
              default='console', show_default=True, help='Output format')
def simulate_cmd(config, mock, record, record_dir, replay_path, runs, workers,
                 max_cost, yes, stage_flag, staged_dir, fmt):
    """Drive multi-turn conversation scenarios against your agent.

    Scripted scenarios (a `turns:` list of user messages) run
    deterministically — the CI path. Generative scenarios (`persona:` +
    `goal:`, no turns) let a cheap persona LLM improvise the user side —
    the finder path, nondeterministic by design. `--record` saves each
    conversation as a golden envelope; `--replay` feeds a recorded
    conversation's user turns back verbatim (the persona is never called,
    only the agent side can vary) and shows a turn-by-turn diff against
    the golden.

    \b
    Spec additions:
        conversation_runner: "myagent.run:respond"   # (messages) -> str | Trace
        persona_config: {model: claude-haiku-4-5, temperature: 0.7}  # optional
        scenarios:
          - turns: ["hi", "i want a refund for order #123"]   # scripted
            max_turns: 8
            per_turn:
              path: {expected_tools: [search_kb]}
            outcome:
              correctness: {any_expected_in_answer: ["refund", "5-7 business days"]}
          - persona: "frustrated customer, discontinued product"  # generative
            goal: "get a refund routed correctly"
            max_turns: 8
            outcome:
              correctness: {any_expected_in_answer: ["refund"]}

    \b
    Found-bug → regression-test in one command:
        ciagent simulate --record            # failing conversation saved as golden
        ciagent simulate --replay ./golden   # the suite now gates on it

    \b
    Termination is deterministic: scripted turns exhausted, max_turns, or an
    explicit stop_when event — the persona never ends a conversation and
    outcome checks are the END-of-conversation verdict, never a stop
    condition. Live sessions show a cost estimate before running; --max-cost
    adds a hard mid-session abort.

    \b
    Exit codes:
        0 — every scenario's checks passed
        1 — a scenario failed its outcome or a per-turn correctness check
        2 — config error, agent exception, or session aborted on --max-cost
    """
    _route_chrome(fmt)
    import json as _json
    from pathlib import Path

    from .conversation import load_envelope
    from .engine.simulate import envelope_to_scenario
    from .exceptions import BaselineError, ConfigError
    from .loader import load_spec

    try:
        spec = load_spec(config)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        sys.exit(2)

    # ── Resolve what to run: spec scenarios, or recorded envelopes (--replay) ─
    from typing import Any as _Any

    goldens: list[_Any] = []  # parallel to scenarios: (path, envelope) when replaying, else None
    if replay_path:
        paths = _collect_replay_paths(Path(replay_path), spec.agent)
        if not paths:
            console.print(
                f"[bold red]Nothing to replay at[/] [cyan]{replay_path}[/] — expected a "
                "golden envelope .json or a directory containing "
                f"[cyan]{spec.agent}/scenarios/*.json[/]."
            )
            sys.exit(2)
        scenarios = []
        for p in paths:
            try:
                env = load_envelope(p)
                scenarios.append(envelope_to_scenario(env))
            except (BaselineError, ValueError) as e:
                console.print(f"[bold red]Replay error:[/] {e}")
                sys.exit(2)
            goldens.append((p, env))
    else:
        if not spec.scenarios:
            console.print(
                "[bold red]No scenarios in spec.[/] Add a [cyan]scenarios:[/] block — "
                "see [cyan]ciagent simulate --help[/] for the shape."
            )
            sys.exit(2)
        scenarios = spec.scenarios
        goldens = [None] * len(scenarios)

    run_mode = "replay" if replay_path else "scripted"
    if replay_path:
        mode_str = "mock replay" if mock else "replay"
    else:
        mode_str = "mock" if mock else "live"
    console.print(
        f"[bold blue]CIAgent v{__version__}[/] │ simulate │ agent: [cyan]{spec.agent}[/] │ "
        f"scenarios: [cyan]{len(scenarios)}[/] │ mode: [cyan]{mode_str}[/]"
    )

    # Resolve the conversation runner (mock synthesizes one per scenario);
    # generative scenarios additionally need a user-turn source (mock persona
    # in --mock, the persona LLM otherwise — the driver builds the latter).
    conv_runner_for = None
    turn_source_for = None
    if mock:
        from .engine.mock_runner import mock_conversation_runner, mock_persona_turn_source

        console.print("[dim]Running with synthetic traces — zero API cost[/]\n")
        conv_runner_for = mock_conversation_runner
        turn_source_for = (
            lambda s: mock_persona_turn_source(s) if not s.turns else None
        )
    else:
        if not spec.conversation_runner:
            console.print(
                "[bold red]No conversation_runner in spec.[/] Add one to run live:\n\n"
                "  [cyan]conversation_runner: \"myagent.run:respond\"[/]\n\n"
                "The function must accept [bold](messages: list[dict]) → str | Trace[/].\n"
                "Or use [cyan]ciagent simulate --mock[/] to validate scenarios without API calls."
            )
            sys.exit(2)
        from .engine.parallel import resolve_runner

        try:
            live_runner = resolve_runner(spec.conversation_runner)
        except (ImportError, AttributeError, ValueError) as e:
            console.print(f"[bold red]Runner error:[/] {e}")
            sys.exit(2)
        # Pre-run cost estimate + confirm gate (binding: the tool that sells
        # cost budgets does not ship a simulator without them)
        agent_turns, persona_turns, judged_turns = _simulation_turn_plan(scenarios)
        if not yes and fmt == "console":
            from rich.prompt import Confirm

            from .engine.cost_estimator import (
                estimate_simulation_cost,
                format_simulation_estimate,
            )

            est = estimate_simulation_cost(
                agent_turns=agent_turns,
                persona_turns=persona_turns,
                judged_turns=judged_turns,
                runs=runs,
                persona_model=(spec.persona_config or {}).get("model"),
                judge_model=(spec.judge_config or {}).get("model"),
            )
            console.print(f"[dim]{format_simulation_estimate(est, len(scenarios), runs)}[/]")
            if max_cost is not None:
                console.print(f"[dim]Hard abort at --max-cost ${max_cost:.2f}[/]")
            if not Confirm.ask("Proceed?", default=True):
                console.print("[yellow]Aborted.[/]")
                sys.exit(0)
        conv_runner_for = lambda scenario: live_runner  # noqa: E731

    from .engine.simulate import CostBudget, run_scenarios_parallel

    budget = CostBudget(max_usd=max_cost) if max_cost is not None else None

    all_runs: list[list] = []
    for _run_index in range(max(1, runs)):
        if budget is not None and budget.exceeded:
            break  # session aborted — completed runs still report
        try:
            run_results = run_scenarios_parallel(
                scenarios,
                conv_runner_for,
                agent_name=spec.agent,
                judge_config=spec.judge_config,
                spec_dir=str(Path(config).parent) if config else None,
                max_workers=workers,
                turn_source_for=turn_source_for,
                persona_config=spec.persona_config,
                budget=budget,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]Simulation error:[/] {e}")
            sys.exit(2)
        all_runs.append(run_results)
    results = all_runs[0]

    cost_aborted = any(r.is_cost_aborted for rr in all_runs for r in rr) or (
        budget is not None and budget.exceeded and len(all_runs) < runs
    )

    # Per-scenario stability across runs (simulation-variance attribution)
    stability = None
    if len(all_runs) > 1:
        from .engine.stability import build_scenario_stability

        stability = build_scenario_stability(all_runs)

    # ── Auto-stage failing conversations (opt-in v1) ─────────────────────────
    # A nondeterministic repro is captured so it is never lost; promoting one to
    # a permanent CI gate stays a human `ciagent promote`. Best-effort: a
    # StageStore error prints a warning and NEVER changes the run's exit code.
    staging_enabled = (
        stage_flag if stage_flag is not None
        else getattr(spec.staging, "enabled", False)
    )
    any_failure = any(r.hard_fail for r in results)
    if any_failure and not replay_path:
        if staging_enabled:
            _auto_stage_failures(results, stability, spec, staged_dir, "simulate")
        else:
            console.print(
                "[dim]a repro was found — enable staging (spec staging.enabled or "
                "--stage) to keep it for promotion.[/]"
            )

    # ── Conversation-aware diff vs the golden being replayed ─────────────────
    diffs: list[_Any] = [None] * len(results)
    if replay_path:
        from .engine.diff import diff_envelopes

        diffs = [
            diff_envelopes(golden_env, r.to_envelope(agent=spec.agent, mode="replay"))
            for (_, golden_env), r in zip(goldens, results)
        ]

    # ── Record each conversation as a golden envelope (--record) ─────────────
    recorded_paths = []
    if record or record_dir:
        from .engine.simulate import record_scenario_result

        out_dir = record_dir or spec.baseline_dir
        for r in results:
            try:
                recorded_paths.append(
                    record_scenario_result(
                        r,
                        out_dir,
                        agent=spec.agent,
                        # replay re-records as replay; otherwise mode as produced
                        # (scripted | simulated)
                        mode="replay" if replay_path else None,
                        mock=mock,
                    )
                )
            except ValueError as e:
                # Artifact gate refused (e.g. zero turns after an infra
                # error) — skip this golden, keep recording the rest.
                console.print(f"[yellow]not recorded:[/] {e}")

    # ── Report ────────────────────────────────────────────────────────────────
    if fmt == "json":
        from .engine.reporter import _serialize_result

        payload = {
            "mode": run_mode,
            "runs": len(all_runs),
            "scenarios": [
                {
                    "name": r.scenario.display_name(),
                    "mode": r.mode,
                    "termination": r.termination,
                    "error": r.error,
                    "hard_fail": r.hard_fail,
                    "partial": r.is_partial,
                    "cost_usd": round(r.cost_usd, 6),
                    "turns": [
                        {
                            "turn_index": t.turn_index,
                            "user_message": t.user_message,
                            "answer": (t.trace.metadata or {}).get("final_output"),
                            "checks": _serialize_result(t.checks) if t.checks else None,
                        }
                        for t in r.turns
                    ],
                    "outcome": _serialize_result(r.outcome) if r.outcome else None,
                    "diff": d.summary_json() if d is not None else None,
                }
                for r, d in zip(results, diffs)
            ],
            "recorded": [str(p) for p in recorded_paths],
            "stability": (
                [
                    {
                        "scenario": s.scenario,
                        "verdicts": s.verdicts,
                        "flipped": s.flipped,
                        "flip_source": s.flip_source.value if s.flip_source else None,
                        "flip_reason": s.flip_reason,
                    }
                    for s in stability
                ]
                if stability is not None
                else None
            ),
            "cost_aborted": cost_aborted,
            "spent_usd": round(budget.spent_usd, 6) if budget is not None else None,
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if not r.hard_fail and not r.is_partial),
                "failed": sum(1 for r in results if r.hard_fail),
                "infra_errors": sum(1 for r in results if r.is_infra_error),
                "cost_aborted": sum(1 for r in results if r.is_cost_aborted),
            },
        }
        print(_json.dumps(payload, indent=2))
    else:
        if cost_aborted and budget is not None:
            console.print(
                f"\n[bold red]SESSION ABORTED — --max-cost ${budget.max_usd:.2f} breached "
                f"(spent ~${budget.spent_usd:.4f}). Results below are PARTIAL.[/]"
            )
        for r, d in zip(results, diffs):
            icon = "❌" if r.hard_fail else ("⚠️ " if r.is_partial else "✅")
            tag = " [magenta](simulated)[/]" if r.mode == "simulated" else ""
            console.print(
                f"\n{icon} [bold]{r.scenario.display_name()}[/]{tag} — "
                f"{len(r.turns)} turn(s), ended: [cyan]{r.termination}[/]"
            )
            if r.is_partial:
                console.print(
                    "   [yellow]PARTIAL — conversation did not complete; "
                    "outcome verdict not evaluated[/]"
                )
            if r.error:
                console.print(f"   [red]{r.error}[/]")
            for t in r.turns:
                mark = ""
                if t.checks is not None:
                    mark = " ❌" if t.checks.hard_fail else " ✅"
                answer = ((t.trace.metadata or {}).get("final_output") or "")[:70]
                console.print(f"   [dim]turn {t.turn_index + 1}:[/] {t.user_message[:50]!r} → {answer!r}{mark}")
            if r.outcome is not None:
                status = "FAIL" if r.outcome.hard_fail else "PASS"
                color = "red" if r.outcome.hard_fail else "green"
                console.print(f"   outcome: [{color}]{status}[/] {'; '.join(r.outcome.correctness.messages[:2])}")
            if r.cost_usd > 0:
                console.print(f"   [dim]cost: ${r.cost_usd:.4f}[/]")
            if d is not None:
                _print_conversation_diff(d)
        if stability is not None:
            console.print(f"\n[bold]Stability across {len(all_runs)} runs:[/]")
            for s in stability:
                flip = ""
                if s.flipped and s.flip_source is not None:
                    flip = f"  [yellow]{s.flip_source.value}[/] — {s.flip_reason}"
                console.print(f"   {s.verdict_string}  {s.scenario}{flip}")
            flaky = sum(1 for s in stability if s.flipped)
            console.print(
                f"   [dim]{len(stability) - flaky}/{len(stability)} stable[/]"
                + (f"  |  [yellow]{flaky} flaky[/]" if flaky else "")
            )
        if recorded_paths:
            console.print(f"\n[dim]Recorded {len(recorded_paths)} golden envelope(s):[/]")
            for p in recorded_paths:
                console.print(f"   [cyan]{p}[/]")
        passed = sum(1 for r in results if not r.hard_fail and not r.is_partial)
        console.print(
            f"\nScenarios: {passed}/{len(results)} passed"
            + (f"  |  {sum(1 for r in results if r.is_partial)} partial/infra" if any(r.is_partial for r in results) else "")
        )

    # Exit codes consider every run, not just the reported first one
    if any(r.hard_fail for rr in all_runs for r in rr):
        sys.exit(1)
    if cost_aborted or any(r.is_infra_error for rr in all_runs for r in rr):
        sys.exit(2)
    sys.exit(0)


@cli.command(name="eval")
@click.option('--config', '-c', default='agentci_spec.yaml',
              help='Path to agentci_spec.yaml', show_default=True)
@click.option('--tags', '-t', multiple=True, help='Only evaluate queries with these tags')
@click.option('--format', 'fmt',
              type=click.Choice(['console', 'github', 'json', 'prometheus', 'html']),
              default='console', show_default=True, help='Output format')
@click.option('--output', '-o', default=None, type=click.Path(),
              help='Output file path (used with --format html, default: agentci-report.html)')
@click.option('--workers', '-w', default=4, show_default=True, type=int,
              help='Max parallel workers for query execution')
@click.option('--sample-ensemble', default=None, type=float,
              help='Fraction of queries to use ensemble judging (0.0-1.0, e.g. 0.2)')
def eval_cmd(config, tags, fmt, output, workers, sample_ensemble):
    """Run evaluation WITHOUT requiring golden baselines.

    Useful for correctness-only checks or absolute cost/path boundaries.
    Skips relative assertions like max_cost_multiplier and min_sequence_similarity.
    """
    _route_chrome(fmt)
    from .loader import load_spec, filter_by_tags
    from .engine.reporter import report_results
    from .engine.parallel import run_spec_parallel, resolve_runner
    from .engine.runner import evaluate_spec
    from .exceptions import ConfigError, AgentCIError

    try:
        spec = load_spec(config)
    except AgentCIError as e:
        _print_error_panel(e)
        sys.exit(2)

    if tags:
        spec = filter_by_tags(spec, list(tags))
        if not spec.queries:
            console.print(f"[yellow]No queries match tags: {tags}[/]")
            sys.exit(0)

    if not spec.runner:
        console.print(
            "[yellow]ℹ[/] No [bold]runner[/] declared in spec. Add one to run locally:\n\n"
            "  [cyan]runner: \"myagent.run:run_agent\"[/]\n"
        )
        sys.exit(0)

    try:
        runner_fn = resolve_runner(spec.runner)
    except (ImportError, AttributeError, ValueError) as e:
        console.print(f"[bold red]Runner error:[/] {e}")
        sys.exit(2)

    if sample_ensemble is not None:
        if not (0.0 <= sample_ensemble <= 1.0):
            console.print("[bold red]--sample-ensemble must be between 0.0 and 1.0[/]")
            sys.exit(2)
        spec.judge_config = spec.judge_config or {}
        spec.judge_config["sample_ensemble"] = sample_ensemble

    console.print(
        f"[bold blue]CIAgent v{__version__} Eval[/] │ agent: [cyan]{spec.agent}[/] │ "
        f"queries: [cyan]{len(spec.queries)}[/] │ workers: [cyan]{workers}[/]"
    )
    if fmt in ("console", "github"):
        console.print("")

    try:
        traces = run_spec_parallel(spec, runner_fn, max_workers=workers)
    except Exception as e:
        console.print(f"[bold red]Infrastructure error:[/] {e}")
        sys.exit(2)

    if not traces:
        console.print("[bold red]Error:[/] No traces captured.")
        sys.exit(1)

    # Note: Explicitly passing None for baselines
    try:
        results = evaluate_spec(spec, traces, None)
    except Exception as e:
        console.print(f"[bold red]Evaluation error:[/] {e}")
        sys.exit(2)

    exit_code = report_results(results, format=fmt, spec_file=config, output_path=output)
    sys.exit(exit_code)



@cli.command(name="save")
@click.option('--agent', required=True, help='Agent identifier (matches spec agent field)')
@click.option('--version', required=True, help='Version tag, e.g. v1-broken or v2-fixed')
@click.option('--query', 'query_text', default='', help='Query text this baseline corresponds to')
@click.option('--config', '-c', default='agentci_spec.yaml', show_default=True)
@click.option('--baseline-dir', default=None, help='Override baseline directory')
@click.option('--force-save', is_flag=True,
              help='Bypass correctness precheck and save anyway')
@click.option('--trace-file', type=click.Path(exists=True), required=True,
              help='Path to trace JSON file to save as baseline')
def save_cmd(agent, version, query_text, config, baseline_dir, force_save, trace_file):
    """Save a trace as a versioned golden baseline.

    By default runs a correctness precheck against the spec before saving.
    Use --force-save to bypass (e.g. for intentional "broken" demo baselines).
    """
    import json
    from .loader import load_spec
    from .baselines import save_baseline
    from .models import Trace
    from .exceptions import ConfigError

    try:
        spec = load_spec(config)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        sys.exit(2)

    try:
        trace_data = json.loads(open(trace_file).read())
        trace = Trace.model_validate(trace_data)
    except Exception as e:
        console.print(f"[bold red]Failed to load trace:[/] {e}")
        sys.exit(2)

    effective_dir = baseline_dir or spec.baseline_dir

    try:
        out_path = save_baseline(
            trace=trace,
            agent=agent,
            version=version,
            spec=spec,
            query_text=query_text,
            baseline_dir=effective_dir,
            force=force_save,
        )
        console.print(f"[green]✅ Saved baseline:[/] {out_path}")
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        if e.recovery_hint:
            console.print(f"  [yellow]Fix:[/] {e.recovery_hint}")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[bold red]Precheck failed:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(2)


@cli.command(name="baselines")
@click.option('--agent', required=True, help='Agent identifier')
@click.option('--config', '-c', default='agentci_spec.yaml', show_default=True)
@click.option('--baseline-dir', default=None)
def baselines_cmd(agent, config, baseline_dir):
    """List available baseline versions for an agent."""
    from .baselines import list_baselines
    from .loader import load_spec
    from .exceptions import ConfigError

    effective_dir = baseline_dir
    if not effective_dir:
        try:
            spec = load_spec(config)
            effective_dir = spec.baseline_dir
        except ConfigError:
            effective_dir = "./baselines"

    entries = list_baselines(agent, effective_dir)
    if not entries:
        console.print(f"[yellow]No baselines found for agent '{agent}' in {effective_dir}[/]")
        return

    table = Table(title=f"Baselines: {agent}")
    table.add_column("Version", style="cyan")
    table.add_column("Captured At")
    table.add_column("Query")
    table.add_column("Precheck")
    table.add_column("Spec Hash")

    for e in entries:
        meta = e.get("metadata", {})
        precheck = "[green]✅[/]" if meta.get("precheck_passed") else "[yellow]⚠ forced[/]"
        table.add_row(
            e["version"],
            e.get("captured_at", "—")[:19],  # Trim to datetime only
            (e.get("query") or "—")[:50],
            precheck,
            (meta.get("spec_hash") or "—")[:16],
        )

    console.print(table)


# ── Golden Promotion Pipeline: stage + promote (0.11) ───────────────────────────


def _stage_store(config, staged_dir):
    """Resolve (spec, StageStore) from config + staged-dir. Exit 2 on config error."""
    from .exceptions import ConfigError
    from .loader import load_spec
    from .promotion import DEFAULT_STAGED_DIR, StageStore
    from pathlib import Path

    try:
        spec = load_spec(config)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        sys.exit(2)
    root = Path(staged_dir or DEFAULT_STAGED_DIR)
    store = StageStore(
        root,
        cap=getattr(spec.staging, "cap", 10),
        max_age_days=getattr(spec.staging, "max_age_days", 30),
        redactor=_resolve_redactor(spec),
    )
    return spec, store


@cli.group(name="stage")
def stage():
    """Inspect, verify, and clean up auto-staged failing conversations.

    Every failing simulate conversation is captured under
    .ciagent/staged/<agent>/<scenario-id>/<run-ts>.json when staging is enabled.
    Staging never loses a repro; promoting one to a permanent CI gate is
    `ciagent promote`. This group is for looking at what's staged and pruning it.
    """


@stage.command(name="list")
@click.option('--config', '-c', default='agentci_spec.yaml', show_default=True,
              help='Spec used to resolve agent + staging config')
@click.option('--staged-dir', default=None, type=click.Path(),
              help='Staging root (default: .ciagent/staged)')
@click.option('--agent', default=None, help='Filter to one agent')
@click.option('--classification', 'klass', default=None,
              type=click.Choice(['consistent', 'flaky-agent', 'held',
                                  'held-infra', 'unverified']),
              help='Show only entries with this triage classification')
@click.option('--format', 'fmt', type=click.Choice(['console', 'json']),
              default='console', show_default=True)
def stage_list(config, staged_dir, agent, klass, fmt):
    """List staged conversations, best-to-promote first.

    Sort order: consistent → flaky-agent → unverified → held → held-infra,
    then newest run-ts first.
    \b
    Exit codes:
        0 — listed (including empty)
        2 — config error
    """
    _route_chrome(fmt)
    import json as _json

    _spec, store = _stage_store(config, staged_dir)
    entries = [e for e in store.list(agent=agent) if not klass or e.classification == klass]

    if fmt == "json":
        print(_json.dumps([
            {
                "id": e.stage_id,
                "agent": e.agent,
                "scenario_id": e.scenario_id,
                "classification": e.classification,
                "runs_observed": e.staging.get("runs_observed"),
                "verdicts": e.staging.get("verdicts"),
                "failure_summary": e.staging.get("failure_summary"),
            }
            for e in entries
        ], indent=2))
        sys.exit(0)

    if not entries:
        console.print("[yellow]No staged conversations.[/]")
        sys.exit(0)
    table = Table(title="Staged conversations (best to promote first)")
    table.add_column("id", style="cyan")
    table.add_column("class")
    table.add_column("verdicts")
    table.add_column("runs")
    table.add_column("summary")
    for e in entries:
        verdicts = "".join("✅" if v else "❌" for v in (e.staging.get("verdicts") or []))
        table.add_row(
            e.stage_id,
            e.classification,
            verdicts or "—",
            str(e.staging.get("runs_observed", "—")),
            (e.staging.get("failure_summary") or "—")[:50],
        )
    console.print(table)
    sys.exit(0)


@stage.command(name="show")
@click.argument('stage_id')
@click.option('--config', '-c', default='agentci_spec.yaml', show_default=True)
@click.option('--staged-dir', default=None, type=click.Path())
@click.option('--export', 'export_path', default=None, type=click.Path(),
              help='Write a REDACTED copy of the staged envelope here for '
                   'sharing (issue/PR). Redaction runs with the current config '
                   'on every show/export, so pre-0.12 unredacted staged files '
                   'are scrubbed too.')
@click.option('--format', 'fmt', type=click.Choice(['console', 'json']),
              default='console', show_default=True)
def stage_show(stage_id, config, staged_dir, export_path, fmt):
    """Show one staged conversation (turns, checks, triage block).

    \b
    Exit codes:
        0 — shown (and exported, if --export)
        1 — no staged entry matches STAGE_ID
        2 — config error
    """
    _route_chrome(fmt)
    import json as _json
    from pathlib import Path

    from .conversation import save_envelope
    from .promotion import StageAmbiguous, StageNotFound, _identity

    _spec, store = _stage_store(config, staged_dir)
    try:
        path, env = store.load(stage_id)
    except StageNotFound as e:
        console.print(f"[bold red]{e}[/]")
        sys.exit(1)
    except StageAmbiguous as e:
        console.print(f"[bold red]{e}[/]")
        sys.exit(1)

    # Everything `stage show` emits is a sharing path (ADR A11): re-redact
    # with the current config. Covers pre-0.12 files staged before redaction
    # existed; on already-redacted files this is a no-op by construction.
    redactor = _resolve_redactor(_spec)
    if redactor is not _identity:
        env = redactor(env)

    if fmt == "json":
        print(_json.dumps(_json.loads(env.model_dump_json()), indent=2))
    else:
        st = env.staging or {}
        console.print(f"[bold]{path}[/]")
        console.print(f"  agent: [cyan]{env.agent}[/]  class: [cyan]{st.get('classification')}[/]")
        console.print(f"  runs: {st.get('runs_observed')}  flip: {st.get('flip_source')}")
        console.print(f"  summary: {st.get('failure_summary')}")
        for t in env.turns:
            answer = ((t.trace.metadata or {}).get("final_output") or "")[:70]
            console.print(f"  [dim]turn {t.turn_index + 1}:[/] {t.user_message[:50]!r} → {answer!r}")

    if export_path:
        out = save_envelope(env, Path(export_path))
        if redactor is _identity:
            console.print(f"[green]exported:[/] {out}")
            console.print(
                "[yellow]note:[/] redact is disabled in your spec — this file "
                "contains the raw conversation text. Review it before sharing."
            )
        else:
            console.print(f"[green]exported redacted copy:[/] {out}")
    sys.exit(0)


@stage.command(name="verify")
@click.argument('stage_id')
@click.option('--config', '-c', default='agentci_spec.yaml', show_default=True)
@click.option('--staged-dir', default=None, type=click.Path())
@click.option('--runs', default=3, show_default=True, type=int,
              help='Re-run this scenario N times, replaying the staged user turns '
                   'verbatim (persona NOT re-rolled — verifies agent-side '
                   'reproducibility, not simulation luck).')
@click.option('--mock', is_flag=True, help='Verify with synthetic traces (zero keys)')
@click.option('--workers', '-w', default=4, show_default=True, type=int)
@click.option('--yes', '-y', is_flag=True, help='Skip the live-run confirmation')
def stage_verify(stage_id, config, staged_dir, runs, mock, workers, yes):
    """Re-run one staged scenario N times and re-classify it in place.

    The cheap path from `unverified` → `consistent` (or a downgrade). Zero-key
    verification only in --mock.
    \b
    Exit codes:
        0 — verified and re-classified (regardless of new class)
        1 — no staged entry matches STAGE_ID
        2 — config/runner error or cost abort
    """
    from .engine.simulate import envelope_to_scenario, run_scenario
    from .engine.stability import build_scenario_stability
    from .promotion import (
        StageAmbiguous,
        StageNotFound,
        TriageClassifier,
        build_staging_block,
    )

    spec, store = _stage_store(config, staged_dir)
    try:
        _path, env = store.load(stage_id)
    except (StageNotFound, StageAmbiguous) as e:
        console.print(f"[bold red]{e}[/]")
        sys.exit(1)

    try:
        scenario = envelope_to_scenario(env)
    except ValueError as e:
        console.print(f"[bold red]Cannot replay:[/] {e}")
        sys.exit(2)
    _warn_redacted_check_literals(env)

    if mock:
        from .engine.mock_runner import mock_conversation_runner
        conv_runner = mock_conversation_runner(scenario)
    else:
        if not spec.conversation_runner:
            console.print(
                "[bold red]No conversation_runner in spec.[/] Use --mock for "
                "zero-key verification, or add a conversation_runner."
            )
            sys.exit(2)
        from .engine.parallel import resolve_runner
        try:
            conv_runner = resolve_runner(spec.conversation_runner)
        except (ImportError, AttributeError, ValueError) as e:
            console.print(f"[bold red]Runner error:[/] {e}")
            sys.exit(2)
        if not yes:
            from rich.prompt import Confirm
            if not Confirm.ask(f"Re-run this scenario {runs}× live?", default=True):
                console.print("[yellow]Aborted.[/]")
                sys.exit(0)

    all_runs = []
    for _ in range(max(1, runs)):
        try:
            r = run_scenario(
                scenario, conv_runner,
                agent_name=spec.agent, judge_config=spec.judge_config,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]Verify error:[/] {e}")
            sys.exit(2)
        all_runs.append([r])

    stabilities = build_scenario_stability(all_runs)
    stab = stabilities[0] if stabilities else None
    n = stab.runs if stab is not None else 1
    klass = TriageClassifier.classify(stab, runs=n)

    st = dict(env.staging or {})
    summary = st.get("failure_summary", "") or "scenario re-verified"
    block = build_staging_block(
        run_id=st.get("run_id", "verify"),
        scenario_id=st.get("scenario_id", ""),
        source=st.get("source", "simulate"),
        classification=klass,
        stability=stab,
        runs_observed=n,
        failure_summary=summary,
    )
    store.update_staging_block(stage_id, block)
    console.print(
        f"[green]re-classified[/] {stage_id} → [cyan]{klass.value}[/] "
        f"({stab.verdict_string if stab else '—'})"
    )
    sys.exit(0)


@stage.command(name="drop")
@click.argument('stage_id', required=False)
@click.option('--config', '-c', default='agentci_spec.yaml', show_default=True)
@click.option('--staged-dir', default=None, type=click.Path())
@click.option('--agent', default=None, help='Scope --held/--all to one agent')
@click.option('--held', is_flag=True, help='Drop everything classified held/held-infra')
@click.option('--all', 'drop_all', is_flag=True, help='Drop the whole staging area')
@click.option('--yes', '-y', is_flag=True, help='Skip the confirmation prompt')
def stage_drop(stage_id, config, staged_dir, agent, held, drop_all, yes):
    """Delete staged conversations (a single id, all held, or everything).

    \b
    Exit codes:
        0 — dropped (or nothing matched)
        2 — bad flag combination / config error
    """
    from .promotion import StageAmbiguous, StageNotFound

    if sum(bool(x) for x in (stage_id, held, drop_all)) != 1:
        console.print("[bold red]Give exactly one of:[/] STAGE_ID, --held, or --all.")
        sys.exit(2)

    _spec, store = _stage_store(config, staged_dir)

    if stage_id:
        try:
            store.drop(stage_id)
            console.print(f"[green]dropped[/] {stage_id}")
        except (StageNotFound, StageAmbiguous) as e:
            console.print(f"[yellow]{e}[/]")
        sys.exit(0)

    targets = store.list(agent=agent)
    if held:
        targets = [e for e in targets if e.classification in ("held", "held-infra")]
    if not targets:
        console.print("[yellow]Nothing matched.[/]")
        sys.exit(0)
    if not yes:
        from rich.prompt import Confirm
        if not Confirm.ask(f"Drop {len(targets)} staged conversation(s)?", default=False):
            console.print("[yellow]Aborted.[/]")
            sys.exit(0)
    for e in targets:
        store.drop(e.stage_id)
    console.print(f"[green]dropped {len(targets)} staged conversation(s).[/]")
    sys.exit(0)


@stage.command(name="gc")
@click.option('--config', '-c', default='agentci_spec.yaml', show_default=True)
@click.option('--staged-dir', default=None, type=click.Path())
def stage_gc(config, staged_dir):
    """Run retention GC (age + global caps) across the staging area.

    \b
    Exit codes:
        0 — GC ran
        2 — config error
    """
    _spec, store = _stage_store(config, staged_dir)
    removed = store.gc()
    console.print(f"[green]gc complete[/] — removed {removed} file(s).")
    sys.exit(0)


@cli.command(name="promote")
@click.argument('stage_id', required=False)
@click.option('--config', '-c', default='agentci_spec.yaml', show_default=True)
@click.option('--staged-dir', default=None, type=click.Path())
@click.option('--baseline-dir', default=None,
              help='Override where the golden is written (default: spec baseline_dir)')
@click.option('--force', is_flag=True,
              help='Promote a held/held-infra/unverified entry anyway (prints why '
                   'it was gated).')
@click.option('--yes', '-y', is_flag=True, help='Skip the interactive picker/confirm')
@click.option('--format', 'fmt', type=click.Choice(['console', 'json']),
              default='console', show_default=True)
def promote_cmd(stage_id, config, staged_dir, baseline_dir, force, yes, fmt):
    """Promote a staged failing conversation to a permanent golden CI gate.

    The human "yes" — and only that. `ciagent promote` with no id opens an
    interactive picker sorted best-first; `promote <id>` promotes one. The
    envelope moves into <baseline_dir>/<agent>/scenarios/ (exactly where
    --record writes) with an additive `provenance:` block; `simulate --replay`
    then gates on it unchanged (the `gate` lifecycle: replay exits 1 while the
    bug reproduces).
    \b
    Exit codes:
        0 — promoted
        1 — refused: classification gated and --force not given, or STAGE_ID
            not found
        2 — config error
    """
    _route_chrome(fmt)
    import json as _json

    from .promotion import (
        PromotionRefused,
        PromotionService,
        StageAmbiguous,
        StageNotFound,
    )

    spec, store = _stage_store(config, staged_dir)
    effective_baseline = baseline_dir or spec.baseline_dir

    if not stage_id:
        entries = store.list()
        if not entries:
            console.print("[yellow]Nothing staged to promote.[/]")
            sys.exit(0)
        console.print("[bold]Staged conversations (best to promote first):[/]")
        for e in entries:
            console.print(f"  [cyan]{e.stage_id}[/]  [{e.classification}]  "
                          f"{(e.staging.get('failure_summary') or '')[:50]}")
        if not yes:
            from rich.prompt import Prompt
            stage_id = Prompt.ask("Promote which id? (blank to cancel)", default="")
        if not stage_id:
            console.print("[yellow]Cancelled.[/]")
            sys.exit(0)

    svc = PromotionService(store)
    try:
        _, env_preview = store.load(stage_id)
        _warn_redacted_check_literals(env_preview)
    except Exception:  # noqa: BLE001 — warning only; promote() re-resolves
        pass
    try:
        out = svc.promote(stage_id, baseline_dir=effective_baseline, force=force)
    except (StageNotFound, StageAmbiguous) as e:
        console.print(f"[bold red]{e}[/]")
        sys.exit(1)
    except PromotionRefused as e:
        console.print(f"[bold red]Refused:[/] {e}")
        sys.exit(1)

    if fmt == "json":
        print(_json.dumps({"promoted": str(out), "stage_id": stage_id}, indent=2))
    else:
        console.print(f"[green]✅ promoted[/] {stage_id} → [cyan]{out}[/]")
        console.print("[dim]`ciagent simulate --replay` now gates on it (exit 1 while the bug reproduces).[/]")
    sys.exit(0)


if __name__ == '__main__':
    cli()
