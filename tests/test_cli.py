"""
Tests for the CLI.
"""
import json
import os
from pathlib import Path

from click.testing import CliRunner
from agentci.cli import (
    cli,
    _scan_project,
    _detect_agent_type,
    _detect_agent_type_from_code,
    _detect_tools_from_code,
    _detect_kb_dir,
    _load_golden_pairs,
    _generate_smoke_queries,
    _generate_full_queries,
    _prompt_for_queries_interactive,
    _generate_skeleton_spec,
    _build_next_steps,
    _calibrate_spec_from_traces,
)

def test_init_command_interactive(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ['init'], input="my.custom.runner:run\n")
        assert result.exit_code == 0
        assert "AgentCI Setup" in result.output
        assert "What is the import path" in result.output
        assert os.path.exists("agentci_spec.yaml")
        
        with open("agentci_spec.yaml") as f:
            content = f.read()
            assert "runner: \"my.custom.runner:run\"" in content
            assert "queries:" in content
            assert "How do I reset my password?" not in content  # default spec is empty

def test_init_command_example_flag(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ['init', '--example'], input="\n") # use default default="myagent.run:run_agent"
        assert result.exit_code == 0
        assert os.path.exists("agentci_spec.yaml")
        
        with open("agentci_spec.yaml") as f:
            content = f.read()
            assert "runner: \"myagent.run:run_agent\"" in content
            assert "How do I reset my password?" in content

def test_bootstrap_command_interactive(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create a dummy runner
        with open("dummy.py", "w") as f:
            f.write("""
from agentci.models import Trace, Span, SpanKind, ToolCall
def run(query: str) -> Trace:
    trace = Trace(test_name="test", agent_name="dummy")
    span = Span(kind=SpanKind.AGENT, name="dummy")
    span.tool_calls.append(ToolCall(tool_name="dummy_tool", arguments={}))
    trace.spans.append(span)
    trace.compute_metrics()
    return trace
""")
        
        # We need to answer: Query 1, Query 2 (empty), Accept trace (y)
        # Sequence: "Hello\n\ny\n"
        result = runner.invoke(cli, ['bootstrap', '--runner', 'dummy:run'], input="Hello\n\ny\n")
        assert result.exit_code == 0
        assert os.path.exists("agentci_spec.yaml")
        assert os.path.exists("baselines/my-agent/v1-hello.json")
        
        with open("agentci_spec.yaml") as f:
            content = f.read()
            assert "query: Hello" in content
            assert "max_tool_calls: 2" in content
            assert "expected_tools:" in content
            assert "- dummy_tool" in content

def test_eval_command(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with open("agentci_spec.yaml", "w") as f:
            f.write("""
version: 1.0
agent: test-agent
runner: dummy:run
queries:
  - query: "test query"
""")
        with open("dummy.py", "w") as f:
            f.write("""
from agentci.models import Trace
def run(query: str) -> Trace:
    return Trace(test_name="test", agent_name="dummy")
""")
        result = runner.invoke(cli, ['eval'])
        assert result.exit_code == 0
        assert "AgentCI v" in result.output
        assert "Eval" in result.output


# ── Tests for guided init helpers ─────────────────────────────────────────────


class TestDetectAgentType:
    """Tests for _detect_agent_type()."""

    def test_rag_keywords(self):
        assert _detect_agent_type("Answers questions using a knowledge base") == "rag"
        assert _detect_agent_type("RAG-based document retrieval agent") == "rag"
        assert _detect_agent_type("FAQ support bot") == "rag"

    def test_tool_keywords(self):
        assert _detect_agent_type("Books flights and manages reservations via API") == "tool"
        assert _detect_agent_type("Search and booking tool agent") == "tool"

    def test_conversational_fallback(self):
        assert _detect_agent_type("Chats with users about their day") == "conversational"
        assert _detect_agent_type("General assistant") == "conversational"

    def test_case_insensitive(self):
        assert _detect_agent_type("KNOWLEDGE BASE retrieval") == "rag"
        assert _detect_agent_type("API TOOL caller") == "tool"


class TestDetectToolsFromCode:
    """Tests for _detect_tools_from_code()."""

    def test_detects_tool_decorator(self, tmp_path):
        (tmp_path / "agent.py").write_text(
            "@tool\ndef search_flights(query: str):\n    pass\n\n"
            "@tool()\ndef book_ticket(flight_id: str):\n    pass\n"
        )
        tools = _detect_tools_from_code(tmp_path)
        assert "search_flights" in tools
        assert "book_ticket" in tools

    def test_detects_bind_tools(self, tmp_path):
        (tmp_path / "agent.py").write_text(
            "llm.bind_tools([search_docs, grade_answer])\n"
        )
        tools = _detect_tools_from_code(tmp_path)
        assert "search_docs" in tools
        assert "grade_answer" in tools

    def test_skips_venv_and_tests(self, tmp_path):
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "lib.py").write_text("@tool\ndef hidden(): pass\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_agent.py").write_text("@tool\ndef test_tool(): pass\n")
        tools = _detect_tools_from_code(tmp_path)
        assert tools == []

    def test_deduplicates(self, tmp_path):
        (tmp_path / "a.py").write_text("@tool\ndef search(): pass\n")
        (tmp_path / "b.py").write_text("@tool\ndef search(): pass\n")
        tools = _detect_tools_from_code(tmp_path)
        assert tools.count("search") == 1


class TestDetectKbDir:
    """Tests for _detect_kb_dir()."""

    def test_finds_knowledge_base(self, tmp_path):
        kb = tmp_path / "knowledge_base"
        kb.mkdir()
        (kb / "doc.md").write_text("# Hello")
        assert _detect_kb_dir(tmp_path) is not None

    def test_finds_docs(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "readme.txt").write_text("readme")
        assert _detect_kb_dir(tmp_path) is not None

    def test_returns_none_when_empty(self, tmp_path):
        assert _detect_kb_dir(tmp_path) is None

    def test_ignores_dirs_without_text_files(self, tmp_path):
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "image.png").write_bytes(b"\x89PNG")
        assert _detect_kb_dir(tmp_path) is None


class TestLoadGoldenPairs:
    """Tests for _load_golden_pairs()."""

    def test_loads_json(self, tmp_path):
        data = [
            {"question": "What is X?", "answer": "X is Y"},
            {"question": "How to Z?", "answer": "Do A then B"},
        ]
        path = tmp_path / "golden.json"
        path.write_text(json.dumps(data))
        pairs = _load_golden_pairs(str(path))
        assert len(pairs) == 2
        assert pairs[0]["question"] == "What is X?"

    def test_loads_csv(self, tmp_path):
        path = tmp_path / "golden.csv"
        path.write_text("question,answer\nWhat is X?,X is Y\nHow to Z?,Do A then B\n")
        pairs = _load_golden_pairs(str(path))
        assert len(pairs) == 2
        assert pairs[1]["answer"] == "Do A then B"

    def test_returns_empty_for_missing_file(self):
        assert _load_golden_pairs("/nonexistent/file.json") == []

    def test_filters_invalid_entries(self, tmp_path):
        data = [
            {"question": "Valid?", "answer": "Yes"},
            {"foo": "bar"},  # missing question/answer
        ]
        path = tmp_path / "golden.json"
        path.write_text(json.dumps(data))
        pairs = _load_golden_pairs(str(path))
        assert len(pairs) == 1


class TestScanProjectDeepKB:
    """Tests for deeper KB sampling in _scan_project()."""

    def test_reads_full_small_files(self, tmp_path):
        kb = tmp_path / "knowledge_base"
        kb.mkdir()
        content = "# Pricing\n\nBusiness plan: $199/month\nEnterprise: $499/month"
        (kb / "pricing.md").write_text(content)
        result = _scan_project(tmp_path)
        # Full content should be preserved (under 2000 chars)
        assert result["knowledge_base"][0]["snippet"] == content

    def test_truncates_large_files(self, tmp_path):
        kb = tmp_path / "knowledge_base"
        kb.mkdir()
        content = "A" * 3000
        (kb / "big.md").write_text(content)
        result = _scan_project(tmp_path)
        snippet = result["knowledge_base"][0]["snippet"]
        # Should be truncated: first 1000 + "\n...\n" + last 500
        assert len(snippet) < 3000
        assert "..." in snippet

    def test_kb_override(self, tmp_path):
        # Default KB dir exists but override points elsewhere
        default_kb = tmp_path / "knowledge_base"
        default_kb.mkdir()
        (default_kb / "default.md").write_text("default content")

        custom_kb = tmp_path / "my_docs"
        custom_kb.mkdir()
        (custom_kb / "custom.md").write_text("custom content")

        result = _scan_project(tmp_path, kb_override=str(custom_kb))
        paths = [f["path"] for f in result["knowledge_base"]]
        assert any("custom.md" in p for p in paths)
        assert not any("default.md" in p for p in paths)

    def test_sorts_by_size_ascending(self, tmp_path):
        kb = tmp_path / "knowledge_base"
        kb.mkdir()
        (kb / "big.md").write_text("B" * 500)
        (kb / "small.md").write_text("S" * 50)
        result = _scan_project(tmp_path)
        # Small file should come first
        assert "small.md" in result["knowledge_base"][0]["path"]


class TestMockTestCommand:
    """Tests for agentci test --mock."""

    def test_mock_mode_no_runner_needed(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with open("agentci_spec.yaml", "w") as f:
                f.write(
                    "version: 1.0\nagent: test-agent\n"
                    "queries:\n  - query: hello\n  - query: goodbye\n"
                )
            result = runner.invoke(cli, ["test", "--mock"])
            assert "mock" in result.output.lower()
            assert result.exit_code in (0, 1)  # should run, not crash

    def test_mock_mode_with_expected_tools(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with open("agentci_spec.yaml", "w") as f:
                f.write(
                    "version: 1.0\nagent: test-agent\n"
                    "queries:\n"
                    "  - query: test\n"
                    "    path:\n"
                    "      expected_tools: [search_docs]\n"
                )
            result = runner.invoke(cli, ["test", "--mock"])
            assert result.exit_code in (0, 1)


# ── Tests for v0.4.1 improvements ───────────────────────────────────────────


class TestDetectAgentTypeFromCode:
    """Tests for _detect_agent_type_from_code()."""

    def test_rag_from_kb_dir(self, tmp_path):
        kb = tmp_path / "knowledge_base"
        kb.mkdir()
        (kb / "doc.md").write_text("# Hello")
        context = _scan_project(tmp_path)
        detected_kb = _detect_kb_dir(tmp_path)
        tools = _detect_tools_from_code(tmp_path)

        result = _detect_agent_type_from_code(context, tools, detected_kb)
        assert result == "rag"

    def test_rag_from_retrieval_keywords(self, tmp_path):
        (tmp_path / "agent.py").write_text(
            "from langchain import ChatOpenAI\n"
            "def retrieve_docs(query):\n    pass\n"
        )
        context = _scan_project(tmp_path)
        tools = _detect_tools_from_code(tmp_path)
        detected_kb = _detect_kb_dir(tmp_path)

        result = _detect_agent_type_from_code(context, tools, detected_kb)
        assert result == "rag"

    def test_tool_from_detected_tools(self, tmp_path):
        (tmp_path / "agent.py").write_text(
            "from langchain import ChatOpenAI\n"
            "@tool\ndef search_flights(): pass\n"
        )
        context = _scan_project(tmp_path)
        tools = _detect_tools_from_code(tmp_path)
        detected_kb = _detect_kb_dir(tmp_path)

        result = _detect_agent_type_from_code(context, tools, detected_kb)
        assert result == "tool"

    def test_conversational_fallback(self, tmp_path):
        (tmp_path / "agent.py").write_text("def chat(msg): pass\n")
        context = _scan_project(tmp_path)
        tools = _detect_tools_from_code(tmp_path)
        detected_kb = _detect_kb_dir(tmp_path)

        result = _detect_agent_type_from_code(context, tools, detected_kb)
        assert result == "conversational"

    def test_kb_takes_priority_over_tools(self, tmp_path):
        """If both KB and tools are present, agent type should be RAG."""
        kb = tmp_path / "knowledge_base"
        kb.mkdir()
        (kb / "doc.md").write_text("# FAQ")
        (tmp_path / "agent.py").write_text(
            "from langchain import ChatOpenAI\n"
            "@tool\ndef search(): pass\n"
        )
        context = _scan_project(tmp_path)
        tools = _detect_tools_from_code(tmp_path)
        detected_kb = _detect_kb_dir(tmp_path)

        result = _detect_agent_type_from_code(context, tools, detected_kb)
        assert result == "rag"


class TestGenerateSkeletonSpec:
    """Tests for _generate_skeleton_spec()."""

    def test_rag_skeleton(self):
        spec = _generate_skeleton_spec("rag", [], "demo:run")
        assert 'runner: "demo:run"' in spec
        assert "TODO" in spec
        assert "Out-of-scope" in spec

    def test_tool_skeleton_includes_tools(self):
        spec = _generate_skeleton_spec("tool", ["search", "book"], "agent:run")
        assert "search" in spec
        assert "book" in spec
        assert "expected_tools" in spec

    def test_conversational_skeleton(self):
        spec = _generate_skeleton_spec("conversational", [], "chat:run")
        assert "TODO" in spec
        assert 'runner: "chat:run"' in spec

    def test_tool_skeleton_caps_at_five(self):
        tools = [f"tool_{i}" for i in range(10)]
        spec = _generate_skeleton_spec("tool", tools, "agent:run")
        # Should only include first 5 tools
        assert "tool_0" in spec
        assert "tool_4" in spec
        assert "tool_5" not in spec


class TestBuildNextSteps:
    """Tests for _build_next_steps()."""

    def test_mock_with_queries(self):
        steps = _build_next_steps(run_mode="mock", created_workflow=False, has_queries=True)
        combined = "\n".join(steps)
        assert "agentci test --mock" in combined
        assert "git push" not in combined

    def test_mock_without_queries(self):
        steps = _build_next_steps(run_mode="mock", created_workflow=False, has_queries=False)
        combined = "\n".join(steps)
        assert "TODO" in combined or "Fill in" in combined
        assert "agentci test --mock" in combined

    def test_live_with_workflow(self):
        steps = _build_next_steps(run_mode="live", created_workflow=True, has_queries=True)
        combined = "\n".join(steps)
        assert "agentci test" in combined
        assert "git push" in combined
        assert "API key" in combined

    def test_live_without_workflow(self):
        steps = _build_next_steps(run_mode="live", created_workflow=False, has_queries=True)
        combined = "\n".join(steps)
        assert "agentci test" in combined
        assert "git push" not in combined


class TestInitMockZeroKey:
    """Tests for zero-API-key mock mode in init --generate."""

    def test_mock_mode_no_api_key_skeleton(self, tmp_path, monkeypatch):
        """Mock mode with no queries should generate skeleton without API keys."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            (Path("agent.py")).write_text("def chat(msg): pass\n")

            # Input: Q2a (topics handle), Q2b (topics decline), runner, decline interactive queries
            result = runner.invoke(
                cli,
                ["init", "--generate", "--force", "--mode", "mock"],
                input="\n\nmyagent:run\nn\n",
            )
            assert result.exit_code == 0
            assert os.path.exists("agentci_spec.yaml")
            with open("agentci_spec.yaml") as f:
                content = f.read()
                assert "queries:" in content

    def test_mock_mode_with_golden_file(self, tmp_path, monkeypatch):
        """--golden-file should load queries from file in mock mode."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            (Path("agent.py")).write_text("def chat(msg): pass\n")

            golden_data = [
                {"question": "What is X?", "answer": "X is Y"},
                {"question": "How to Z?", "answer": "Z via A"},
            ]
            golden_file = Path("golden.json")
            golden_file.write_text(json.dumps(golden_data))

            # Input: Q2a (topics handle), Q2b (topics decline), runner
            result = runner.invoke(
                cli,
                ["init", "--generate", "--force", "--mode", "mock",
                 "--golden-file", str(golden_file)],
                input="\n\nmyagent:run\n",
            )
            assert result.exit_code == 0
            with open("agentci_spec.yaml") as f:
                content = f.read()
                assert "What is X?" in content
                assert "How to Z?" in content

    def test_live_mode_requires_api_key(self, tmp_path, monkeypatch):
        """Live mode without API keys should fail."""
        # Patch os.environ.get to return None for API keys, even if dotenv loaded them
        _orig_get = os.environ.get
        def _patched_get(key, default=None):
            if key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
                return None
            return _orig_get(key, default)
        monkeypatch.setattr(os.environ, "get", _patched_get)

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            (Path("agent.py")).write_text("def chat(msg): pass\n")

            result = runner.invoke(
                cli,
                ["init", "--generate", "--force", "--mode", "live"],
            )
            assert result.exit_code != 0


class TestInitDeprecatedAgentDescription:
    """Tests for deprecated --agent-description flag."""

    def test_shows_deprecation_warning(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            (Path("agent.py")).write_text("def chat(msg): pass\n")

            # Input: Q2a (topics), Q2b (topics), runner, decline interactive queries
            result = runner.invoke(
                cli,
                ["init", "--generate", "--force", "--mode", "mock",
                 "--agent-description", "RAG bot"],
                input="\n\nmyagent:run\nn\n",
            )
            assert "deprecated" in result.output.lower()


class TestInitContextAwareNextSteps:
    """Tests for context-aware Next Steps output."""

    def test_mock_mode_no_git_push(self, tmp_path):
        """Mock mode should NOT suggest git push."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            (Path("agent.py")).write_text("def chat(msg): pass\n")

            # Input: Q2a (topics), Q2b (topics), runner, decline interactive queries
            result = runner.invoke(
                cli,
                ["init", "--generate", "--force", "--mode", "mock"],
                input="\n\nmyagent:run\nn\n",
            )
            assert result.exit_code == 0
            assert "agentci test --mock" in result.output


# ── Tests for calibration pass (v0.4.2) ──────────────────────────────────────


class TestCalibrateSpecFromTraces:
    """Tests for _calibrate_spec_from_traces()."""

    def _make_trace(self, llm_call_count: int):
        """Create a Trace with the given number of LLM calls."""
        from agentci.models import Trace, Span, LLMCall
        span = Span(
            name="agent",
            llm_calls=[LLMCall(model="test", input_tokens=10, output_tokens=10)
                       for _ in range(llm_call_count)],
        )
        return Trace(spans=[span])

    def test_calibrates_max_llm_calls(self):
        """Calibration should set max_llm_calls based on observed trace."""
        from io import StringIO
        from rich.console import Console

        queries = [
            {"query": "What is X?", "tags": ["in-scope"], "cost": {"max_llm_calls": 3}},
            {"query": "What is Y?", "tags": ["in-scope"], "cost": {"max_llm_calls": 3}},
            {"query": "Hi there!", "tags": ["greeting"], "cost": {"max_llm_calls": 1}},
        ]

        def mock_runner(query):
            return self._make_trace(6)

        console = Console(file=StringIO())
        result = _calibrate_spec_from_traces(queries, mock_runner, console)

        # In-scope: max(3, int(6 * 1.5)) = 9
        assert result[0]["cost"]["max_llm_calls"] == 9
        assert result[1]["cost"]["max_llm_calls"] == 9
        # Greeting: max(2, min(observed)) = max(2, 6) = 6
        assert result[2]["cost"]["max_llm_calls"] == 6

    def test_skips_out_of_scope_for_sampling(self):
        """Out-of-scope queries should not be used as calibration samples."""
        from io import StringIO
        from rich.console import Console

        queries = [
            {"query": "What is the weather?", "tags": ["out-of-scope"], "cost": {}},
            {"query": "What is X?", "tags": ["in-scope"], "cost": {"max_llm_calls": 3}},
        ]

        call_log = []

        def mock_runner(query):
            call_log.append(query)
            return self._make_trace(4)

        console = Console(file=StringIO())
        _calibrate_spec_from_traces(queries, mock_runner, console)

        # Only the in-scope query should have been run
        assert len(call_log) == 1
        assert call_log[0] == "What is X?"

    def test_graceful_failure_returns_unchanged(self):
        """If runner fails, queries should be returned unchanged."""
        from io import StringIO
        from rich.console import Console

        queries = [
            {"query": "What is X?", "tags": ["in-scope"], "cost": {"max_llm_calls": 3}},
        ]

        def failing_runner(query):
            raise RuntimeError("Agent crashed")

        console = Console(file=StringIO())
        result = _calibrate_spec_from_traces(queries, failing_runner, console)

        # Unchanged
        assert result[0]["cost"]["max_llm_calls"] == 3

    def test_empty_queries_returns_unchanged(self):
        """No candidates should return queries unchanged."""
        from io import StringIO
        from rich.console import Console

        queries = [
            {"query": "Hi!", "tags": ["greeting"], "cost": {"max_llm_calls": 1}},
            {"query": "Weather?", "tags": ["out-of-scope"], "cost": {"max_llm_calls": 1}},
        ]

        def mock_runner(query):
            return self._make_trace(2)

        console = Console(file=StringIO())
        result = _calibrate_spec_from_traces(queries, mock_runner, console)

        # No in-scope candidates → unchanged
        assert result[0]["cost"]["max_llm_calls"] == 1
        assert result[1]["cost"]["max_llm_calls"] == 1


# ── Tests for `agentci calibrate` command ────────────────────────────────────


class TestCalibrateCommand:
    """Tests for the `agentci calibrate` CLI command."""

    def _write_spec(self, tmp_path, *, runner="my.agent:run", queries=None):
        """Write a minimal spec file and return its path."""
        import yaml

        if queries is None:
            queries = [
                {"query": "What is X?", "tags": ["in-scope"],
                 "cost": {"max_llm_calls": 3}},
            ]
        spec = {
            "agent": "test-agent",
            "runner": runner,
            "queries": queries,
        }
        spec_path = tmp_path / "agentci_spec.yaml"
        spec_path.write_text(yaml.dump(spec, sort_keys=False))
        return spec_path

    def _make_trace(self, llm_calls=4, tool_calls=2, tokens=500, cost=0.01):
        from agentci.models import Trace, Span, LLMCall, ToolCall as TC
        span = Span(
            name="agent",
            llm_calls=[
                LLMCall(model="test", tokens_in=tokens // llm_calls,
                        tokens_out=tokens // llm_calls)
                for _ in range(llm_calls)
            ],
            tool_calls=[TC(tool_name=f"tool_{i}") for i in range(tool_calls)],
        )
        trace = Trace(spans=[span])
        trace.compute_metrics()
        return trace

    def test_dry_run_does_not_modify_spec(self, tmp_path):
        import yaml
        from unittest.mock import patch

        spec_path = self._write_spec(tmp_path)
        original = spec_path.read_text()

        mock_trace = self._make_trace()

        with patch("agentci.engine.parallel.resolve_runner",
                    return_value=lambda q: mock_trace):
            runner = CliRunner()
            result = runner.invoke(cli, [
                "calibrate", "--spec", str(spec_path), "--dry-run",
            ])

        assert result.exit_code == 0
        assert "Dry-run" in result.output
        assert spec_path.read_text() == original

    def test_updates_spec_with_yes_flag(self, tmp_path):
        import yaml
        from unittest.mock import patch

        spec_path = self._write_spec(tmp_path)
        mock_trace = self._make_trace(llm_calls=6)

        with patch("agentci.engine.parallel.resolve_runner",
                    return_value=lambda q: mock_trace):
            runner = CliRunner()
            result = runner.invoke(cli, [
                "calibrate", "--spec", str(spec_path), "--yes",
            ])

        assert result.exit_code == 0
        assert "Updated" in result.output

        with spec_path.open() as f:
            updated = yaml.safe_load(f)
        # max(10, int(6 * 1.5)) = 10
        assert updated["queries"][0]["cost"]["max_llm_calls"] == 10

    def test_no_runner_exits_with_error(self, tmp_path):
        spec_path = self._write_spec(tmp_path, runner=None)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "calibrate", "--spec", str(spec_path),
        ])

        assert result.exit_code != 0
        assert "No runner" in result.output

    def test_shows_calibration_table(self, tmp_path):
        from unittest.mock import patch

        spec_path = self._write_spec(tmp_path)
        mock_trace = self._make_trace()

        with patch("agentci.engine.parallel.resolve_runner",
                    return_value=lambda q: mock_trace):
            runner = CliRunner()
            result = runner.invoke(cli, [
                "calibrate", "--spec", str(spec_path), "--dry-run",
            ])

        assert result.exit_code == 0
        assert "LLM Calls" in result.output
        assert "Tool Calls" in result.output
