# AgentCI Developer Workflow & UX Improvements

## Current Status: Building & Testing an Agent with AgentCI

Currently, when a developer decides to use AgentCI to test their AI agent, they must navigate a multi-step manual process. The framework provides the execution engine, but the scaffolding and configuration burden rests heavily on the user.

Here is the step-by-step reality of the current developer experience:

### 1. Installation & Initialization
1. The developer installs the wheel package `pip install agentci`.
2. They run `agentci init` in their terminal.
3. **What happens:** AgentCI detects the dependency file (e.g., `reqs.txt` or `pyproject.toml`) and directory structure. It creates a `.github/workflows/agentci.yml` file for CI/CD and an optional git pre-push hook. It then prints next steps to the console.
4. **The Gap:** No actual AgentCI test configuration files are created in the project. The developer is left with CI infrastructure but no local tests.

### 2. Manual Test Configuration
5. The developer must manually create an `agentci_spec.yaml` file from scratch.
6. They must crack open the AgentCI documentation to understand the schema: `agent:`, `tests/queries:`, `correctness:` assertions, etc.
7. They manually write their test cases (queries like "How do I install X?") and formulate JSONPath or string-matching assertions by guessing what the agent's internal trace state might look like.
8. They must correctly wire up the `runner:` field to their Python agent's entrypoint function.

### 3. Golden Baseline Generation
9. The developer runs a command like `agentci record my_test_name` (or uses the v2 CLI `save` commands).
10. They wait while the agent executes the query live against the LLM provider.
11. If the result looks correct, they confirm saving it. This saves a massive JSON `Trace` file under `baselines/<agent>/<version>.json`.
12. This baseline becomes the ground truth that future tests will be diffed against.

### 4. Continuous Testing
13. Finally, the developer runs `agentci test`.
14. The engine executes the queries, captures new traces, compares them against the baselines (checking for path changes, token cost regressions, and correctness rules), and prints a console report.

---

## Room for Improvements (UX/DX Proposals)

After reviewing the codebase (specifically `cli.py`, `loader.py`, and the runner logic), it is clear that we can build abstractions that dramatically lower the barrier to entry and reduce developer friction. Here are the highest-impact proposals:

### 1. Auto-Generate `agentci_spec.yaml`
**The Friction:** `agentci init` leaves the repo without the core spec file.
**The Solution:** 
* When a user runs `agentci init`, auto-generate a generic, heavily-commented `agentci_spec.yaml` file in the project directory. 
* *Interactive Mode:* Prompt the user during `init`: *"What is the import path for your agent runner function? (e.g. `src.my_agent:run_agent`)"*. Auto-populate the `runner:` field in the YAML so it works out-of-the-box.

### 2. "Zero-to-Golden" Interactive Bootstrapper
**The Friction:** Writing test YAML, guessing assertions, and manually recording baselines is tedious.
**The Solution:** Create an `agentci bootstrap --agent <import_path>` command.
* The CLI asks the user to enter 3-5 example queries interactively in the terminal.
* AgentCI instantiates their agent, runs the queries live, and streams the output.
* If the user approves the output (e.g., presses 'y'), AgentCI **automatically** writes the queries into `agentci_spec.yaml`, saves the baselines to disk, and infers basic `correctness` rules based on the agent's output. 

### 3. "Fix My Spec" AI Assistant
**The Friction:** Writing precise JSONPath or `grade_artifacts` assertions is prone to syntax errors and brittleness.
**The Solution:** Add an `--auto-fix` flag to the testing pipeline.
* If a test fails due to a broken spec (e.g., the developer changed an internal data structure so a span assertion fails), AgentCI uses an LLM to inspect the recorded *Trace* under the hood, compares it to the *Broken Spec*, and automatically suggests the corrected YAML assertion.

### 4. Interactive Trace Inspector in the CLI 
**The Friction:** When a test fails or a regression is caught, the user has to open a massive, dense JSON trace file to debug *why*.
**The Solution:** Enhance `agentci diff` and `agentci test`.
* On failure, offer an interactive prompt: `[Press 't' to view Trace, 'd' to view Diff, 'q' to quit]`.
* Use the `rich` library to render a collapsible, tree-view of the Trace execution graph directly in the terminal, keeping the developer in their flow state instead of switching to a JSON IDE.

### 5. Drop-in PyTest Native Integration
**The Friction:** Asking developers to integrate a standalone `agentci test` command introduces friction for teams already deeply embedded in standard testing ecosystems.
**The Solution:** Build a native pytest plugin (`pytest-agentci`).
* The developer just runs `pytest`. 
* The plugin auto-discovers `agentci_spec.yaml`, dynamically generates parameterised pytest cases for each declarative query, and reports the AgentCI diffs/layers natively inside PyTest's UI and reporters.
