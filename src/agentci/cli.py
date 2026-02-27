"""
Agent CI Command Line Interface.

Commands:
  agentci init          Scaffold a new test suite
  agentci run           Execute test suite
  agentci run --runs N  Statistical mode (run N times)
  agentci record        Run agent live, save golden trace
  agentci diff          Compare latest run against golden
  agentci report        Generate HTML report from last run
"""

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

@click.group()
@click.version_option()
def cli():
    """Agent CI — Continuous Integration for AI Agents"""
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


@cli.command()
@click.option('--hook', is_flag=True, help='Also install a .git/hooks/pre-push script')
@click.option('--force', is_flag=True, help='Overwrite existing files')
def init(hook, force):
    """Scaffold a new AgentCI test suite and CI/CD pipeline."""
    import jinja2
    
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
    
    # If templates aren't packaged (e.g. during dev), fallback to basic string replacement
    # We'll use a simple manual replacement if jinja2 fails to load the file
    github_action_dest = Path(".github/workflows/agentci.yml")
    pre_push_dest = Path(".git/hooks/pre-push")
    
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
            content = f"# Scaffolded by AgentCI\n# Test Path: {test_path}\n# Deps: {dependency_file}\n"
            
        with open(github_action_dest, "w") as f:
            f.write(content)
        console.print(f"[green]✓ Created[/] {github_action_dest}")
        
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

    console.print("\n[bold green]AgentCI Initialization Complete! 🚀[/]")
    console.print("\n[bold]Next Steps:[/]")
    console.print("1. Commit the newly generated files: [cyan]git add .github/[/]")
    console.print("2. Add [cyan]ANTHROPIC_API_KEY[/] to your GitHub repository secrets.")
    console.print(f"3. Push your code to see the CI run: [cyan]git push[/]")



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
    """Execute the test suite."""
    if not output_json:
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
        if Confirm.ask(f"Save golden trace to [yellow]{save_path}[/]?", default=True):
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
        agentci diff --baseline v1-broken --compare v2-fixed --agent rag-agent

    Exit codes:
        0  No regressions detected
        1  Correctness regression (pass → fail)
        2  Error loading baselines
    """
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
            title = f"AgentCI Diff: {report.agent} ({report.from_version} → {report.to_version})"
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
@click.option('--input', '-i', type=click.Path(exists=True), required=True)
@click.option('--output', '-o', type=click.Path(), required=True)
def report(input, output):
    """Generate an HTML report from a JSON results file."""
    pass


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
        console.print(
            f"[green]✅ Valid:[/] {len(spec.queries)} queries, agent='{spec.agent}'"
        )
        sys.exit(0)
    except (ConfigError, ValidationError) as e:
        console.print(f"[bold red]❌ Validation failed:[/]\n{e}", err=True)
        sys.exit(1)


@cli.command(name="test")
@click.option('--config', '-c', default='agentci_spec.yaml',
              help='Path to agentci_spec.yaml', show_default=True)
@click.option('--tags', '-t', multiple=True, help='Only evaluate queries with these tags')
@click.option('--format', 'fmt',
              type=click.Choice(['console', 'github', 'json', 'prometheus']),
              default='console', show_default=True, help='Output format')
@click.option('--baseline-dir', default=None,
              help='Override baseline directory from spec')
@click.option('--workers', '-w', default=4, show_default=True, type=int,
              help='Max parallel workers for query execution')
@click.option('--sample-ensemble', default=None, type=float,
              help='Fraction of queries to use ensemble judging (0.0-1.0, e.g. 0.2)')
def test_cmd(config, tags, fmt, baseline_dir, workers, sample_ensemble):
    """Run AgentCI v2 evaluation against a spec file.

    Loads agentci_spec.yaml, runs the agent for each query (capturing traces),
    evaluates all three layers (Correctness / Path / Cost), and reports results.

    Requires the spec to declare a runner:

    \b
        runner: "myagent.run:run_agent"

    The runner function must accept (query: str) and return an agentci.models.Trace.
    Without a runner, agentci test prints instructions for API usage.

    \b
    Exit codes:
        0 — all correctness checks pass (warnings emitted as annotations)
        1 — one or more correctness failures
        2 — runtime / infrastructure error
    """
    import json as _json
    from pathlib import Path

    from .loader import load_spec, filter_by_tags
    from .engine.reporter import report_results
    from .engine.parallel import run_spec_parallel, resolve_runner
    from .engine.runner import evaluate_spec
    from .exceptions import ConfigError

    try:
        spec = load_spec(config)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}", err=True)
        sys.exit(2)

    if tags:
        spec = filter_by_tags(spec, list(tags))
        if not spec.queries:
            console.print(f"[yellow]No queries match tags: {tags}[/]")
            sys.exit(0)

    effective_baseline_dir = baseline_dir or spec.baseline_dir

    # ── Check for runner ──────────────────────────────────────────────────────
    if not spec.runner:
        console.print(
            f"[bold blue]AgentCI v2[/] spec has [cyan]{len(spec.queries)}[/] "
            f"queries for agent '[cyan]{spec.agent}[/]'\n"
        )
        console.print(
            "[yellow]ℹ[/] No [bold]runner[/] declared in spec. Add one to run live:\n\n"
            "  [cyan]runner: \"myagent.run:run_agent\"[/]\n\n"
            "The function must accept [bold](query: str) → Trace[/].\n\n"
            "Or use the Python API in your test suite:\n"
            "  [cyan]from agentci import load_spec, run_spec[/]\n"
            "  [cyan]results = run_spec(spec, my_agent_fn)[/]"
        )
        sys.exit(0)

    # ── Resolve runner ────────────────────────────────────────────────────────
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

    # ── Run queries in parallel ───────────────────────────────────────────────
    console.print(
        f"[bold blue]AgentCI v2[/] │ agent: [cyan]{spec.agent}[/] │ "
        f"queries: [cyan]{len(spec.queries)}[/] │ workers: [cyan]{workers}[/]"
    )
    if fmt in ("console", "github"):
        console.print("")

    try:
        traces = run_spec_parallel(spec, runner_fn, max_workers=workers)
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Infrastructure error:[/] {e}")
        sys.exit(2)

    if not traces:
        console.print("[bold red]Error:[/] No traces captured — runner may have failed for all queries.")
        sys.exit(1)

    # ── Load baselines (optional) ─────────────────────────────────────────────
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

    # ── Evaluate ──────────────────────────────────────────────────────────────
    try:
        results = evaluate_spec(spec, traces, baselines or None)
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Evaluation error:[/] {e}", err=True)
        sys.exit(2)

    # ── Report + exit ─────────────────────────────────────────────────────────
    exit_code = report_results(results, format=fmt, spec_file=config)
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
        console.print(f"[bold red]Config error:[/] {e}", err=True)
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


if __name__ == '__main__':
    cli()
