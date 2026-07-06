# Continuous Integration with GitHub Actions

AgentCI is designed to run in your CI/CD pipeline to catch regressions before
they hit production.

## Quick Setup

1. Generate the workflow (plus a pre-push hook) in your repository:

   ```bash
   agentci init
   ```

   This writes `.github/workflows/agentci.yml`, tailored to your project's
   dependency file and Python version.

2. Set your API keys in GitHub → Settings → Security → Secrets and variables →
   Actions:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`

3. The workflow now runs on every push and pull request.

## What the Workflow Does

```yaml
- name: Validate spec
  run: agentci validate agentci_spec.yaml

- name: Run AgentCI evaluation
  run: agentci test --config agentci_spec.yaml --format github --workers 4
```

- `--format github` emits GitHub annotations, so failures appear inline in the
  PR's "Files changed" tab.
- `agentci test` exits 1 on any correctness failure, failing the build. Path
  and cost exceedances surface as warning annotations without blocking.

## Zero-Cost PR Gating

If you don't want live LLM calls on every PR, run mock mode instead — no API
keys required:

```yaml
- name: Run AgentCI evaluation (mock)
  run: agentci test --mock --yes --format github
```

## Artifacts

The generated workflow also exports results as JSON and uploads them as a
build artifact, so you can download a full report from the Action run summary
or render it later:

```bash
agentci report -i agentci-eval-results.json -o report.html
```
