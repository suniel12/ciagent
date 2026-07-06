# Manual Verification of CIAgent in GitHub Actions

Since CIAgent is a CI tool, the ultimate test is running it in a real GitHub Actions environment.

## Prerequisites
- A GitHub account.
- An OpenAI or Anthropic API key.

## Steps

1.  **Create a new empty GitHub repository** (e.g., `ciagent-demo`).

2.  **Initialize a new project locally**:
    ```bash
    mkdir ciagent-demo
    cd ciagent-demo
    git init
    
    # Install ciagent (if not installed globally)
    # pip install ciagent
    # OR if you are developing locally, point to your local version
    
    # Initialize CIAgent project
    ciagent init
    ```

3.  **Add the GitHub Action Workflow**:
    ```bash
    mkdir -p .github/workflows
    # Copy the template we just created
    cp /path/to/ciagent/.github/workflows/agentci-template.yml .github/workflows/agentci.yml
    ```

4.  **Update `agentci.yml` dependency**:
    *Crucial Step for Testing:* Since `ciagent` isn't on PyPI yet (or is an old version), you need to tell the workflow where to install it from.
    
    Edit `.github/workflows/agentci.yml`:
    ```yaml
    - name: Install dependencies
      run: |
        pip install --upgrade pip
        # For testing, install directly from the main branch of your fork/repo
        # pip install git+https://github.com/YOUR_USERNAME/ciagent.git@main
        
        # OR if you just want to test that the workflow runs (it will fail to install ciagent if not on PyPI)
        # You can comment out the install step and just echo "Installing..."
    ```

5.  **Push to GitHub**:
    ```bash
    git add .
    git commit -m "Setup CIAgent demo"
    git remote add origin https://github.com/YOUR_USERNAME/ciagent-demo.git
    git push -u origin main
    ```

6.  **Add Secrets**:
    Go to your repository on GitHub -> **Settings** -> **Secrets and variables** -> **Actions**.
    Add a new repository secret:
    - Name: `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`)
    - Value: `sk-...`

7.  **Check Actions Tab**:
    Go to the **Actions** tab in your repository. You should see the "CIAgent Tests" workflow running.

## What Success Looks Like
- The workflow should turn **Green (Success)**.
- Click on the run, then "CIAgent Tests" job.
- Expand "Run CIAgent tests". You should see the output of `ciagent run`.
- Check **Artifacts** at the bottom of the summary page. There should be a `ciagent-report` zip file containing the HTML report.
