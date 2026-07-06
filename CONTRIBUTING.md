# Contributing to Agent CI

Thank you for your interest in contributing to Agent CI!

## Getting Started

1. Fork the repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/AgentCI.git
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .[dev]
   ```

## Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Make your changes and commit them with descriptive messages.
3. Push your branch to your fork:
   ```bash
   git push origin feature/my-feature
   ```
4. Open a Pull Request on GitHub.

## Testing

Run tests with pytest:
```bash
pytest
```

## Code Style

We use `ruff` for linting and formatting. Please ensure your code passes checks before submitting a PR.

## Releasing (maintainers)

Releases are tag-driven — never upload to PyPI by hand:

1. Bump `version` in `pyproject.toml` and move the `[Unreleased]` CHANGELOG section
   under the new version heading (in a normal PR).
2. After merge: `git tag vX.Y.Z && git push --tags`

The release workflow builds the package, refuses to publish if the tag doesn't match
`pyproject.toml`, runs the test suite against the built wheel, publishes to PyPI via
trusted publishing, and creates the GitHub release with the built artifacts attached.
