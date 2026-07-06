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
