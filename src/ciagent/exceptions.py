# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
AgentCI custom exceptions with actionable fix suggestions.

All exceptions include a `fix` attribute that tells the user (or coding agent)
exactly how to resolve the issue. This makes AgentCI errors self-documenting
and agent-friendly.
"""


class AgentCIError(Exception):
    """Base exception for all AgentCI errors.

    Args:
        message: What went wrong.
        fix: How to fix it (included in the error message).

    Example:
        >>> raise AgentCIError(
        ...     "No spans found in trace",
        ...     fix="Register AgentCITraceProcessor: add_trace_processor(AgentCITraceProcessor())"
        ... )
    """

    def __init__(self, message: str, fix: str = ""):
        self.fix = fix
        full_message = f"{message}\n  Fix: {fix}" if fix else message
        super().__init__(full_message)


class TraceError(AgentCIError):
    """Errors related to trace capture and processing.

    Raised when traces are empty, malformed, or missing expected data.
    """
    pass


class ConfigError(AgentCIError):
    """Errors related to AgentCI configuration.

    Raised when agentci.yaml is missing, malformed, or has invalid values.
    """
    pass


class MockError(AgentCIError):
    """Errors related to mock setup and execution.

    Raised when mock tools are missing, mock sequences are exhausted,
    or mock response formats are invalid.
    """
    pass


class ImportError_(AgentCIError):
    """Errors related to agent function imports.

    Raised when the agent function specified in agentci.yaml cannot be imported.
    Named with trailing underscore to avoid shadowing the builtin ImportError.
    """
    pass


class BaselineError(AgentCIError):
    """Errors related to golden trace baselines.

    Raised when baseline files are missing or cannot be loaded.
    """
    pass


class JudgeError(AgentCIError):
    """Errors related to LLM-as-a-judge evaluation.

    Raised when judge API calls fail, responses cannot be parsed,
    or required API keys are missing.
    """
    pass


class SchemaError(AgentCIError):
    """Errors related to agentci_spec.yaml schema validation.

    Raised when spec files fail Pydantic validation or contain
    unsupported field values.
    """
    pass


class EngineError(AgentCIError):
    """Errors raised by the evaluation engine.

    Raised when the correctness, path, or cost evaluation layers
    encounter an unexpected runtime error.
    """
    pass
