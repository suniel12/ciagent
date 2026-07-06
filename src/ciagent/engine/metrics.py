# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Shared metric computation functions for the AgentCI path evaluation engine.

All functions are pure (no side-effects, no I/O) and operate on plain Python
lists / sets so they are trivially testable.

Mathematical definitions
------------------------
tool_recall    = |E ∩ U| / |E|           (E=expected, U=used)
tool_precision = |E ∩ U| / |U|
tool_f1        = 2·P·R / (P+R)

sequence_lcs (normalised LCS):
    LCS = longest common subsequence of (predicted, reference)
    similarity = 2·|LCS| / (|predicted| + |reference|)

sequence_edit (normalised Levenshtein):
    ED = min edits (insert/delete/substitute) predicted → reference
    similarity = 1 - ED / max(|predicted|, |reference|)

loop_count = number of consecutive identical tool calls
"""

from __future__ import annotations


# ── Set-based Metrics ──────────────────────────────────────────────────────────


def compute_tool_recall(expected: set[str], used: set[str]) -> float:
    """Return |E ∩ U| / |E|.  Returns 1.0 when expected is empty."""
    if not expected:
        return 1.0
    return len(expected & used) / len(expected)


def compute_tool_precision(expected: set[str], used: set[str]) -> float:
    """Return |E ∩ U| / |U|.  Returns 1.0 when both sets are empty."""
    if not used:
        return 1.0 if not expected else 0.0
    return len(expected & used) / len(used)


def compute_tool_f1(expected: set[str], used: set[str]) -> float:
    """Return 2·P·R / (P+R).  Returns 0.0 when both are empty."""
    p = compute_tool_precision(expected, used)
    r = compute_tool_recall(expected, used)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ── Sequence-based Metrics ─────────────────────────────────────────────────────


def compute_sequence_lcs(seq_a: list[str], seq_b: list[str]) -> float:
    """Normalised LCS similarity: 2·|LCS(A,B)| / (|A| + |B|).

    - Both empty → 1.0  (identical)
    - One empty  → 0.0  (completely different)
    - Range: [0, 1]
    """
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0

    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return (2 * dp[m][n]) / (m + n)


def compute_edit_distance_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """Normalised Levenshtein similarity: 1 - ED(A,B) / max(|A|, |B|).

    - Both empty → 1.0
    - One empty  → 0.0
    - Range: [0, 1]
    """
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0

    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return 1.0 - dp[m][n] / max(m, n)


# ── Loop Detection ─────────────────────────────────────────────────────────────


def detect_loops(tool_sequence: list[str]) -> int:
    """Count consecutive repeated tool invocations.

    Example: ['a', 'a', 'b', 'b', 'b'] → 3  (a-a is 1, b-b-b is 2)
    """
    if not tool_sequence:
        return 0
    loops = 0
    for i in range(1, len(tool_sequence)):
        if tool_sequence[i] == tool_sequence[i - 1]:
            loops += 1
    return loops
