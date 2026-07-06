"""
Unit tests for ciagent.engine.metrics — pure metric functions.
All tests are deterministic and require no mocks or I/O.
"""

import pytest

from ciagent.engine.metrics import (
    compute_edit_distance_similarity,
    compute_sequence_lcs,
    compute_tool_f1,
    compute_tool_precision,
    compute_tool_recall,
    detect_loops,
)


# ── tool_recall ───────────────────────────────────────────────────────────────


class TestToolRecall:
    def test_full_recall(self):
        assert compute_tool_recall({"a", "b"}, {"a", "b"}) == 1.0

    def test_partial_recall(self):
        assert compute_tool_recall({"a", "b"}, {"a"}) == pytest.approx(0.5)

    def test_zero_recall(self):
        assert compute_tool_recall({"a", "b"}, {"c"}) == 0.0

    def test_empty_expected_returns_one(self):
        assert compute_tool_recall(set(), {"a", "b"}) == 1.0

    def test_empty_used_returns_zero(self):
        assert compute_tool_recall({"a"}, set()) == 0.0

    def test_superset_used(self):
        # extras in used don't affect recall
        assert compute_tool_recall({"a"}, {"a", "b", "c"}) == 1.0

    def test_single_element_match(self):
        assert compute_tool_recall({"x"}, {"x"}) == 1.0

    def test_single_element_no_match(self):
        assert compute_tool_recall({"x"}, {"y"}) == 0.0


# ── tool_precision ────────────────────────────────────────────────────────────


class TestToolPrecision:
    def test_full_precision(self):
        assert compute_tool_precision({"a", "b"}, {"a", "b"}) == 1.0

    def test_partial_precision(self):
        # used = {a, b, c}, expected = {a, b} → 2/3
        assert compute_tool_precision({"a", "b"}, {"a", "b", "c"}) == pytest.approx(2 / 3)

    def test_zero_precision(self):
        assert compute_tool_precision({"a"}, {"b", "c"}) == 0.0

    def test_empty_used_empty_expected(self):
        assert compute_tool_precision(set(), set()) == 1.0

    def test_empty_used_nonempty_expected(self):
        assert compute_tool_precision({"a"}, set()) == 0.0

    def test_subset_used(self):
        # subset of expected used → precision = 1.0
        assert compute_tool_precision({"a", "b", "c"}, {"a"}) == 1.0


# ── tool_f1 ───────────────────────────────────────────────────────────────────


class TestToolF1:
    def test_perfect_f1(self):
        assert compute_tool_f1({"a", "b"}, {"a", "b"}) == 1.0

    def test_zero_f1_disjoint(self):
        assert compute_tool_f1({"a"}, {"b"}) == 0.0

    def test_both_empty_returns_one(self):
        # Nothing expected, nothing used → perfect match
        assert compute_tool_f1(set(), set()) == 1.0

    def test_partial_f1(self):
        # expected={a,b}, used={a,c} → P=0.5, R=0.5, F1=0.5
        f1 = compute_tool_f1({"a", "b"}, {"a", "c"})
        assert f1 == pytest.approx(0.5)


# ── sequence_lcs ─────────────────────────────────────────────────────────────


class TestSequenceLCS:
    def test_identical_sequences(self):
        assert compute_sequence_lcs(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_both_empty(self):
        assert compute_sequence_lcs([], []) == 1.0

    def test_one_empty(self):
        assert compute_sequence_lcs(["a"], []) == 0.0
        assert compute_sequence_lcs([], ["a"]) == 0.0

    def test_disjoint(self):
        assert compute_sequence_lcs(["a", "b"], ["c", "d"]) == 0.0

    def test_reversed_partial(self):
        # LCS of [a,b] and [b,a] = 1 element → 2*1/(2+2) = 0.5
        assert compute_sequence_lcs(["a", "b"], ["b", "a"]) == pytest.approx(0.5)

    def test_subsequence(self):
        # LCS([a,b,c], [a,c]) = 2 → 2*2/(3+2) = 0.8
        assert compute_sequence_lcs(["a", "b", "c"], ["a", "c"]) == pytest.approx(0.8)

    def test_single_element_match(self):
        assert compute_sequence_lcs(["x"], ["x"]) == 1.0

    def test_single_element_no_match(self):
        assert compute_sequence_lcs(["x"], ["y"]) == 0.0

    def test_long_common_prefix(self):
        a = ["a", "b", "c", "d"]
        b = ["a", "b", "c", "e"]
        # LCS = 3 → 2*3/(4+4) = 0.75
        assert compute_sequence_lcs(a, b) == pytest.approx(0.75)


# ── edit_distance_similarity ──────────────────────────────────────────────────


class TestEditDistanceSimilarity:
    def test_identical(self):
        assert compute_edit_distance_similarity(["a", "b"], ["a", "b"]) == 1.0

    def test_both_empty(self):
        assert compute_edit_distance_similarity([], []) == 1.0

    def test_one_empty(self):
        assert compute_edit_distance_similarity(["a"], []) == 0.0
        assert compute_edit_distance_similarity([], ["a"]) == 0.0

    def test_one_substitution(self):
        # ED([a,b],[a,c]) = 1 → 1 - 1/2 = 0.5
        assert compute_edit_distance_similarity(["a", "b"], ["a", "c"]) == pytest.approx(0.5)

    def test_completely_different(self):
        # ED([a,b],[c,d]) = 2 → 1 - 2/2 = 0.0
        assert compute_edit_distance_similarity(["a", "b"], ["c", "d"]) == 0.0

    def test_insertion(self):
        # ED([a,b],[a,b,c]) = 1 → 1 - 1/3 = 0.666...
        result = compute_edit_distance_similarity(["a", "b"], ["a", "b", "c"])
        assert result == pytest.approx(1 - 1 / 3)

    def test_single_match(self):
        assert compute_edit_distance_similarity(["x"], ["x"]) == 1.0


# ── detect_loops ──────────────────────────────────────────────────────────────


class TestDetectLoops:
    def test_no_loops(self):
        assert detect_loops(["a", "b", "c"]) == 0

    def test_empty_sequence(self):
        assert detect_loops([]) == 0

    def test_single_element(self):
        assert detect_loops(["a"]) == 0

    def test_all_same(self):
        # [a, a, a] → 2 consecutive repeats
        assert detect_loops(["a", "a", "a"]) == 2

    def test_alternating_no_loops(self):
        assert detect_loops(["a", "b", "a", "b"]) == 0

    def test_one_consecutive_pair(self):
        assert detect_loops(["a", "a", "b"]) == 1

    def test_two_separate_pairs(self):
        assert detect_loops(["a", "a", "b", "b"]) == 2

    def test_triple_consecutive(self):
        assert detect_loops(["x", "x", "x"]) == 2

    def test_mixed(self):
        # [a, a, b, b, b, c] → a-a (1), b-b (1), b-b (1) = 3
        assert detect_loops(["a", "a", "b", "b", "b", "c"]) == 3
