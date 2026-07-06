# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Judge Audit Engine — meta-evaluation of the LLM judge.

Re-scores recorded answers (no agent re-runs) three ways:

  Mode 1 — Judge vs. deterministic checks (zero labels required):
      On queries that have BOTH deterministic checks and judge rubrics, run
      both independently and report the disagreement matrix. The row that
      matters: judge PASSED an answer a deterministic fact-check FAILED.

  Mode 2 — Judge retest stability (--repeats K):
      Same answer, same rubric, K times. Verdict flips on identical input
      measure the judge's own noise floor.

  Mode 3 — Judge vs. hand labels (optional):
      Agreement + Cohen's kappa against a user-supplied labels file.

Scoped claim (by design, stated in every report): Mode 1 measures the judge
only on fact-checkable queries. A judge that fails where you CAN check it
should not be trusted where you can't — a one-directional, disqualifying
signal. It is a smoke test, not a guarantee, for judgment-only queries;
for those, label a sample and use Mode 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentci.engine.correctness import evaluate_correctness
from agentci.engine.results import LayerStatus
from agentci.schema.spec_models import AgentCISpec, CorrectnessSpec, GoldenQuery

# Verdict thresholds (documented in docs/judge-audit.md)
FLIP_RATE_UNRELIABLE = 0.20        # >20% of judged queries flip on retest
FLIP_RATE_CALIBRATE = 0.05
FALSE_PASS_RATE_UNRELIABLE = 0.15  # >15% of det-failed answers passed by judge
LABEL_AGREEMENT_UNRELIABLE = 0.75  # standard guidance floor
LABEL_AGREEMENT_CALIBRATE = 0.90
MIN_CHECKABLE_SAMPLE = 5           # below this, Mode 1 rates are anecdotes

JudgeFn = Callable[..., dict[str, Any]]  # signature of judge.run_judge


@dataclass
class QueryAudit:
    """Audit record for a single query's recorded answer."""
    query: str
    answer: str
    det_verdict: Optional[bool] = None     # None = no deterministic checks
    judge_verdicts: list[bool] = field(default_factory=list)  # one per repeat
    judge_rationales: list[str] = field(default_factory=list)
    judge_errors: int = 0                  # judge calls that raised (API down, no key)
    label: Optional[bool] = None           # hand label, if provided

    @property
    def has_judge(self) -> bool:
        return bool(self.judge_verdicts)

    @property
    def judge_verdict(self) -> Optional[bool]:
        """Majority verdict across repeats (ties fail — conservative)."""
        if not self.judge_verdicts:
            return None
        passes = sum(self.judge_verdicts)
        return passes * 2 > len(self.judge_verdicts)

    @property
    def judge_flipped(self) -> bool:
        return len(set(self.judge_verdicts)) > 1

    @property
    def checkable(self) -> bool:
        return self.det_verdict is not None and self.has_judge

    @property
    def false_pass(self) -> bool:
        """The killer row: judge passed what a deterministic check failed."""
        return self.checkable and self.judge_verdict is True and self.det_verdict is False


@dataclass
class JudgeAuditReport:
    """Aggregated judge audit across all judged queries."""
    repeats: int
    queries: list[QueryAudit]
    judgment_only_count: int = 0    # judged queries with no deterministic check

    @property
    def judged(self) -> list[QueryAudit]:
        return [q for q in self.queries if q.has_judge]

    @property
    def total_judge_errors(self) -> int:
        return sum(q.judge_errors for q in self.queries)

    @property
    def all_judge_calls_errored(self) -> bool:
        """Every judge call raised — the judge never actually ran. No verdict
        about the judge can honestly be made from this."""
        total_calls = sum(len(q.judge_verdicts) for q in self.judged)
        return total_calls > 0 and self.total_judge_errors >= total_calls

    @property
    def checkable_queries(self) -> list[QueryAudit]:
        return [q for q in self.queries if q.checkable]

    @property
    def false_passes(self) -> list[QueryAudit]:
        return [q for q in self.checkable_queries if q.false_pass]

    @property
    def false_alarms(self) -> list[QueryAudit]:
        """Judge failed what deterministic checks passed (may be judge noise
        or a genuine quality issue keywords can't see — listed, not counted
        against the judge)."""
        return [
            q for q in self.checkable_queries
            if q.judge_verdict is False and q.det_verdict is True
        ]

    @property
    def agreement_rate(self) -> Optional[float]:
        cq = self.checkable_queries
        if not cq:
            return None
        agree = sum(1 for q in cq if q.judge_verdict == q.det_verdict)
        return round(agree / len(cq), 3)

    @property
    def false_pass_rate(self) -> Optional[float]:
        """Of answers deterministic checks FAILED, how many did the judge pass?"""
        det_failed = [q for q in self.checkable_queries if q.det_verdict is False]
        if not det_failed:
            return None
        return round(sum(1 for q in det_failed if q.judge_verdict) / len(det_failed), 3)

    @property
    def flip_rate(self) -> Optional[float]:
        judged = self.judged
        if not judged or self.repeats < 2:
            return None
        return round(sum(1 for q in judged if q.judge_flipped) / len(judged), 3)

    @property
    def labeled_queries(self) -> list[QueryAudit]:
        return [q for q in self.judged if q.label is not None]

    @property
    def label_agreement(self) -> Optional[float]:
        lq = self.labeled_queries
        if not lq:
            return None
        agree = sum(1 for q in lq if q.judge_verdict == q.label)
        return round(agree / len(lq), 3)

    @property
    def cohens_kappa(self) -> Optional[float]:
        """Cohen's kappa between judge verdicts and hand labels (binary)."""
        lq = self.labeled_queries
        if len(lq) < 2:
            return None
        n = len(lq)
        po = sum(1 for q in lq if q.judge_verdict == q.label) / n
        judge_pass = sum(1 for q in lq if q.judge_verdict) / n
        label_pass = sum(1 for q in lq if q.label) / n
        pe = judge_pass * label_pass + (1 - judge_pass) * (1 - label_pass)
        if pe == 1.0:
            return 1.0 if po == 1.0 else 0.0
        return round((po - pe) / (1 - pe), 3)

    @property
    def low_sample(self) -> bool:
        return len(self.checkable_queries) < MIN_CHECKABLE_SAMPLE

    @property
    def verdict(self) -> str:
        """TRUSTWORTHY / NEEDS CALIBRATION / UNRELIABLE, most severe wins.
        ERROR when the judge never actually ran (no honest verdict possible)."""
        if self.all_judge_calls_errored:
            return "ERROR"
        fp = self.false_pass_rate
        fr = self.flip_rate
        la = self.label_agreement
        kappa = self.cohens_kappa

        if (
            (fp is not None and fp > FALSE_PASS_RATE_UNRELIABLE)
            or (fr is not None and fr > FLIP_RATE_UNRELIABLE)
            or (la is not None and la < LABEL_AGREEMENT_UNRELIABLE)
            or (kappa is not None and kappa < 0.4)
        ):
            return "UNRELIABLE"
        if (
            self.false_passes
            or (fr is not None and fr > FLIP_RATE_CALIBRATE)
            or (la is not None and la < LABEL_AGREEMENT_CALIBRATE)
        ):
            return "NEEDS CALIBRATION"
        return "TRUSTWORTHY"

    @property
    def scope_note(self) -> str:
        """The claim this report is allowed to make — and the one it isn't."""
        n_checkable = len(self.checkable_queries)
        n_judgment = self.judgment_only_count
        note = (
            f"Judge agreement measured on {n_checkable} fact-checkable "
            f"quer{'y' if n_checkable == 1 else 'ies'}. This calibrates the judge "
            f"where ground truth exists — a judge that fails where you CAN check "
            f"it should not be trusted where you can't."
        )
        if n_judgment:
            note += (
                f" It is a smoke test, not a guarantee, for the {n_judgment} "
                f"judgment-only quer{'y' if n_judgment == 1 else 'ies'} it cannot "
                f"cover; to measure those, hand-label a sample (--labels)."
            )
        if self.low_sample:
            note += (
                f" NOTE: fewer than {MIN_CHECKABLE_SAMPLE} checkable queries — "
                f"treat rates as anecdotes, not statistics."
            )
        return note


def run_judge_audit(
    spec: AgentCISpec,
    answers: dict[str, str],
    repeats: int = 3,
    labels: Optional[dict[str, bool]] = None,
    sample: Optional[int] = None,
    judge_fn: Optional[JudgeFn] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> JudgeAuditReport:
    """Audit the spec's LLM judge against recorded answers.

    Args:
        spec:     Loaded AgentCISpec (source of rubrics + deterministic checks).
        answers:  query text → recorded answer (from golden baselines).
        repeats:  Judge calls per rubric per query (Mode 2 retest stability).
        labels:   Optional query text → human pass/fail (Mode 3).
        sample:   Cap on judged queries (cost control); first N in spec order.
        judge_fn: Injectable judge callable (tests); defaults to judge.run_judge.
        progress: Optional callback(str) for per-query progress output.

    Returns:
        JudgeAuditReport.
    """
    if judge_fn is None:
        from agentci.engine.judge import run_judge as judge_fn  # type: ignore[no-redef]

    labels = labels or {}
    audits: list[QueryAudit] = []
    judgment_only = 0
    judged_count = 0

    for gq in spec.queries:
        answer = answers.get(gq.query)
        if answer is None or gq.correctness is None:
            continue
        rubrics = _judge_rubrics(gq.correctness)
        if not rubrics:
            continue  # nothing to audit without a judge
        if sample is not None and judged_count >= sample:
            break
        judged_count += 1

        audit = QueryAudit(query=gq.query, answer=answer, label=labels.get(gq.query))
        audit.det_verdict = _deterministic_verdict(answer, gq.correctness)
        if audit.det_verdict is None:
            judgment_only += 1

        for _ in range(max(1, repeats)):
            verdict, rationale, errored = _judge_all_rubrics(
                answer, rubrics, spec.judge_config, gq, judge_fn,
            )
            audit.judge_verdicts.append(verdict)
            audit.judge_rationales.append(rationale)
            if errored:
                audit.judge_errors += 1

        if progress is not None:
            progress(gq.query)
        audits.append(audit)

    return JudgeAuditReport(
        repeats=repeats, queries=audits, judgment_only_count=judgment_only,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

_DETERMINISTIC_FIELDS = (
    "expected_in_answer",
    "any_expected_in_answer",
    "not_in_answer",
    "exact_match",
    "regex_match",
    "json_schema",
)


def _judge_rubrics(c: CorrectnessSpec) -> list:
    rubrics = list(c.llm_judge or [])
    if c.safety_check:
        rubrics.append(c.safety_check)
    if c.hallucination_check:
        rubrics.append(c.hallucination_check)
    return rubrics


def _deterministic_verdict(answer: str, c: CorrectnessSpec) -> Optional[bool]:
    """Evaluate ONLY the deterministic checks. None if there are none."""
    if not any(getattr(c, f, None) is not None and getattr(c, f) != [] for f in _DETERMINISTIC_FIELDS):
        return None
    stripped = c.model_copy(
        update={
            "llm_judge": None,
            "safety_check": None,
            "hallucination_check": None,
            "refutes_premise": False,
        }
    )
    result = evaluate_correctness(answer=answer, spec=stripped)
    return result.status == LayerStatus.PASS


def _judge_all_rubrics(
    answer: str,
    rubrics: list,
    judge_config: Optional[dict[str, Any]],
    gq: GoldenQuery,
    judge_fn: JudgeFn,
) -> tuple[bool, str, bool]:
    """One judge pass over all rubrics: verdict = all rubrics pass.

    Returns (verdict, rationale, errored). An erroring judge fails the pass
    but is tracked separately — a judge that never ran must not be scored.
    """
    rationales: list[str] = []
    verdict = True
    errored = False
    for rubric in rubrics:
        try:
            result = judge_fn(
                answer=answer, rubric=rubric, config=judge_config, query=gq.query,
            )
        except Exception as e:  # noqa: BLE001
            result = {"passed": False, "rationale": f"judge error: {e}", "error": str(e)}
        if result.get("error"):
            errored = True
        if not result.get("passed", False):
            verdict = False
        if result.get("rationale"):
            rationales.append(str(result["rationale"]))
    return verdict, " | ".join(rationales), errored


def load_answers_from_baselines(baseline_dir: str) -> dict[str, str]:
    """Collect query → final answer from golden baseline JSON files.

    Tolerates both shapes on disk: a bare trace dict (``agentci record``) and
    a wrapper with a ``trace`` key (versioned ``agentci save`` baselines).
    """
    import glob
    import json
    from pathlib import Path

    answers: dict[str, str] = {}
    for f in sorted(glob.glob(str(Path(baseline_dir) / "**" / "*.json"), recursive=True)):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        trace_dict = data.get("trace") if isinstance(data.get("trace"), dict) else data
        if not isinstance(trace_dict, dict) or "spans" not in trace_dict:
            continue
        query = (
            trace_dict.get("query")
            or trace_dict.get("test_name")
            or (trace_dict.get("metadata") or {}).get("query")
        )
        if not query:
            continue
        answer = (trace_dict.get("metadata") or {}).get("final_output")
        if not answer:
            spans = trace_dict.get("spans") or []
            for span in reversed(spans):
                if span.get("output_data"):
                    answer = str(span["output_data"])
                    break
        if answer:
            answers[str(query)] = str(answer)
    return answers


def load_labels_file(path: str) -> dict[str, bool]:
    """Parse a hand-labels file: YAML/JSON mapping of query → pass|fail.

    Accepted values: pass/fail, true/false, 1/0 (case-insensitive).
    """
    import json
    from pathlib import Path

    import yaml

    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(
            f"Labels file must be a mapping of query → pass|fail, got {type(data).__name__}"
        )
    labels: dict[str, bool] = {}
    for query, value in data.items():
        if isinstance(value, bool):
            labels[str(query)] = value
        elif str(value).strip().lower() in ("pass", "true", "1", "yes"):
            labels[str(query)] = True
        elif str(value).strip().lower() in ("fail", "false", "0", "no"):
            labels[str(query)] = False
        else:
            raise ValueError(f"Label for '{query}' must be pass|fail, got: {value!r}")
    return labels
