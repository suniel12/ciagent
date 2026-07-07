# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""Importers — convert exported production traces into CIAgent goldens.

`import_trace_file` sniffs the format and dispatches:
  - LangSmith run objects (run_type) → importers.langsmith
  - OTel GenAI spans (OTLP envelope / span lists) → importers.otel
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union


class TraceImportError(ValueError):
    """The file cannot be read as any supported trace export."""


def import_trace_file(path: Union[str, Path]) -> tuple[Any, Optional[str], str]:
    """Read a trace export and map it. Returns (trace, query, source_format).

    Raises TraceImportError (or a format importer's subclass of ValueError)
    when the file isn't a readable export — distinct from the artifact
    gate's rejection of readable-but-partial traces.
    """
    from ciagent.importers.langsmith import (
        LangsmithImportError,
        load_runs,
        looks_like_runs,
        trace_from_langsmith,
    )
    from ciagent.importers.otel import OtelImportError, load_spans, trace_from_otel

    try:
        parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        parsed = None  # maybe JSONL — the LangSmith loader handles that
    except OSError as e:
        raise TraceImportError(f"cannot read '{path}': {e}") from e

    if parsed is None or looks_like_runs(parsed):
        try:
            return (*trace_from_langsmith(load_runs(path)), "langsmith-runs")
        except LangsmithImportError:
            if parsed is not None:
                raise
            # fell through: not JSONL runs either — report as OTel below

    try:
        return (*trace_from_otel(load_spans(path)), "otel-genai")
    except OtelImportError as e:
        raise TraceImportError(str(e)) from e
