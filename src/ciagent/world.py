# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Simulated World MVP: world-from-failure (Plan_docs/world_sim_mvp.md).

A World is a set of frozen tool fixtures extracted from a recorded failing
run. During replay, `world_tool`-wrapped tools serve frozen responses instead
of hitting real backends, fail-closed: a call with no matching fixture raises
``WorldMiss`` AND is recorded (the recorded miss list is the authoritative
signal — frameworks like openai-agents convert tool exceptions into
error strings fed back to the model, so the raise alone proves nothing; A3).

Matching crosses the framework's validation layer (A1): the frozen ``match``
dict is the LLM's raw tool JSON, the runtime call arrives after pydantic
validation (defaults filled, scalars coerced, positional). So matching is:
every non-ignored match key present and scalar-normalized-equal; extra
offered keys are a miss unless they equal the parameter's signature default.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

WORLD_SCHEMA_VERSION = 1


# ── Exceptions ──────────────────────────────────────────────────────────────────


class WorldMiss(Exception):
    """A wrapped tool was called with arguments no fixture matches.

    Fail-closed: the world never invents a response and never falls through
    to the real function. The message carries the nearest fixture's
    field-level diff and a ready-to-paste ``ignore:`` suggestion (A5a).
    """

    def __init__(self, tool: str, offered: dict[str, Any], detail: str) -> None:
        super().__init__(f"world miss: {tool}({_short(offered)}) — {detail}")
        self.tool = tool
        self.offered = offered
        self.detail = detail


class WorldError(Exception):
    """World file invalid (schema, ambiguity invariant)."""


def _short(d: dict[str, Any], limit: int = 120) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ── Matching (A1) ───────────────────────────────────────────────────────────────


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def _scalar_eq(a: Any, b: Any) -> bool:
    """Equality across the framework's type-coercion boundary."""
    if a == b:
        return True
    if _canon(a) == _canon(b):
        return True
    if not isinstance(a, (dict, list)) and not isinstance(b, (dict, list)):
        return str(a) == str(b)
    return False


@dataclass
class Fixture:
    match: dict[str, Any]
    response: Any = None
    ignore: list[str] = field(default_factory=list)
    turn: Optional[int] = None            # informational (A10a)
    notes: str = ""

    def effective_match(self) -> dict[str, Any]:
        return {k: v for k, v in self.match.items() if k not in self.ignore}

    def matches(self, offered: dict[str, Any],
                defaults: Optional[dict[str, Any]] = None) -> bool:
        eff = self.effective_match()
        for k, v in eff.items():
            if k not in offered or not _scalar_eq(offered[k], v):
                return False
        # Extra offered keys: a miss unless the value equals the signature
        # default (the framework fills defaults the LLM omitted; A1).
        for k, v in offered.items():
            if k in eff or k in self.ignore:
                continue
            if defaults is not None and k in defaults and _scalar_eq(v, defaults[k]):
                continue
            return False
        return True

    def diff(self, offered: dict[str, Any]) -> tuple[int, str, list[str]]:
        """(mismatch_count, human diff, value-only-differing field names)."""
        eff = self.effective_match()
        lines: list[str] = []
        value_only: list[str] = []
        mismatches = 0
        for k, v in eff.items():
            if k not in offered:
                lines.append(f"    {k}: missing (fixture: {v!r})")
                mismatches += 1
            elif not _scalar_eq(offered[k], v):
                lines.append(f"    {k}: offered {offered[k]!r} != fixture {v!r}")
                value_only.append(k)
                mismatches += 1
        for k in offered:
            if k not in eff and k not in self.ignore:
                lines.append(f"    {k}: unexpected (offered {offered[k]!r})")
                mismatches += 1
        return mismatches, "\n".join(lines), value_only


@dataclass
class ToolWorld:
    fixtures: list[Fixture] = field(default_factory=list)
    sequence: bool = False
    suggested_ignore: list[str] = field(default_factory=list)  # A5b, informational
    _consumed: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._consumed:
            self._consumed = [False] * len(self.fixtures)


@dataclass
class MissRecord:
    tool: str
    offered: dict[str, Any]
    detail: str


@dataclass
class WorldReport:
    served: dict[str, int]
    misses: list[MissRecord]
    unconsumed: dict[str, int]        # sequence fixtures never reached
    gaps: list[dict[str, Any]]

    @property
    def miss_count(self) -> int:
        return len(self.misses)


class World:
    """Frozen tool fixtures + per-clone serving state (misses, consumption)."""

    def __init__(self, tools: dict[str, ToolWorld], *,
                 name: str = "", agent: str = "",
                 frozen_from: Optional[dict[str, Any]] = None,
                 gaps: Optional[list[dict[str, Any]]] = None) -> None:
        self.tools = tools
        self.name = name
        self.agent = agent
        self.frozen_from = frozen_from or {}
        self.gaps = gaps or []
        self._served: dict[str, int] = {}
        self._misses: list[MissRecord] = []
        self._lock = threading.Lock()   # in-turn parallel tool calls (A12)
        self._validate()

    # -- ambiguity invariant (A10b) ---------------------------------------------

    def _validate(self) -> None:
        for tool, tw in self.tools.items():
            if tw.sequence:
                continue
            seen: dict[str, Any] = {}
            for f in tw.fixtures:
                key = _canon(f.effective_match())
                if key in seen and _canon(seen[key]) != _canon(f.response):
                    raise WorldError(
                        f"ambiguous fixtures for reusable tool '{tool}': two "
                        f"fixtures share match {key} with different responses. "
                        "Set `sequence: true` for this tool, or differentiate "
                        "the matches."
                    )
                seen[key] = f.response

    # -- serve ------------------------------------------------------------------

    def serve(self, tool: str, offered: dict[str, Any],
              defaults: Optional[dict[str, Any]] = None) -> Any:
        with self._lock:
            tw = self.tools.get(tool)
            if tw is None:
                detail = (
                    f"tool '{tool}' has no fixtures in world '{self.name}' "
                    f"(frozen tools: {sorted(self.tools) or 'none'})"
                )
                if any(g.get("tool") == tool for g in self.gaps):
                    detail += (
                        " — this tool's calls were recorded WITHOUT a result "
                        "at freeze time (world gaps); they can never match."
                    )
                self._misses.append(MissRecord(tool, offered, detail))
                raise WorldMiss(tool, offered, detail)

            candidates = [
                (i, f) for i, f in enumerate(tw.fixtures)
                if not (tw.sequence and tw._consumed[i])
            ]
            for i, f in candidates:
                if f.matches(offered, defaults):
                    if tw.sequence:
                        tw._consumed[i] = True
                    self._served[tool] = self._served.get(tool, 0) + 1
                    return f.response

            detail = self._miss_detail(tool, tw, offered, candidates)
            self._misses.append(MissRecord(tool, offered, detail))
            raise WorldMiss(tool, offered, detail)

    def _miss_detail(self, tool: str, tw: ToolWorld, offered: dict[str, Any],
                     candidates: list[tuple[int, Fixture]]) -> str:
        gap_hits = [g for g in self.gaps if g.get("tool") == tool]
        if not candidates:
            base = f"all {len(tw.fixtures)} sequence fixture(s) already consumed"
            if gap_hits:
                base += " (note: this tool had result-less calls at freeze time)"
            return base
        ranked = sorted(
            (f.diff(offered) + (f,) for _, f in candidates),
            key=lambda t: t[0],
        )
        mismatches, diff_text, value_only, nearest = ranked[0]
        parts = [f"nearest fixture ({mismatches} field(s) differ):\n{diff_text}"]
        if value_only:
            parts.append(
                "if these fields are mutable (free text, ids, timestamps), "
                f"add to that fixture: \"ignore\": {json.dumps(sorted(value_only))}"
            )
        if gap_hits:
            parts.append(
                "note: this tool had calls recorded WITHOUT a result at freeze "
                "time (world gaps) — those calls can never match."
            )
        return "\n  ".join(parts)

    # -- lifecycle ---------------------------------------------------------------

    def clone(self) -> "World":
        """Fresh serving state for one scenario-run (A4)."""
        tools = {
            name: ToolWorld(
                fixtures=tw.fixtures,          # immutable during replay
                sequence=tw.sequence,
                suggested_ignore=list(tw.suggested_ignore),
            )
            for name, tw in self.tools.items()
        }
        return World(tools, name=self.name, agent=self.agent,
                     frozen_from=self.frozen_from, gaps=self.gaps)

    def report(self) -> WorldReport:
        unconsumed = {
            name: tw._consumed.count(False)
            for name, tw in self.tools.items()
            if tw.sequence and not all(tw._consumed)
        }
        return WorldReport(
            served=dict(self._served),
            misses=list(self._misses),
            unconsumed=unconsumed,
            gaps=list(self.gaps),
        )

    # -- persistence -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_schema": WORLD_SCHEMA_VERSION,
            "name": self.name,
            "agent": self.agent,
            "frozen_from": self.frozen_from,
            "gaps": self.gaps,
            "tools": {
                name: {
                    **({"sequence": True} if tw.sequence else {}),
                    **({"suggested_ignore": tw.suggested_ignore}
                       if tw.suggested_ignore else {}),
                    "fixtures": [
                        {
                            "match": f.match,
                            **({"ignore": f.ignore} if f.ignore else {}),
                            "response": f.response,
                            **({"turn": f.turn} if f.turn is not None else {}),
                            **({"notes": f.notes} if f.notes else {}),
                        }
                        for f in tw.fixtures
                    ],
                }
                for name, tw in self.tools.items()
            },
        }

    def save(self, path: Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path) -> "World":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise WorldError(f"cannot read world file {path}: {e}") from e
        declared = data.get("world_schema")
        if declared != WORLD_SCHEMA_VERSION:
            raise WorldError(
                f"unsupported world_schema {declared!r} (this ciagent reads "
                f"{WORLD_SCHEMA_VERSION})"
            )
        tools: dict[str, ToolWorld] = {}
        for name, td in (data.get("tools") or {}).items():
            fixtures = [
                Fixture(
                    match=dict(fd.get("match") or {}),
                    response=fd.get("response"),
                    ignore=list(fd.get("ignore") or []),
                    turn=fd.get("turn"),
                    notes=fd.get("notes") or "",
                )
                for fd in (td.get("fixtures") or [])
            ]
            tools[name] = ToolWorld(
                fixtures=fixtures,
                sequence=bool(td.get("sequence")),
                suggested_ignore=list(td.get("suggested_ignore") or []),
            )
        return cls(
            tools,
            name=data.get("name") or "",
            agent=data.get("agent") or "",
            frozen_from=data.get("frozen_from") or {},
            gaps=list(data.get("gaps") or []),
        )


# ── Freeze (D3, A9-A11) ─────────────────────────────────────────────────────────


def freeze_envelope(envelope: Any, *, name: str = "",
                    tools_filter: Optional[list[str]] = None,
                    allow_gaps: bool = False) -> World:
    """Build a World from a (already-redacted, A8) envelope's tool traffic.

    Raises WorldError on zero usable tool calls, or on result-less calls
    without allow_gaps (a gap WILL miss on replay; the caller must opt in).
    Call order is span-END order as recorded (A12).
    """
    calls: list[tuple[str, dict[str, Any], Any, int, bool]] = []
    for turn in envelope.turns:
        for span in turn.trace.spans or []:
            for tc in span.tool_calls or []:
                if tools_filter and tc.tool_name not in tools_filter:
                    continue
                args = tc.arguments if isinstance(tc.arguments, dict) else {}
                calls.append(
                    (tc.tool_name, args, tc.result, turn.turn_index,
                     tc.result is None)
                )

    if not calls:
        raise WorldError(
            "envelope has no tool calls to freeze — an empty world would make "
            "every wrapped tool miss on replay."
        )

    gaps = [
        {"tool": t, "args": a, "turn": turn}
        for t, a, _r, turn, is_gap in calls if is_gap
    ]
    if gaps and not allow_gaps:
        listing = ", ".join(f"{g['tool']} (turn {g['turn'] + 1})" for g in gaps)
        raise WorldError(
            f"{len(gaps)} tool call(s) were recorded WITHOUT a result and "
            f"WILL miss on replay: {listing}. Re-run with --allow-gaps to "
            "freeze anyway (the gaps are recorded in the world file)."
        )

    by_tool: dict[str, list[tuple[dict[str, Any], Any, int]]] = {}
    for t, a, r, turn, is_gap in calls:
        if is_gap:
            continue
        by_tool.setdefault(t, []).append((a, r, turn))

    tools: dict[str, ToolWorld] = {}
    for tool, entries in by_tool.items():
        fixtures: list[Fixture] = []
        seen: dict[str, Any] = {}          # canon(match) -> canon(response)
        sequence = False
        for args, result, turn in entries:
            key = _canon(args)
            if key in seen:
                if seen[key] == _canon(result):
                    continue               # exact duplicate → dedupe (reusable)
                sequence = True            # same args, new result → stateful
            seen[key] = _canon(result)
            fixtures.append(Fixture(match=args, response=result, turn=turn))
        if sequence:
            # Rebuild without dedupe: every call is a step in the state
            # machine, consumed FIFO.
            fixtures = [Fixture(match=a, response=r, turn=turn)
                        for a, r, turn in entries]
        tools[tool] = ToolWorld(
            fixtures=fixtures,
            sequence=sequence,
            suggested_ignore=_suggest_ignore(entries),
        )

    return World(tools, name=name, agent=envelope.agent or "",
                 frozen_from={}, gaps=gaps)


_FREE_TEXT_MIN_LEN = 60


def _suggest_ignore(entries: list[tuple[dict[str, Any], Any, int]]) -> list[str]:
    """A5b heuristics, informational only: fields that differ between calls
    whose OTHER fields agree, plus long free-text strings."""
    suggestions: set[str] = set()
    for i, (a1, _r1, _t1) in enumerate(entries):
        for a2, _r2, _t2 in entries[i + 1:]:
            if set(a1) != set(a2):
                continue
            differing = [k for k in a1 if not _scalar_eq(a1[k], a2[k])]
            if len(differing) == 1:
                suggestions.add(differing[0])
    for args, _r, _t in entries:
        for k, v in args.items():
            if isinstance(v, str) and len(v) >= _FREE_TEXT_MIN_LEN:
                suggestions.add(k)
    return sorted(suggestions)


# ── Activation + the decorator seam (D1, A2, A4) ────────────────────────────────


_active_world: contextvars.ContextVar[Optional[World]] = contextvars.ContextVar(
    "_ciagent_active_world", default=None
)


def active_world() -> Optional[World]:
    return _active_world.get()


@contextmanager
def activate(world: World) -> Iterator[None]:
    token = _active_world.set(world)
    try:
        yield
    finally:
        _active_world.reset(token)


def _is_context_param(param: inspect.Parameter) -> bool:
    """Detect framework context params (e.g. openai-agents RunContextWrapper)
    by annotation name, without importing any framework."""
    ann = param.annotation
    name = getattr(ann, "__name__", None) or str(ann)
    return "RunContextWrapper" in name or "ToolContext" in name


def world_tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The interposition seam (D1). MUST be the innermost decorator, directly
    on the plain function, underneath any framework decorator.

    No active world: zero-overhead passthrough. Active world: serves the
    frozen response for (tool_name, offered args), fail-closed.
    """
    if not inspect.isfunction(fn):
        raise TypeError(
            "world_tool must wrap a plain function and be the INNERMOST "
            f"decorator (got {type(fn).__name__}). Apply it directly on the "
            "def, underneath @function_tool or similar."
        )

    sig = inspect.signature(fn)
    context_params = {n for n, p in sig.parameters.items() if _is_context_param(p)}
    defaults = {
        n: p.default for n, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }

    def _offered(args: tuple, kwargs: dict) -> dict[str, Any]:
        # Bind WITHOUT apply_defaults (A1): a default the framework filled in
        # is indistinguishable from one the LLM passed, so absence stays
        # absent and the extra-key rule uses `defaults` at match time.
        bound = sig.bind(*args, **kwargs)
        return {
            k: v for k, v in bound.arguments.items() if k not in context_params
        }

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            world = _active_world.get()
            if world is None:
                return await fn(*args, **kwargs)
            return world.serve(fn.__name__, _offered(args, kwargs), defaults)
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        world = _active_world.get()
        if world is None:
            return fn(*args, **kwargs)
        return world.serve(fn.__name__, _offered(args, kwargs), defaults)
    return wrapper
