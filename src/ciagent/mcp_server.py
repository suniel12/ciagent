# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
MCP server: the found-bug-to-CI-gate loop, operable by coding agents.

Design: Plan_docs/mcp_server.md (adversarially reviewed, A1-A13 binding).
Architecture: every tool shells out to the tested CLI surface
(`sys.executable -m ciagent.cli ...`, A2) with per-command capability tables
(A1), parses the #39 one-JSON-document stdout where it exists, and returns a
uniform envelope. The server, not the CLI, is the sole cost gate (A4): json
mode plus --yes skips every CLI confirm, so live runs are refused locally
unless the caller passes max_cost / allow_live.

All tool logic lives in plain async functions so the test suite exercises
them without the `mcp` package; `build_server()` is the only place FastMCP
is imported (A5).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# A1: per-command flag capabilities — appending a flag a command lacks is a
# guaranteed click UsageError (exit 2).
SUPPORTS_JSON = {
    ("test",), ("simulate",), ("stage", "list"), ("stage", "show"),
    ("promote",), ("world", "show"),
}
SUPPORTS_YES = {
    ("test",), ("simulate",), ("stage", "verify"), ("stage", "drop"),
    ("promote",),
}

DEFAULT_TIMEOUT_S = 600
DEFAULT_DATA_CAP_BYTES = 50_000


class GuardrailRefused(Exception):
    """The server refused locally, before any subprocess ran."""


@dataclass
class ServerConfig:
    project_root: Path
    timeout_s: int = DEFAULT_TIMEOUT_S
    data_cap_bytes: int = DEFAULT_DATA_CAP_BYTES
    # A11: import rewrites the spec non-atomically; mutating tools serialize.
    mutate_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ── Guardrails ──────────────────────────────────────────────────────────────────


def jail(cfg: ServerConfig, value: str, *, must_exist: bool = False) -> str:
    """A9: resolve a path argument and refuse escapes from the project root.

    Symlink-safe (resolve + commonpath). Constrains path ARGUMENTS only; the
    CLI still executes the project's own runner code by design.
    """
    root = cfg.project_root.resolve()
    p = Path(value)
    resolved = (p if p.is_absolute() else root / p).resolve()
    try:
        if os.path.commonpath([str(root), str(resolved)]) != str(root):
            raise ValueError
    except ValueError:
        raise GuardrailRefused(
            f"path '{value}' escapes the project root {root} — refused."
        ) from None
    if must_exist and not resolved.exists():
        raise GuardrailRefused(f"path '{value}' does not exist under {root}.")
    return str(resolved)


def require_live_ack(*, mock: bool, tool: str,
                     max_cost: Optional[float] = None,
                     allow_live: bool = False,
                     needs_max_cost: bool = False) -> None:
    """A4: the sole cost gate. The CLI's confirms are all skipped under the
    server (json mode + --yes), so refusing here is the only speed bump."""
    if mock:
        return
    if needs_max_cost:
        if max_cost is None:
            raise GuardrailRefused(
                f"{tool}: live runs spend money and the CLI confirm is "
                "bypassed under MCP. Pass max_cost (USD hard abort) or "
                "mock=true."
            )
        return
    if not allow_live:
        raise GuardrailRefused(
            f"{tool}: live runs spend money and have no --max-cost abort. "
            "Pass allow_live=true to acknowledge, or mock=true."
        )


# ── Subprocess + envelope ───────────────────────────────────────────────────────


async def run_cli(cfg: ServerConfig, args: list[str],
                  timeout_s: Optional[int] = None) -> tuple[int, str, str]:
    """A2/A6: same-interpreter subprocess, own session, killpg on timeout."""
    cmd = [sys.executable, "-m", "ciagent.cli", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cfg.project_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out_b, err_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s or cfg.timeout_s
        )
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        await proc.wait()
        raise GuardrailRefused(
            f"command timed out after {timeout_s or cfg.timeout_s}s and was "
            "killed (pass timeout_s to raise the limit)."
        )
    return proc.returncode or 0, out_b.decode(errors="replace"), err_b.decode(errors="replace")


def _shape_simulate(data: dict[str, Any]) -> dict[str, Any]:
    """A7: keep the load-bearing keys verbatim, compress per-scenario noise."""
    return {
        "summary": data.get("summary"),
        "stability": data.get("stability"),
        "recorded": data.get("recorded"),
        "cost_aborted": data.get("cost_aborted"),
        "spent_usd": data.get("spent_usd"),
        "scenarios": [
            {k: s.get(k) for k in (
                "name", "hard_fail", "partial", "termination", "lifecycle",
                "xpass", "world_misses") if k in s}
            for s in (data.get("scenarios") or [])
        ],
    }


def make_envelope(cfg: ServerConfig, subcmd: tuple[str, ...], args: list[str],
                  exit_code: int, stdout: str, stderr: str) -> dict[str, Any]:
    """A3/A7/A12 envelope. `ok` = completed + parsed-when-expected; exit 1 is
    often a successful detection and is NOT an envelope failure."""
    expect_json = subcmd in SUPPORTS_JSON
    env: dict[str, Any] = {
        "exit_code": exit_code,
        "command": "ciagent " + " ".join(args),
        "stderr_tail": stderr[-2000:],
    }
    data: Any = None
    parse_ok = True
    if expect_json and stdout.strip():
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            parse_ok = False
            env["stdout_text"] = stdout[-4000:]
    elif not expect_json:
        env["stdout_text"] = stdout[-4000:]

    if data is not None:
        raw = json.dumps(data)
        if len(raw) > cfg.data_cap_bytes:
            out_dir = cfg.project_root / ".ciagent" / "mcp"
            out_dir.mkdir(parents=True, exist_ok=True)
            data_file = out_dir / f"{'-'.join(subcmd)}-{int(time.time())}.json"
            data_file.write_text(raw, encoding="utf-8")
            env["data_truncated"] = True
            env["data_file"] = str(data_file)
            data = _shape_simulate(data) if subcmd == ("simulate",) else {
                k: data[k] for k in list(data)[:10]
                if len(json.dumps(data[k])) < 2000
            }
    env["data"] = data
    env["ok"] = parse_ok
    return env


async def invoke(cfg: ServerConfig, subcmd: tuple[str, ...], args: list[str],
                 *, timeout_s: Optional[int] = None,
                 mutating: bool = False) -> dict[str, Any]:
    full = list(subcmd) + args
    if subcmd in SUPPORTS_JSON:
        full += ["--format", "json"]
    if subcmd in SUPPORTS_YES:
        full += ["--yes"]
    try:
        if mutating:
            async with cfg.mutate_lock:
                code, out, err = await run_cli(cfg, full, timeout_s)
        else:
            code, out, err = await run_cli(cfg, full, timeout_s)
    except GuardrailRefused as e:
        return {"ok": False, "exit_code": None, "error": str(e),
                "command": "ciagent " + " ".join(full), "data": None}
    return make_envelope(cfg, subcmd, full, code, out, err)


def _refused(cmd: str, e: GuardrailRefused) -> dict[str, Any]:
    return {"ok": False, "exit_code": None, "error": str(e), "command": cmd,
            "data": None}


# ── Tool implementations (plain async functions; FastMCP wraps these) ───────────


async def tool_test(cfg: ServerConfig, *, mock: bool = True, runs: int = 1,
                    stage: Optional[bool] = None, tags: Optional[str] = None,
                    allow_live: bool = False,
                    timeout_s: Optional[int] = None) -> dict[str, Any]:
    try:
        require_live_ack(mock=mock, tool="ciagent_test", allow_live=allow_live)
    except GuardrailRefused as e:
        return _refused("ciagent test", e)
    args: list[str] = []
    if mock:
        args.append("--mock")
    if runs > 1:
        args += ["--runs", str(runs)]
    if stage is True:
        args.append("--stage")
    elif stage is False:
        args.append("--no-stage")
    if tags:
        for t in tags.split(","):
            args += ["--tags", t.strip()]
    return await invoke(cfg, ("test",), args, timeout_s=timeout_s)


async def tool_simulate(cfg: ServerConfig, *, mock: bool = True, runs: int = 1,
                        stage: Optional[bool] = None, record: bool = False,
                        replay: Optional[str] = None,
                        world: Optional[str] = None,
                        max_cost: Optional[float] = None,
                        timeout_s: Optional[int] = None) -> dict[str, Any]:
    try:
        # A4: --world replay goes live after a miss; it is NOT free.
        require_live_ack(mock=mock, tool="ciagent_simulate",
                         max_cost=max_cost, needs_max_cost=True)
        args: list[str] = []
        if mock:
            args.append("--mock")
        if runs > 1:
            args += ["--runs", str(runs)]
        if stage is True:
            args.append("--stage")
        elif stage is False:
            args.append("--no-stage")
        if record:
            args.append("--record")
        if replay:
            args += ["--replay", jail(cfg, replay, must_exist=True)]
        if world:
            args += ["--world", jail(cfg, world, must_exist=True)]
        if max_cost is not None:
            args += ["--max-cost", str(max_cost)]
    except GuardrailRefused as e:
        return _refused("ciagent simulate", e)
    return await invoke(cfg, ("simulate",), args, timeout_s=timeout_s)


async def tool_stage_list(cfg: ServerConfig, *,
                          classification: Optional[str] = None,
                          agent: Optional[str] = None) -> dict[str, Any]:
    args: list[str] = []
    if classification:
        args += ["--classification", classification]
    if agent:
        args += ["--agent", agent]
    return await invoke(cfg, ("stage", "list"), args)


async def tool_stage_show(cfg: ServerConfig, *, stage_id: str) -> dict[str, Any]:
    return await invoke(cfg, ("stage", "show"), [stage_id])


async def tool_stage_verify(cfg: ServerConfig, *, stage_id: str, runs: int = 3,
                            mock: bool = True, reroll: bool = False,
                            world: Optional[str] = None,
                            allow_live: bool = False,
                            timeout_s: Optional[int] = None) -> dict[str, Any]:
    try:
        require_live_ack(mock=mock, tool="ciagent_stage_verify",
                         allow_live=allow_live)
        args = [stage_id, "--runs", str(runs)]
        if mock:
            args.append("--mock")
        if reroll:
            args.append("--reroll")
        if world:
            args += ["--world", jail(cfg, world, must_exist=True)]
    except GuardrailRefused as e:
        return _refused("ciagent stage verify", e)
    return await invoke(cfg, ("stage", "verify"), args, timeout_s=timeout_s)


async def tool_stage_drop(cfg: ServerConfig, *, stage_id: str) -> dict[str, Any]:
    return await invoke(cfg, ("stage", "drop"), [stage_id], mutating=True)


async def tool_promote(cfg: ServerConfig, *, stage_id: str, xfail: bool = False,
                       force: bool = False) -> dict[str, Any]:
    # A8: stage_id is REQUIRED — the interactive picker with --yes exits 0
    # "Cancelled", which an agent would misread as success.
    if not stage_id:
        return _refused("ciagent promote", GuardrailRefused(
            "promote requires a stage_id (use ciagent_stage_list first)."))
    args = [stage_id]
    if xfail:
        args.append("--xfail")
    if force:
        args.append("--force")
    return await invoke(cfg, ("promote",), args, mutating=True)


async def tool_flip(cfg: ServerConfig, *, golden: str) -> dict[str, Any]:
    return await invoke(cfg, ("promote",), ["--flip", golden], mutating=True)


async def tool_world_freeze(cfg: ServerConfig, *, source: str,
                            output: Optional[str] = None,
                            allow_gaps: bool = False,
                            force_redact: bool = False) -> dict[str, Any]:
    try:
        # A9: SOURCE is a stage id OR a path; jail only when it is a file.
        src = source
        candidate = (cfg.project_root / source)
        if Path(source).is_absolute() or candidate.is_file():
            src = jail(cfg, source, must_exist=True)
        # A3: always pass --output so the world path is known a priori.
        out = jail(cfg, output) if output else str(
            cfg.project_root / "worlds" / "frozen.world.json")
        args = [src, "--output", out]
        if allow_gaps:
            args.append("--allow-gaps")
        if force_redact:
            args.append("--force-redact")
    except GuardrailRefused as e:
        return _refused("ciagent world freeze", e)
    env = await invoke(cfg, ("world", "freeze"), args, mutating=True)
    env["world_file"] = out
    return env


async def tool_world_show(cfg: ServerConfig, *, path: str) -> dict[str, Any]:
    try:
        p = jail(cfg, path, must_exist=True)
    except GuardrailRefused as e:
        return _refused("ciagent world show", e)
    return await invoke(cfg, ("world", "show"), [p])


async def tool_import(cfg: ServerConfig, *, trace_file: str,
                      dry_run: bool = False, force_save: bool = False,
                      allow_live: bool = False) -> dict[str, Any]:
    try:
        # A4: without dry_run/force_save the save-baseline precheck can make
        # a judge call — that is spend.
        if not dry_run and not force_save:
            require_live_ack(mock=False, tool="ciagent_import",
                             allow_live=allow_live)
        p = jail(cfg, trace_file, must_exist=True)
        args = [p]
        if dry_run:
            args.append("--dry-run")
        if force_save:
            args.append("--force-save")
    except GuardrailRefused as e:
        return _refused("ciagent import", e)
    return await invoke(cfg, ("import",), args, mutating=True)


# ── FastMCP registration (the only place `mcp` is imported; A5) ─────────────────


def build_server(cfg: ServerConfig):
    from mcp.server.fastmcp import FastMCP

    s = FastMCP(
        "ciagent",
        instructions=(
            "Regression-test AI agents: run suites (mock is free; live runs "
            "REQUIRE max_cost or allow_live), inspect auto-staged failures, "
            "promote one to a permanent CI gate, freeze a failing run's tool "
            "traffic into a deterministic world, and replay against it. Exit "
            "code 1 usually means 'the gate detected a failure' — that is "
            "the tool working, not an error."
        ),
    )

    @s.tool(name="ciagent_test")
    async def _test(mock: bool = True, runs: int = 1,
                    stage: Optional[bool] = None, tags: Optional[str] = None,
                    allow_live: bool = False,
                    timeout_s: Optional[int] = None) -> dict:
        """Run the single-turn suite. Exit 0 pass / 1 correctness failure / 2 infra."""
        return await tool_test(cfg, mock=mock, runs=runs, stage=stage,
                               tags=tags, allow_live=allow_live,
                               timeout_s=timeout_s)

    @s.tool(name="ciagent_simulate")
    async def _simulate(mock: bool = True, runs: int = 1,
                        stage: Optional[bool] = None, record: bool = False,
                        replay: Optional[str] = None,
                        world: Optional[str] = None,
                        max_cost: Optional[float] = None,
                        timeout_s: Optional[int] = None) -> dict:
        """Run scenario suite / replay / frozen-world replay. Live requires
        max_cost (USD hard abort) — including --world replay. Exit 0 pass /
        1 gate failure or world miss / 2 config-cost-infra."""
        return await tool_simulate(cfg, mock=mock, runs=runs, stage=stage,
                                   record=record, replay=replay, world=world,
                                   max_cost=max_cost, timeout_s=timeout_s)

    @s.tool(name="ciagent_stage_list")
    async def _stage_list(classification: Optional[str] = None,
                          agent: Optional[str] = None) -> dict:
        """List auto-staged failing conversations, best-to-promote first.
        classification: consistent|flaky-agent|held|held-infra|unverified."""
        return await tool_stage_list(cfg, classification=classification,
                                     agent=agent)

    @s.tool(name="ciagent_stage_show")
    async def _stage_show(stage_id: str) -> dict:
        """Show one staged conversation (redacted)."""
        return await tool_stage_show(cfg, stage_id=stage_id)

    @s.tool(name="ciagent_stage_verify")
    async def _stage_verify(stage_id: str, runs: int = 3, mock: bool = True,
                            reroll: bool = False, world: Optional[str] = None,
                            allow_live: bool = False,
                            timeout_s: Optional[int] = None) -> dict:
        """Re-run a staged scenario N times and re-classify. Live requires allow_live."""
        return await tool_stage_verify(cfg, stage_id=stage_id, runs=runs,
                                       mock=mock, reroll=reroll, world=world,
                                       allow_live=allow_live,
                                       timeout_s=timeout_s)

    @s.tool(name="ciagent_stage_drop")
    async def _stage_drop(stage_id: str) -> dict:
        """Delete one staged conversation."""
        return await tool_stage_drop(cfg, stage_id=stage_id)

    @s.tool(name="ciagent_promote")
    async def _promote(stage_id: str, xfail: bool = False,
                       force: bool = False) -> dict:
        """Promote a staged failure to a golden CI gate. xfail: replay stays
        green while the bug reproduces. Exit 1 = refused (gated class)."""
        return await tool_promote(cfg, stage_id=stage_id, xfail=xfail,
                                  force=force)

    @s.tool(name="ciagent_flip")
    async def _flip(golden: str) -> dict:
        """Flip a passing xfail golden to a normal gate golden."""
        return await tool_flip(cfg, golden=golden)

    @s.tool(name="ciagent_world_freeze")
    async def _world_freeze(source: str, output: Optional[str] = None,
                            allow_gaps: bool = False,
                            force_redact: bool = False) -> dict:
        """Freeze a staged entry's or golden's tool traffic into a world file."""
        return await tool_world_freeze(cfg, source=source, output=output,
                                       allow_gaps=allow_gaps,
                                       force_redact=force_redact)

    @s.tool(name="ciagent_world_show")
    async def _world_show(path: str) -> dict:
        """Show a world file's tool surface."""
        return await tool_world_show(cfg, path=path)

    @s.tool(name="ciagent_import")
    async def _import(trace_file: str, dry_run: bool = False,
                      force_save: bool = False,
                      allow_live: bool = False) -> dict:
        """Import a production trace (OTel/Langfuse/LangSmith) as a gated
        regression test. Without dry_run/force_save a judge precheck may
        spend — requires allow_live."""
        return await tool_import(cfg, trace_file=trace_file, dry_run=dry_run,
                                 force_save=force_save, allow_live=allow_live)

    return s


def main(project: str = ".", timeout: int = DEFAULT_TIMEOUT_S) -> None:
    cfg = ServerConfig(project_root=Path(project).resolve(),
                       timeout_s=timeout)
    build_server(cfg).run()
