# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
LLM persona for `ciagent simulate` (F6 Phase 3 — the finder path).

A persona LLM role-plays the user described by a scenario's `persona:` and
`goal:`, generating a fresh user turn from the conversation so far. Binding
rules (eng review 2026-07-05 / ADR):

- The persona is nondeterministic BY DESIGN — it finds bugs. The regression
  gate is replay mode, which never calls the persona.
- The persona never terminates the conversation: termination stays with
  max_turns / stop_when in the driver. A persona that produces unusable
  output (empty, whitespace) raises PersonaError — the driver marks the
  scenario infra-error and keeps completed turns; a derailed simulated user
  must never silently grade the agent.
- Defaults to a cheap haiku-class model: simulation multiplies LLM calls,
  and the persona is the multiplier that isn't the product under test.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

DEFAULT_PERSONA_TEMPERATURE = 0.7
_MAX_TURN_CHARS = 2000  # a "user message" longer than this is a derailed essay

# (system_prompt, user_prompt, model, temperature) -> completion text.
# Injectable so tests and mock mode never touch the network.
CompleteFn = Callable[[str, str, str, float], str]


class PersonaError(Exception):
    """The persona LLM produced no usable user turn (derail guard)."""


def default_persona_model() -> str:
    """Haiku-class default, picked by available API key (binding: cheap model)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    return "claude-haiku-4-5"  # fails later with the client's clear key error


def build_persona_system_prompt(persona: str, goal: str) -> str:
    return (
        "You are role-playing a USER talking to a customer-facing AI agent. "
        "Stay in character for the whole conversation.\n\n"
        f"Your character: {persona or 'an ordinary user'}\n"
        f"Your goal: {goal or 'get your problem resolved'}\n\n"
        "Rules:\n"
        "- Write ONLY the user's next message — no narration, no quotes, "
        "no explanations of what you are doing.\n"
        "- Keep it short and natural, the way real users type.\n"
        "- Pursue your goal persistently; react to what the agent just said.\n"
        "- Never break character, never mention being an AI or a test."
    )


def _render_transcript(messages: list[dict[str, str]]) -> str:
    """Render history from the simulated user's perspective.

    In driver history, 'user' is the persona's own past messages and
    'assistant' is the agent under test — rendered as a transcript instead of
    swapping roles, so it reads the same for every provider.
    """
    if not messages:
        return "The conversation has not started. Write your opening message."
    lines = []
    for m in messages:
        speaker = "You" if m.get("role") == "user" else "Agent"
        lines.append(f"{speaker}: {m.get('content', '')}")
    lines.append("\nWrite your next message as the user.")
    return "\n".join(lines)


def _default_complete(system: str, user: str, model: str, temperature: float) -> str:
    # Same provider clients the judge uses — model prefix picks the API.
    from .judge import _call_anthropic, _call_openai

    if model.startswith(("claude", "anthropic:")):
        return _call_anthropic(model, system, user, temperature)
    return _call_openai(model, system, user, temperature)


def generate_user_turn(
    persona: str,
    goal: str,
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = DEFAULT_PERSONA_TEMPERATURE,
    complete_fn: Optional[CompleteFn] = None,
) -> str:
    """Generate the simulated user's next message. Raises PersonaError on derail."""
    system = build_persona_system_prompt(persona, goal)
    prompt = _render_transcript(messages)
    complete = complete_fn or _default_complete

    try:
        raw = complete(system, prompt, model or default_persona_model(), temperature)
    except Exception as exc:
        raise PersonaError(f"persona LLM call failed: {type(exc).__name__}: {exc}") from exc

    text = (raw or "").strip().strip('"').strip()
    if not text:
        raise PersonaError("persona LLM returned an empty user turn")
    if len(text) > _MAX_TURN_CHARS:
        raise PersonaError(
            f"persona LLM derailed: produced {len(text)} chars for one user turn "
            f"(max {_MAX_TURN_CHARS})"
        )
    return text


def persona_turn_source(
    scenario,
    persona_config: Optional[dict] = None,
    complete_fn: Optional[CompleteFn] = None,
):
    """Build a driver turn source that asks the persona LLM for each turn.

    Returns ``(messages, turn_index) -> str``; never returns None — a
    generative conversation only ends via max_turns or stop_when.
    """
    cfg = persona_config or {}
    model = cfg.get("model")
    temperature = cfg.get("temperature", DEFAULT_PERSONA_TEMPERATURE)

    def next_turn(messages: list[dict[str, str]], turn_index: int) -> str:
        return generate_user_turn(
            persona=scenario.persona or "",
            goal=scenario.goal or "",
            messages=messages,
            model=model,
            temperature=temperature,
            complete_fn=complete_fn,
        )

    return next_turn
