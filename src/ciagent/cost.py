# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Token cost computation.

Pricing is hardcoded (updated monthly). This is intentional —
an API call to fetch pricing would be a dependency and a failure point.
Users can override with custom pricing in ciagent.yaml.
"""

# Prices per 1M tokens as of Feb 2026 (update monthly)
# Source: provider pricing pages
PRICING: dict[str, dict[str, tuple[float, float]]] = {
    # provider -> model -> (input_per_1M, output_per_1M)
    "openai": {
        "gpt-4o":       (2.50, 10.00),
        "gpt-4o-mini":  (0.15, 0.60),
        "gpt-4.1":      (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "o3-mini":      (1.10, 4.40),
    },
    "anthropic": {
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-haiku-4-5-20251001": (0.80, 4.00),
        "claude-opus-4-6":  (15.00, 75.00),
    },
}


def compute_cost(
    provider: str, 
    model: str, 
    tokens_in: int, 
    tokens_out: int
) -> float:
    """Compute USD cost for a single LLM call."""
    provider_pricing = PRICING.get(provider, {})
    
    # Try exact match first, then prefix match
    pricing = provider_pricing.get(model)
    if pricing is None:
        for model_key, price in provider_pricing.items():
            if model.startswith(model_key):
                pricing = price
                break
    
    if pricing is None:
        return 0.0  # Unknown model — don't crash, just skip cost
    
    input_cost = (tokens_in / 1_000_000) * pricing[0]
    output_cost = (tokens_out / 1_000_000) * pricing[1]
    return round(float(input_cost + output_cost), 6)


def infer_provider(model: str) -> str:
    """Best-effort provider from a model name.

    Used when usage comes from provider-agnostic metadata (e.g. LangChain
    AIMessage.usage_metadata) that doesn't say which SDK produced it.
    Returns "" when the model name doesn't identify a known provider.
    """
    name = (model or "").lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    return ""


def compute_cost_for_model(
    model: str,
    tokens_in: int,
    tokens_out: int,
    provider: str = "",
) -> float:
    """Compute cost when the provider may be unknown.

    Tries the given provider first, then falls back to inferring it from
    the model name, then to searching every pricing table. Returns 0.0 for
    models with no known pricing (same contract as compute_cost).
    """
    candidates = [p for p in (provider, infer_provider(model)) if p]
    candidates += [p for p in PRICING if p not in candidates]
    for candidate in candidates:
        cost = compute_cost(candidate, model, tokens_in, tokens_out)
        if cost:
            return cost
    return 0.0
