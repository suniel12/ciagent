# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Generate agentci_spec.schema.json from the Pydantic AgentCISpec model.

Usage:
    python -m ciagent.schema.generate_schema
"""

import json
from pathlib import Path

from ciagent.schema.spec_models import AgentCISpec


def generate() -> None:
    schema = AgentCISpec.model_json_schema()
    out = Path(__file__).parent / "agentci_spec.schema.json"
    out.write_text(json.dumps(schema, indent=2))
    print(f"Schema written to {out}")


if __name__ == "__main__":
    generate()
