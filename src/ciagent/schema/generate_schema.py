# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Generate ciagent_spec.schema.json from the Pydantic CIAgentSpec model.

Usage:
    python -m ciagent.schema.generate_schema
"""

import json
from pathlib import Path

from ciagent.schema.spec_models import CIAgentSpec


def generate() -> None:
    schema = CIAgentSpec.model_json_schema()
    out = Path(__file__).parent / "ciagent_spec.schema.json"
    out.write_text(json.dumps(schema, indent=2))
    print(f"Schema written to {out}")


if __name__ == "__main__":
    generate()
