#!/usr/bin/env bash
# Reproduce the prompt-injection gate end to end (no API key; toy agent).
set -euo pipefail
ciagent simulate --yes --record --record-dir ./golden          # 1. clean golden
GOLDEN=$(ls golden/atlas-inject/scenarios/*.json | head -1)
ciagent world freeze "$GOLDEN" -o clean.world.json             # 2. freeze the clean backend
ciagent simulate --yes --replay ./golden --world clean.world.json  # 3. clean replay passes (exit 0)
ciagent world mutate clean.world.json --op inject \
    --payload-id role-override -o evil.world.json              # 4. inject an override into the tool result
echo "--- injected replay (expect exit 1: the agent obeyed the tool output) ---"
ciagent simulate --yes --replay ./golden --world evil.world.json   # 5. GATE FIRES
