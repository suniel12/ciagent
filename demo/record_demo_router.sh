#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CIAgent Demo Recording Script
#
# Record with:
#   cd CIAgent && asciinema rec demo/ciagent-demo.cast -c "./demo/record_demo.sh"
#
# Upload with:
#   asciinema upload demo/ciagent-demo.cast
# ─────────────────────────────────────────────────────────────────────────────

set -e

# Ensure TERM is set (needed for headless asciinema)
export TERM="${TERM:-xterm-256color}"

DEMO_DIR="/Users/sunilpandey/startup/github/Agents/DemoAgents/examples/support-router"
TRIAGE="$DEMO_DIR/support_router/agents/triage.py"

# Init conda
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"

# Helper: simulate typing
type_cmd() {
    echo ""
    echo -n "$ "
    for (( i=0; i<${#1}; i++ )); do
        echo -n "${1:$i:1}"
        sleep 0.03
    done
    echo ""
    sleep 0.4
}

pause() { sleep "${1:-1.5}"; }

clear

echo ""
echo "  CIAgent — Regression Testing for AI Agents"
echo "  pip install ciagent | github.com/suniel12/CIAgent"
echo ""
pause 2

# ── Step 1 ──
echo "# 1. Install CIAgent and clone the demo agents"
type_cmd "pip install ciagent"
echo "Successfully installed ciagent-0.5.1"
pause 0.8

type_cmd "git clone https://github.com/suniel12/DemoAgents.git && cd DemoAgents/examples/support-router"
echo "Cloning into 'DemoAgents'... done."
cd "$DEMO_DIR"
pause 1

# ── Step 2: Run tests (all green) ──
echo ""
echo "# 2. Run the support router tests — 56 tests, no API keys needed"
type_cmd "pytest tests/ -v"
pause 0.3

conda run -n ciagent python -m pytest tests/ -v --tb=short 2>&1 | grep -E "(::.*PASSED|passed in)" | tail -15
echo ""
conda run -n ciagent python -m pytest tests/ -v --tb=short 2>&1 | grep "passed"
pause 3

# ── Step 3: Break it ──
echo ""
echo "# 3. Break it: remove Account Agent from the routing handoffs"
type_cmd "vim support_router/agents/triage.py"
pause 0.5

# Make the actual edit
cp "$TRIAGE" "$TRIAGE.bak"
python3 -c "
content = open('$TRIAGE').read()
content = content.replace(
    'handoffs=[billing_agent, technical_agent, account_agent, general_agent],',
    'handoffs=[billing_agent, technical_agent, general_agent],'
)
open('$TRIAGE', 'w').write(content)
"

echo ""
echo "  # Before:"
echo "  handoffs=[billing_agent, technical_agent, account_agent, general_agent]"
echo ""
echo "  # After:"
echo "  handoffs=[billing_agent, technical_agent, general_agent]"
echo "  #                                        ^^^^^^^^^^^^^^ removed!"
pause 2

# ── Step 4: Run tests again (failures!) ──
echo ""
echo "# 4. Run tests again — CIAgent catches the routing drift"
type_cmd "pytest tests/test_routing.py -v"
pause 0.3

conda run -n ciagent python -m pytest tests/test_routing.py -v --tb=line 2>&1 | grep -E "(::.*PASSED|::.*FAILED|passed|failed)" | head -30
pause 3

echo ""
echo "  CIAgent caught it: 6 tests failed."
echo "  Account queries can no longer reach Account Agent."
echo "  \"Tool transfer_to_account_agent not found in agent Triage Agent\""
pause 3

# ── Step 5: Revert ──
echo ""
echo "# 5. Revert and you're back to green"
type_cmd "git checkout -- support_router/"
mv "$TRIAGE.bak" "$TRIAGE" 2>/dev/null || true
pause 0.5

type_cmd "pytest tests/ -q"
conda run -n ciagent python -m pytest tests/ -q --tb=no 2>&1 | grep "passed"
pause 2

echo ""
echo ""
echo "  Code changes. Prompt changes. Model swaps. Tool path drift."
echo "  CIAgent catches it before you push."
echo ""
pause 3

echo "  pip install ciagent"
echo "  Pytest-native. Get started without API keys. Apache 2.0."
echo "  github.com/suniel12/CIAgent"
echo ""
pause 3
