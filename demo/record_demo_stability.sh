#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Stability Report hero recording — the "stable score / unstable system" shot.
#
# Records the zero-key bundled demo (`ciagent test --mock --runs 3`): three
# identical 88% suite scores, then the stability report showing 3/8 verdicts
# flipped with flip-source attribution. This is the artifact the README hero
# and launch post lead with.
#
# Record with (155 cols: the flip-attribution lines must not wrap;
# asciinema 3.x uses --window-size, not --cols/--rows):
#   cd AgentCI && asciinema rec demo/stability-demo.cast \
#     --window-size 155x30 --overwrite -c "./demo/record_demo_stability.sh"
#
# Convert to GIF (the ✅/❌ flip markers need an emoji font agg can rasterize;
# Apple Color Emoji and Noto COLOR Emoji are bitmap fonts agg can't draw —
# use the monochrome Noto Emoji outline font with the fontdue renderer):
#   curl -sL -o /tmp/fonts/NotoEmoji.ttf \
#     "https://github.com/google/fonts/raw/main/ofl/notoemoji/NotoEmoji%5Bwght%5D.ttf"
#   agg --font-size 14 --last-frame-duration 10 --renderer fontdue \
#     --font-dir /tmp/fonts --font-family "JetBrains Mono,Menlo,Noto Emoji" \
#     demo/stability-demo.cast demo/stability-report.gif
#
# The static PNG (demo/stability-report.png) is the GIF's final frame —
# the post/social-embed version of the same artifact.
# ─────────────────────────────────────────────────────────────────────────────

set -e

export TERM="${TERM:-xterm-256color}"
export FORCE_COLOR=1
export COLUMNS=155

# Init conda (same pattern as record_demo_router.sh)
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate agentci

# Helper: simulate typing
type_cmd() {
    echo -n "$ "
    for (( i=0; i<${#1}; i++ )); do
        echo -n "${1:$i:1}"
        sleep 0.04
    done
    echo ""
    sleep 0.4
}

type_cmd "ciagent test --mock --runs 3"
ciagent test --mock --runs 3
sleep 2
