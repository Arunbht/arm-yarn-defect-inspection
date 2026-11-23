#!/usr/bin/env bash
# Automatically launches side-by-side split screen using tmux (if available)
ROOT="$(cd "$(dirname "$0")" && pwd)"

if command -v tmux >/dev/null 2>&1; then
    SESSION="yarn_inspection"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -n "YarnVision"
    
    # Left pane: Camera vision stream
    tmux send-keys -t "$SESSION:0.0" "cd \"$ROOT\" && (source rpi_env/bin/activate 2>/dev/null || source ../llmenv/bin/activate 2>/dev/null || true); python3 inspect_live.py -x 1.5" C-m
    
    # Right pane: Llama operator advisory monitor
    tmux split-window -h -t "$SESSION:0"
    tmux send-keys -t "$SESSION:0.1" "cd \"$ROOT\" && python3 view_advisory.py" C-m
    
    # Attach to session
    tmux attach-session -t "$SESSION"
else
    echo "=========================================================="
    echo "  Strategy 3: Split-Terminal Mode                         "
    echo "=========================================================="
    echo "  To view Vision & Llama side-by-side, open two terminals:"
    echo ""
    echo "  Terminal 1 (Left - Live Camera Vision):"
    echo "    cd $ROOT"
    echo "    python3 inspect_live.py -x 1.5"
    echo ""
    echo "  Terminal 2 (Right - GenAI Operator Guidance):"
    echo "    cd $ROOT"
    echo "    python3 view_advisory.py"
    echo "=========================================================="
fi
