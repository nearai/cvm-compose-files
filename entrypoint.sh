#!/bin/bash
# Reads /sgl-workspace/tuned_config.json (baked in by build.sh) and starts
# sglang serve with the optimal parameters. Falls back to safe defaults.
set -euo pipefail

CONFIG="/sgl-workspace/tuned_config.json"
MODEL_PATH=$(cat /models/.model_path 2>/dev/null || echo "")

if [ -z "$MODEL_PATH" ]; then
    echo "[entrypoint] ERROR: /models/.model_path not found. Was the model baked in?" >&2
    exit 1
fi

# Read tuned config or use defaults
if [ -f "$CONFIG" ]; then
    echo "[entrypoint] Loading tuned config from $CONFIG"
    MEM_FRACTION=$(python3 -c "import json; print(json.load(open('$CONFIG'))['mem_fraction_static'])")
    MAX_RUNNING=$(python3 -c "import json; print(json.load(open('$CONFIG'))['max_running_requests'])")
    DECODE_STEPS=$(python3 -c "import json; print(json.load(open('$CONFIG'))['num_continuous_decode_steps'])")
    TP=$(python3 -c "import json; print(json.load(open('$CONFIG'))['tp'])")
else
    echo "[entrypoint] No tuned config found, using defaults"
    MEM_FRACTION=0.85
    MAX_RUNNING=55
    DECODE_STEPS=5
    TP=8
fi

# Allow env var overrides
MEM_FRACTION="${SGLANG_MEM_FRACTION:-$MEM_FRACTION}"
MAX_RUNNING="${SGLANG_MAX_RUNNING:-$MAX_RUNNING}"
DECODE_STEPS="${SGLANG_DECODE_STEPS:-$DECODE_STEPS}"
TP="${SGLANG_TP:-$TP}"
PORT="${SGLANG_PORT:-8000}"
HOST="${SGLANG_HOST:-0.0.0.0}"

echo "[entrypoint] Model:       /models/$MODEL_PATH"
echo "[entrypoint] TP:          $TP"
echo "[entrypoint] mem_frac:    $MEM_FRACTION"
echo "[entrypoint] max_running: $MAX_RUNNING"
echo "[entrypoint] decode_steps: $DECODE_STEPS"
echo "[entrypoint] port:        $PORT"

exec python3 -m sglang.launch_server \
    --model-path "/models/$MODEL_PATH" \
    --tp "$TP" \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    --num-continuous-decode-steps "$DECODE_STEPS" \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --context-length 200000 \
    --enable-cache-report \
    --enable-metrics \
    --log-requests-level 0 \
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
    --port "$PORT" \
    --host "$HOST" \
    "$@"
