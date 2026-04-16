#!/bin/bash
# Reads /sgl-workspace/tuned_config.json (baked in by build.sh) and starts
# sglang serve with the optimal parameters. Falls back to safe defaults.
set -euo pipefail

CONFIG="/sgl-workspace/tuned_config.json"

# Model path: env var > baked-in > default
MODEL_PATH="${SGLANG_MODEL_PATH:-$(cat /models/.model_path 2>/dev/null || echo "")}"
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="zai-org/GLM-5.1-FP8"
fi

# Read tuned config or use defaults
if [ -f "$CONFIG" ]; then
    echo "[entrypoint] Loading tuned config from $CONFIG"
    eval "$(python3 << 'PYEOF'
import json, sys
try:
    c = json.load(open("/sgl-workspace/tuned_config.json"))
    print(f"MEM_FRACTION={c.get('mem_fraction_static', 0.88)}")
    print(f"MAX_RUNNING={c.get('max_running_requests', 55)}")
    print(f"DECODE_STEPS={c.get('num_continuous_decode_steps', 5)}")
    print(f"TP={c.get('tp', 8)}")
except Exception as e:
    print(f"MEM_FRACTION=0.88", file=sys.stderr)
    print(f"MAX_RUNNING=55")
    print(f"DECODE_STEPS=5")
    print(f"TP=8")
PYEOF
)"
else
    echo "[entrypoint] No tuned config found, using defaults"
    MEM_FRACTION=0.88
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

echo "[entrypoint] Model:        $MODEL_PATH"
echo "[entrypoint] TP:           $TP"
echo "[entrypoint] mem_frac:     $MEM_FRACTION"
echo "[entrypoint] max_running:  $MAX_RUNNING"
echo "[entrypoint] decode_steps: $DECODE_STEPS"
echo "[entrypoint] port:         $PORT"

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
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
