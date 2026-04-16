#!/bin/bash
# Build a fully self-contained SGLang GLM-5/5.1 image.
#
# Must run on a GPU host. Produces an image with model weights,
# autotuned kernels, and optimal serve parameters baked in.
#
# Usage:
#   ./build.sh zai-org/GLM-5.1-FP8 nearaidev/sglang:glm5.1-tuned
#   ./build.sh zai-org/GLM-5-FP8   nearaidev/sglang:glm5-tuned
#
# Optional env vars:
#   HF_TOKEN          — HuggingFace token for gated models
#   SGLANG_COMMIT     — SGLang git commit (default: ce31934ca80e)
#   TP                — tensor parallel size (default: 8)
#   PHASES            — warmup phases (default: "deepgemm memory bench")
#   SKIP_WARMUP       — set to 1 to skip warmup (just build base + model)
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model-path> <image-tag>}"
IMAGE_TAG="${2:?Usage: $0 <model-path> <image-tag>}"
SGLANG_COMMIT="${SGLANG_COMMIT:-ce31934ca80e}"
TP="${TP:-8}"
PHASES="${PHASES:-deepgemm memory bench}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BASE_TAG="${IMAGE_TAG}-base"
BAKE_CONTAINER="sglang-bake-$$"
BUILD_DIR=$(mktemp -d)

trap 'rm -rf "$BUILD_DIR"' EXIT

echo "============================================================"
echo "  Model:    $MODEL_PATH"
echo "  Image:    $IMAGE_TAG"
echo "  Commit:   $SGLANG_COMMIT"
echo "  TP:       $TP"
echo "  Phases:   $PHASES"
echo "============================================================"

# ── Step 1: Clone SGLang ─────────────────────────────────────────────────────
echo ""
echo ">>> Step 1/4: Cloning SGLang at $SGLANG_COMMIT..."
git clone --depth 200 https://github.com/sgl-project/sglang.git "$BUILD_DIR/sglang"
cd "$BUILD_DIR/sglang"
git checkout "$SGLANG_COMMIT"

# Copy our files into the build context
cp "$SCRIPT_DIR/warmup.py" "$BUILD_DIR/sglang/warmup.py"
cp "$SCRIPT_DIR/entrypoint.sh" "$BUILD_DIR/sglang/entrypoint.sh"
cp "$SCRIPT_DIR/Dockerfile.sglang-glm5" "$BUILD_DIR/sglang/Dockerfile"

# ── Step 2: Build base image (no GPU required) ───────────────────────────────
echo ""
echo ">>> Step 2/4: Building base image..."
HF_TOKEN_ARG=""
if [ -n "${HF_TOKEN:-}" ]; then
    HF_TOKEN_ARG="--build-arg HF_TOKEN=$HF_TOKEN"
fi

docker build \
    --build-arg MODEL_PATH="$MODEL_PATH" \
    $HF_TOKEN_ARG \
    -t "$BASE_TAG" \
    "$BUILD_DIR/sglang"

echo "Base image built: $BASE_TAG"

if [ "${SKIP_WARMUP:-0}" = "1" ]; then
    echo "SKIP_WARMUP=1, tagging base as final."
    docker tag "$BASE_TAG" "$IMAGE_TAG"
    echo "Done: $IMAGE_TAG (no warmup)"
    exit 0
fi

# ── Step 3: Warmup + tune (GPU required) ─────────────────────────────────────
echo ""
echo ">>> Step 3/4: Running warmup & auto-tuning (this takes a while)..."
docker run \
    --gpus all \
    --ipc=host \
    --name "$BAKE_CONTAINER" \
    "$BASE_TAG" \
    python3 /sgl-workspace/warmup.py \
        --model-path "/models/$MODEL_PATH" \
        --tp "$TP" \
        --phases $PHASES

# ── Step 4: Commit the baked container as the final image ────────────────────
echo ""
echo ">>> Step 4/4: Committing tuned image..."
docker commit \
    -c 'CMD ["/sgl-workspace/entrypoint.sh"]' \
    -c 'ENV HF_HUB_OFFLINE=1' \
    "$BAKE_CONTAINER" \
    "$IMAGE_TAG"

docker rm "$BAKE_CONTAINER"
docker rmi "$BASE_TAG" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  Done: $IMAGE_TAG"
echo ""
echo "  Test locally:"
echo "    docker run --gpus all --ipc=host -p 8000:8000 $IMAGE_TAG"
echo ""
echo "  Use in docker-compose:"
echo "    services:"
echo "      glm:"
echo "        image: $IMAGE_TAG"
echo "        runtime: nvidia"
echo "        ipc: host"
echo "        command: /sgl-workspace/entrypoint.sh"
echo "============================================================"
