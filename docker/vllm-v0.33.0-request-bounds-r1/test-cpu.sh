#!/usr/bin/env bash
# CPU-only checks for this image; the README has the run commands. Optional step 3 needs network and a
# v0.33.0 vLLM tests/ tree at $VLLM_TESTS. The script stops at the first failure.
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T="$RECIPE_DIR/tests"
SITE=/usr/local/lib/python3.12/dist-packages
export CUDA_VISIBLE_DEVICES=

step() { echo "== $1"; }

step "1 patched files match PATCHED-FILES.sha256"
(cd "$SITE" && sha256sum --check --strict --quiet "$RECIPE_DIR/PATCHED-FILES.sha256")
echo ok

step "2 request-bound checks (tests/test_request_bounds.py; each fails on stock v0.33.0)"
(cd /tmp && python3 -m pytest -q -p no:cacheprovider -p no:warnings "$T/test_request_bounds.py")

if [ -z "${VLLM_TESTS:-}" ]; then
  echo "step 3 skipped (set VLLM_TESTS to a v0.33.0 tests/ directory to run the upstream tests)"
  echo "vLLM v0.33.0 image CPU checks passed"
  exit 0
fi

step "3 upstream unit tests the patches add or edit"
work=$(mktemp -d)
cp -a "$VLLM_TESTS" "$work/tests"
(cd "$work" && patch -p1 -s --no-backup-if-mismatch < "$T/upstream-tests.diff")
python3 -m pip install -q --no-cache-dir pytest-asyncio==1.3.0 tblib==3.2.2 pytest-timeout==2.4.0
upstream() {  # upstream <log> <pytest args...>; pytest rc 0 or 1 only (anything else is a broken run)
  local log=$1 rc; shift
  set +e
  (cd "$work" && VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS=1 python3 -m pytest -q -p no:cacheprovider \
    -p no:warnings -rfE --timeout 900 --continue-on-collection-errors "$@" > "$log" 2>&1)
  rc=$?
  set -e
  tail -n 1 "$log"
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
    echo "step 3: pytest exited $rc (not a test-failure result): failing the gate" >&2; exit 1
  fi
}
upstream /tmp/near-upstream-1.txt \
  tests/entrypoints/serve/exception_handling/test_error_sanitization.py \
  tests/test_sampling_params.py \
  tests/entrypoints/openai/responses/test_sampling_params.py \
  tests/v1/engine/test_output_processor.py \
  tests/test_request_input_bounds.py \
  tests/entrypoints/unit_tests/test_sampling_int_bounds.py \
  tests/v1/engine/test_add_request_rollback.py \
  tests/entrypoints/pooling/test_pooling_prompt_limit.py \
  tests/utils_/test_vision_rope_shape_cache.py \
  tests/multimodal/media \
  tests/test_connections.py
# Only the detokenizer tests around patch 04; the full file downloads dozens of tokenizers.
upstream /tmp/near-upstream-2.txt tests/tokenizers_/test_detokenize.py \
  -k "malformed or incomplete_utf8 or unrepresentable"
# Not run here: the chat test_video.py suite (07) needs a running server. Step 2 and
# tests/multimodal/media cover it.
{ grep -hE '^(FAILED|ERROR) tests/' /tmp/near-upstream-1.txt /tmp/near-upstream-2.txt || true; } \
  | sed 's/ - .*//' | sort -u > /tmp/near-fails.txt
# Environment-only failures, recorded on gpu-less CPU containers with network: fixtures that need
# the gated meta-llama/Llama-3.2-1B or a GPU device; they fail identically on stock v0.33.0.
new=$(comm -23 /tmp/near-fails.txt <(sort -u "$T/stock-v0330-known-failures.txt"))
if [ -n "$new" ]; then
  echo "failures not in tests/stock-v0330-known-failures.txt:"; echo "$new"; exit 1
fi

echo "vLLM v0.33.0 image CPU checks passed"
