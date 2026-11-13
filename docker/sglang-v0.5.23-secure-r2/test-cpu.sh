#!/usr/bin/env bash
# CPU-only checks for the SGLang v0.5.23 secure image (no GPU needed). Run inside the built image:
#   docker run --rm --runtime runc --entrypoint bash <image> /usr/share/nearai/sglang-v0523-secure/test-cpu.sh
# Every check must pass; the script stops at the first failure.
# Step 6 needs network: about 50 upstream tests fetch Hugging Face configs (DeepSeek-V3.2 and others). Under
# --network none they fail the same way on stock v0.5.23, but they are not in the known-failures list.
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T="$RECIPE_DIR/tests"
cd /sgl-workspace/sglang
SRT="$PWD/python/sglang/srt"
# Upstream test utilities index the first visible-device digit: use an unavailable ordinal.
export CUDA_VISIBLE_DEVICES=9

step() { echo "== $1"; }

step "1 admin auth + credential redaction (admin-auth-redaction; upstream #41322 + #48518)"
python3 "$T/test_admin_auth_redaction.py" --root "$SRT" --live | tail -n 1

step "2 customer-data log redaction (customer-data-log-redaction)"
python3 "$T/test_customer_data_log_redaction.py" --root "$SRT"

step "3 upstream unit tests the patches add or edit (rid-abort-exact-match, admin-auth-redaction, log-format-followup)"
SGLANG_USE_CPU_ENGINE=1 PYTHONPATH="$T" python3 -m pytest -q -p no:cacheprovider -p no:warnings -p near_stub_plugin \
  test/registered/unit/managers/test_abort_rid_match.py \
  test/registered/unit/entrypoints/openai/test_serving_chat.py \
  test/registered/unit/constrained/test_grammar_manager.py \
  test/registered/unit/mem_cache/test_dllm_fdfo_kv_reuse.py \
  test/registered/unit/entrypoints/test_server_info.py \
  test/registered/unit/server_args/test_server_args.py \
  test/registered/unit/function_call/test_unknown_tool_name.py

step "4 security backports (patches 05-14): recipe checks + the upstream unit tests the backports add or edit"
SGLANG_USE_CPU_ENGINE=1 PYTHONPATH="$T" python3 -m pytest -q -p no:cacheprovider -p no:warnings -p near_stub_plugin \
  "$T/test_security_backports.py" \
  test/registered/unit/entrypoints/openai/test_protocol.py \
  test/registered/unit/entrypoints/openai/test_serving_completions.py \
  test/registered/unit/entrypoints/openai/test_transcription_upload_limit.py \
  test/registered/unit/managers/test_io_struct.py \
  test/registered/unit/managers/test_sampling_mask_validation.py \
  test/registered/unit/managers/test_expert_distribution_control.py \
  test/registered/unit/multimodal/test_precomputed_embedding_validation.py \
  test/registered/unit/beam_search/test_beam_search_core.py \
  test/registered/unit/sampling/test_penaltylib.py \
  test/registered/unit/sampling/test_sampling_params.py \
  test/registered/unit/spec/test_spec_utils_traverse_tree.py \
  test/registered/unit/model_executor/model_runner_components/test_weight_exporter.py

step "5 sglang-diffusion patches 15-16 (media URL policy, operator-only request fields)"
MMG=python/sglang/multimodal_gen/test/unit
python3 -m pytest -q -p no:cacheprovider -p no:warnings "$T/test_diffusion_security.py" \
  $MMG/test_openai_security.py $MMG/test_runtime_utils_regressions.py $MMG/test_openai_image_api.py \
  $MMG/test_video_api_profiling.py $MMG/test_perf_dump_write.py $MMG/test_hf_diffusers_utils.py \
  $MMG/test_multiple_task_types.py

step "6 upstream unit suites around the patches: no failure beyond stock v0.5.23's own"
# Stock v0.5.23 already fails/errors some of these on a CPU-only container (collection errors,
# CUDA-only fixtures); tests/stock-v0523-known-failures.txt is that list, recorded on the pinned base
# image (same list as docker/sglang-v0.5.23-glm53-r2). Any other failing test id is a regression from the patches.
# A few upstream multimodal tests read local media files on purpose; media-local-path-reject (09)
# refuses those unless SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1. The default (off) is covered by
# tests/test_security_backports.py, so these suites run with the opt-in.
set +e
SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1 \
SGLANG_USE_CPU_ENGINE=1 PYTHONPATH="$T" python3 -m pytest -q -p no:cacheprovider -p no:warnings -p near_stub_plugin \
  -rfE --continue-on-collection-errors \
  test/registered/unit/mem_cache test/registered/unit/managers test/registered/unit/entrypoints \
  test/registered/unit/constrained test/registered/unit/multimodal test/registered/unit/function_call \
  > /tmp/near-suites.txt 2>&1
rc=$?
set -e
tail -n 1 /tmp/near-suites.txt
# pytest rc 0 = all passed, 1 = some tests failed (compared below). Anything else (2 interrupted,
# 3 internal error, 4 usage error, 5 nothing collected) is a broken run, never a pass.
if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
  echo "step 6: pytest exited $rc (not a test-failure result): failing the gate" >&2; exit 1
fi
if ! tail -n 1 /tmp/near-suites.txt | grep -Eq '[0-9]+ (passed|failed)'; then
  echo "step 6: no pytest summary line: failing the gate" >&2; exit 1
fi
# An all-green run has no FAILED/ERROR lines: grep then exits 1, which is fine here.
{ grep -E '^(FAILED|ERROR) test/' /tmp/near-suites.txt || true; } | sed 's/ - .*//' | sort -u > /tmp/near-fails.txt
new=$(comm -23 /tmp/near-fails.txt <(sort -u "$T/stock-v0523-known-failures.txt"))
if [ -n "$new" ]; then
  echo "regressions vs stock v0.5.23:"; echo "$new"; exit 1
fi

echo "SGLang v0.5.23 secure image CPU checks passed"
