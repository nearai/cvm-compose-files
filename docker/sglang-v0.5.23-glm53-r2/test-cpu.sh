#!/usr/bin/env bash
# CPU-only checks for the GLM-5.3 v0.5.23 image (no GPU needed). Run inside the built image:
#   docker run --rm --runtime runc --entrypoint bash <image> /usr/share/nearai/glm53-v0523/test-cpu.sh
# Every check must pass; the script stops at the first failure.
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T="$RECIPE_DIR/tests"
cd /sgl-workspace/sglang
SRT="$PWD/python/sglang/srt"
# Upstream test utilities index the first visible-device digit: use an unavailable ordinal.
export CUDA_VISIBLE_DEVICES=9

step() { echo "== $1"; }

step "1 event-loop stall dump (event-loop-stall-dump)"
python3 -W ignore::DeprecationWarning "$T/test_stall_dump.py" \
  --module "$SRT/utils/event_loop_stall_dump.py" --out /tmp/stall_report.txt > /dev/null

step "2 ghost prefix cache (ghost-prefix-cache, opt-in)"
python3 "$T/test_ghost_cache.py" --module "$SRT/observability/ghost_cache.py" \
  --aggregator "$SRT/observability/ghost_aggregator.py"

step "3 KV tier metrics (kv-tier-metrics, opt-in), alone and under the upstream unified-cache tests in strict mode"
python3 "$T/test_kv_tier_metrics.py" --module "$SRT/observability/kv_tier_metrics.py" \
  --tree-core "$SRT/mem_cache/unified_cache/unified_tree_core.py"
SGLANG_KV_TIER_METRICS=1 SGLANG_KV_TIER_METRICS_FORCE=1 SGLANG_KV_TIER_METRICS_STRICT=1 \
SGLANG_KV_TIER_METRICS_RESYNC_S=0 SGLANG_USE_CPU_ENGINE=1 PYTHONPATH="$T" \
  python3 -m pytest -q -p no:cacheprovider -p no:warnings -p near_stub_plugin -p kvtm_check_plugin \
  test/registered/unit/mem_cache/test_unified_radix_allocation_eviction.py \
  test/registered/unit/mem_cache/test_unified_radix_hicache_dispatch.py \
  test/registered/unit/mem_cache/test_unified_hicache_regressions.py \
  test/registered/unit/mem_cache/unified_radix_cache

step "4 self-profiler (near-self-profile, opt-in)"
python3 "$T/test_self_profile.py" --module "$SRT/utils/near_self_profile.py" \
  --scheduler "$SRT/managers/scheduler.py" \
  --profiler-manager "$SRT/managers/scheduler_components/profiler_manager.py" | tail -n 1

step "5 FP8 NoPE DSA KV paths (fp8kv-flashmla)"
python3 "$T/test_fp8kv_paths.py" --root "$SRT" | tail -n 1

step "6 preprocess pool (preprocess-pool, opt-in)"
python3 "$T/test_preprocess_pool.py" "$SRT/managers/preprocess_pool.py"

step "7 tool-schema size cap (tool-schema-depth-cap, opt-in)"
python3 "$T/test_tool_schema_guard.py" --module "$SRT/utils/tool_schema_guard.py" \
  --serving-chat "$SRT/entrypoints/openai/serving_chat.py"

step "8 KDA host-side token count (upstream #39688; kda-zero-sync dropped)"
python3 "$T/test_kda_zero_sync.py" "$SRT/layers/attention/linear/kda_backend.py" \
  "$SRT/layers/attention/hybrid_linear_attn_backend.py"

step "9 Q8 DSA prefill on the 656 B FP8 NoPE row (dsa-q8-flashmla-sparse)"
python3 "$T/test_dsa_q8_flashmla_sparse.py" "$SRT/layers/attention/dsa_backend.py" \
  "$SRT/layers/attention/dsa/dsa_backend_kpool.py"

step "10 DSA indexer query split (dsa-indexer-qsplit, opt-in)"
python3 "$T/test_dsa_indexer_qsplit.py" "$SRT/layers/attention/dsa/dsa_indexer_kpool.py"

step "11 admin auth + credential redaction (admin-auth-redaction; upstream #41322 + #48518)"
python3 "$T/test_admin_auth_redaction.py" --root "$SRT" --live | tail -n 1

step "12 customer-data log redaction (customer-data-log-redaction)"
python3 "$T/test_customer_data_log_redaction.py" --root "$SRT"

step "13 upstream unit tests the series adds or edits"
SGLANG_USE_CPU_ENGINE=1 PYTHONPATH="$T" python3 -m pytest -q -p no:cacheprovider -p no:warnings -p near_stub_plugin \
  test/registered/unit/entrypoints/openai/test_thinking_token_budget.py \
  test/registered/unit/entrypoints/openai/test_serving_chat.py \
  test/registered/unit/function_call/test_unknown_tool_name.py \
  test/registered/unit/managers/test_scheduler_chunked_req_gate.py \
  test/registered/unit/managers/test_batch_result_processor_hidden_states.py \
  test/registered/unit/utils/test_glm5_next_perception_gate.py \
  test/registered/unit/managers/test_abort_rid_match.py \
  test/registered/unit/constrained/test_grammar_manager.py \
  test/registered/unit/entrypoints/openai/test_serving_base_event_loop.py \
  test/registered/unit/entrypoints/test_server_info.py \
  test/registered/unit/managers/test_tokenizer_manager_event_loop.py \
  test/registered/unit/mem_cache/test_dllm_fdfo_kv_reuse.py \
  test/registered/unit/mem_cache/test_hicache_draft_plan.py \
  test/registered/unit/mem_cache/test_hicache_hcc_managed_allocator.py \
  test/registered/unit/mem_cache/test_startup_ram_budget.py \
  test/registered/unit/server_args/test_server_args.py

step "14 security backports (patches 27-36): recipe checks + the upstream unit tests the backports add or edit"
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

step "15 upstream unit suites around the series: no failure beyond stock v0.5.23's own"
# Stock v0.5.23 already fails/errors some of these on a CPU-only container (collection errors,
# CUDA-only fixtures); tests/stock-v0523-known-failures.txt is that list, recorded on the pinned base
# image. Any other failing test id is a regression from the series. One entry is order-dependent upstream:
# test_http_server_liveness::test_ping_... calls /ping, which reads the module-global server state the test
# never sets; it passes only when an earlier test leaves that state behind (it fails alone on stock).
# A few upstream multimodal tests read local media files on purpose; media-local-path-reject (31)
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
  echo "step 15: pytest exited $rc (not a test-failure result): failing the gate" >&2; exit 1
fi
if ! tail -n 1 /tmp/near-suites.txt | grep -Eq '[0-9]+ (passed|failed)'; then
  echo "step 15: no pytest summary line: failing the gate" >&2; exit 1
fi
# An all-green run has no FAILED/ERROR lines: grep then exits 1, which is fine here.
{ grep -E '^(FAILED|ERROR) test/' /tmp/near-suites.txt || true; } | sed 's/ - .*//' | sort -u > /tmp/near-fails.txt
new=$(comm -23 /tmp/near-fails.txt <(sort -u "$T/stock-v0523-known-failures.txt"))
if [ -n "$new" ]; then
  echo "regressions vs stock v0.5.23:"; echo "$new"; exit 1
fi

echo "GLM-5.3 v0.5.23 image CPU checks passed"
