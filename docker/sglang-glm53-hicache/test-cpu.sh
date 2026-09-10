#!/usr/bin/env bash
set -euo pipefail
cd /sgl-workspace/sglang
# This pinned upstream test utility indexes the first visible-device digit.
# Use an unavailable ordinal, with the CPU container's runtime set to runc.
export CUDA_VISIBLE_DEVICES=9
python3 -m pytest -q -p no:cacheprovider \
  test/registered/unit/mem_cache/test_hybrid_pool_assembler.py \
  test/registered/unit/mem_cache/test_hybrid_dsa_hicache.py \
  test/registered/unit/mem_cache/test_compressed_dsa_checkpoint_roundtrip.py \
  test/registered/unit/mem_cache/test_hicache_draft_plan.py \
  test/registered/unit/mem_cache/test_unified_cache_linker.py
