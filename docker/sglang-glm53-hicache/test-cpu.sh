#!/usr/bin/env bash
set -euo pipefail
cd /sgl-workspace/sglang
export CUDA_VISIBLE_DEVICES=9
python3 -m pytest -q -p no:cacheprovider \
  test/registered/unit/mem_cache/test_hicache_hcc_managed_allocator.py
