CPU tests for `scripts/kvq/glm53_shared_kv_startup_patch.py`, run against a v6 source tree the patch
was applied to (rebuild v6 from fork fc91d24 + the three recipes, then
`KVQ_SHARED_KV_PATCH_ROOT=<tree>/python/sglang/srt python3 scripts/kvq/glm53_shared_kv_startup_patch.py`):

    for t in scripts/kvq/tests/t*.py; do PYTHONPATH=<tree>/python python3 $t; done

Needs torch (CPU) and prometheus_client. t1-t5 are the shared-kv recipe's tests; t6/t7 cover the
port additions (per-rank mamba keys, non-owner adoption, transfer-page hash chain); t8 the
self/peer attribution counters.
