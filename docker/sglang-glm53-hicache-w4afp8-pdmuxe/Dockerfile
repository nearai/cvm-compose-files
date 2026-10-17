# syntax=docker/dockerfile:1.7
FROM docker.io/nearaidev/sglang@sha256:3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash HiCache W4AFP8 engine, PD-Multiplexing + EAGLE canary"
LABEL org.opencontainers.image.description="HCC-safe HiCache engine with admission reserve v10, the W4AFP8 loader fix, the unconditional chunked-prefill pool clamp, the opt-in DSA indexer query split, request preprocessing and the multimodal shared-memory copy moved off the HTTP event loop, an event-loop stall dump, an opt-in ghost prefix cache that measures would-be KV hits from keyed page hashes, opt-in rank-0 KV tier (VRAM/DRAM) eviction metrics, an opt-in one-shot self-profiler that prints a compact NEAR_PROFILE decode-step summary to stdout, FP8 KV cache support for GLM's NoPE DSA (active only with --kv-cache-dtype fp8_e4m3 on flashmla_kv), an opt-in process pool with a per-request deadline for chat preprocessing, an opt-in tool-schema size cap, and an opt-in PD-Multiplexing (pdmux) + EAGLE path for GLM-5.3 (v7 plus one patch; inert unless --enable-pdmux)"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/nearaidev/sglang"
LABEL org.opencontainers.image.base.digest="sha256:3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-hicache-w4afp8-pdmuxe"
LABEL nearai.build.recipe="docker/sglang-glm53-hicache-w4afp8-pdmuxe/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_revision="fc91d2403cc210a86bd5fc715c6609f58c932e5a"
LABEL nearai.sglang.admission_reserve="v10"
LABEL nearai.sglang.admission_reserve_revision="fb944723035d6c8b8135fd3c6ffbdc21a4615409"
LABEL nearai.sglang.hicache_fix_revision="230102db838cbbe6eca7952c729518ee6d93c3a9"
LABEL nearai.sglang.pooled_transfers="v1-opt-in"
LABEL nearai.sglang.hcc_host_memory="v4-cuda-hostalloc-opt-in"
LABEL nearai.sglang.hcc_managed_memory="v3-host-prefetch-opt-in"
LABEL nearai.sglang.w4afp8_loader_patch="modules-to-not-convert-v1"
LABEL nearai.sglang.chunked_prefill_pool_clamp="v1"
LABEL nearai.sglang.dsa_indexer_qsplit="v1-opt-in"
LABEL nearai.sglang.request_preprocessing_off_loop="pr30771-60a56aa"
LABEL nearai.sglang.mm_shm_wrap_off_loop="v1"
LABEL nearai.sglang.event_loop_stall_dump="v1-on-30s"
LABEL nearai.sglang.ghost_prefix_cache="v1-opt-in"
LABEL nearai.sglang.kv_tier_metrics="v1-opt-in"
LABEL nearai.sglang.self_profile="v1-opt-in"
LABEL nearai.sglang.fp8_kv_nope_dsa="v1"
LABEL nearai.sglang.preprocess_pool="v1-opt-in"
LABEL nearai.sglang.tool_schema_depth_cap="v1-opt-in"
LABEL nearai.sglang.pdmux_eagle="v2-opt-in-canary-draft-graphs"
LABEL nearai.sglang.pdmux_config="/etc/glm53/pdmux-ov96-l4.yaml"
LABEL nearai.security_remediation="CVE-2026-102268:pyjwt-2.14.0"

COPY Dockerfile \
     PROVENANCE \
     README.md \
     SHA256SUMS \
     apply-patches.py \
     chunked-prefill-pool-clamp.diff \
     dsa-indexer-qsplit.diff \
     event-loop-stall-dump.diff \
     fp8kv-flashmla.diff \
     ghost-prefix-cache.diff \
     kv-tier-metrics.diff \
     modules-to-not-convert.diff \
     pdmux-eagle.diff \
     pdmux-ov96-l4.yaml \
     pdmux-canary-argv.json \
     near-self-profile.diff \
     preprocess-pool.diff \
     sglang-pr30771.diff \
     shm-off-loop.diff \
     source-manifest.json \
     tool-schema-depth-cap.diff \
     test-cpu.sh \
     test_stall_dump.py \
     test_ghost_cache.py \
     test_kv_tier_metrics.py \
     test_self_profile.py \
     test_preprocess_pool.py \
     test_tool_schema_guard.py \
     test_fp8kv_paths.py \
     test_eagle_pdmux.py \
     test_eagle_pdmux_draft_graphs.py \
     test_glm5_next_pdmux.py \
     test_pdmux_backlog.py \
     test_pdmux_group_alias.py \
     test_pdmux_hicache_events.py \
     test_pdmux_overlap_streams.py \
     test_pdmux_scheduler.py \
     test_w4afp8_pdmux_buffers.py \
     kvtm_check_plugin.py \
     security-requirements.txt \
     /usr/share/nearai/glm53-hicache-w4afp8-pdmuxe/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-hicache-w4afp8 \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# The PD-Multiplexing config for the one-replica canary, at a fixed path the compose file passes as
# --pdmux-config-path. sha256sum --check above has already verified the file; it is read-only here.
RUN install -D -m 0444 /usr/share/nearai/glm53-hicache-w4afp8-pdmuxe/pdmux-ov96-l4.yaml /etc/glm53/pdmux-ov96-l4.yaml

# Security remediation: the base ships PyJWT 2.13.0 (CVE-2026-102268, fixed in 2.14.0). It is only a
# dependency of msal; SGLang never imports it. Install the fixed wheel by exact hash, without deps.
RUN /opt/sglang/bin/python3 -m pip install --no-cache-dir --no-deps --require-hashes \
        -r /usr/share/nearai/glm53-hicache-w4afp8-pdmuxe/security-requirements.txt \
    && /opt/sglang/bin/python3 -c "import jwt; assert jwt.__version__ == '2.14.0', jwt.__version__"

# The loader and pool-clamp patches are required for the W4AFP8 image and are unconditional. The DSA
# indexer query split is inert unless SGLANG_DSA_INDEXER_QSPLIT=1. The two event-loop patches are
# unconditional: upstream sglang PR #30771 runs chat template rendering and tokenization in a
# request-preprocessor thread, and shm-off-loop moves the dispatch-time /dev/shm copy of multimodal
# features to its own thread, so /health keeps answering during multi-MB and image-heavy requests.
# The event-loop stall dump is on by default: after 30 s without a loop tick it writes every
# thread's stack to stderr with the prefix [event-loop-stall]. SGLANG_EVENT_LOOP_STALL_DUMP_SECS=0
# disables it; SGLANG_EVENT_LOOP_STALL_DUMP_REPEAT_SECS (default 60) spaces the repeats.
# The ghost prefix cache is inert unless SGLANG_GHOST_CACHE=1: then TP rank 0 hashes each finished
# request's 64-token pages with a per-process random key on a background thread and exports only
# aggregate counters (sglang:ghost_*): prompt tokens seen before, by LRU reuse distance and by age.
# With a shared SGLANG_GHOST_CACHE_KEY_FILE and SGLANG_GHOST_CACHE_SOCKET the replicas of one CVM also
# send sampled digests to a sidecar (python3 -m sglang.srt.observability.ghost_aggregator) that keeps
# a pooled ghost cache and counts tokens seen before only on another replica.
# The KV tier metrics are inert unless SGLANG_KV_TIER_METRICS=1: then TP rank 0 counts radix-tree
# VRAM evictions (demoted to DRAM vs deleted), DRAM evictions (duplicate of VRAM vs the only copy),
# DRAM->VRAM load-backs and their idle-time buckets, and at most every SGLANG_KV_TIER_METRICS_RESYNC_S
# (default 30) walks the tree for the sglang:kv_tier_* VRAM / DRAM-duplicate / DRAM-only gauges.
# The self-profiler is inert unless NEAR_SELF_PROFILE=1: then TP rank 0 profiles
# NEAR_SELF_PROFILE_STEPS (default 50) scheduler steps once per process, starting at the first decode
# batch after NEAR_SELF_PROFILE_AFTER_S (default 600) seconds, through the scheduler's existing
# SchedulerProfilerManager. A niced child process parses the trace, prints at most about 30 stdout
# lines prefixed "NEAR_PROFILE " and deletes /tmp/near_selfprof. NEAR_SELF_PROFILE_ACTIVITIES=cpu
# forces the CPU-only profile that is also the automatic fallback when CUPTI yields no kernel events
# (a CUPTI crash or hang is not caught: enable it on one restartable replica). Rollback: unset it.
# FP8 KV cache for GLM's NoPE DSA (fp8kv-flashmla.diff) is active only with --kv-cache-dtype fp8_e4m3
# and the flashmla_kv DSA backends: the pool row becomes 656 B (512 fp8 + 16 B scales + 128 B zero
# rope slots) instead of 1024 B, q is zero-padded 512 -> 576 and the kpool index table is padded
# with -1 to a multiple of 128. With the default bf16 KV cache every hunk is unreachable or
# arithmetically identical (see README "FP8 KV cache").
# The preprocess pool is inert unless SGLANG_PREPROCESS_WORKERS=N with N > 0: then N forked worker
# processes run chat template rendering and tokenization, and a request over
# SGLANG_PREPROCESS_TIMEOUT_S (default 60) fails alone with HTTP 422 while its worker is killed and
# replaced. Unset or 0 forks nothing and keeps the single request-preprocessor thread.
# The tool-schema cap is inert unless SGLANG_TOOL_SCHEMA_MAX_DEPTH and/or SGLANG_TOOL_SCHEMA_MAX_NODES
# is a positive integer: a tool whose parameters schema exceeds it is rejected with HTTP 400 before
# jsonschema's check_schema runs on the event loop.
# PD-Multiplexing + EAGLE (pdmux-eagle.diff, the last patch) is inert unless the server is started with
# --enable-pdmux: 19 files change, every pdmux branch is gated on that flag, and the ported tests (test-cpu.sh
# step 14) check the non-pdmux paths keep the v7 behaviour where they are covered. With it,
# --sm-group-num 3 and --pdmux-config-path /etc/glm53/pdmux-ov96-l4.yaml (baked above: prefill green context
# capped at 96 SMs, decode on the full device, at most 4 prefill layers per scheduler iteration) prefill runs
# in slices on its own stream beside decode. EAGLE 4/1/5 is supported; its draft decode and draft-extend forwards
# replay per-stream draft CUDA graphs (one set per decode stream group). SGLANG_PDMUX_EAGLE_DRAFT_GRAPHS=0 falls
# back to the eager draft forwards (the prototype behaviour). The backlog mode in multiplex/ is code only: no
# backlog_* key is in the baked YAML, it is off and must stay off (rejected in exp 45, decode worse than v7; the
# uncapped variant, backlog_max_split_forward_layers 0, deadlocked across TP ranks). The validation hook rejects
# every other speculative setup, the overlap scheduler and mixed chunking.
# SGLANG_PDMUX_EAGLE_DECODE_FENCE=0 removes an extra decode fence after each prefill's draft extend (A/B knob,
# default 1). This is a prototype: see README "PD-Multiplexing + EAGLE" before use.
# Everything inherited from the base stays opt-in: admission reserve v10 via
# SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE, and HiCache via --enable-hierarchical-cache plus
# SGLANG_HICACHE_*. On an HCC/PPCIe (TEE) host SGLANG_HICACHE_CUDA_HOST_MEMORY must stay 1; 0 selects
# the anonymous-mmap plus cudaHostRegister path, which fails with CUDA error 801.
