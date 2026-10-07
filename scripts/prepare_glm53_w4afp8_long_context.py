#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run scripts/prepare_glm53_w4afp8_long_context.py --write
"""Generate the W4AFP8 + HiCache long-context file from the long-context file.

Both replicas run the gpu31 campaign-2 arm L2. Everything outside the two engines and
their truthful telemetry stays byte-identical to the source, so the long-domain routing
contract (nginx, the :8001 discovery stub, registrar, proxy pooling) cannot drift.
`--write` regenerates the target; `--check` prints a diff and exits non-zero when the
committed target is stale.
"""

import argparse
import difflib
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import glm53_observability as obs  # noqa: E402
from scripts import glm53_v8_bundle as v8  # noqa: E402

SOURCE = Path("prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml")
TARGET = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml")

CHECKPOINT: Final = "graphistry/GLM-5.3-Flash-W4AFP8"
CHECKPOINT_REVISION: Final = "99f1fa70408c52b007d4fd69e02e5a522422e755"
FP8_REVISION: Final = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
SOURCE_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7"
SOURCE_HICACHE_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c"
# Every replica runs the released glm53-hicache-w4afp8-v6 image (the #308 offloop-v3 off-loop path,
# the opt-in ghost prefix cache and KV tier metrics, and the PyJWT fix) and carries the DSA
# indexer split, which is enabled by the shared QSPLIT environment.
IMAGE: Final = "docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"
ENGINE_IMAGE_LABEL: Final = IMAGE.split(":")[-1][:12]
R2_IMAGE: Final = IMAGE
R2_ENGINE_IMAGE_LABEL: Final = ENGINE_IMAGE_LABEL
REPLICA_IMAGE: Final = {1: IMAGE, 2: R2_IMAGE}
REPLICA_IMAGE_LABEL: Final = {1: ENGINE_IMAGE_LABEL, 2: R2_ENGINE_IMAGE_LABEL}
# BOTH replicas run 8192. The 2026-09-25 canary ran the split at 16384 and inverted the lab
# result at the tail: over a 4 h steady-state window TTFT p95 was 76.9 s against r1's 47.3 s
# (1.62x), while the decode win did reproduce (ITL p90 0.87x, p50 0.97x). It was not explained by
# traffic volume (681 vs 688 req/h), cache warmth (79.7% vs 81.6% hit) or queueing (r2's average
# queue depth was LOWER, 1.23 vs 1.56). r2 did draw ~23% more uncached prefill tokens per request,
# which it absorbed for +3% median TTFT but +62% at p95 -- worse only at the tail, i.e. only on the
# largest prefills.
#
# Two candidates bite exactly there, and the canary changed both at once:
#   a) the 16384 chunk, which doubles how long one prefill chunk occupies the scheduler;
#   b) the indexer split's all-gather, which trades per-rank compute for a collective. gpu31 is
#      bare metal with CC off, so its all-gather is cheap; under CC/PPCIe memory encryption a
#      collective costs more and the trade can invert. This was flagged untested in #300.
#
# 8192 is the chunk with a stable history on this tier, so both replicas hold there while the
# split provides the crash headroom. The validator gate is one-directional -- c16384 REQUIRES the
# split, the split does not require c16384 -- so split-at-8192 is permitted by design.
CHUNKED_PREFILL_SIZE: Final = {1: 8192, 2: 8192}
SOURCE_SERVICE_PREFIX: Final = "model-sg-glm53-fp8-tp4-r"
SERVICE_PREFIX: Final = "model-sg-glm53-w4afp8-tp4-r"
PRECISION: Final = "int4-weights-fp8-activations-bf16-kv"
SOURCE_VARIANTS: Final = (
    "fc91d24-long-context-admission-reserve-disabled-hicache-disabled-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192",
    "fc91d24-long-context-admission-reserve-disabled-hicache-cuda-host-pooled-v1-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192",
)
# Both replicas carry the split at chunk 8192. The marker makes their unconditional v3
# off-loop implementation observable without introducing an activation flag. They differ in
# pdi1/pdi2 and in r2's larger HiCache host tier (host650g), which is how dashboards separate them.
# Both also carry the opt-in observability marker (obs.VARIANT_SUFFIX).
VARIANTS: Final = {
    1: "fc91d24-long-context-w4afp8-c8192-qsplit-offloop-v3-hicache-cuda-host-pooled-v1-admission-reserve-disabled"
    "-pool-clamp-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192" + obs.VARIANT_SUFFIX,
    2: "fc91d24-long-context-w4afp8-c8192-qsplit-offloop-v3-hicache-cuda-host-pooled-v1-host650g-admission-reserve-disabled"
    "-pool-clamp-pdi2-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192" + obs.VARIANT_SUFFIX,
}
DEPLOYMENT: Final = "glm53-flash-sgl-tp4"
# HiCache host-tier canary: r2's startup budget. With write_through the host tier is an inclusive
# copy of the GPU pool, so only (host - device) tokens are extra. W4AFP8 grew the device pool to
# ~3.52M tokens while 406 GiB holds ~4.99M, leaving ~1.5M extra; on 2026-09-25 host hits were ~1%
# of cached tokens on both replicas. 650 GiB holds ~8M (~4.5M extra). The gpu02 CVM had ~690 GiB
# free with both replicas at 406 GiB, so r2 at 650 still leaves ~450 GiB.
ANCHOR_BUDGET_LINE: Final = "    - SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-406GiB}\n"
R2_BUDGET_LINE: Final = "    - SGLANG_HICACHE_RAM_BUDGET=${GLM53_R2_HICACHE_RAM_BUDGET:-650GiB}\n"
PREFILL_DECODE_INTERVAL: Final = {1: 1, 2: 2}
SOURCE_DIST_INIT: Final = "127.0.0.1:29510"
DIST_INIT: Final = {1: "127.0.0.1:29510", 2: "127.0.0.1:29511"}
HICACHE_FLAGS: Final = (
    "--enable-hierarchical-cache",
    "--hicache-write-policy write_through",
    "--hicache-io-backend direct",
    "--hicache-mem-layout page_first_direct",
)
FP8_MODEL_PATH: Final = f"--model-path /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/{FP8_REVISION}"
MODEL_PATH: Final = (
    "--model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/"
    f"snapshots/{CHECKPOINT_REVISION}"
)

HEADER: Final = (
    "# GLM-5.3 Flash dedicated long-context tier (gpu02) on W4AFP8 + HiCache, generated from\n"
    "# prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml by scripts/prepare_glm53_w4afp8_long_context.py.\n"
    "# Both replicas run the W4AFP8 checkpoint graphistry/GLM-5.3-Flash-W4AFP8@99f1fa7 with\n"
    "# --max-prefill-tokens 32768, HiCache with CUDA-owned host memory and a fixed 406 GiB\n"
    "# startup host-memory budget per replica, and no admission reserve.\n"
    "#\n"
    "# Both replicas run chunk 8192 with SGLANG_DSA_INDEXER_QSPLIT=1 and pin the\n"
    "# glm53-hicache-w4afp8-v6 image (the #308 offloop-v3 no-flag path, plus the ghost cache and KV tier metrics):\n"
    f"#   {R2_IMAGE}\n"
    "# (glm53-hicache-w4afp8-v6, source 678a6b5ee3e4e7b83340f810c76530958eecc499; workflow 37505271073 SUCCESS).\n"
    "# The v3 off-loop path is unconditional: no dynamic-batch-tokenizer activation variable\n"
    "# or CLI flag is added. r1/r2 keep --prefill-decode-interval 1/2 respectively.\n"
    "#\n"
    "# WHY THE SPLIT IS ON: r1 crashed on 2026-09-25 without it. deep_gemm.fp8_mqa_logits\n"
    "# (dsa_indexer_kpool.py:919) asked for a single fp32 buffer of new_tokens x total_context\n"
    "# -- 11.96 GiB at ~392K context with an 8192 chunk -- against 11.88 GiB free. An ordinary\n"
    "# request for this tier. The split has each TP rank score 1/TP of the rows, taking that\n"
    "# allocation to roughly 3 GiB.\n"
    "#\n"
    "# THE SPLIT IS A MITIGATION, NOT THE FIX. It divides the buffer by TP; it does not bound\n"
    "# it. The same crash returns at ~4x the context, or at TP1. The fix is to call\n"
    "# _should_chunk_mqa_logits -- defined but never called, at dsa_indexer_kpool.py:862 --\n"
    "# and chunk the logits against free memory, as dsa_indexer.py already does for the\n"
    "# non-kpool path at :1165. That bounds the buffer at any context, chunk size and TP.\n"
    "#\n"
    "# WHY 8192 AND NOT 16384: the 2026-09-25 canary ran the split at chunk 16384 and regressed\n"
    "# the tail -- TTFT p95 76.9 s against r1's 47.3 s (1.62x) over a 4 h steady-state window --\n"
    "# though the decode win did reproduce (ITL p90 0.87x). Traffic volume, cache warmth and\n"
    "# queue depth did not explain it. gpu02 was rolled back. 8192 is the chunk that has run\n"
    "# stably on this tier; 16384 must never be set without the split in any case, because\n"
    "# without it a concurrent long burst left 0.04-0.65 GB free per GPU.\n"
    "# validate_glm53_prod_config.rb enforces that pairing.\n"
    "#\n"
    "# Note this leaves no unsplit control on the tier. The split's cost under CC/PPCIe is\n"
    "# therefore measured against the pre-change history, not against a live sibling. Roll out\n"
    "# with docs/glm53-w4afp8-long-context-rollout.md, one replica at a time, and watch p95.\n"
    "#\n"
    "# 2xTP2 memory-optimized replicas (docs/gpu02-glm53-2xtp2-memopt-canary.md,\n"
    "# docs/long-context-glm53-2xtp2-rollout.md): this file is shared by gpu02 and gpu23. The TP4 r1/r2\n"
    "# services are kept as-is, and four TP2 services are added to start in their place, per host and\n"
    "# per stage, through a scoped services list: -r2a (GPUs 4,5) and -r2b (6,7) replace r2; -r1a (0,1)\n"
    "# and -r1b (2,3) replace r1. proxy-glm53's pool is ${GLM53_BACKEND_URLS:-<r1,r2>}; a host sets it\n"
    "# only once it runs TP2 services (the per-host values are HOST_POOLS in the generator).\n"
    "#\n"
    "# v8-BUNDLE CANARY SLOT (docs/glm53-v8-bundle-canary.md): -r2a alone reads the ${GLM53_V8_R2A_*} override\n"
    "# variables (image, kv dtype, DSA backend, running and queued caps, extra args, environment prefix, telemetry\n"
    "# suffix). Each is empty or today's value unless gpu02's compose-manager env map sets it, so every other service\n"
    "# and gpu23 render exactly what they did without them. gpu23 must never set them; no other replica reads them.\n"
    "#\n"
    "# OBSERVABILITY: every replica (TP4 r1/r2 and TP2 r1a/r1b/r2a/r2b) enables the opt-in ghost\n"
    "# prefix cache in shared mode (SGLANG_GHOST_CACHE_REPLICA = its own name, one key and socket\n"
    "# on the in-memory ghost volume) and the KV tier metrics; the glm53-ghost-aggregator sidecar\n"
    "# (one per CVM: include it in each host's scoped service list) pools the digests and is\n"
    "# scraped like the engines. Both need the docker/sglang-glm53-hicache-w4afp8 image with\n"
    "# ghost-prefix-cache.diff and kv-tier-metrics.diff (glm53-hicache-w4afp8-v6, pinned here).\n"
    "# Do not hand-edit this file.\n"
)

HEADER_REPLACEMENTS: Final = (
    (
        "# Hand-derived from prod/GLM-5.3-Flash-SGL-TP4.yaml. Routing remains the dedicated\n"
        "# long-context contract below, while the engines intentionally form an r1 control / r2\n"
        "# HiCache experiment and therefore are not byte-identical to the canonical file.\n",
        "# Generated from the long-context file. Routing remains the dedicated long-context\n"
        "# contract below; both replicas run a W4AFP8 + HiCache engine at chunk 8192 with the\n"
        "# DSA indexer split and the offloop-v3 no-flag image at pdi1/r1 and pdi2/r2.\n",
    ),
    (
        "#   tier; every other host keeps the canonical file. The inference-proxy and routing\n"
        "#   behavior stay unchanged; only r2's engine and the two replicas' truthful telemetry\n"
        "#   variants differ for this experiment:\n",
        "#   tier; every other host keeps its own file. The inference-proxy and routing\n"
        "#   behavior stay unchanged; only the two engines and their truthful telemetry differ\n"
        "#   from the long-context file:\n",
    ),
    (
        "#   Non-HiCache engine flags remain identical between replicas: chunk 8192 OOMs in the\n"
        "#   DSA indexer under concurrent 400K+ contexts (exactly this tier's load), chunk 2048\n"
        "#   halves prefill speed, and no other per-replica flag moved the tail. Conversation\n"
        "#   affinity stays on because the prefix cache is worth ~3x in request capacity.\n",
        "#   The replicas' engine flags differ only in --dist-init-addr and\n"
        "#   --prefill-decode-interval (1 on r1, 2 on r2). Both run chunk 8192 and\n"
        "#   SGLANG_DSA_INDEXER_QSPLIT=1; both pin the offloop-v3 image. The 8192 chunk is the\n"
        "#   W4AFP8 setting: FP8 at 8192 OOMed in the DSA indexer under concurrent 400K+\n"
        "#   contexts, while W4AFP8's 2.4x larger device KV pool kept 3.7 GiB free through 8\n"
        "#   concurrent 647K-756K-token cold prefills on gpu31. 16384 was rejected then because\n"
        "#   it fell to 0.04 GiB free; the indexer split shrinks that scratch by TP and restores\n"
        "#   9.3-9.5 GB, which is what makes r2's 16384 admissible. Conversation affinity stays\n"
        "#   on because the prefix cache is worth ~3x in request capacity.\n",
    ),
    (
        "#   Experiment rollback: redeploy the prior long-context tag, or remove r2's HiCache\n"
        "#   image/command/environment override so it inherits the r1 control settings. Routing\n"
        "#   rollback remains LONG_TIER_ONLY=false + registrar restart (host rejoins the base\n"
        "#   pool while still serving the long domain), or remove the cloud-api long_context block.\n",
        "#   Engine rollback: redeploy the previous tag and file scoped to the same services,\n"
        "#   after stopping each W4AFP8 engine with this file (never rely on orphan removal;\n"
        "#   docs/glm53-w4afp8-long-context-rollout.md). Routing rollback remains\n"
        "#   LONG_TIER_ONLY=false + registrar restart (host rejoins the base pool while still\n"
        "#   serving the long domain), or remove the cloud-api long_context block.\n",
    ),
    (
        "# DSA import-cycle fixes. r1 pins the published base engine as the ordinary GPU-prefix-\n"
        "# cache control; r2 pins the published HCC-safe HiCache derivative and uses an 80%\n"
        "# startup host-memory budget across all four TP ranks by default. The deployment may\n"
        "# override that percentage with GLM53_HICACHE_RAM_BUDGET.\n",
        "# DSA import-cycle fixes. Both replicas pin a published split-capable\n"
        "# docker/sglang-glm53-hicache-w4afp8 derivative (glm53-hicache-w4afp8-v6 on every replica):\n"
        "# the HCC-safe HiCache image (CUDA-owned\n"
        "# host memory) plus the W4AFP8 loader fix and the unconditional chunked-prefill pool\n"
        "# clamp. Two replicas share the CVM's RAM, so each replica's startup host-memory\n"
        "# budget defaults to a fixed 406 GiB across its four TP ranks (the qualified L2 value;\n"
        "# a percentage resolves against MemAvailable at each start and would split unevenly).\n"
        "# GLM53_HICACHE_RAM_BUDGET overrides it for r1. r2 is the host-tier canary: 650 GiB,\n"
        "# overridable with GLM53_R2_HICACHE_RAM_BUDGET.\n",
    ),
    (
        "# Admission reserve is deliberately disabled on both replicas: reserve v10 admitted a\n"
        "# 292K-cached waiter beside an in-flight chunked prefill on gpu02 and exhausted the\n"
        "# long-context token pool. Keep the reserve environment unset until the pool-clamp\n"
        "# image is published and qualified. The published images' reserve patch is inert while\n"
        "# those variables are unset; --prefill-decode-interval 1 remains part of both otherwise\n"
        "# identical serving contracts. The official CUDA 13 SGLang image is the pinned build\n"
        "# base. The serving envelope is\n",
        "# Admission reserve is deliberately disabled on both replicas: reserve v10 admitted a\n"
        "# 292K-cached waiter beside an in-flight chunked prefill on gpu02 and exhausted the\n"
        "# long-context token pool, and the qualified arm (L2) ran without it. The image's\n"
        "# reserve patch is inert while those variables are unset; --prefill-decode-interval\n"
        "# (1 on r1, 2 on the r2 canary) is part of the serving contract. The official CUDA 13\n"
        "# SGLang image is the\n"
        "# pinned build base. The serving envelope is\n",
    ),
    (
        "# intentionally conservative: BF16 KV, 0.80 static memory, 32 running requests, a\n"
        "# bounded 8-request queue, 4096-token prefill chunks, decode graphs capped at batch 32,\n"
        "# TileLang DSA, DeepGEMM, and adaptive EAGLE 5/1/6. This retains the full\n"
        "# 1,048,576-token model context while avoiding the late K-pool workspace exhaustion\n"
        "# observed with larger prefill chunks.\n",
        "# intentionally conservative: BF16 KV, 0.80 static memory, 32 running requests, a\n"
        "# bounded 8-request queue, 8192-token prefill chunks with --max-prefill-tokens 32768,\n"
        "# decode graphs capped at batch 32, TileLang DSA, the CUTLASS W4A8 MoE path (hence no\n"
        "# --moe-runner-backend and no FP8 --revision), and adaptive EAGLE 5/1/6. This retains\n"
        "# the full 1,048,576-token model context.\n",
    ),
)

ANCHOR_ENV_OLD: Final = (
    "    # Admission reserve stays unset pending a published, qualified pool-clamp image.\n"
    "    # --prefill-decode-interval 1 is retained independently of the inert reserve patch.\n"
    "    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n"
    "  restart: unless-stopped\n"
)
ANCHOR_ENV_NEW: Final = (
    "    # No admission reserve on the long tier (see the header); --prefill-decode-interval\n"
    "    # is set per replica, independently of the inert reserve patch.\n"
    "    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n"
    "    # HiCache host tier: a fixed 406 GiB per replica across its four TP ranks (the\n"
    "    # qualified L2 value). A percentage would resolve against MemAvailable at each start,\n"
    "    # so the replica started second would get less. r2 overrides this with its own\n"
    "    # 650 GiB canary budget; startup fails if it exceeds available RAM minus 10 GiB.\n"
    "    - SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-406GiB}\n"
    "    # CUDA-owned host memory (cudaMallocHost). Must stay 1 on TEE hosts: 0 selects\n"
    "    # cudaHostRegister, which fails with CUDA error 801 under HCC/PPCIe.\n"
    "    - SGLANG_HICACHE_CUDA_HOST_MEMORY=${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}\n"
    "    - SGLANG_HICACHE_POOLED_TRANSFERS=1\n"
    "    - SGLANG_HICACHE_STAGING_PAGES=64\n"
    "    # DSA indexer query split, on BOTH replicas. Each TP rank scores 1/TP of a prefill\n"
    "    # chunk's rows instead of all of them, so the fp32 logits buffer -- sized\n"
    "    # new_tokens x total_context -- shrinks by TP. r1 crashed on 2026-09-25 without it:\n"
    "    # deep_gemm.fp8_mqa_logits asked for 11.96 GiB with 11.88 GiB free at ~392K context\n"
    "    # (dsa_indexer_kpool.py:919). At TP4 that allocation becomes ~3 GiB.\n"
    "    # This is a MITIGATION, not the fix. It divides the buffer by TP; it does not bound\n"
    "    # it. The same crash returns at ~4x the context, or at TP1. The real fix is calling\n"
    "    # _should_chunk_mqa_logits (defined, unused, at dsa_indexer_kpool.py:862) and chunking\n"
    "    # the logits against free memory, which holds at any context, chunk size and TP.\n"
    "    - SGLANG_DSA_INDEXER_QSPLIT=1\n"
    + obs.engine_environment("r1", 4)
    + "  restart: unless-stopped\n"
)
ANCHOR_VOLUMES: Final = "  volumes:\n    - kernel_cache:/root/.cache\n    - huggingface_cache:/root/.cache/huggingface\n"


# --- 2xTP2 memory-optimized replicas (replace r2's GPUs 4-7 and, later, r1's GPUs 0-3) ---------------
# This file is shared by gpu02 and gpu23 (each host's compose-manager scopes its own services).
# The TP4 r1/r2 definitions above are KEPT unchanged so a host not yet converted keeps deploying them.
# TP2 replicas are ADDED to start in their place (2a/2b over r2's GPUs, 1a/1b over r1's), and the proxy backend list becomes
# host-overridable: a host still on TP4 never sets GLM53_BACKEND_URLS, so its effective value is unchanged.
# docs/gpu02-glm53-2xtp2-memopt-canary.md and docs/long-context-glm53-2xtp2-rollout.md are the runbooks.
R2_SERVICE: Final = f"{SERVICE_PREFIX}2"
TP2_SERVICE_PREFIX: Final = "model-sg-glm53-w4afp8-tp2-r"
# suffix -> (GPU pair, distinct --dist-init-addr port, HiCache budget variable)
TP2_REPLICAS: Final = {
    "2a": {"devices": ("4", "5"), "gpu_pair": "4-5", "dist_init": "127.0.0.1:29512", "budget_var": "GLM53_R2A_HICACHE_RAM_BUDGET"},
    "2b": {"devices": ("6", "7"), "gpu_pair": "6-7", "dist_init": "127.0.0.1:29513", "budget_var": "GLM53_R2B_HICACHE_RAM_BUDGET"},
    # Long-context 2xTP2 rollout: the same pair, in r1's place on GPUs 0-3 (docs/long-context-glm53-2xtp2-rollout.md).
    "1a": {"devices": ("0", "1"), "gpu_pair": "0-1", "dist_init": "127.0.0.1:29514", "budget_var": "GLM53_R1A_HICACHE_RAM_BUDGET"},
    "1b": {"devices": ("2", "3"), "gpu_pair": "2-3", "dist_init": "127.0.0.1:29515", "budget_var": "GLM53_R1B_HICACHE_RAM_BUDGET"},
}
# Half of r2's 650 GiB budget each, so the two replicas together take what r2 took.
TP2_HICACHE_BUDGET: Final = "325GiB"
TP2_MAMBA_CACHE: Final = "330"
TP2_VARIANT: Final = (
    "fc91d24-long-context-w4afp8-c8192-qsplit-offloop-v3-hicache-cuda-host-pooled-v1-host325g"
    "-memopt-mamba330-bf16state-admission-reserve-disabled-pool-clamp-pdi2-h200-tp2-ep2-eagle-fixed-4-1-5-mr12q4-strict-budget8192"
    + obs.VARIANT_SUFFIX
)
TP2_FLAG_CHANGES: Final = {
    "--tp-size 4": "--tp-size 2",
    "--ep-size 4": "--ep-size 2",
    "--mem-fraction-static 0.80": "--mem-fraction-static 0.86",
    "--max-running-requests 32": "--max-running-requests 12",
    "--max-queued-requests 8": "--max-queued-requests 4",
    "--cuda-graph-max-bs-decode 32": "--cuda-graph-max-bs-decode 12",
    "--speculative-num-steps 5": "--speculative-num-steps 4",
    "--speculative-num-draft-tokens 6": "--speculative-num-draft-tokens 5",
    "--speculative-adaptive": None,
}
# v8-bundle canary slot (docs/glm53-v8-bundle-canary.md): exactly one TP2 replica of this shared file (gpu02 and gpu23
# both deploy it) reads per-replica override variables. The canary is gpu02's r2a (GPUs 4,5), with r1a, r1b and the
# island-mate r2b as same-host controls and gpu23's r2a as the cross-host reference. Each variable is empty, or today's
# prod value, unless gpu02's compose-manager env map sets it; every other service and gpu23 render byte-for-byte what
# they rendered before.
V8_SLOT: Final = "2a"
V8_PREFIX: Final = "GLM53_V8_R2A_"
# THE CAPS UNDER TEST. They are the only place the canary's running/queued caps live: the env-map printer
# (scripts/glm53_v8_canary_env.py), the runbook table and the tests all read these two constants. Pending the lab result
# (2026-10-07 ~21:30 UTC); change them here, regenerate nothing (the file only holds the 12/4 defaults), and re-run the tests.
V8_LONG_MAX_RUNNING: Final = 16
V8_LONG_MAX_QUEUED: Final = 6
# Arguments of the TP2 argv whose value becomes a variable, with the variable and its default (today's value).
V8_ARGUMENT_VARIABLES: Final = {
    "--kv-cache-dtype bfloat16": ("--kv-cache-dtype", "KV_DTYPE", "bfloat16"),
    "--dsa-prefill-backend tilelang": ("--dsa-prefill-backend", "DSA_BACKEND", "tilelang"),
    "--dsa-decode-backend tilelang": ("--dsa-decode-backend", "DSA_BACKEND", "tilelang"),
    "--max-running-requests 12": ("--max-running-requests", "MAX_RUNNING", "12"),
    "--cuda-graph-max-bs-decode 12": ("--cuda-graph-max-bs-decode", "MAX_RUNNING", "12"),
    "--max-queued-requests 4": ("--max-queued-requests", "MAX_QUEUED", "4"),
}
V8_IMAGE_EXPRESSION: Final = v8.expression(V8_PREFIX, "IMAGE", R2_IMAGE)
V8_IMAGE_LABEL_EXPRESSION: Final = v8.expression(V8_PREFIX, "IMAGE_LABEL", R2_ENGINE_IMAGE_LABEL)
V8_PRECISION_EXPRESSION: Final = v8.expression(V8_PREFIX, "PRECISION", PRECISION)
V8_VARIANT_EXPRESSION: Final = v8.expression(V8_PREFIX, "VARIANT_SUFFIX", "")
V8_EXTRA_ARGS_EXPRESSION: Final = v8.expression(V8_PREFIX, "EXTRA_ARGS", "")
V8_ENV_PREFIX_EXPRESSION: Final = v8.expression(V8_PREFIX, "ENV_PREFIX", "")
BACKEND_URLS_R1_R2: Final = f"http://{SERVICE_PREFIX}1:8000,http://{SERVICE_PREFIX}2:8000"
# The value gpu02's compose-manager env map sets for GLM53_BACKEND_URLS during the 2xTP2 canary.
GPU02_BACKEND_URLS: Final = (
    f"http://{SERVICE_PREFIX}1:8000,http://{TP2_SERVICE_PREFIX}2a:8000,http://{TP2_SERVICE_PREFIX}2b:8000"
)
TP2_URL: Final = {suffix: f"http://{TP2_SERVICE_PREFIX}{suffix}:8000" for suffix in TP2_REPLICAS}
R1_URL: Final = f"http://{SERVICE_PREFIX}1:8000"
# Effective proxy pool per host and stage: the value that host's compose-manager env map sets for
# GLM53_BACKEND_URLS. A host not listed (or a stage not reached) leaves the variable UNSET, which
# resolves to BACKEND_URLS_R1_R2. gpu13 is deliberately absent: it deploys prod/small-models.yaml.
HOST_POOLS: Final = {
    "gpu02": {
        # r2 -> r2a + r2b, r1 stays TP4 (#332, deployed).
        "r2-pair": ",".join((R1_URL, TP2_URL["2a"], TP2_URL["2b"])),
        # r1 -> r1a + r1b as well: no TP4 replica left on the host.
        "all-tp2": ",".join((TP2_URL["1a"], TP2_URL["1b"], TP2_URL["2a"], TP2_URL["2b"])),
    },
    "gpu23": {
        "r2-pair": ",".join((R1_URL, TP2_URL["2a"], TP2_URL["2b"])),
        "all-tp2": ",".join((TP2_URL["1a"], TP2_URL["1b"], TP2_URL["2a"], TP2_URL["2b"])),
    },
}
GPU02_ALL_TP2_BACKEND_URLS: Final = HOST_POOLS["gpu02"]["all-tp2"]
GPU23_R2_PAIR_BACKEND_URLS: Final = HOST_POOLS["gpu23"]["r2-pair"]
GPU23_ALL_TP2_BACKEND_URLS: Final = HOST_POOLS["gpu23"]["all-tp2"]
PROXY_BACKEND_OLD: Final = f"      - VLLM_BACKEND_URLS={BACKEND_URLS_R1_R2}\n"
PROXY_BACKEND_NEW: Final = (
    "      # Host-overridable pool. Unset, this is exactly r1 + r2 (the TP4 replicas). A host whose\n"
    "      # compose-manager env map starts TP2 replicas sets GLM53_BACKEND_URLS to that stage's pool\n"
    "      # (HOST_POOLS in scripts/prepare_glm53_w4afp8_long_context.py); a host still on TP4 must not.\n"
    "      # docs/long-context-glm53-2xtp2-rollout.md lists the value per host and stage.\n"
    f"      - VLLM_BACKEND_URLS=${{GLM53_BACKEND_URLS:-{BACKEND_URLS_R1_R2}}}\n"
)
TP2_BUDGET_COMMENT_OLD_START: Final = "      # HiCache host tier: a fixed 406 GiB per replica"


class GenerationError(ValueError):
    pass


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    actual = text.count(old)
    if actual != expected:
        raise GenerationError(f"{label}: expected {expected} matches, found {actual}")
    return text.replace(old, new)


def section(text: str, start_marker: str, end_marker: str, label: str) -> tuple[int, int, str]:
    try:
        start = text.index(start_marker)
        end = text.index(end_marker, start + len(start_marker))
    except ValueError as error:
        raise GenerationError(f"{label}: source structure changed") from error
    return start, end, text[start:end]


def engine_arguments(source: list[str], replica: int) -> list[str]:
    """The source anchor's FP8 argv turned into campaign-2 arm L2, order preserved."""
    if not source or source[0] != "sglang serve":
        raise GenerationError("engine command must start with sglang serve")
    expected = {
        FP8_MODEL_PATH,
        f"--revision {FP8_REVISION}",
        "--moe-runner-backend deep_gemm",
        "--chunked-prefill-size 4096",
        "--prefill-decode-interval 1",
        f"--dist-init-addr {SOURCE_DIST_INIT}",
    }
    missing = sorted(argument for argument in expected if source.count(argument) != 1)
    if missing:
        raise GenerationError(f"source engine command changed: {missing}")
    if any(argument.startswith(("--hicache", "--enable-hierarchical-cache", "--max-prefill-tokens")) for argument in source):
        raise GenerationError("source engine command already carries L2 options")
    arguments: list[str] = []
    for argument in source:
        if argument in (f"--revision {FP8_REVISION}", "--moe-runner-backend deep_gemm"):
            continue
        if argument == FP8_MODEL_PATH:
            arguments.append(MODEL_PATH)
        elif argument == "--chunked-prefill-size 4096":
            arguments.extend(
                (f"--chunked-prefill-size {CHUNKED_PREFILL_SIZE[replica]}", "--max-prefill-tokens 32768")
            )
        elif argument == "--prefill-decode-interval 1":
            arguments.append(f"--prefill-decode-interval {PREFILL_DECODE_INTERVAL[replica]}")
        elif argument == f"--dist-init-addr {SOURCE_DIST_INIT}":
            arguments.append(f"--dist-init-addr {DIST_INIT[replica]}")
        else:
            arguments.append(argument)
    arguments.extend(HICACHE_FLAGS)
    return arguments


def render_command(arguments: list[str], indent: int) -> str:
    return " " * indent + "command: >\n" + "".join(f"{' ' * (indent + 4)}{argument}\n" for argument in arguments)


def resolve_engine_image_labels(text: str) -> str:
    """Bind each ENGINE_IMAGE_PLACEHOLDER to the image its own replica runs.

    The replacements above are replica-agnostic, so both replicas get a placeholder. Each site
    is disambiguated by a marker that is already replica-specific: the Datadog/OTel label blocks
    carry instance:N, and the scrape job carries the replica's service name.
    """
    for replica, label in REPLICA_IMAGE_LABEL.items():
        for marker, expected in (
            (f'"instance:{replica}"', 1),
            (f'nearai.otel.instance: "{replica}"', 1),
            (f"sglang-{SERVICE_PREFIX}{replica}", 1),
        ):
            # Enforce the count rather than letting find() silently take the first hit. If a
            # marker ever stops being replica-unique, binding the nearest placeholder could
            # quietly attach the wrong replica's image label; fail loudly instead, matching the
            # exact-count discipline replace_exact uses everywhere else in this script.
            actual = text.count(marker)
            if actual != expected:
                raise GenerationError(
                    f"engine_image: marker {marker!r} for replica {replica} must appear "
                    f"{expected} time(s), found {actual}"
                )
            index = text.find(marker)
            head, tail = text[:index], text[index:]
            placeholder_in_head = head.rfind("ENGINE_IMAGE_PLACEHOLDER")
            placeholder_in_tail = tail.find("ENGINE_IMAGE_PLACEHOLDER")
            if placeholder_in_head == -1 and placeholder_in_tail == -1:
                raise GenerationError(f"engine_image: no placeholder near {marker}")
            # The label precedes its instance marker in the Datadog tag list and follows it in
            # the OTel blocks; take whichever is closer to the marker.
            if placeholder_in_tail != -1 and (
                placeholder_in_head == -1 or placeholder_in_tail < len(head) - placeholder_in_head
            ):
                text = head + tail.replace("ENGINE_IMAGE_PLACEHOLDER", label, 1)
            else:
                text = head[:placeholder_in_head] + label + head[placeholder_in_head + len("ENGINE_IMAGE_PLACEHOLDER"):] + tail
    if "ENGINE_IMAGE_PLACEHOLDER" in text:
        raise GenerationError("engine_image: unresolved placeholder remains")
    return text


def tp2_service(r2: str, suffix: str) -> str:
    """One TP2 replica of the gpu02 2xTP2 canary, derived from the generated r2 service text."""
    spec = TP2_REPLICAS[suffix]
    name = f"{TP2_SERVICE_PREFIX}{suffix}"
    command_start = r2.index("    command: >\n") + len("    command: >\n")
    env_start = r2.index("    environment:\n")
    arguments = [line.strip() for line in r2[command_start:env_start].splitlines() if line.strip()]
    rewritten: list[str] = []
    for argument in arguments:
        if argument in TP2_FLAG_CHANGES:
            replacement = TP2_FLAG_CHANGES[argument]
            if replacement is not None:
                rewritten.append(replacement)
        elif argument.startswith("--dist-init-addr "):
            rewritten.append(f"--dist-init-addr {spec['dist_init']}")
        else:
            rewritten.append(argument)
    for argument in TP2_FLAG_CHANGES:
        if arguments.count(argument) != 1:
            raise GenerationError(f"tp2 canary: r2 argv must carry {argument!r} exactly once")
    rewritten.extend((f"--max-mamba-cache-size {TP2_MAMBA_CACHE}", "--mamba-ssm-dtype bfloat16"))
    v8_slot = suffix == V8_SLOT
    if v8_slot:
        missing = sorted(argument for argument in V8_ARGUMENT_VARIABLES if rewritten.count(argument) != 1)
        if missing:
            raise GenerationError(f"tp2 canary: the TP2 argv changed, cannot derive the v8 slot argv: {missing}")
        rewritten = [
            f"{V8_ARGUMENT_VARIABLES[argument][0]} {v8.expression(V8_PREFIX, V8_ARGUMENT_VARIABLES[argument][1], V8_ARGUMENT_VARIABLES[argument][2])}"
            if argument in V8_ARGUMENT_VARIABLES
            else argument
            for argument in rewritten
        ]
        rewritten = [V8_ENV_PREFIX_EXPRESSION, *rewritten, V8_EXTRA_ARGS_EXPRESSION]
    image = V8_IMAGE_EXPRESSION if v8_slot else R2_IMAGE
    variant = TP2_VARIANT + (V8_VARIANT_EXPRESSION if v8_slot else "")
    precision = V8_PRECISION_EXPRESSION if v8_slot else PRECISION
    engine_label = V8_IMAGE_LABEL_EXPRESSION if v8_slot else R2_ENGINE_IMAGE_LABEL

    env_end = r2.index("    depends_on:\n")
    environment = r2[env_start:env_end]
    budget_line = f"      - SGLANG_HICACHE_RAM_BUDGET=${{GLM53_R2_HICACHE_RAM_BUDGET:-650GiB}}\n"
    comment_start = environment.index(TP2_BUDGET_COMMENT_OLD_START)
    if environment.count(budget_line) != 1 or environment.index(budget_line) < comment_start:
        raise GenerationError("tp2 canary: r2 HiCache budget block changed")
    comment_end = environment.index(budget_line) + len(budget_line)
    environment = (
        environment[:comment_start]
        + "      # HiCache host tier: 325 GiB per TP2 replica across its two TP ranks (half of r2's 650 GiB; the\n"
        + "      # pair takes what the TP4 r2 took, and two pairs take 1300 GiB per host, more than 406 + 650 for r1 + r2). Fixed (not a percentage) so the replica started\n"
        + "      # second does not get less; startup fails if it exceeds available RAM minus 10 GiB.\n"
        + f"      - SGLANG_HICACHE_RAM_BUDGET=${{{spec['budget_var']}:-{TP2_HICACHE_BUDGET}}}\n"
        + environment[comment_end:]
    )
    # Each TP2 replica reports to the CVM's ghost aggregator under its own name.
    environment = replace_exact(
        environment, obs.replica_line("r2"), obs.replica_line(f"r{suffix}"), 1, f"tp2 {suffix} ghost replica"
    )
    devices = ",".join(f'"{device}"' for device in spec["devices"])
    log_tags = (
        '"model:z-ai/glm-5.3-flash","model_path:graphistry/GLM-5.3-Flash-W4AFP8","served_model:z-ai/glm-5.3-flash",'
        f'"precision:{precision}","deployment:glm53-flash-sgl-tp4","config_variant:{variant}",'
        f'"request_logging:disabled","engine_image:{engine_label}","env:${{ENV}}","host:${{CVM_HOST}}",'
        f'"ip:${{HOST_IP}}","port:8000","instance:{suffix}","gpu_pair:{spec["gpu_pair"]}"'
    )
    return (
        f"  # --- GLM-5.3-Flash 2xTP2 memory-optimized canary replica {suffix} (SGLang TP2, GPUs {spec['gpu_pair']}) ---\n"
        f"  # Started only by a scoped services list naming it (never by an unscoped up, which would collide with the TP4 replica on these GPUs).\n"
        f"  {name}:\n"
        "    <<: *sg-glm53-flash-common\n"
        f"    container_name: {name}\n"
        + (
            "    # v8 canary slot (docs/glm53-v8-bundle-canary.md): every value the bundle changes is behind a per-replica variable that\n"
            "    # is empty or today's value unless gpu02's env map sets it. ENV_PREFIX is `env NAME=value ...`, EXTRA_ARGS carries\n"
            "    # the scheduler flag; neither flag nor environment is ever a literal in this file.\n"
            if v8_slot
            else ""
        )
        + f"    image: {image}\n"
        + render_command(rewritten, 4)
        + environment
        + "    depends_on:\n      model-downloader:\n        condition: service_completed_successfully\n"
        "    deploy:\n      resources:\n        reservations:\n          devices:\n"
        f"            - driver: nvidia\n              device_ids: [{devices}]\n              capabilities: [gpu]\n"
        "    labels:\n"
        f"      com.datadoghq.ad.logs: '[{{\"source\":\"sglang\",\"service\":\"sglang\",\"tags\":[{log_tags}]}}]'\n"
        '      nearai.otel.scrape: "true"\n      nearai.otel.job: "sglang"\n      nearai.otel.service: "sglang"\n'
        '      nearai.otel.source: "sglang"\n'
        f'      nearai.otel.container_name: "{name}"\n'
        '      nearai.otel.port: "8000"\n      nearai.otel.path: "/metrics"\n'
        '      nearai.otel.model: "z-ai/glm-5.3-flash"\n'
        f'      nearai.otel.model_path: "{CHECKPOINT}"\n'
        '      nearai.otel.served_model: "z-ai/glm-5.3-flash"\n'
        '      nearai.otel.deployment: "glm53-flash-sgl-tp4"\n'
        f'      nearai.otel.config_variant: "{variant}"\n'
        '      nearai.otel.thinking_budget_policy: "default8192-public-to-native"\n'
        '      nearai.otel.request_logging: "disabled"\n'
        f'      nearai.otel.engine_image: "{engine_label}"\n'
        f'      nearai.otel.instance: "{suffix}"\n'
        f'      nearai.otel.gpu_pair: "{spec["gpu_pair"]}"\n'
        '      nearai.otel.env: "${ENV}"\n      nearai.otel.host: "${CVM_HOST}"\n'
        '      nearai.otel.host_machine: "${CVM_HOST}"\n      nearai.otel.cvm_name: "${CVM_NAME}"\n'
        '      nearai.otel.ip: "${HOST_IP}"\n'
    )


def tp2_scrape_job(suffix: str) -> str:
    spec = TP2_REPLICAS[suffix]
    name = f"{TP2_SERVICE_PREFIX}{suffix}"
    v8_slot = suffix == V8_SLOT
    variant = TP2_VARIANT + (V8_VARIANT_EXPRESSION if v8_slot else "")
    precision = V8_PRECISION_EXPRESSION if v8_slot else PRECISION
    engine_label = V8_IMAGE_LABEL_EXPRESSION if v8_slot else R2_ENGINE_IMAGE_LABEL
    return (
        f"              - job_name: sglang-{name}\n"
        "                scrape_interval: 15s\n"
        "                metrics_path: /metrics\n"
        "                static_configs:\n"
        f"                  - targets: ['{name}:8000']\n"
        "                    labels:\n"
        '                      service: "sglang"\n'
        '                      source: "sglang"\n'
        f'                      container_name: "{name}"\n'
        '                      model: "z-ai/glm-5.3-flash"\n'
        f'                      model_path: "{CHECKPOINT}"\n'
        '                      served_model: "z-ai/glm-5.3-flash"\n'
        f'                      precision: "{precision}"\n'
        '                      deployment: "glm53-flash-sgl-tp4"\n'
        '                      env: "${ENV}"\n'
        '                      host: "${CVM_HOST}"\n'
        '                      host_machine: "${CVM_HOST}"\n'
        '                      cvm_name: "${CVM_NAME}"\n'
        '                      ip: "${HOST_IP}"\n'
        '                      port: "8000"\n'
        f'                      instance: "{suffix}"\n'
        f'                      gpu_pair: "{spec["gpu_pair"]}"\n'
        f'                      config_variant: "{variant}"\n'
        '                      thinking_budget_policy: "default8192-public-to-native"\n'
        '                      request_logging: "disabled"\n'
        f'                      engine_image: "{engine_label}"\n'
    )


def add_tp2_canary(text: str) -> str:
    """Add the gpu02 2xTP2 pair beside the untouched TP4 r2 and make the proxy pool overridable."""
    text = replace_exact(text, PROXY_BACKEND_OLD, PROXY_BACKEND_NEW, 1, "proxy backend list")
    r2_start, r2_end, r2 = section(
        text, f"  {R2_SERVICE}:\n", "\n  # Explicit operator-only semantic check;", "replica 2 service"
    )
    pair = "".join("\n" + tp2_service(r2, suffix) for suffix in TP2_REPLICAS)
    text = text[:r2_end] + pair + text[r2_end:]
    job_marker = "              - job_name: dcgm-dcgm-glm53\n"
    jobs = "".join(tp2_scrape_job(suffix) for suffix in TP2_REPLICAS)
    return replace_exact(text, job_marker, jobs + job_marker, 1, "tp2 scrape jobs")


def generate(source: str) -> str:
    if SERVICE_PREFIX in source or f"models--{CHECKPOINT.replace('/', '--')}/" in source:
        raise GenerationError("source compose already contains W4AFP8 engines")

    updated = source
    for old, new in HEADER_REPLACEMENTS:
        updated = replace_exact(updated, old, new, 1, "long-context header")
    updated = replace_exact(updated, SOURCE_SERVICE_PREFIX, SERVICE_PREFIX, 17, "replica service identity")

    anchor_start, anchor_end, anchor = section(
        updated,
        "x-sg-glm53-flash-common: &sg-glm53-flash-common\n",
        "\nx-dcgm-common: &dcgm-common\n",
        "shared engine anchor",
    )
    anchor = replace_exact(anchor, f"\n  image: {SOURCE_IMAGE}\n", f"\n  image: {IMAGE}\n", 1, "anchor image")
    command_start, command_end, command = section(anchor, "  command: >\n", "  volumes:\n", "anchor command")
    source_arguments = [line.strip() for line in command.splitlines()[1:] if line.strip()]
    anchor = anchor[:command_start] + render_command(engine_arguments(source_arguments, 1), 2) + anchor[command_end:]
    anchor = replace_exact(anchor, ANCHOR_ENV_OLD, ANCHOR_ENV_NEW, 1, "anchor environment")
    anchor = replace_exact(anchor, ANCHOR_VOLUMES, f"{ANCHOR_VOLUMES}    - {obs.MOUNT}\n", 1, "anchor volumes")
    updated = updated[:anchor_start] + anchor + updated[anchor_end:]

    r2_start, r2_end, r2 = section(
        updated, f"  {SERVICE_PREFIX}2:\n", "\n  # Explicit operator-only semantic check;", "replica 2 service"
    )
    container = f"    container_name: {SERVICE_PREFIX}2\n"
    override_start = r2.index(container) + len(container)
    override_end = r2.index("    depends_on:\n", override_start)
    override = r2[override_start:override_end]
    if not override.startswith(f"    image: {SOURCE_HICACHE_IMAGE}\n    command: >\n") or "\n    environment:\n" not in override:
        raise GenerationError("replica 2 no longer carries the source HiCache override")
    # r2 must override image AND environment, not just command: a YAML merge key replaces a
    # list wholesale rather than deep-merging it, so r2 cannot inherit the anchor's environment
    # and add one variable. The block is derived from the anchor here rather than duplicated as
    # a literal, so the two can never drift apart.
    _, _, anchor_environment = section(anchor, "  environment:\n", "  restart: unless-stopped\n", "anchor environment block")
    r2_environment = "".join(
        f"  {line}\n" if line.strip() else "\n" for line in anchor_environment.splitlines()
    )
    r2_environment = replace_exact(r2_environment, f"  {ANCHOR_BUDGET_LINE}", f"  {R2_BUDGET_LINE}", 1, "r2 host budget")
    r2_environment = replace_exact(r2_environment, obs.replica_line("r1"), obs.replica_line("r2"), 1, "r2 ghost replica")
    r2 = (
        r2[:override_start]
        + f"    image: {REPLICA_IMAGE[2]}\n"
        + render_command(engine_arguments(source_arguments, 2), 4)
        + r2_environment
        + r2[override_end:]
    )
    updated = updated[:r2_start] + r2 + updated[r2_end:]

    for old, new, count, label in (
        ("model_path:zai-org/GLM-5.3-Flash", f"model_path:{CHECKPOINT}", 3, "log model_path"),
        ('model_path: "zai-org/GLM-5.3-Flash"', f'model_path: "{CHECKPOINT}"', 6, "metric model_path"),
        ("precision:fp8-weights-bf16-kv", f"precision:{PRECISION}", 2, "log precision"),
        ('precision: "fp8-weights-bf16-kv"', f'precision: "{PRECISION}"', 2, "scrape precision"),
        (SOURCE_VARIANTS[0], VARIANTS[1], 3, "replica 1 config_variant"),
        (SOURCE_VARIANTS[1], VARIANTS[2], 3, "replica 2 config_variant"),
        ('"request_logging:disabled",', '"request_logging:disabled","engine_image:ENGINE_IMAGE_PLACEHOLDER",', 2, "log engine_image"),
        (
            '      nearai.otel.request_logging: "disabled"\n',
            '      nearai.otel.request_logging: "disabled"\n      nearai.otel.engine_image: "ENGINE_IMAGE_PLACEHOLDER"\n',
            2,
            "engine_image label",
        ),
        (
            '                      request_logging: "disabled"\n',
            '                      request_logging: "disabled"\n                      engine_image: "ENGINE_IMAGE_PLACEHOLDER"\n',
            2,
            "scrape engine_image",
        ),
    ):
        updated = replace_exact(updated, old, new, count, label)
    updated = resolve_engine_image_labels(updated)
    updated = add_tp2_canary(updated)

    # Observability: the per-CVM ghost aggregator, its in-memory volume and its scrape job.
    engines = "The engines"
    for old, new, label in (
        ("  # --- Full-host GPU telemetry ---\n", obs.sidecar_service(IMAGE, DEPLOYMENT, engines) + "  # --- Full-host GPU telemetry ---\n", "ghost sidecar"),
        ("\n  kernel_cache:\n", f"\n  kernel_cache:\n{obs.volume_declaration()}", "ghost volume"),
        ("              - job_name: dcgm-dcgm-glm53\n", obs.scrape_job(DEPLOYMENT) + "              - job_name: dcgm-dcgm-glm53\n", "ghost scrape job"),
    ):
        updated = replace_exact(updated, old, new, 1, label)
    return HEADER + updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--write", action="store_true", help="Write the target file")
    _ = parser.add_argument("--check", action="store_true", help="Fail with a diff when the target is stale")
    args = parser.parse_args()
    if args.write and args.check:
        parser.error("--write and --check are mutually exclusive")

    expected = generate((ROOT / SOURCE).read_text())
    actual = (ROOT / TARGET).read_text() if (ROOT / TARGET).exists() else ""
    if args.check:
        if actual == expected:
            return 0
        diff = difflib.unified_diff(
            actual.splitlines(keepends=True), expected.splitlines(keepends=True), fromfile=f"a/{TARGET}", tofile=f"b/{TARGET}"
        )
        print("".join(diff), end="")
        return 1
    if args.write:
        _ = (ROOT / TARGET).write_text(expected)
        return 0
    print(expected, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
