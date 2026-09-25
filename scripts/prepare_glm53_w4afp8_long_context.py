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
SOURCE = Path("prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml")
TARGET = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml")

CHECKPOINT: Final = "graphistry/GLM-5.3-Flash-W4AFP8"
CHECKPOINT_REVISION: Final = "99f1fa70408c52b007d4fd69e02e5a522422e755"
FP8_REVISION: Final = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
SOURCE_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7"
SOURCE_HICACHE_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c"
IMAGE: Final = "docker.io/nearaidev/sglang@sha256:fde25985aea3ebabf1eb581ae21d53be8540e32933eef942ee8b962a1bfbea20"
ENGINE_IMAGE_LABEL: Final = "fde25985aea3"
# r2 only: glm53-hicache-w4afp8-v2, which adds the opt-in DSA indexer query split (#300,
# recipe commit 556482c). r1 stays on IMAGE so the control arm is byte-identical to #294 and a
# targeted `compose up` recreates r2 alone. The split is inert unless SGLANG_DSA_INDEXER_QSPLIT=1.
R2_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:8ff1a487b98a52fe08b781715bebd7c8c445d4fe068f312f03f527d5a3c77e84"
R2_ENGINE_IMAGE_LABEL: Final = "8ff1a487b98a"
REPLICA_IMAGE: Final = {1: IMAGE, 2: R2_IMAGE}
REPLICA_IMAGE_LABEL: Final = {1: ENGINE_IMAGE_LABEL, 2: R2_ENGINE_IMAGE_LABEL}
# r2 is back to 8192 while keeping the split. This is an ISOLATION ARM, not the shipping config.
#
# The 2026-09-25 gpu02 canary ran split + c16384 + pdi 2 and produced the opposite of the lab
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
# Holding the split at 8192 separates them: if p95 recovers, the chunk was at fault; if it stays
# bad, the all-gather under CC is. Note the validator gate is one-directional -- c16384 REQUIRES
# the split, but the split does not require c16384 -- so this combination is permitted by design.
CHUNKED_PREFILL_SIZE: Final = {1: 8192, 2: 8192}
R2_EXTRA_ENV: Final = ("      - SGLANG_DSA_INDEXER_QSPLIT=1\n",)
SOURCE_SERVICE_PREFIX: Final = "model-sg-glm53-fp8-tp4-r"
SERVICE_PREFIX: Final = "model-sg-glm53-w4afp8-tp4-r"
PRECISION: Final = "int4-weights-fp8-activations-bf16-kv"
SOURCE_VARIANTS: Final = (
    "fc91d24-long-context-admission-reserve-disabled-hicache-disabled-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192",
    "fc91d24-long-context-admission-reserve-disabled-hicache-cuda-host-pooled-v1-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192",
)
# r1 is the unchanged #294 control; r2 is the --prefill-decode-interval 2 canary. The
# variants differ only in pdi1/pdi2 so dashboards split the replicas by config_variant.
VARIANTS: Final = {
    1: "fc91d24-long-context-w4afp8-c8192-hicache-cuda-host-pooled-v1-admission-reserve-disabled"
    "-pool-clamp-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192",
    2: "fc91d24-long-context-w4afp8-c8192-qsplit-hicache-cuda-host-pooled-v1-admission-reserve-disabled"
    "-pool-clamp-pdi2-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192",
}
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
    "# THE REPLICAS ARE NOT IDENTICAL. r1 is the live control, r2 the canary:\n"
    "#   r1: gpu31 campaign-2 arm L2 (2026-09-23), chunk 8192, --prefill-decode-interval 1\n"
    f"#       {IMAGE}\n"
    "#       published by workflow run 35903077821 from recipe merge commit f8106f096e9c83.\n"
    "#   r2: chunk 8192, --prefill-decode-interval 2, SGLANG_DSA_INDEXER_QSPLIT=1\n"
    f"#       {R2_IMAGE}\n"
    "#       published by workflow run 36073196105 from recipe merge commit 556482c78cd3,\n"
    "#       which adds the opt-in DSA indexer query split (cvm-compose-files#300).\n"
    "#\n"
    "# r2 IS AN ISOLATION ARM, NOT A SHIPPING CONFIG. The 2026-09-25 canary ran the split WITH\n"
    "# chunk 16384 and regressed the tail: TTFT p95 76.9 s against r1's 47.3 s (1.62x) over a\n"
    "# 4 h steady-state window, though the decode win did reproduce (ITL p90 0.87x). Traffic\n"
    "# volume, cache warmth and queue depth did not explain it. The chunk and the split both\n"
    "# bite hardest on the largest prefills, and the canary changed both at once, so this arm\n"
    "# holds the split at 8192 to tell them apart: if p95 recovers the chunk was at fault, if\n"
    "# it stays bad the all-gather under CC/PPCIe is. Do not read this file as an endorsement\n"
    "# of either setting until that run reports.\n"
    "#\n"
    "# chunk 16384 must never be set without the split: it left 0.04-0.65 GB free per GPU on a\n"
    "# concurrent long burst, the condition that preceded the gpu02 crash. The reverse is\n"
    "# allowed, which is what makes this arm possible. validate_glm53_prod_config.rb enforces it.\n"
    "# The HCC/PPCIe host-memory path is confirmed working in a CVM as of 2026-09-25; the\n"
    "# split's all-gather under CC is what this arm measures. Roll out with\n"
    "# docs/glm53-w4afp8-long-context-rollout.md, one replica at a time.\n"
    "# Do not hand-edit this file.\n"
)

HEADER_REPLACEMENTS: Final = (
    (
        "# Hand-derived from prod/GLM-5.3-Flash-SGL-TP4.yaml. Routing remains the dedicated\n"
        "# long-context contract below, while the engines intentionally form an r1 control / r2\n"
        "# HiCache experiment and therefore are not byte-identical to the canonical file.\n",
        "# Generated from the long-context file. Routing remains the dedicated long-context\n"
        "# contract below; both replicas run a W4AFP8 + HiCache engine, r2 as an isolation arm\n"
        "# at --prefill-decode-interval 2 with the DSA indexer split and chunk 8192.\n",
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
        "#   The replicas' engine flags differ in --dist-init-addr, --prefill-decode-interval\n"
        "#   (1 on r1, 2 on r2); both now run --chunked-prefill-size 8192. r2 additionally sets\n"
        "#   SGLANG_DSA_INDEXER_QSPLIT=1 and runs a different image. The 8192 chunk is the\n"
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
        "# DSA import-cycle fixes. Both replicas pin a published\n"
        "# docker/sglang-glm53-hicache-w4afp8 derivative (r1 v1, r2 v2 with the opt-in DSA\n"
        "# indexer split): the HCC-safe HiCache image (CUDA-owned\n"
        "# host memory) plus the W4AFP8 loader fix and the unconditional chunked-prefill pool\n"
        "# clamp. Two replicas share the CVM's RAM, so each replica's startup host-memory\n"
        "# budget defaults to a fixed 406 GiB across its four TP ranks (the qualified L2 value;\n"
        "# a percentage resolves against MemAvailable at each start and would split unevenly).\n"
        "# GLM53_HICACHE_RAM_BUDGET overrides it for BOTH replicas.\n",
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
    "    # so the replica started second would get less. The override applies to BOTH\n"
    "    # replicas; startup fails if it exceeds available RAM minus the 10 GiB reserve.\n"
    "    - SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-406GiB}\n"
    "    # CUDA-owned host memory (cudaMallocHost). Must stay 1 on TEE hosts: 0 selects\n"
    "    # cudaHostRegister, which fails with CUDA error 801 under HCC/PPCIe.\n"
    "    - SGLANG_HICACHE_CUDA_HOST_MEMORY=${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}\n"
    "    - SGLANG_HICACHE_POOLED_TRANSFERS=1\n"
    "    - SGLANG_HICACHE_STAGING_PAGES=64\n"
    "  restart: unless-stopped\n"
)


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
    ) + "".join(R2_EXTRA_ENV)
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
