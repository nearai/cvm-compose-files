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
    2: "fc91d24-long-context-w4afp8-c8192-hicache-cuda-host-pooled-v1-admission-reserve-disabled"
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
    "# Both replicas run the gpu31 campaign-2 arm L2 (2026-09-23): the W4AFP8 checkpoint\n"
    "# graphistry/GLM-5.3-Flash-W4AFP8@99f1fa7, 8192-token prefill chunks with\n"
    "# --max-prefill-tokens 32768, HiCache with CUDA-owned host memory and a fixed 406 GiB\n"
    "# startup host-memory budget per replica, and no admission reserve. r2 is a canary at\n"
    "# --prefill-decode-interval 2 (two decode steps between prefill chunks); r1 stays at 1 as\n"
    "# the live control. On stored long-tier traffic (gpu31, 2026-09-24) pdi 2 cut TPOT p50 to\n"
    "# 0.53-0.72x of pdi 1 at a paired TTFT cost of at most 2%, with 100% completion. Both pin\n"
    f"# {IMAGE}\n"
    "# (docker/sglang-glm53-hicache-w4afp8), published by workflow run 35903077821 from recipe\n"
    "# merge commit f8106f096e9c838b283a0f79a452d6c43e470641. Its HCC/PPCIe host-memory path\n"
    "# has not run in a CVM before this file, so gpu02 is its first soak. Roll out with\n"
    "# docs/glm53-w4afp8-long-context-rollout.md, one replica at a time.\n"
    "# Do not hand-edit this file.\n"
)

HEADER_REPLACEMENTS: Final = (
    (
        "# Hand-derived from prod/GLM-5.3-Flash-SGL-TP4.yaml. Routing remains the dedicated\n"
        "# long-context contract below, while the engines intentionally form an r1 control / r2\n"
        "# HiCache experiment and therefore are not byte-identical to the canonical file.\n",
        "# Generated from the long-context file. Routing remains the dedicated long-context\n"
        "# contract below; both replicas run the same W4AFP8 + HiCache engine, r2 at\n"
        "# --prefill-decode-interval 2 as a canary.\n",
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
        "#   The replicas' engine flags differ only in --dist-init-addr and r2's canary\n"
        "#   --prefill-decode-interval 2. The\n"
        "#   8192-token chunk is the W4AFP8 setting: FP8 at 8192 OOMed in the DSA indexer under\n"
        "#   concurrent 400K+ contexts, while W4AFP8's 2.4x larger device KV pool kept 3.7 GiB\n"
        "#   free through 8 concurrent 647K-756K-token cold prefills on gpu31 (16384 fell to\n"
        "#   0.04 GiB and is rejected). Conversation affinity stays on because the prefix cache\n"
        "#   is worth ~3x in request capacity.\n",
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
        "# DSA import-cycle fixes. Both replicas pin the published\n"
        "# docker/sglang-glm53-hicache-w4afp8 derivative: the HCC-safe HiCache image (CUDA-owned\n"
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
            arguments.extend(("--chunked-prefill-size 8192", "--max-prefill-tokens 32768"))
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
    r2 = r2[:override_start] + render_command(engine_arguments(source_arguments, 2), 4) + r2[override_end:]
    updated = updated[:r2_start] + r2 + updated[r2_end:]

    for old, new, count, label in (
        ("model_path:zai-org/GLM-5.3-Flash", f"model_path:{CHECKPOINT}", 3, "log model_path"),
        ('model_path: "zai-org/GLM-5.3-Flash"', f'model_path: "{CHECKPOINT}"', 6, "metric model_path"),
        ("precision:fp8-weights-bf16-kv", f"precision:{PRECISION}", 2, "log precision"),
        ('precision: "fp8-weights-bf16-kv"', f'precision: "{PRECISION}"', 2, "scrape precision"),
        (SOURCE_VARIANTS[0], VARIANTS[1], 3, "replica 1 config_variant"),
        (SOURCE_VARIANTS[1], VARIANTS[2], 3, "replica 2 config_variant"),
        ('"request_logging:disabled",', f'"request_logging:disabled","engine_image:{ENGINE_IMAGE_LABEL}",', 2, "log engine_image"),
        (
            '      nearai.otel.request_logging: "disabled"\n',
            f'      nearai.otel.request_logging: "disabled"\n      nearai.otel.engine_image: "{ENGINE_IMAGE_LABEL}"\n',
            2,
            "engine_image label",
        ),
        (
            '                      request_logging: "disabled"\n',
            f'                      request_logging: "disabled"\n                      engine_image: "{ENGINE_IMAGE_LABEL}"\n',
            2,
            "scrape engine_image",
        ),
    ):
        updated = replace_exact(updated, old, new, count, label)
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
