#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run scripts/prepare_glm53_w4afp8_base.py --write
"""Generate the W4AFP8 base-tier file from the canonical GLM-5.3 Flash file.

Both replicas run the gpu31 campaign-2 arm B5: the argv gpu02's W4AFP8 r1 runs, with the
canonical admission reserve. Everything outside the two engines and their truthful
telemetry stays byte-identical to the canonical file. `--write` regenerates the target;
`--check` prints a diff and exits non-zero when the committed target is stale.
"""

import argparse
import difflib
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("prod/GLM-5.3-Flash-SGL-TP4.yaml")
TARGET = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml")

CHECKPOINT: Final = "graphistry/GLM-5.3-Flash-W4AFP8"
CHECKPOINT_REVISION: Final = "99f1fa70408c52b007d4fd69e02e5a522422e755"
FP8_REVISION: Final = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
SOURCE_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7"
IMAGE: Final = "docker.io/nearaidev/sglang@sha256:8bce6a7cc872a80faded3bd1ef0a64873a1d7abae34c94e5358775ca21f133cc"
ENGINE_IMAGE_LABEL: Final = "8bce6a7cc872"
SOURCE_SERVICE_PREFIX: Final = "model-sg-glm53-fp8-tp4-r"
SERVICE_PREFIX: Final = "model-sg-glm53-w4afp8-tp4-r"
PRECISION: Final = "int4-weights-fp8-activations-bf16-kv"
SOURCE_VARIANT: Final = "fc91d24-admission-reserve-v10-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
VARIANT: Final = "fc91d24-w4afp8-c4096-admission-reserve-v10-pool-clamp-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
FP8_MODEL_PATH: Final = f"--model-path /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/{FP8_REVISION}"
MODEL_PATH: Final = (
    "--model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/"
    f"snapshots/{CHECKPOINT_REVISION}"
)

HEADER: Final = (
    "# GLM-5.3 Flash base tier on W4AFP8 (gpu03, gpu04, gpu23), generated from\n"
    "# prod/GLM-5.3-Flash-SGL-TP4.yaml by scripts/prepare_glm53_w4afp8_base.py.\n"
    "# Both replicas run the gpu31 campaign-2 arm B5 (2026-09-23): the W4AFP8 checkpoint\n"
    "# graphistry/GLM-5.3-Flash-W4AFP8@99f1fa7 with the argv gpu02's W4AFP8 r1 runs\n"
    "# (4096-token prefill chunks, --max-prefill-tokens 32768, no --revision, no\n"
    "# --moe-runner-backend) plus the canonical admission reserve (4096, max fraction 0.75).\n"
    "# No HiCache. Both pin\n"
    f"# {IMAGE}\n"
    "# (docker/sglang-glm53-w4afp8), published by workflow run 35659748426 from recipe merge\n"
    "# commit 7c473970af2ac040afb233df8b274aa0cf8ebbcb. The canonical FP8 file is the\n"
    "# rollback. Roll out with docs/glm53-w4afp8-base-rollout.md, one host at a time.\n"
    "# Do not hand-edit this file.\n"
)

HEADER_REPLACEMENTS: Final = (
    (
        "# Canonical GLM-5.3 Flash production deployment for 8x H200 hosts.\n"
        "# Run the exact same committed tag and file on every matching host: two independent\n"
        "# TP4/EP4 replicas, one per four-GPU NVLink island, behind one inference proxy.\n",
        "# W4AFP8 variant of the canonical GLM-5.3 Flash deployment for 8x H200 base-tier hosts.\n"
        "# Run the exact same committed tag and file on every base-tier host: two independent\n"
        "# TP4/EP4 replicas, one per four-GPU NVLink island, behind one inference proxy.\n",
    ),
    (
        "# DSA import-cycle fixes. The production digest is the\n"
        "# docker/sglang-glm53-admission-reserve derivative of that build, which adds the opt-in\n"
        "# chunked-prefill admission-reserve v10 patch (nearai/inference-optimizer@fb94472),\n"
        "# enabled below with SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096,\n"
        "# SGLANG_ADMISSION_RESERVE_MAX_FRACTION=0.75 and --prefill-decode-interval 1. The\n"
        "# official CUDA 13 SGLang image is the pinned build base. The serving envelope is\n"
        "# intentionally conservative: BF16 KV, 0.80 static memory, 32 running requests, a\n"
        "# bounded 8-request queue, 4096-token prefill chunks, decode graphs capped at batch 32,\n"
        "# TileLang DSA, DeepGEMM, and adaptive EAGLE 5/1/6. This retains the full\n"
        "# 1,048,576-token model context while avoiding the late K-pool workspace exhaustion\n"
        "# observed with larger prefill chunks.\n",
        "# DSA import-cycle fixes. The production digest is the docker/sglang-glm53-w4afp8\n"
        "# derivative: the docker/sglang-glm53-admission-reserve build (opt-in chunked-prefill\n"
        "# admission-reserve v10 patch, nearai/inference-optimizer@fb94472) plus the W4AFP8\n"
        "# loader fix and the unconditional chunked-prefill pool clamp. The reserve is enabled\n"
        "# below with SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096,\n"
        "# SGLANG_ADMISSION_RESERVE_MAX_FRACTION=0.75 and --prefill-decode-interval 1. The\n"
        "# official CUDA 13 SGLang image is the pinned build base. The serving envelope is\n"
        "# intentionally conservative: BF16 KV, 0.80 static memory, 32 running requests, a\n"
        "# bounded 8-request queue, 4096-token prefill chunks with --max-prefill-tokens 32768,\n"
        "# decode graphs capped at batch 32, TileLang DSA, the CUTLASS W4A8 MoE path (hence no\n"
        "# --moe-runner-backend and no FP8 --revision), and adaptive EAGLE 5/1/6. This retains\n"
        "# the full 1,048,576-token model context.\n",
    ),
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


def engine_arguments(source: list[str]) -> list[str]:
    """The canonical FP8 argv turned into campaign-2 arm B5, order preserved."""
    if not source or source[0] != "sglang serve":
        raise GenerationError("engine command must start with sglang serve")
    expected = (FP8_MODEL_PATH, f"--revision {FP8_REVISION}", "--moe-runner-backend deep_gemm", "--chunked-prefill-size 4096")
    missing = sorted(argument for argument in expected if source.count(argument) != 1)
    if missing:
        raise GenerationError(f"source engine command changed: {missing}")
    if any(argument.startswith(("--hicache", "--enable-hierarchical-cache", "--max-prefill-tokens")) for argument in source):
        raise GenerationError("source engine command already carries B5 or HiCache options")
    arguments: list[str] = []
    for argument in source:
        if argument in (f"--revision {FP8_REVISION}", "--moe-runner-backend deep_gemm"):
            continue
        if argument == FP8_MODEL_PATH:
            arguments.append(MODEL_PATH)
        elif argument == "--chunked-prefill-size 4096":
            arguments.extend(("--chunked-prefill-size 4096", "--max-prefill-tokens 32768"))
        else:
            arguments.append(argument)
    return arguments


def generate(source: str) -> str:
    if SERVICE_PREFIX in source or f"models--{CHECKPOINT.replace('/', '--')}/" in source:
        raise GenerationError("source compose already contains W4AFP8 engines")

    updated = source
    for old, new in HEADER_REPLACEMENTS:
        updated = replace_exact(updated, old, new, 1, "canonical header")
    updated = replace_exact(updated, SOURCE_SERVICE_PREFIX, SERVICE_PREFIX, 17, "replica service identity")

    anchor_start, anchor_end, anchor = section(
        updated,
        "x-sg-glm53-flash-common: &sg-glm53-flash-common\n",
        "\nx-dcgm-common: &dcgm-common\n",
        "shared engine anchor",
    )
    anchor = replace_exact(anchor, f"\n  image: {SOURCE_IMAGE}\n", f"\n  image: {IMAGE}\n", 1, "anchor image")
    command_start, command_end, command = section(anchor, "  command: >\n", "  volumes:\n", "anchor command")
    arguments = engine_arguments([line.strip() for line in command.splitlines()[1:] if line.strip()])
    rendered = "  command: >\n" + "".join(f"      {argument}\n" for argument in arguments)
    anchor = anchor[:command_start] + rendered + anchor[command_end:]
    if "SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096" not in anchor or "SGLANG_HICACHE_" in anchor:
        raise GenerationError("canonical engine environment changed: B5 keeps the reserve and has no HiCache")
    updated = updated[:anchor_start] + anchor + updated[anchor_end:]

    for old, new, count, label in (
        ("model_path:zai-org/GLM-5.3-Flash", f"model_path:{CHECKPOINT}", 3, "log model_path"),
        ('model_path: "zai-org/GLM-5.3-Flash"', f'model_path: "{CHECKPOINT}"', 6, "metric model_path"),
        ("precision:fp8-weights-bf16-kv", f"precision:{PRECISION}", 2, "log precision"),
        ('precision: "fp8-weights-bf16-kv"', f'precision: "{PRECISION}"', 2, "scrape precision"),
        (SOURCE_VARIANT, VARIANT, 6, "config_variant"),
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
