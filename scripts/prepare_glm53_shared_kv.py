#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run scripts/prepare_glm53_shared_kv.py --write
"""Generate the shared-KV-cache variant of the gpu02 long-context file.

Source: the live W4AFP8 long-context file. The two replicas keep their own engine flags (r1 the
chunk-8192 control, r2 the chunk-16384 + DSA-indexer-split canary); the only change is that both run
the shared KV cache image and share one CPU-RAM KV store:

  * both engines run the published docker/sglang-glm53-v0520-shared-kv image
    (docker/sglang-glm53-v0520-shared-kv/RELEASED_IMAGE, written after the publish workflow);
  * HiCache gains `--hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete`;
  * the store is one tmpfs volume (`shared_kv`) mounted at /shared-kv in both engines: guest RAM,
    nothing leaves the CVM;
  * the startup RAM budget keeps 406 GiB per replica and reserves the store's growth
    (SGLANG_HICACHE_SHARED_STORE_BUDGET); each replica's evictors are capped so both fit it;
  * config_variant and engine_image telemetry say so.

Everything else stays byte-identical to the source. `--write` regenerates the target; `--check`
prints a diff and exits non-zero when the committed target is stale; `--image` overrides the
released image (lab soaks only; never commit a target generated with it).
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml")
TARGET = Path("experiments/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext-SharedKV.yaml")
RELEASED_IMAGE = Path("docker/sglang-glm53-v0520-shared-kv/RELEASED_IMAGE")

SOURCE_IMAGES: Final = {
    "docker.io/nearaidev/sglang@sha256:fde25985aea3ebabf1eb581ae21d53be8540e32933eef942ee8b962a1bfbea20": "fde25985aea3",
    "docker.io/nearaidev/sglang@sha256:8ff1a487b98a52fe08b781715bebd7c8c445d4fe068f312f03f527d5a3c77e84": "8ff1a487b98a",
}
SOURCE_VARIANT_PREFIX: Final = "fc91d24-long-context-w4afp8-"
VARIANT_PREFIX: Final = "v0520-shared-kv-long-context-w4afp8-"
SOURCE_HICACHE_TAG: Final = "hicache-cuda-host-pooled-v1-"
HICACHE_TAG: Final = "hicache-cuda-host-pooled-v1-l3-shared-file-"
STORE_MOUNT: Final = "/shared-kv"
STORE_BUDGET: Final = "${GLM53_SHARED_KV_STORE_BUDGET:-400GiB}"
# HiCacheFile's size syntax (Gi/G, not GiB). Rank 0 is capped at this; each other TP rank bounds its own
# mamba sidecars at 25% of it, so a replica uses at most cap x (1 + 3 x 0.25) = 1.75 cap. Two replicas:
# 3.5 cap <= store budget, i.e. <= 114 GiB for 400 GiB.
REPLICA_CAP: Final = "${GLM53_SHARED_KV_REPLICA_CAP:-110Gi}"
# Above the store budget so the budget and the evictors bind first, never tmpfs ENOSPC.
TMPFS_SIZE: Final = "${GLM53_SHARED_KV_TMPFS_SIZE:-440g}"

HEADER: Final = (
    "# GLM-5.3 Flash long-context tier with an in-CVM SHARED KV cache, generated from\n"
    "# prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml by scripts/prepare_glm53_shared_kv.py.\n"
    "# Both replicas run docker/sglang-glm53-v0520-shared-kv and share one tmpfs KV store\n"
    "# (volume shared_kv at /shared-kv): a prompt one replica computed restores on the other\n"
    "# instead of re-prefilling (gpu32: 3.3x faster at 131K, 4.0x at 262K, logprobs at the noise\n"
    "# floor). Each replica keeps its 406 GiB private host tier; the startup budget reserves the\n"
    "# store's growth (GLM53_SHARED_KV_STORE_BUDGET, default 400GiB) and each replica's evictor is\n"
    "# capped so both replicas fit it (GLM53_SHARED_KV_REPLICA_CAP). Engine flags are otherwise unchanged from\n"
    "# the source, so this isolates the sharing change. Not yet soaked on an HCC/PPCIe host: see\n"
    "# docs/glm53-shared-kv-rollout.md before deploying.\n"
    "#\n"
    "# ---- source header follows ----\n"
)


def replace(text: str, old: str, new: str, count: int) -> str:
    found = text.count(old)
    if found != count:
        raise SystemExit(f"expected {count} of {old!r} in {SOURCE}, found {found}")
    return text.replace(old, new)


def render(source: str, image: str) -> str:
    if not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", image):
        raise SystemExit(f"image must be a digest-pinned reference, got {image!r}")
    short = image.rsplit(":", 1)[1][:12]
    out = source
    for src_image, src_short in SOURCE_IMAGES.items():
        out = replace(out, f"image: {src_image}\n", f"image: {image}\n", 1)
        # Datadog tag (inside the JSON label), then the OTel label and the collector attribute.
        out = replace(out, f"engine_image:{src_short}", f"engine_image:{short}", 1)
        out = replace(out, f'engine_image: "{src_short}"', f'engine_image: "{short}"', 2)
    for indent in ("      ", "        "):
        out = replace(
            out,
            f"\n{indent}--hicache-mem-layout page_first_direct\n",
            f"\n{indent}--hicache-mem-layout page_first_direct\n"
            f"{indent}--hicache-storage-backend file\n"
            f"{indent}--hicache-storage-prefetch-policy wait_complete\n",
            1,
        )
    for indent in ("    ", "      "):
        out = replace(
            out,
            f"\n{indent}- SGLANG_HICACHE_STAGING_PAGES=64\n",
            f"\n{indent}- SGLANG_HICACHE_STAGING_PAGES=64\n"
            f"{indent}# Shared KV store: one tmpfs volume in guest RAM, mounted by both replicas.\n"
            f"{indent}- SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR={STORE_MOUNT}\n"
            f"{indent}- SGLANG_HICACHE_SHARED_STORE_BUDGET={STORE_BUDGET}\n"
            f"{indent}- SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE={REPLICA_CAP}\n",
            1,
        )
    out = replace(
        out,
        "    - kernel_cache:/root/.cache\n    - huggingface_cache:/root/.cache/huggingface\n  environment:\n",
        "    - kernel_cache:/root/.cache\n    - huggingface_cache:/root/.cache/huggingface\n"
        f"    - shared_kv:{STORE_MOUNT}\n  environment:\n",
        1,
    )
    out = replace(
        out,
        "volumes:\n  otelcol_app_storage:\n  huggingface_cache:\n  kernel_cache:\n",
        "volumes:\n  otelcol_app_storage:\n  huggingface_cache:\n  kernel_cache:\n"
        "  shared_kv:\n"
        "    driver: local\n"
        "    driver_opts:\n"
        "      type: tmpfs\n"
        "      device: tmpfs\n"
        f'      o: "size={TMPFS_SIZE},mode=0700"\n',
        1,
    )
    variants = sorted(set(re.findall(re.escape(SOURCE_VARIANT_PREFIX) + r"[a-z0-9-]+", out)))
    if len(variants) != 2:
        raise SystemExit(f"expected the two replica config variants in {SOURCE}, found {variants}")
    for variant in variants:
        new = variant.replace(SOURCE_VARIANT_PREFIX, VARIANT_PREFIX, 1).replace(SOURCE_HICACHE_TAG, HICACHE_TAG, 1)
        out = out.replace(variant, new)
    return HEADER + out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--stdout", action="store_true", help="print the rendering instead of writing it")
    ap.add_argument("--image", help="digest-pinned image; defaults to RELEASED_IMAGE")
    a = ap.parse_args()
    if a.image:
        image = a.image
    elif (ROOT / RELEASED_IMAGE).exists():
        image = (ROOT / RELEASED_IMAGE).read_text().strip()
    else:
        raise SystemExit(f"{RELEASED_IMAGE} is absent: publish docker/sglang-glm53-v0520-shared-kv first, or pass --image")
    rendered = render((ROOT / SOURCE).read_text(), image)
    if a.stdout:
        sys.stdout.write(rendered)
        return 0
    target = ROOT / TARGET
    if a.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered)
        print(f"wrote {TARGET}")
        return 0
    current = target.read_text() if target.exists() else ""
    if current == rendered:
        print(f"{TARGET} is up to date")
        return 0
    sys.stdout.writelines(difflib.unified_diff(current.splitlines(True), rendered.splitlines(True), str(TARGET), "rendered"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
