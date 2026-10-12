#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/prepare_glm53_v8_kda_canary.py --write   (or --check)
"""Generate the GLM-5.3 v8 (zero-sync KDA) one-replica canary file for gpu03 from the committed base fleet file.

The output is a full copy of prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml with ONLY replica r4 changed: the
glm53-hicache-w4afp8-v8 image (v7 + kda-zero-sync.diff, #352) and r4's engine_image / config_variant telemetry labels
(service block and its OTel scrape job). The argv, environment, GPUs, every other service and every other label are
byte-identical to the fleet file, so r1-r3 on the same host are the v7 controls and the dashboard keeps selecting r4.

`--write` regenerates the target; `--check` prints a diff and exits non-zero when the committed target is stale.
Runbook: docs/glm53-v8-kda-canary.md.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml")
TARGET: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V8KdaCanary.yaml")
SERVICE: Final = "model-sg-glm53-w4afp8-tp2-r4"
HOST: Final = "gpu03"

V7_IMAGE_DIGEST: Final = "sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5"
# glm53-hicache-w4afp8-v8 (published from main 650932a, #352; publish run 38210800164).
V8_IMAGE_DIGEST: Final = "sha256:59700363c0b6885301ea1f344fe3d45c8335597fdef4ffd49def79dca347f91d"
V8_IMAGE: Final = f"docker.io/nearaidev/sglang@{V8_IMAGE_DIGEST}"
V7_LABEL: Final = V7_IMAGE_DIGEST.split(":")[1][:12]
V8_LABEL: Final = V8_IMAGE_DIGEST.split(":")[1][:12]
SUFFIX: Final = "-kda"


class GenerationError(ValueError):
    pass


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    actual = text.count(old)
    if actual != expected:
        raise GenerationError(f"{label}: expected {expected} matches, found {actual}")
    return text.replace(old, new)


def span(text: str, start_marker: str, end_pattern: str, label: str) -> tuple[int, int]:
    if text.count(start_marker) != 1:
        raise GenerationError(f"{label}: start marker not found exactly once")
    start = text.find(start_marker)
    match = re.search(end_pattern, text[start + len(start_marker):])
    if not match:
        raise GenerationError(f"{label}: end marker not found")
    return start, start + len(start_marker) + match.start()


def generate(source: str) -> str:
    if "kda-zero-sync" in source or V8_IMAGE in source:
        raise GenerationError("source already carries the v8 canary")
    s_start, s_end = span(source, f"\n  {SERVICE}:\n", r"\n  [^ \n]", f"service {SERVICE}")
    block = source[s_start:s_end]
    if "\n    image:" in block:
        raise GenerationError("r4 sets its own image; the canary expects it to inherit the v7 anchor image")
    block = replace_exact(
        block,
        f"    container_name: {SERVICE}\n",
        f"    container_name: {SERVICE}\n"
        "    # v8 zero-sync KDA canary (docs/glm53-v8-kda-canary.md): v7 + kda-zero-sync.diff, nothing else changes.\n"
        f"    image: {V8_IMAGE}\n",
        1,
        "image",
    )
    variant = re.search(r'nearai\.otel\.config_variant: "([^"]+)"', block).group(1)
    block = replace_exact(block, f'"engine_image:{V7_LABEL}"', f'"engine_image:{V8_LABEL}"', 1, "log engine_image")
    block = replace_exact(block, f'nearai.otel.engine_image: "{V7_LABEL}"', f'nearai.otel.engine_image: "{V8_LABEL}"', 1, "otel engine_image")
    block = replace_exact(block, f'{variant}"', f'{variant}{SUFFIX}"', 2, "service config_variant")
    out = source[:s_start] + block + source[s_end:]
    j_start, j_end = span(out, f"\n              - job_name: sglang-{SERVICE}\n", r"\n              - job_name:", "scrape job")
    job = out[j_start:j_end]
    job = replace_exact(job, f'engine_image: "{V7_LABEL}"', f'engine_image: "{V8_LABEL}"', 1, "scrape engine_image")
    job = replace_exact(job, f'config_variant: "{variant}"', f'config_variant: "{variant}{SUFFIX}"', 1, "scrape config_variant")
    out = out[:j_start] + job + out[j_end:]
    header = (
        f"# GLM-5.3 Flash v8 zero-sync KDA CANARY file for {HOST}, generated from {SOURCE} by\n"
        "# scripts/prepare_glm53_v8_kda_canary.py (do not hand-edit). Full copy of the base fleet file with ONLY\n"
        f"# {SERVICE} changed: the glm53-hicache-w4afp8-v8 image ({V8_IMAGE_DIGEST[:19]}...) and its engine_image /\n"
        f"# config_variant ({SUFFIX}) labels. Deploy it ONLY to {HOST}, scoped to that one service; roll back by redeploying\n"
        f"# {SOURCE} for the same service. docs/glm53-v8-kda-canary.md.\n"
        "#\n"
    )
    return header + out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--write", action="store_true", help="Write the target")
    _ = parser.add_argument("--check", action="store_true", help="Fail with a diff when the committed target is stale")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("pass exactly one of --write / --check")
    expected = generate((ROOT / SOURCE).read_text())
    target = ROOT / TARGET
    if args.write:
        _ = target.write_text(expected)
        return 0
    actual = target.read_text() if target.exists() else ""
    if actual != expected:
        print("".join(difflib.unified_diff(actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                                           fromfile=f"a/{TARGET}", tofile=f"b/{TARGET}")), end="")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
