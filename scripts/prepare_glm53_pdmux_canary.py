#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/prepare_glm53_pdmux_canary.py --write   (or --check)
"""Generate the GLM-5.3 PD-Multiplexing + EAGLE (pdmux) one-replica canary file for gpu04 from the committed base fleet file.

The output is a full copy of prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml with ONLY replica r4 changed:
  - the glm53-hicache-w4afp8-v7-pdmuxe image (the v7 recipe plus pdmux-eagle.diff, docker/sglang-glm53-hicache-w4afp8-pdmuxe),
  - its argv: the v7 fleet argv unchanged plus `--enable-pdmux --sm-group-num 3 --pdmux-config-path /etc/glm53/pdmux-ov96-l4.yaml`
    (the YAML is baked into the image), and
  - r4's engine_image / config_variant telemetry labels (service block and its OTel scrape job).
Every other flag, the environment, GPUs, every other service and every other label are byte-identical to the fleet
file, so r1-r3 on the same host are the v7 controls and the dashboard keeps selecting r4.

`--write` regenerates the target; `--check` prints a diff and exits non-zero when the committed target is stale.
Runbook: docs/glm53-pdmux-canary.md.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml")
TARGET: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-PdmuxCanary.yaml")
SERVICE: Final = "model-sg-glm53-w4afp8-tp2-r4"
HOST: Final = "gpu04"
ANCHOR: Final = "x-sg-glm53-flash-candidate"

V7_IMAGE_DIGEST: Final = "sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5"
# PLACEHOLDER until the image is published. After a human runs the "Publish GLM-5.3 admission-reserve engine images"
# workflow with variant hicache-w4afp8-pdmuxe from a merged main commit, replace it with the digest of
# docker.io/nearaidev/sglang:glm53-hicache-w4afp8-v7-pdmuxe (the publish run's "immutable" line) and note the run id below.
# Record it like prepare_glm53_v8_kda_canary.py does: "published from main <sha>, #<PR>; publish run <id>".
# docs/glm53-pdmux-canary.md, "Publish the image".
PDMUX_IMAGE_DIGEST: Final = "sha256:0000000000000000000000000000000000000000000000000000000000000000"  # TODO(publish): real digest
PDMUX_IMAGE: Final = f"docker.io/nearaidev/sglang@{PDMUX_IMAGE_DIGEST}"
V7_LABEL: Final = V7_IMAGE_DIGEST.split(":")[1][:12]
PDMUX_LABEL: Final = PDMUX_IMAGE_DIGEST.split(":")[1][:12]
SUFFIX: Final = "-pdmuxe-ov96-l4"
PDMUX_ARGS: Final = (
    "--enable-pdmux",
    "--sm-group-num 3",
    "--pdmux-config-path /etc/glm53/pdmux-ov96-l4.yaml",
)


class GenerationError(ValueError):
    pass


def is_placeholder() -> bool:
    return set(PDMUX_IMAGE_DIGEST.split(":")[1]) == {"0"}


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


def anchor_command(source: str) -> list[str]:
    """The anchor's folded `command: >` lines, still at their anchor indentation (6 spaces)."""
    a_start, a_end = span(source, f"\n{ANCHOR}: &sg-glm53-flash-candidate\n", r"\n\S", f"anchor {ANCHOR}")
    block = source[a_start:a_end]
    marker = "\n  command: >\n"
    if block.count(marker) != 1:
        raise GenerationError(f"{ANCHOR}: command block not found exactly once")
    lines = []
    for line in block[block.index(marker) + len(marker):].split("\n"):
        if not line.startswith("      "):
            break
        lines.append(line)
    if not lines or lines[0].strip() != "sglang serve":
        raise GenerationError(f"{ANCHOR}: unexpected command block")
    for flag in ("--speculative-algorithm EAGLE", "--disable-overlap-schedule", "--max-queued-requests 32"):
        if not any(line.strip() == flag for line in lines):
            raise GenerationError(f"{ANCHOR}: expected `{flag}` in the fleet argv")
    if any("pdmux" in line or "--sm-group-num" in line for line in lines):
        raise GenerationError(f"{ANCHOR}: the fleet argv already mentions pdmux")
    return lines


def generate(source: str) -> str:
    if "pdmux" in source or PDMUX_IMAGE in source:
        raise GenerationError("source already carries the pdmux canary")
    command = anchor_command(source)
    s_start, s_end = span(source, f"\n  {SERVICE}:\n", r"\n  [^ \n]", f"service {SERVICE}")
    block = source[s_start:s_end]
    if "\n    image:" in block or "\n    command:" in block:
        raise GenerationError("r4 sets its own image or command; the canary expects it to inherit both from the v7 anchor")
    # Indent the anchor command two more spaces (service keys sit at 4, anchor keys at 2) and append the pdmux flags.
    emitted = ["    command: >"] + ["  " + line for line in command] + [f"        {arg}" for arg in PDMUX_ARGS]
    block = replace_exact(
        block,
        f"    container_name: {SERVICE}\n",
        f"    container_name: {SERVICE}\n"
        "    # PD-Multiplexing + EAGLE canary (docs/glm53-pdmux-canary.md): the v7 image plus pdmux-eagle.diff and the v7 fleet argv\n"
        "    # plus the three pdmux flags; the pdmux YAML is baked into the image. Nothing else changes.\n"
        f"    image: {PDMUX_IMAGE}\n" + "\n".join(emitted) + "\n",
        1,
        "image and command",
    )
    found = re.search(r'nearai\.otel\.config_variant: "([^"]+)"', block)
    if found is None:
        raise GenerationError(f"service {SERVICE}: nearai.otel.config_variant label not found")
    variant = found.group(1)
    block = replace_exact(block, f'"engine_image:{V7_LABEL}"', f'"engine_image:{PDMUX_LABEL}"', 1, "log engine_image")
    block = replace_exact(block, f'nearai.otel.engine_image: "{V7_LABEL}"', f'nearai.otel.engine_image: "{PDMUX_LABEL}"', 1, "otel engine_image")
    block = replace_exact(block, f'{variant}"', f'{variant}{SUFFIX}"', 2, "service config_variant")
    out = source[:s_start] + block + source[s_end:]
    j_start, j_end = span(out, f"\n              - job_name: sglang-{SERVICE}\n", r"\n              - job_name:", "scrape job")
    job = out[j_start:j_end]
    job = replace_exact(job, f'engine_image: "{V7_LABEL}"', f'engine_image: "{PDMUX_LABEL}"', 1, "scrape engine_image")
    job = replace_exact(job, f'config_variant: "{variant}"', f'config_variant: "{variant}{SUFFIX}"', 1, "scrape config_variant")
    out = out[:j_start] + job + out[j_end:]
    header = (
        f"# GLM-5.3 Flash PD-Multiplexing + EAGLE (pdmux) CANARY file for {HOST}, generated from {SOURCE} by\n"
        "# scripts/prepare_glm53_pdmux_canary.py (do not hand-edit). Full copy of the base fleet file with ONLY\n"
        f"# {SERVICE} changed: the glm53-hicache-w4afp8-v7-pdmuxe image ({PDMUX_IMAGE_DIGEST[:19]}...), the three pdmux\n"
        f"# flags appended to its argv, and its engine_image / config_variant ({SUFFIX}) labels. Deploy it ONLY to {HOST},\n"
        f"# scoped to that one service; roll back by redeploying {SOURCE} for the same service.\n"
        "# docs/glm53-pdmux-canary.md.\n"
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
    try:
        expected = generate((ROOT / SOURCE).read_text())
    except (GenerationError, OSError) as exc:
        print(f"prepare_glm53_pdmux_canary: {exc}", file=sys.stderr)
        return 2
    if is_placeholder():
        print("prepare_glm53_pdmux_canary: WARNING: PDMUX_IMAGE_DIGEST is still the placeholder; do not deploy this file", file=sys.stderr)
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
