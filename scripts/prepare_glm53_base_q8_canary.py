#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/prepare_glm53_base_q8_canary.py --write   (or --check)
"""Generate the GLM-5.3 base-tier v12 Q8-prefill one-replica canary file for gpu04 r4 from the committed base fleet file.

The output is a full copy of prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml (v7) with ONLY replica r4 changed, to exactly what
r4 runs in the v12 fleet file (scripts/prepare_glm53_base_v12_q8_fleet.py, whose image, argv edits and labels it imports):
- image: glm53-hicache-w4afp8-v12 (v10's Q8 DSA prefill and KDA + the K-pool indexer logits budget + CVE pins);
- command: r4 gets its own `command: >` block, a copy of the x-sg-glm53-flash-candidate anchor argv with the fleet
  generator's ARGV_EDITS applied (today ONLY `--dsa-prefill-backend flashmla_kv` -> `flashmla_sparse_q8`; chunk 8192).
  The anchor itself is not touched, so r1-r3 keep the v7 argv;
- r4's engine_image / config_variant (-v7 -> -v12-q8) telemetry labels (service block and its OTel scrape job).
Environment, GPUs, decode backend, every other service and every other label are byte-identical to the fleet file,
so gpu04 r1-r3 are the same-host v7 controls and the dashboard keeps selecting r4.

`--write` regenerates the target; `--check` prints a diff and exits non-zero when the committed target is stale.
Runbook: docs/glm53-base-v12-q8-rollout.md.
"""

import argparse
import difflib
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import prepare_glm53_base_v12_q8_fleet as fleet  # noqa: E402

SOURCE: Final = fleet.SOURCE
TARGET: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-Q8Canary.yaml")
SERVICE: Final = "model-sg-glm53-w4afp8-tp2-r4"
ANCHOR: Final = fleet.ANCHOR
HOST: Final = "gpu04"
RUNBOOK: Final = "docs/glm53-base-v12-q8-rollout.md"

V7_IMAGE_DIGEST: Final = fleet.V7_IMAGE_DIGEST
V12_IMAGE_DIGEST: Final = fleet.V12_IMAGE_DIGEST
V12_IMAGE: Final = fleet.V12_IMAGE
V7_LABEL: Final = fleet.V7_LABEL
V12_LABEL: Final = fleet.V12_LABEL
V7_VARIANT_SUFFIX: Final = fleet.V7_VARIANT_SUFFIX
V12_VARIANT_SUFFIX: Final = fleet.V12_VARIANT_SUFFIX
Q8_BACKEND: Final = fleet.Q8_BACKEND

GenerationError = fleet.GenerationError
replace_exact = fleet.replace_exact


def generate(source: str) -> str:
    if Q8_BACKEND in source or V12_IMAGE_DIGEST in source:
        raise GenerationError("source already carries the Q8 canary (flashmla_sparse_q8 or the v12 image)")
    argv = fleet.v12_argv(fleet.anchor_argv(source))
    s_start, s_end = fleet.service_span(source, SERVICE)
    block = source[s_start:s_end]
    if "\n    image:" in block or "\n    command:" in block:
        raise GenerationError(f"service {SERVICE} sets its own image or command; the canary expects both to inherit")
    block = replace_exact(
        block,
        f"    container_name: {SERVICE}\n",
        f"    container_name: {SERVICE}\n"
        f"    # Base-tier v12 Q8 canary ({RUNBOOK}): v12 image, and command is the anchor argv with only\n"
        "    # the prefill backend changed to flashmla_sparse_q8 (8K chunk kept). The anchor is not touched.\n"
        f"    image: {V12_IMAGE}\n"
        "    command: >\n"
        f"{argv}",
        1,
        "container_name",
    )
    block = fleet.relabel_service(block, SERVICE)
    out = source[:s_start] + block + source[s_end:]
    out = fleet.replace_in_span(out, fleet.scrape_job_span(out, SERVICE), lambda job: fleet.relabel_scrape_job(job, SERVICE))
    header = (
        f"# GLM-5.3 Flash base-tier v12 Q8-prefill CANARY file for {HOST}, generated from {SOURCE} by\n"
        "# scripts/prepare_glm53_base_q8_canary.py (do not hand-edit). Full copy of the v7 base fleet file with ONLY\n"
        f"# {SERVICE} changed: the glm53-hicache-w4afp8-v12 image ({V12_IMAGE_DIGEST[:19]}...), its own command (the\n"
        "# anchor argv with --dsa-prefill-backend flashmla_sparse_q8, chunk 8192), and its engine_image /\n"
        f"# config_variant ({V12_VARIANT_SUFFIX}) labels; r4 equals r4 of the v12 fleet file. Deploy it ONLY to {HOST}, scoped\n"
        f"# to that one service; roll back by redeploying {SOURCE} for the same service.\n"
        f"# Runbook: {RUNBOOK}.\n"
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
    target = ROOT / TARGET
    try:
        expected = generate((ROOT / SOURCE).read_text())
        if args.write:
            _ = target.write_text(expected)
            return 0
        actual = target.read_text() if target.exists() else ""
    except (GenerationError, OSError) as exc:
        print(f"prepare_glm53_base_q8_canary: {exc}", file=sys.stderr)
        return 2
    if actual != expected:
        print("".join(difflib.unified_diff(actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                                           fromfile=f"a/{TARGET}", tofile=f"b/{TARGET}")), end="")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
