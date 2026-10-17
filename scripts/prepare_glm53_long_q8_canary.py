#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/prepare_glm53_long_q8_canary.py --write   (or --check)
"""Generate the GLM-5.3 long-tier Q8 + 16K-chunk one-replica canary file from the committed long fleet file.

The output is a full copy of prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml with ONLY the TP2 replica r2b
changed:
- image: glm53-hicache-w4afp8-v9 (v8 + dsa-q8-flashmla-sparse.diff, flag-gated);
- argv: --dsa-prefill-backend flashmla_kv -> flashmla_sparse_q8 and --chunked-prefill-size 8192 -> 16384
  (SGLANG_DSA_INDEXER_QSPLIT=1 is already set on r2b, which 16384 requires);
- r2b's engine_image / config_variant telemetry labels (service block and its OTel scrape job).
Decode backend, environment, GPUs, every other service and every other label are byte-identical to the fleet
file, so the other TP2 replicas on the same host are v7 controls and the dashboard keeps selecting r2b.

`--write` regenerates the target; `--check` prints a diff and exits non-zero when the committed target is stale.
Runbook: docs/glm53-long-q8-canary.md.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE: Final = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml")
TARGET: Final = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext-Q8Canary.yaml")
SERVICE: Final = "model-sg-glm53-w4afp8-tp2-r2b"

V7_IMAGE_DIGEST: Final = "sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5"
# glm53-hicache-w4afp8-v9 (v8 + dsa-q8-flashmla-sparse.diff). Fill in from the publish run after the recipe
# PR merges; generation refuses to run while it is empty.
V9_IMAGE_DIGEST: Final = ""
V7_LABEL: Final = V7_IMAGE_DIGEST.split(":")[1][:12]
SUFFIX: Final = "-q8-c16384"


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


def generate(source: str, digest: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest or ""):
        raise GenerationError("no v9 image digest: set V9_IMAGE_DIGEST (or pass --image-digest) after the publish run")
    v9_image = f"docker.io/nearaidev/sglang@{digest}"
    v9_label = digest.split(":")[1][:12]
    if "flashmla_sparse_q8" in source or v9_image in source:
        raise GenerationError("source already carries the Q8 canary")
    s_start, s_end = span(source, f"\n  {SERVICE}:\n", r"\n  [^ \n]", f"service {SERVICE}")
    block = source[s_start:s_end]
    block = replace_exact(
        block,
        f"    image: docker.io/nearaidev/sglang@{V7_IMAGE_DIGEST}\n",
        "    # Long-tier Q8 + 16K-chunk canary (docs/glm53-long-q8-canary.md): v9 image, two argv flags.\n"
        f"    image: {v9_image}\n",
        1,
        "image",
    )
    block = replace_exact(block, "        --chunked-prefill-size 8192\n", "        --chunked-prefill-size 16384\n", 1, "chunk")
    block = replace_exact(
        block, "        --dsa-prefill-backend flashmla_kv\n", "        --dsa-prefill-backend flashmla_sparse_q8\n", 1, "prefill backend"
    )
    if "      - SGLANG_DSA_INDEXER_QSPLIT=1\n" not in block:
        raise GenerationError("r2b must keep SGLANG_DSA_INDEXER_QSPLIT=1 (required with chunk 16384)")
    found = re.search(r'nearai\.otel\.config_variant: "([^"]+)"', block)
    if found is None:
        raise GenerationError(f"service {SERVICE}: nearai.otel.config_variant label not found")
    variant = found.group(1)
    block = replace_exact(block, f'"engine_image:{V7_LABEL}"', f'"engine_image:{v9_label}"', 1, "log engine_image")
    block = replace_exact(block, f'nearai.otel.engine_image: "{V7_LABEL}"', f'nearai.otel.engine_image: "{v9_label}"', 1, "otel engine_image")
    block = replace_exact(block, f'{variant}"', f'{variant}{SUFFIX}"', 2, "service config_variant")
    out = source[:s_start] + block + source[s_end:]
    j_start, j_end = span(out, f"\n              - job_name: sglang-{SERVICE}\n", r"\n              - job_name:", "scrape job")
    job = out[j_start:j_end]
    job = replace_exact(job, f'engine_image: "{V7_LABEL}"', f'engine_image: "{v9_label}"', 1, "scrape engine_image")
    job = replace_exact(job, f'config_variant: "{variant}"', f'config_variant: "{variant}{SUFFIX}"', 1, "scrape config_variant")
    out = out[:j_start] + job + out[j_end:]
    header = (
        "# GLM-5.3 Flash long-tier Q8 + 16K-chunk CANARY file, generated from\n"
        f"# {SOURCE} by scripts/prepare_glm53_long_q8_canary.py (do not hand-edit).\n"
        f"# Full copy of the long fleet file with ONLY {SERVICE} changed: the glm53-hicache-w4afp8-v9 image\n"
        f"# ({digest[:19]}...), --dsa-prefill-backend flashmla_sparse_q8, --chunked-prefill-size 16384, and its\n"
        f"# engine_image / config_variant ({SUFFIX}) labels. Deploy it ONLY to the one host named in the runbook, scoped\n"
        f"# to that one service; roll back by redeploying {SOURCE} for the same service.\n"
        "# docs/glm53-long-q8-canary.md.\n"
        "#\n"
    )
    return header + out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--write", action="store_true", help="Write the target")
    _ = parser.add_argument("--check", action="store_true", help="Fail with a diff when the committed target is stale")
    _ = parser.add_argument("--image-digest", default=V9_IMAGE_DIGEST, help="v9 image digest (sha256:...)")
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("pass exactly one of --write / --check")
    try:
        expected = generate((ROOT / SOURCE).read_text(), args.image_digest)
    except (GenerationError, OSError) as exc:
        print(f"prepare_glm53_long_q8_canary: {exc}", file=sys.stderr)
        return 2
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
