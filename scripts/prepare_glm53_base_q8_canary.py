#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/prepare_glm53_base_q8_canary.py --write   (or --check)
"""Generate the GLM-5.3 base-tier v10 Q8-prefill one-replica canary file for gpu04 r4 from the committed base fleet file.

The output is a full copy of prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml with ONLY replica r4 changed:
- image: glm53-hicache-w4afp8-v10 (v8 + flag-gated Q8 DSA patch #363 + linux-libc-dev CVE fix #365);
- command: r4 gets its own `command: >` block, a copy of the x-sg-glm53-flash-candidate anchor argv with ONLY
  `--dsa-prefill-backend flashmla_kv` -> `flashmla_sparse_q8`. The anchor itself is not touched, so r1-r3 keep
  the v7 argv. The chunk stays `--chunked-prefill-size 8192` (16K hurt base decode in the lab);
- r4's engine_image / config_variant telemetry labels (service block and its OTel scrape job).
Environment, GPUs, decode backend, every other service and every other label are byte-identical to the fleet file,
so gpu04 r1-r3 are the same-host v7 controls and the dashboard keeps selecting r4.

`--write` regenerates the target; `--check` prints a diff and exits non-zero when the committed target is stale.
Runbook: docs/glm53-base-q8-canary.md.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml")
TARGET: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-Q8Canary.yaml")
SERVICE: Final = "model-sg-glm53-w4afp8-tp2-r4"
ANCHOR: Final = "x-sg-glm53-flash-candidate"
HOST: Final = "gpu04"

V7_IMAGE_DIGEST: Final = "sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5"
# glm53-hicache-w4afp8-v10: v8 + dsa-q8-flashmla-sparse.diff (#363) plus the linux-libc-dev CVE fix (#365); published
# from main 913d6995 by publish run 39094781691 (attested and signed).
V10_IMAGE_DIGEST: Final = "sha256:c15a33672ade5a9f2b4f7874a746eb4114952dae7230d5baa563a02eca081b78"
V10_IMAGE: Final = f"docker.io/nearaidev/sglang@{V10_IMAGE_DIGEST}"
V7_LABEL: Final = V7_IMAGE_DIGEST.split(":")[1][:12]
V10_LABEL: Final = V10_IMAGE_DIGEST.split(":")[1][:12]
SUFFIX: Final = "-q8"

Q8_BACKEND: Final = "flashmla_sparse_q8"
OLD_BACKEND_LINE: Final = "      --dsa-prefill-backend flashmla_kv\n"
NEW_BACKEND_LINE: Final = "      --dsa-prefill-backend flashmla_sparse_q8\n"
CHUNK_LINE: Final = "      --chunked-prefill-size 8192\n"


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


def command_body(text: str) -> str:
    """The folded `command: >` body: the run of lines indented 6 spaces that follows the header."""
    body: list[str] = []
    for line in text.splitlines(keepends=True):
        if not line.startswith("      "):
            break
        body.append(line)
    return "".join(body)


def anchor_argv(source: str) -> str:
    a_start, a_end = span(source, f"\n{ANCHOR}: &sg-glm53-flash-candidate\n", r"\n[^ \n#]", f"anchor {ANCHOR}")
    anchor = source[a_start:a_end]
    if anchor.count("\n  command: >\n") != 1:
        raise GenerationError(f"anchor {ANCHOR}: expected exactly one `  command: >` block")
    body = command_body(anchor.split("\n  command: >\n", 1)[1])
    if body.count(OLD_BACKEND_LINE) != 1:
        raise GenerationError(f"anchor {ANCHOR}: expected `--dsa-prefill-backend flashmla_kv` exactly once")
    if body.count(CHUNK_LINE) != 1:
        raise GenerationError(f"anchor {ANCHOR}: expected `--chunked-prefill-size 8192` exactly once")
    return body


def generate(source: str) -> str:
    if Q8_BACKEND in source or V10_IMAGE in source:
        raise GenerationError("source already carries the Q8 canary (flashmla_sparse_q8 or the v10 image)")
    argv = replace_exact(anchor_argv(source), OLD_BACKEND_LINE, NEW_BACKEND_LINE, 1, "anchor prefill backend")
    s_start, s_end = span(source, f"\n  {SERVICE}:\n", r"\n  [^ \n]", f"service {SERVICE}")
    block = source[s_start:s_end]
    if "\n    image:" in block or "\n    command:" in block:
        raise GenerationError(f"service {SERVICE} sets its own image or command; the canary expects both to inherit")
    block = replace_exact(
        block,
        f"    container_name: {SERVICE}\n",
        f"    container_name: {SERVICE}\n"
        "    # Base-tier Q8 canary (docs/glm53-base-q8-canary.md): v10 image, and command is the anchor argv with only\n"
        "    # the prefill backend changed to flashmla_sparse_q8 (8K chunk kept). The anchor is not touched.\n"
        f"    image: {V10_IMAGE}\n"
        "    command: >\n"
        f"{argv}",
        1,
        "container_name",
    )
    found = re.search(r'nearai\.otel\.config_variant: "([^"]+)"', block)
    if found is None:
        raise GenerationError(f"service {SERVICE}: nearai.otel.config_variant label not found")
    variant = found.group(1)
    block = replace_exact(block, f'"engine_image:{V7_LABEL}"', f'"engine_image:{V10_LABEL}"', 1, "log engine_image")
    block = replace_exact(block, f'nearai.otel.engine_image: "{V7_LABEL}"', f'nearai.otel.engine_image: "{V10_LABEL}"', 1, "otel engine_image")
    block = replace_exact(block, f'{variant}"', f'{variant}{SUFFIX}"', 2, "service config_variant")
    out = source[:s_start] + block + source[s_end:]
    j_start, j_end = span(out, f"\n              - job_name: sglang-{SERVICE}\n", r"\n              - job_name:", "scrape job")
    job = out[j_start:j_end]
    job = replace_exact(job, f'engine_image: "{V7_LABEL}"', f'engine_image: "{V10_LABEL}"', 1, "scrape engine_image")
    job = replace_exact(job, f'config_variant: "{variant}"', f'config_variant: "{variant}{SUFFIX}"', 1, "scrape config_variant")
    out = out[:j_start] + job + out[j_end:]
    header = (
        f"# GLM-5.3 Flash base-tier v10 Q8-prefill CANARY file for {HOST}, generated from {SOURCE} by\n"
        "# scripts/prepare_glm53_base_q8_canary.py (do not hand-edit). Full copy of the base fleet file with ONLY\n"
        f"# {SERVICE} changed: the glm53-hicache-w4afp8-v10 image ({V10_IMAGE_DIGEST[:19]}...), its own command (the\n"
        "# anchor argv with --dsa-prefill-backend flashmla_sparse_q8, chunk 8192), and its engine_image /\n"
        f"# config_variant ({SUFFIX}) labels. Deploy it ONLY to {HOST}, scoped to that one service; roll back by\n"
        f"# redeploying {SOURCE} for the same service. Runbook: docs/glm53-base-q8-canary.md.\n"
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
