#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/prepare_glm53_base_v12_q8_fleet.py --write   (or --check)
"""Generate the GLM-5.3 base-tier v12 + Q8-prefill fleet file from the committed v7 base fleet file.

The output is a full copy of prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml (the v7 fleet file, kept unchanged as the
rollback) with ONLY:
- the engine image in the shared engine anchor (so all four replicas) and the glm53-ghost-aggregator image (it runs
  its engines' image): v7 -> glm53-hicache-w4afp8-v12, the image the long tier runs;
- the x-sg-glm53-flash-candidate argv, which every replica merges: ARGV_EDITS below, today only
  `--dsa-prefill-backend flashmla_kv` -> `flashmla_sparse_q8` (chunk 8192, queue 32 and everything else kept);
- the four replicas' engine_image / config_variant (-v7 -> -v12-q8) labels, in their service blocks and OTel scrape jobs;
- comments only: the generated header, the anchor comment and COMMENT_EDITS (two v7 comment lines that name the image).

ARGV_EDITS / ARGV_APPENDED are the one place to add a lever that wins later (exp 53 rejected EAGLE 3/1/4, chunk 4096, queue 64): the fleet anchor and
the gpu04 r4 canary (scripts/prepare_glm53_base_q8_canary.py imports them) both follow. The tests pin the exact
argv difference, so a new lever also needs its test line.

`--write` regenerates the target; `--check` prints a diff and exits non-zero when the committed target is stale.
Runbook: docs/glm53-base-v12-q8-rollout.md.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml")
TARGET: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml")
COMMON_ANCHOR: Final = "x-sg-glm53-flash-common: &sg-glm53-flash-common"
ANCHOR: Final = "x-sg-glm53-flash-candidate"
AGGREGATOR: Final = "glm53-ghost-aggregator"
SERVICE_PREFIX: Final = "model-sg-glm53-w4afp8-tp2-r"
REPLICAS: Final = tuple(f"{SERVICE_PREFIX}{replica}" for replica in (1, 2, 3, 4))

V7_IMAGE_DIGEST: Final = "sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5"
V7_IMAGE: Final = f"docker.io/nearaidev/sglang@{V7_IMAGE_DIGEST}"
# glm53-hicache-w4afp8-v12 = v10 (v8 zero-sync KDA + flag-gated flashmla_sparse_q8 DSA prefill #363 + CVE fix #365) plus the
# K-pool indexer logits budget (#370) and CVE pins (#384); publish run 39987811670, source 79de066, attested and signed. The
# long tier runs it since #385. Base-tier evidence: tee-bench exp 53 (branch exp53-glm53-base-fp8-load-ladder).
V12_IMAGE_DIGEST: Final = "sha256:da8d4e4b5ec667499b58a21e56050aae9bcfc4df9669b3ac27e2e20af033d681"
V12_IMAGE: Final = f"docker.io/nearaidev/sglang@{V12_IMAGE_DIGEST}"
V7_LABEL: Final = V7_IMAGE_DIGEST.split(":")[1][:12]
V12_LABEL: Final = V12_IMAGE_DIGEST.split(":")[1][:12]
# config_variant ends -v7 on the v7 fleet; the v12 base replicas end -v12-q8, as the long tier's do.
V7_VARIANT_SUFFIX: Final = "-v7"
V12_VARIANT_SUFFIX: Final = "-v12-q8"

Q8_BACKEND: Final = "flashmla_sparse_q8"
# Whole argv lines of the v7 anchor, replaced in place. Each old line must occur exactly once.
ARGV_EDITS: Final = (
    ("--dsa-prefill-backend flashmla_kv", f"--dsa-prefill-backend {Q8_BACKEND}"),
)
# New argv lines, appended after the v7 argv's last line (none today).
ARGV_APPENDED: Final = ()
# Must survive the edits unchanged: 16K chunks hurt base decode (exp 46c) and queue 64 lost to 32 (exp 53).
ARGV_KEPT: Final = ("--chunked-prefill-size 8192", "--max-queued-requests 32", "--dsa-decode-backend flashmla_kv")

ANCHOR_COMMENT: Final = (
    "  # v12 base fleet (docs/glm53-base-v12-q8-rollout.md): the v7 argv below with ONLY --dsa-prefill-backend\n"
    "  # flashmla_sparse_q8 (FP8/Q8 DSA prefill, 8K chunks kept); the v12 image comes from the common anchor.\n"
)

# Whole comment lines of the v7 file that would misstate this file's image, reworded in place. Each must occur exactly once.
COMMENT_EDITS: Final = (
    (
        "# (4 x 325 GiB; OFF in the v7 fleet config) and the DSA indexer split. All four pin the v7 image:\n",
        "# (4 x 325 GiB; OFF in the v7 fleet config) and the DSA indexer split. The v7 file pins the v7 image (this file: v12, top):\n",
    ),
    (
        "  # the environment and the v7 image, is inherited from the common anchor.\n",
        "  # the environment and the engine image (v12 in this file), is inherited from the common anchor.\n",
    ),
)

HEADER: Final = (
    "# GLM-5.3 Flash BASE TIER, v12 + Q8 DSA PREFILL FLEET FILE (gpu03 and gpu04), generated from\n"
    f"# {SOURCE} (the v7 fleet file) by scripts/prepare_glm53_base_v12_q8_fleet.py (do not hand-edit).\n"
    "# Full copy of the v7 fleet file with ONLY: the engine image of all four replicas and of glm53-ghost-aggregator moved to\n"
    f"# glm53-hicache-w4afp8-v12 ({V12_IMAGE_DIGEST[:19]}..., the long tier's image), every replica's argv with\n"
    "# --dsa-prefill-backend flashmla_sparse_q8 (chunk 8192, queue 32, flashmla_kv decode unchanged), and the replicas'\n"
    f"# engine_image / config_variant ({V12_VARIANT_SUFFIX}) labels. Deploy one replica at a time with a scoped compose/up\n"
    f"# (docs/glm53-base-v12-q8-rollout.md). ROLLBACK: redeploy {SOURCE} (v7) for the same service.\n"
    "# The v7 header below describes everything else; where it names the v7 image or flashmla_kv prefill, read v12 and\n"
    "# flashmla_sparse_q8 for the engines.\n"
    "#\n"
)


# --- SGLang v0.5.23 (docker/sglang-glm53-v0523) on all four base replicas ---------------------------------------------
# The common anchor's image and the candidate argv carry it, so r1-r4 stay identical. Keep in sync with
# W4AFP8_TP2X4_V12Q8_V0523 in scripts/validate_glm53_prod_config.rb.
V0523_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:050b1adf961c8621264d14696d24645f70d1a675509ad400a72ccfb1a3fddcb2"
V0523_LABEL: Final = V0523_IMAGE.split(":")[-1][:12]
# REQUIRED on v0.5.23: its default breakable prefill CUDA graphs OOM at warmup at --mem-fraction-static 0.86.
V0523_FLAGS: Final = ("--cuda-graph-backend-prefill disabled",)
V0523_SUFFIX: Final = "-v0523"
V0523_HEADER: Final = (
    f"# SGLang v0.5.23: r1-r4 run {V0523_IMAGE}\n"
    "#   (docker/sglang-glm53-v0523) with the v12 argv below plus --cuda-graph-backend-prefill disabled; the ghost aggregator\n"
    "#   stays on v12. One engine default differs: v0.5.23 turns SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION on (=0 restores it).\n"
    f"#   Labels: engine_image {V0523_LABEL}, config_variant <v12 variant without {V12_VARIANT_SUFFIX}>{V0523_SUFFIX}.\n"
    "#   DEPLOY: from a main tag, gpu03 only, one replica per scoped compose/up, then otelcol-contrib. gpu04 does not deploy this\n"
    "#   file (scripts/prepare_glm53_gpu04_3b1l.py renders it on v12). ROLLBACK: the previous tag's copy, same single service.\n"
)


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


def service_span(text: str, name: str) -> tuple[int, int]:
    return span(text, f"\n  {name}:\n", r"\n  [^ \n]", f"service {name}")


def scrape_job_span(text: str, name: str) -> tuple[int, int]:
    return span(text, f"\n              - job_name: sglang-{name}\n", r"\n              - job_name:", f"scrape job {name}")


def command_body(text: str) -> str:
    """The folded `command: >` body: the run of lines indented 6 spaces that follows the header.

    Strict, so the captured argv can never be silently truncated: every body line must be an argv line
    (`sglang serve` first, then `--flag ...`), and the body must end at the anchor's next 2-space key or at
    the end of the anchor. A blank, re-indented or comment line inside the block is an error.
    """
    body: list[str] = []
    lines = text.splitlines(keepends=True)
    for line in lines:
        if line.startswith("      "):
            token = line.strip()
            if not (token.startswith("--") or (not body and token == "sglang serve")):
                raise GenerationError(f"unexpected line inside the anchor's command block: {line.rstrip()!r}")
            body.append(line)
            continue
        if not re.match(r"  [A-Za-z_<]", line):
            raise GenerationError(f"the anchor's command block must end at a 2-space key, got {line.rstrip()!r}")
        break
    if not body or body[0].strip() != "sglang serve":
        raise GenerationError("the anchor's command block must start with `sglang serve`")
    return "".join(body)


def anchor_span(source: str) -> tuple[int, int]:
    return span(source, f"\n{ANCHOR}: &sg-glm53-flash-candidate\n", r"\n[^ \n#]", f"anchor {ANCHOR}")


def anchor_argv(source: str) -> str:
    """The v7 candidate anchor's argv body (6-space lines), validated."""
    a_start, a_end = anchor_span(source)
    anchor = source[a_start:a_end]
    if anchor.count("\n  command: >\n") != 1:
        raise GenerationError(f"anchor {ANCHOR}: expected exactly one `  command: >` block")
    return command_body(anchor.split("\n  command: >\n", 1)[1])


def v12_argv(v7_body: str) -> str:
    """The v7 argv body with ARGV_EDITS and ARGV_APPENDED applied, as 6-space argv lines."""
    lines = [line.strip() for line in v7_body.splitlines()]
    for old, new in ARGV_EDITS:
        if lines.count(old) != 1:
            raise GenerationError(f"anchor {ANCHOR}: expected `{old}` exactly once")
        lines[lines.index(old)] = new
    for kept in ARGV_KEPT:
        if lines.count(kept) != 1:
            raise GenerationError(f"anchor {ANCHOR}: expected `{kept}` exactly once")
    for added in ARGV_APPENDED:
        flag = added.split()[0]
        if any(line.split()[0] == flag for line in lines):
            raise GenerationError(f"anchor {ANCHOR}: `{flag}` is already set; edit it with ARGV_EDITS instead")
        lines.append(added)
    return "".join(f"      {line}\n" for line in lines)


def v12_variant(variant: str) -> str:
    if not variant.endswith(V7_VARIANT_SUFFIX):
        raise GenerationError(f"config_variant {variant!r} does not end {V7_VARIANT_SUFFIX}")
    return variant[: -len(V7_VARIANT_SUFFIX)] + V12_VARIANT_SUFFIX


def relabel_service(block: str, name: str) -> str:
    """A replica's service block with the v12 engine_image and config_variant labels (log tags and OTel labels)."""
    found = re.search(r'nearai\.otel\.config_variant: "([^"]+)"', block)
    if found is None:
        raise GenerationError(f"service {name}: nearai.otel.config_variant label not found")
    variant = found.group(1)
    block = replace_exact(block, f'"engine_image:{V7_LABEL}"', f'"engine_image:{V12_LABEL}"', 1, f"{name} log engine_image")
    block = replace_exact(block, f'nearai.otel.engine_image: "{V7_LABEL}"', f'nearai.otel.engine_image: "{V12_LABEL}"', 1, f"{name} otel engine_image")
    return replace_exact(block, f'{variant}"', f'{v12_variant(variant)}"', 2, f"{name} config_variant")


def relabel_scrape_job(job: str, name: str) -> str:
    found = re.search(r'config_variant: "([^"]+)"', job)
    if found is None:
        raise GenerationError(f"scrape job {name}: config_variant label not found")
    variant = found.group(1)
    job = replace_exact(job, f'engine_image: "{V7_LABEL}"', f'engine_image: "{V12_LABEL}"', 1, f"{name} scrape engine_image")
    return replace_exact(job, f'config_variant: "{variant}"', f'config_variant: "{v12_variant(variant)}"', 1, f"{name} scrape config_variant")


def replace_in_span(text: str, bounds: tuple[int, int], edit) -> str:
    start, end = bounds
    return text[:start] + edit(text[start:end]) + text[end:]


def add_v0523(out: str) -> str:
    """r1-r4 on the v0.5.23 image: common anchor image, candidate argv + V0523_FLAGS, engine_image/config_variant labels."""
    common = span(out, f"\n{COMMON_ANCHOR}\n", r"\n[^ \n#]", "common engine anchor")
    out = replace_in_span(out, common, lambda anchor: replace_exact(anchor, f"\n  image: {V12_IMAGE}\n", f"\n  image: {V0523_IMAGE}\n", 1, "common anchor image"))
    body = anchor_argv(out)
    lines = [line.strip() for line in body.splitlines()]
    for flag in V0523_FLAGS:
        if any(line.split()[0] == flag.split()[0] for line in lines):
            raise GenerationError(f"v0.5.23 flag {flag!r} is already in the fleet argv")
    indent = body.splitlines(keepends=True)[-1][: len(body.splitlines()[-1]) - len(body.splitlines()[-1].lstrip())]
    out = replace_in_span(out, anchor_span(out), lambda anchor: replace_exact(anchor, body, body + "".join(f"{indent}{f}\n" for f in V0523_FLAGS), 1, "anchor argv"))

    def edit_service(block: str, name: str) -> str:
        variant = re.search(r'nearai\.otel\.config_variant: "([^"]+)"', block).group(1)
        block = replace_exact(block, f'"engine_image:{V12_LABEL}"', f'"engine_image:{V0523_LABEL}"', 1, f"{name} log engine_image")
        block = replace_exact(block, f'nearai.otel.engine_image: "{V12_LABEL}"', f'nearai.otel.engine_image: "{V0523_LABEL}"', 1, f"{name} otel engine_image")
        return replace_exact(block, f'{variant}"', f'{v0523_variant(variant)}"', 2, f"{name} config_variant")

    def edit_job(job: str, name: str) -> str:
        variant = re.search(r'config_variant: "([^"]+)"', job).group(1)
        job = replace_exact(job, f'engine_image: "{V12_LABEL}"', f'engine_image: "{V0523_LABEL}"', 1, f"{name} scrape engine_image")
        return replace_exact(job, f'config_variant: "{variant}"', f'config_variant: "{v0523_variant(variant)}"', 1, f"{name} scrape config_variant")

    for name in REPLICAS:
        out = replace_in_span(out, service_span(out, name), lambda b, n=name: edit_service(b, n))
        out = replace_in_span(out, scrape_job_span(out, name), lambda j, n=name: edit_job(j, n))
    return out


def v0523_variant(variant: str) -> str:
    if not variant.endswith(V12_VARIANT_SUFFIX):
        raise GenerationError(f"config_variant {variant!r} does not end {V12_VARIANT_SUFFIX}")
    return variant[: -len(V12_VARIANT_SUFFIX)] + V0523_SUFFIX


def generate(source: str, v0523: bool = True) -> str:
    """The target file. v0523=False renders it with all four replicas on v12 (the previous tag's file, and the
    input scripts/prepare_glm53_gpu04_3b1l.py builds gpu04's file from)."""
    if Q8_BACKEND in source or V12_IMAGE_DIGEST in source:
        raise GenerationError("source already carries v12 or flashmla_sparse_q8; it must be the v7 fleet file")
    out = source

    # Engine image: the common anchor, which every replica inherits through the candidate anchor.
    out = replace_in_span(
        out,
        span(out, f"\n{COMMON_ANCHOR}\n", r"\n[^ \n#]", "common engine anchor"),
        lambda anchor: replace_exact(anchor, f"\n  image: {V7_IMAGE}\n", f"\n  image: {V12_IMAGE}\n", 1, "common anchor image"),
    )

    # Every replica's argv: the candidate anchor.
    v7_body = anchor_argv(out)
    a_start, a_end = anchor_span(out)
    anchor = out[a_start:a_end]
    anchor = replace_exact(anchor, v7_body, v12_argv(v7_body), 1, "anchor argv")
    anchor = replace_exact(
        anchor,
        f"\n{ANCHOR}: &sg-glm53-flash-candidate\n",
        f"\n{ANCHOR}: &sg-glm53-flash-candidate\n{ANCHOR_COMMENT}",
        1,
        "anchor comment",
    )
    out = out[:a_start] + anchor + out[a_end:]

    for name in REPLICAS:
        bounds = service_span(out, name)
        block = out[bounds[0]:bounds[1]]
        if "\n    image:" in block or "\n    command:" in block or "<<: *sg-glm53-flash-candidate" not in block:
            raise GenerationError(f"service {name} must inherit image and command from the candidate anchor")
        out = replace_in_span(out, bounds, lambda b, n=name: relabel_service(b, n))
        out = replace_in_span(out, scrape_job_span(out, name), lambda j, n=name: relabel_scrape_job(j, n))

    # The ghost aggregator runs its engines' image so both move together.
    out = replace_in_span(
        out,
        service_span(out, AGGREGATOR),
        lambda block: replace_exact(block, f"\n    image: {V7_IMAGE}\n", f"\n    image: {V12_IMAGE}\n", 1, "aggregator image"),
    )

    for old, new in COMMENT_EDITS:
        out = replace_exact(out, old, new, 1, "v7 comment")

    live = "".join(line for line in out.splitlines(keepends=True) if not line.lstrip().startswith("#"))
    if V7_IMAGE_DIGEST in live or V7_LABEL in live:
        raise GenerationError("a v7 image or engine_image label survived outside comments")
    if not v0523:
        return HEADER + out
    return HEADER[: -len("#\n")] + V0523_HEADER + "#\n" + add_v0523(out)


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
        print(f"prepare_glm53_base_v12_q8_fleet: {exc}", file=sys.stderr)
        return 2
    if actual != expected:
        print("".join(difflib.unified_diff(actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                                           fromfile=f"a/{TARGET}", tofile=f"b/{TARGET}")), end="")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
