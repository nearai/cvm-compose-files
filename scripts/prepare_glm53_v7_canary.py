#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/prepare_glm53_v7_canary.py --write   (or --check)
"""Generate the two GLM-5.3 v7 bundle canary compose files from the committed host files.

Each output is a full copy of the file its host already deploys, with ONLY the canary replica changed (hardcoded, no
variables): the image, a handful of flags, the bundle environment, and the replica's telemetry labels
(precision / engine_image / config_variant) in its service block and its OTel scrape job. Every other service, the
`deployment` label and every other label are untouched, so the prod Grafana dashboard keeps selecting them.

  base  gpu03  prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml           -> ...-V7Canary.yaml  (replica r4)
  long  gpu02  prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml -> ...-V7Canary.yaml  (replica r2a)

`--write` regenerates the targets; `--check` prints a diff and exits non-zero when a committed target is stale.
Docs: docs/glm53-v7-canary.md.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]

# glm53-hicache-w4afp8-v7 (published).
V7_IMAGE_DIGEST: Final = "sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5"
V7_IMAGE: Final = f"docker.io/nearaidev/sglang@{V7_IMAGE_DIGEST}"
V7_IMAGE_LABEL: Final = V7_IMAGE_DIGEST.split(":")[1][:12]
V6_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"
V6_IMAGE_LABEL: Final = "9c6ddd4319c4"
PRECISION: Final = "int4-weights-fp8-activations-bf16-kv"
V7_PRECISION: Final = "int4-weights-fp8-activations-fp8-kv"

# THE LONG CAPS UNDER TEST (the base canary is 64 running / 380 mamba slots). Change here and run --write.
V7_LONG_MAX_RUNNING: Final = 16
V7_LONG_MAX_QUEUED: Final = 4
V7_BASE_MAX_RUNNING: Final = 64
V7_BASE_MAMBA_SLOTS: Final = 380
MAMBA_SLOTS_PER_REQUEST: Final = 5

# The bundle environment, as ordinary environment entries of the canary service (the file is canary-only).
V7_ENVIRONMENT: Final = (
    ("SGLANG_PREPROCESS_WORKERS", "4"),
    ("SGLANG_PREPROCESS_TIMEOUT_S", "60"),
    ("SGLANG_PREPROCESS_LOG_SLOW_S", "5"),
    ("SGLANG_TOOL_SCHEMA_MAX_DEPTH", "32"),
    ("SGLANG_TOOL_SCHEMA_MAX_NODES", "25000"),
    ("NEAR_SELF_PROFILE", "1"),
    ("NEAR_SELF_PROFILE_AFTER_S", "900"),
    ("NEAR_SELF_PROFILE_STEPS", "50"),
)
OVERLAP_FLAG: Final = "--disable-overlap-schedule"
SERVICE_PREFIX: Final = "model-sg-glm53-w4afp8-tp2-r"

KINDS: Final = {
    "base": {
        "host": "gpu03",
        "source": Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml"),
        "target": Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V7Canary.yaml"),
        "service": f"{SERVICE_PREFIX}4",
        "suffix": "-v7",
        "flags": {
            "--kv-cache-dtype": ("bfloat16", "fp8_e4m3"),
            "--dsa-prefill-backend": ("tilelang", "flashmla_kv"),
            "--dsa-decode-backend": ("tilelang", "flashmla_kv"),
            "--max-running-requests": ("48", str(V7_BASE_MAX_RUNNING)),
            "--cuda-graph-max-bs-decode": ("48", str(V7_BASE_MAX_RUNNING)),
            "--max-mamba-cache-size": ("330", str(V7_BASE_MAMBA_SLOTS)),
        },
        "running": V7_BASE_MAX_RUNNING,
        "mamba": V7_BASE_MAMBA_SLOTS,
    },
    "long": {
        "host": "gpu02",
        "source": Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml"),
        "target": Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext-V7Canary.yaml"),
        "service": f"{SERVICE_PREFIX}2a",
        "suffix": f"-v7-mr{V7_LONG_MAX_RUNNING}q{V7_LONG_MAX_QUEUED}",
        "flags": {
            "--kv-cache-dtype": ("bfloat16", "fp8_e4m3"),
            "--dsa-prefill-backend": ("tilelang", "flashmla_kv"),
            "--dsa-decode-backend": ("tilelang", "flashmla_kv"),
            "--max-running-requests": ("12", str(V7_LONG_MAX_RUNNING)),
            "--max-queued-requests": ("4", str(V7_LONG_MAX_QUEUED)),
            "--cuda-graph-max-bs-decode": ("12", str(V7_LONG_MAX_RUNNING)),
        },
        "running": V7_LONG_MAX_RUNNING,
        "mamba": 330,
    },
}


class GenerationError(ValueError):
    pass


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    actual = text.count(old)
    if actual != expected:
        raise GenerationError(f"{label}: expected {expected} matches, found {actual}")
    return text.replace(old, new)


def span(text: str, start_marker: str, end_pattern: str, label: str) -> tuple[int, int]:
    start = text.find(start_marker)
    if start == -1 or text.count(start_marker) != 1:
        raise GenerationError(f"{label}: start marker not found exactly once")
    match = re.search(end_pattern, text[start + len(start_marker):])
    if not match:
        raise GenerationError(f"{label}: end marker not found")
    return start, start + len(start_marker) + match.start()


def service_span(text: str, name: str) -> tuple[int, int]:
    return span(text, f"\n  {name}:\n", r"\n  [^ \n]", f"service {name}")


def job_span(text: str, name: str) -> tuple[int, int]:
    return span(text, f"\n              - job_name: sglang-{name}\n", r"\n              - job_name:", f"scrape job {name}")


def retag(block: str, variant: str, suffix: str, with_log_tag: bool) -> str:
    """The canary's three telemetry labels in one block (the service block, or the scrape job)."""
    if with_log_tag:
        block = replace_exact(block, f'"precision:{PRECISION}"', f'"precision:{V7_PRECISION}"', 1, "log precision")
        block = replace_exact(block, f'"engine_image:{V6_IMAGE_LABEL}"', f'"engine_image:{V7_IMAGE_LABEL}"', 1, "log engine_image")
        block = replace_exact(block, f'nearai.otel.engine_image: "{V6_IMAGE_LABEL}"', f'nearai.otel.engine_image: "{V7_IMAGE_LABEL}"', 1, "otel engine_image")
        block = replace_exact(block, f'{variant}"', f'{variant}{suffix}"', 2, "service config_variant")
    else:
        block = replace_exact(block, f'precision: "{PRECISION}"', f'precision: "{V7_PRECISION}"', 1, "scrape precision")
        block = replace_exact(block, f'engine_image: "{V6_IMAGE_LABEL}"', f'engine_image: "{V7_IMAGE_LABEL}"', 1, "scrape engine_image")
        block = replace_exact(block, f'{variant}"', f'{variant}{suffix}"', 1, "scrape config_variant")
    return block


def environment_entries() -> str:
    return (
        "      # v7 bundle canary (docs/glm53-v7-canary.md): preprocessing pool and tool-schema caps (SGLANG_PREPROCESS_WORKERS=0\n"
        "      # or the caps at 0 turn each off) and the one-shot self-profile (inert unless NEAR_SELF_PROFILE is exactly 1).\n"
        + "".join(f"      - {name}={value}\n" for name, value in V7_ENVIRONMENT)
    )


def edit_flag_lines(lines: list[str], flags: dict[str, tuple[str, str]], indent: str) -> list[str]:
    out = list(lines)
    for flag, (old, new) in flags.items():
        wanted = f"{indent}{flag} {old}"
        if out.count(wanted) != 1:
            raise GenerationError(f"canary argv: expected exactly one {wanted.strip()!r}")
        out[out.index(wanted)] = f"{indent}{flag} {new}"
    return out


def generate_base(source: str, spec: dict) -> str:
    # r4 inherits the candidate anchor's image and command; the canary sets its own, copied from the anchor.
    a_start, a_end = span(source, "\nx-sg-glm53-flash-candidate: &sg-glm53-flash-candidate\n", r"\nx-", "candidate anchor")
    anchor = source[a_start:a_end]
    body = anchor.split("  command: >\n", 1)[1].rstrip("\n").split("\n")
    command = edit_flag_lines(body, spec["flags"], "      ")
    if command.count("      --mamba-ssm-dtype bfloat16") != 1:
        raise GenerationError("candidate argv changed")
    command.append(f"      {OVERLAP_FLAG}")
    overrides = (
        "    # v7 bundle canary: the candidate argv with FP8 KV (both flashmla_kv DSA backends), 64 running / 64 graphs / 380 mamba\n"
        "    # slots, the overlap scheduler off, and the v7 image. Every other replica keeps the candidate anchor and the v6 image.\n"
        f"    image: {V7_IMAGE}\n    command: >\n" + "\n".join(command) + "\n"
    )
    name = spec["service"]
    s_start, s_end = service_span(source, name)
    block = source[s_start:s_end]
    block = replace_exact(block, "      - SGLANG_KV_TIER_METRICS=1\n", "      - SGLANG_KV_TIER_METRICS=1\n" + environment_entries(), 1, "environment")
    block = replace_exact(block, "    depends_on:\n", overrides + "    depends_on:\n", 1, "image and command")
    variant = re.search(r'nearai\.otel\.config_variant: "([^"]+)"', block).group(1)
    block = retag(block, variant, spec["suffix"], True)
    out = source[:s_start] + block + source[s_end:]
    j_start, j_end = job_span(out, name)
    return out[:j_start] + retag(out[j_start:j_end], variant, spec["suffix"], False) + out[j_end:]


def generate_long(source: str, spec: dict) -> str:
    name = spec["service"]
    s_start, s_end = service_span(source, name)
    block = source[s_start:s_end]
    block = replace_exact(block, f"    image: {V6_IMAGE}\n", f"    image: {V7_IMAGE}\n", 1, "image")
    lines = block.split("\n")
    lines = edit_flag_lines(lines, spec["flags"], "        ")
    index = lines.index("        --mamba-ssm-dtype bfloat16")
    lines.insert(index + 1, f"        {OVERLAP_FLAG}")
    block = "\n".join(lines)
    block = replace_exact(block, "      - SGLANG_KV_TIER_METRICS=1\n", "      - SGLANG_KV_TIER_METRICS=1\n" + environment_entries(), 1, "environment")
    variant = re.search(r'nearai\.otel\.config_variant: "([^"]+)"', block).group(1)
    block = retag(block, variant, spec["suffix"], True)
    out = source[:s_start] + block + source[s_end:]
    j_start, j_end = job_span(out, name)
    return out[:j_start] + retag(out[j_start:j_end], variant, spec["suffix"], False) + out[j_end:]


def header(kind: str, spec: dict) -> str:
    return (
        f"# GLM-5.3 Flash v7 bundle CANARY file for {spec['host']}, generated from {spec['source']} by\n"
        "# scripts/prepare_glm53_v7_canary.py (do not hand-edit). It is a full copy of the host's file with ONLY the canary replica\n"
        f"# {spec['service']} changed: the glm53-hicache-w4afp8-v7 image ({V7_IMAGE_DIGEST[:19]}...), FP8 KV with flashmla_kv\n"
        f"# DSA backends, {spec['running']} running requests, --disable-overlap-schedule, the bundle environment, and the\n"
        f"# precision / engine_image / config_variant ({spec['suffix']}) telemetry labels. Deploy it ONLY to {spec['host']}, scoped to that one\n"
        "# service and otelcol-contrib; roll back by redeploying the source file for the same service. docs/glm53-v7-canary.md.\n"
        "#\n"
    )


def generate(kind: str, source: str) -> str:
    spec = KINDS[kind]
    if V7_IMAGE in source or "NEAR_SELF_PROFILE" in source:
        raise GenerationError("source already carries the v7 canary")
    if spec["mamba"] < MAMBA_SLOTS_PER_REQUEST * spec["running"]:
        raise GenerationError(f"{spec['mamba']} mamba slots cannot hold {spec['running']} running requests")
    body = generate_base(source, spec) if kind == "base" else generate_long(source, spec)
    return header(kind, spec) + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--write", action="store_true", help="Write the targets")
    _ = parser.add_argument("--check", action="store_true", help="Fail with a diff when a committed target is stale")
    args = parser.parse_args()
    if args.write and args.check:
        parser.error("--write and --check are mutually exclusive")
    status = 0
    for kind, spec in KINDS.items():
        expected = generate(kind, (ROOT / spec["source"]).read_text())
        target = ROOT / spec["target"]
        if args.write:
            _ = target.write_text(expected)
            continue
        actual = target.read_text() if target.exists() else ""
        if actual != expected:
            status = 1
            print("".join(difflib.unified_diff(actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                                               fromfile=f"a/{spec['target']}", tofile=f"b/{spec['target']}")), end="")
    return status


if __name__ == "__main__":
    sys.exit(main())
