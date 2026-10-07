#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/glm53_v8_canary_env.py base|long [--format env|json] [--rollback fp8|overlap|preprocess|profile]
"""Print the compose-manager env-map entries that turn on the GLM-5.3 v8 bundle canary slot of one file.

The generated compose files hold only today's values; the canary lives in the env map of ONE host
(docs/glm53-v8-bundle-canary.md). This prints exactly what to add to that host's map, built from the
generators' constants, so the runbook, the tests and the operator read the same numbers:

  base  -> gpu03, replica model-sg-glm53-w4afp8-tp2-r4   (prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml)
  long  -> gpu02, replica model-sg-glm53-w4afp8-tp2-r2a  (prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml)

It REFUSES (exit 2) while the image digest or the tool-schema depth is still a placeholder
(scripts/glm53_v8_bundle.py), so a placeholder cannot reach an env map. `--allow-placeholder` prints a
clearly marked preview for review only.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import glm53_v8_bundle as v8  # noqa: E402
from scripts import prepare_glm53_w4afp8_long_context as long_context  # noqa: E402
from scripts import prepare_glm53_w4afp8_tp2x4 as base  # noqa: E402

MAMBA_SLOTS_PER_REQUEST = 5

SLOTS = {
    "base": {
        "host": "gpu03",
        "service": f"{base.SERVICE_PREFIX}{base.V8_REPLICA}",
        "file": str(base.TARGET),
        "prefix": base.V8_PREFIX,
    },
    "long": {
        "host": "gpu02",
        "service": f"{long_context.TP2_SERVICE_PREFIX}{long_context.V8_SLOT}",
        "file": str(long_context.TARGET),
        "prefix": long_context.V8_PREFIX,
    },
}


def env_map(kind: str, *, digest: str | None = None) -> dict[str, str]:
    """Every variable the canary host's env map sets for the slot, in emission order."""
    prefix = SLOTS[kind]["prefix"]
    if kind == "base":
        values = {"MAX_RUNNING": base.V8_MAX_RUNNING, "MAMBA_SLOTS": base.V8_MAMBA_SLOTS}
        suffix = v8.VARIANT_SUFFIX
        mamba = int(base.V8_MAMBA_SLOTS)
    else:
        values = {"MAX_RUNNING": str(long_context.V8_LONG_MAX_RUNNING), "MAX_QUEUED": str(long_context.V8_LONG_MAX_QUEUED)}
        suffix = f"{v8.VARIANT_SUFFIX}-mr{long_context.V8_LONG_MAX_RUNNING}q{long_context.V8_LONG_MAX_QUEUED}"
        mamba = int(long_context.TP2_MAMBA_CACHE)
    running = int(values["MAX_RUNNING"])
    if mamba < MAMBA_SLOTS_PER_REQUEST * running:
        raise ValueError(f"{kind}: {mamba} mamba slots cannot hold {running} running requests (needs {MAMBA_SLOTS_PER_REQUEST} each)")
    ordered = {
        "IMAGE": v8.image_reference(digest),
        "IMAGE_LABEL": v8.engine_image_label(digest),
        "PRECISION": v8.FP8_PRECISION,
        "KV_DTYPE": v8.KV_CACHE_DTYPE,
        "DSA_BACKEND": v8.DSA_BACKEND,
        **values,
        "VARIANT_SUFFIX": suffix,
        "EXTRA_ARGS": v8.OVERLAP_FLAG,
        "ENV_PREFIX": v8.env_prefix_value(),
    }
    return {f"{prefix}{name}": value for name, value in ordered.items()}


def rollback_piece(kind: str, piece: str, *, digest: str | None = None) -> dict[str, str | None]:
    """Env-map edits that undo one piece (None = delete the key). The rest of the canary stays as it was."""
    prefix = SLOTS[kind]["prefix"]
    if piece == "fp8":
        # FP8 KV and the FlashMLA-KV backends travel together (flashmla_kv needs the fp8 cache); back to prod values.
        # The caps (and the base's mamba slots) were sized for FP8's larger KV pool, so they go back with it.
        caps = ("MAX_RUNNING", "MAMBA_SLOTS") if kind == "base" else ("MAX_RUNNING", "MAX_QUEUED")
        return {f"{prefix}{name}": None for name in ("KV_DTYPE", "DSA_BACKEND", "PRECISION", *caps)}
    if piece == "overlap":
        return {f"{prefix}EXTRA_ARGS": None}
    if piece == "profile":
        return {f"{prefix}ENV_PREFIX": "env " + " ".join(f"{n}={v}" for n, v in (v8.preprocess_environment() | v8.PROFILE_ENVIRONMENT | {"NEAR_SELF_PROFILE": "0"}).items())}
    if piece == "preprocess":
        environment = v8.preprocess_environment() | v8.PROFILE_ENVIRONMENT | {"SGLANG_PREPROCESS_WORKERS": "0"}
        return {f"{prefix}ENV_PREFIX": "env " + " ".join(f"{name}={value}" for name, value in environment.items())}
    raise ValueError(piece)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("kind", choices=sorted(SLOTS))
    _ = parser.add_argument("--format", choices=("env", "json"), default="env")
    _ = parser.add_argument("--rollback", choices=("fp8", "overlap", "preprocess", "profile"), help="print the edits that undo one piece instead")
    _ = parser.add_argument("--allow-placeholder", action="store_true", help="preview only: print even though a placeholder is unfilled")
    args = parser.parse_args()

    errors = v8.release_errors()
    if errors and not args.allow_placeholder:
        print("REFUSING to print an env map: the v8 bundle values are not filled in.", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    slot = SLOTS[args.kind]
    if errors and args.format == "json":
        print("--format json cannot be combined with --allow-placeholder (no preview marker in JSON)", file=sys.stderr)
        return 2
    if args.rollback:
        edits = rollback_piece(args.kind, args.rollback)
        if args.format == "json":
            print(json.dumps(edits, indent=2))
        else:
            for key, value in edits.items():
                print(f"# DELETE {key}" if value is None else f"{key}={value}")
        return 0
    values = env_map(args.kind)
    if args.format == "json":
        print(json.dumps(values, indent=2))
        return 0
    banner = f"# compose-manager env map additions for {slot['host']} only; replica {slot['service']} ({slot['file']})"
    print(banner)
    if errors:
        print("# PREVIEW ONLY - DO NOT DEPLOY. Unfilled: " + "; ".join(errors))
    for key, value in values.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
