#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/glm53_v8_check_env_map.py --host gpu03 --expect canary env-map.json
"""Operator-side check of a host's compose-manager env map (the repo's validators only see the files).

`--expect none`: no GLM53_V8_* key at all (every host before the canary, every host that is not the canary host,
and the canary host after a full revert). `--expect canary`: the GLM53_V8_* keys are exactly what
`glm53_v8_canary_env.py` prints for that host (gpu03 -> base slot, gpu02 -> long slot) and a host with no slot
(anything else) must have none. Free-text keys (EXTRA_ARGS, ENV_PREFIX) are compared to the printer's value, so a
hand-typed duplicate flag or a stray NEAR_SELF_PROFILE fails. Exit 0 = ok, 1 = mismatch.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import glm53_v8_canary_env as printer  # noqa: E402

HOST_KIND = {slot["host"]: kind for kind, slot in printer.SLOTS.items()}


def problems(env_map: dict[str, str], host: str, expect: str) -> list[str]:
    found = {key: value for key, value in env_map.items() if key.startswith("GLM53_V8_")}
    expected = printer.env_map(HOST_KIND[host]) if expect == "canary" and host in HOST_KIND else {}
    out = [f"{key} must not be set on {host}" for key in sorted(set(found) - set(expected))]
    out += [f"{key} is missing on {host}" for key in sorted(set(expected) - set(found))]
    out += [f"{key}={found[key]!r} must be {expected[key]!r}" for key in sorted(set(found) & set(expected)) if found[key] != expected[key]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("env_map", type=Path, help="JSON object of the host's env map (secrets may be redacted)")
    _ = parser.add_argument("--host", required=True)
    _ = parser.add_argument("--expect", choices=("none", "canary"), required=True)
    args = parser.parse_args()
    errors = problems(json.loads(args.env_map.read_text()), args.host, args.expect)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
