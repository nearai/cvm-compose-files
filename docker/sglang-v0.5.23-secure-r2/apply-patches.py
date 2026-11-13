#!/usr/bin/env python3
"""Apply the NEAR AI security patches to the upstream SGLang v0.5.23 checkout in the image.

The base image (lmsysorg/sglang v0.5.23) ships /sgl-workspace/sglang as a git checkout of tag
v0.5.23 (297e54fd) with an editable install. This script refuses to run on any other source tree:
1. the checkout must be at SOURCE_REVISION with a clean work tree;
2. every patch in SERIES must exist with its pinned sha256 (SHA256SUMS also covers them);
3. each patch must pass `git apply --check` on the tree left by the previous ones, then is applied.
Patches 01-14 are byte-identical to patches of docker/sglang-v0.5.23-glm53-r2 (README maps the numbers)
and apply on clean v0.5.23 without the rest of that series; 15-16 are sglang-diffusion patches.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path("/sgl-workspace/sglang")
HERE = Path(__file__).resolve().parent
SOURCE_REVISION = "297e54fd105e974cdf401e5b4d20253a22c16f90"


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    head = sh("git", "rev-parse", "HEAD")
    if head != SOURCE_REVISION:
        sys.exit(f"refusing: /sgl-workspace/sglang is at {head}, expected {SOURCE_REVISION}")
    if sh("git", "status", "--porcelain", "--untracked-files=no"):
        sys.exit("refusing: the upstream checkout has local modifications")
    series = []
    for line in (HERE / "SERIES").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split()
        series.append((digest, name))
    for digest, name in series:
        path = HERE / "patches" / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            sys.exit(f"refusing: {name} sha256 {actual} != pinned {digest}")
        subprocess.run(["git", "apply", "--check", str(path)], cwd=ROOT, check=True)
        subprocess.run(["git", "apply", str(path)], cwd=ROOT, check=True)
        print(f"applied {name}")
    print(f"applied {len(series)} patches on {SOURCE_REVISION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
