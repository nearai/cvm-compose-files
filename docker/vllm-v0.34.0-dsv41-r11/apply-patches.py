#!/usr/bin/env python3
"""Apply patches/NN-name.diff, in file-name order, to the vLLM v0.34.0 and FlashInfer 0.7.3 packages in the image.

The base image installs vLLM as a wheel in DIST_PACKAGES, with no source checkout. The build checks
SHA256SUMS first; this script then refuses any other vLLM version, dry-runs each patch on the tree
the previous ones left before applying it (a patch may also create a new file), and checks every touched file against PATCHED-FILES.sha256.
"""
import hashlib
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

DIST_PACKAGES = Path("/usr/local/lib/python3.12/dist-packages")
HERE = Path(__file__).resolve().parent
VLLM_VERSION = "0.34.0"
FLASHINFER_VERSION = "0.7.3"
FLASHINFER_FILE = "flashinfer/jit/attention/modules.py"
FLASHINFER_ORIGINAL_SHA256 = "3d9654947a612675cf76c40c3ca90973ef0790d7eebe473b0eec28e77c002441"
NAME = re.compile(r"^[0-9]{2}-[a-z0-9-]+\.diff$")


def main() -> int:
    installed = version("vllm")
    if installed != VLLM_VERSION:
        sys.exit(f"refusing: installed vllm is {installed}, expected {VLLM_VERSION}")
    installed = version("flashinfer-python")
    if installed != FLASHINFER_VERSION:
        sys.exit(f"refusing: installed flashinfer-python is {installed}, expected {FLASHINFER_VERSION}")
    found = hashlib.sha256((DIST_PACKAGES / FLASHINFER_FILE).read_bytes()).hexdigest()
    if found != FLASHINFER_ORIGINAL_SHA256:
        sys.exit(f"refusing: {FLASHINFER_FILE} is {found}, expected {FLASHINFER_ORIGINAL_SHA256}")
    patches = sorted((HERE / "patches").iterdir())
    if not patches or any(not NAME.match(p.name) for p in patches):
        sys.exit("refusing: patches/ must hold only NN-name.diff files")
    if [int(p.name[:2]) for p in patches] != list(range(1, len(patches) + 1)):
        sys.exit("refusing: patches must be numbered 01..N without gaps")
    for path in patches:
        for extra in (["--dry-run"], []):
            subprocess.run(
                ["patch", "-p1", "-s", "--forward", "--no-backup-if-mismatch", *extra, "-i", str(path)],
                cwd=DIST_PACKAGES,
                check=True,
            )
        print(f"applied {path.name}")
    subprocess.run(
        ["sha256sum", "--check", "--strict", "--quiet", str(HERE / "PATCHED-FILES.sha256")],
        cwd=DIST_PACKAGES,
        check=True,
    )
    print(f"applied {len(patches)} patches on vllm {VLLM_VERSION}; patched files match PATCHED-FILES.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
