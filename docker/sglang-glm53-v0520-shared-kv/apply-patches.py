"""Apply the SGLang v0.5.20 port and the in-CVM shared KV cache patch to exact upstream v0.5.20 source bytes."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Final

CONTEXT: Final = Path(__file__).resolve().parent
ROOT: Final = Path("/sgl-workspace/sglang")
# Order matters: the shared-cache patch is written against the ported source.
PATCHES: Final = {
    "v0520-port.diff": "a29b204bdf38f415ef7845bbb7da247e4b5d14a1ba0f56bd8aa4545f714e5ab2",
    "shared-kv.diff": "ed1bb7df493b9c8d6b1d0dc438effc331fc790f4c0d11e2a947d45b4395de7bd",
}
MANIFEST: Final = json.loads((CONTEXT / "source-manifest.json").read_text())


class PatchApplicationError(RuntimeError):
    pass


def digest(path: Path) -> str | None:
    """Return a file's SHA256 digest, or None when it is absent."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


for patch_name, expected in PATCHES.items():
    if digest(CONTEXT / patch_name) != expected:
        raise PatchApplicationError(f"Patch checksum mismatch: {patch_name}")

# A null "before" is a file the port adds; it must not exist in the base image.
for name, entry in MANIFEST.items():
    if digest(ROOT / name) != entry["before"]:
        raise PatchApplicationError(f"Unrecognized base source: {name}")

for patch_name in PATCHES:
    subprocess.run(["git", "apply", "--check", str(CONTEXT / patch_name)], cwd=ROOT, check=True)
    subprocess.run(["git", "apply", str(CONTEXT / patch_name)], cwd=ROOT, check=True)

for name, entry in MANIFEST.items():
    source = ROOT / name
    if digest(source) != entry["after"]:
        raise PatchApplicationError(f"Patched source checksum mismatch: {name}")
    ast.parse(source.read_text(), filename=name)

print(f"Verified and applied {len(PATCHES)} patches across {len(MANIFEST)} source files")
