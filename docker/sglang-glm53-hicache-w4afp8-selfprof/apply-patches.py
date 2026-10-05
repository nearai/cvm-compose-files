"""Apply the reviewed NEAR_SELF_PROFILE self-profiling hook patch to exact production source bytes."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Final

CONTEXT: Final = Path(__file__).resolve().parent
ROOT: Final = Path("/sgl-workspace/sglang")
PATCHES: Final = {
    "near-self-profile.diff": "5a0fbc8e3b0debdd477c94f4bf5cfaa38d58bccd7e4fc9f2d4bb8da0671048ae",
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

for name, entry in MANIFEST.items():
    if digest(ROOT / name) != entry["before"]:
        raise PatchApplicationError(f"Unrecognized base source: {name}")

for patch_name in PATCHES:
    subprocess.run(
        ["git", "apply", "--check", str(CONTEXT / patch_name)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "apply", str(CONTEXT / patch_name)], cwd=ROOT, check=True)

for name, entry in MANIFEST.items():
    source = ROOT / name
    if digest(source) != entry["after"]:
        raise PatchApplicationError(f"Patched source checksum mismatch: {name}")
    ast.parse(source.read_text(), filename=name)

print(f"Verified and applied {len(PATCHES)} patches across {len(MANIFEST)} source files")
