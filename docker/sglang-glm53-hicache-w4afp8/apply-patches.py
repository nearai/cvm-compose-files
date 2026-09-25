"""Apply the reviewed W4AFP8 loader, pool-clamp, DSA indexer query-split, event-loop offload and
event-loop stall-dump patches to exact production source bytes."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Final

CONTEXT: Final = Path(__file__).resolve().parent
ROOT: Final = Path("/sgl-workspace/sglang")
PATCHES: Final = {
    "modules-to-not-convert.diff": "29764baa3e464d2272ea85f2e254392c2a61a3fc61a51f8d33b5910ce0cd8d00",
    "chunked-prefill-pool-clamp.diff": "ba911be688556df0c0b2c9a26cde4c9f38b410a5ba51020d7754fa2e8cd010c3",
    "dsa-indexer-qsplit.diff": "37ad31b95511a831038d2ab572656a59d4ba1c9c21348d9fcb0095cba665aad1",
    "sglang-pr30771.diff": "a102977de7bb02776cee1053ff04979b740526fe99c59d2c63e2f3962df3c95d",
    # Stacked on sglang-pr30771.diff: it edits lines that diff adds, so the order here matters.
    "shm-off-loop.diff": "4bd4cc65d68deacd1b56b95e32a2242853d20bd3cfa4188e8c1a20777cc79a94",
    "event-loop-stall-dump.diff": "367dbcdec432586ae4c8ab984c08caafb2f148d62cebf257d12e89c4c32f72ab",
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

# Check each patch against the tree the previous patches left, then apply it. A stacked patch
# (shm-off-loop.diff on sglang-pr30771.diff) cannot be checked against the unpatched base.
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
