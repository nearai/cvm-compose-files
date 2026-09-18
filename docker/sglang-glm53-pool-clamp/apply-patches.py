"""Apply the reviewed chunked-prefill pool-clamp patch only to the exact production source bytes."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

context = Path(__file__).resolve().parent
root = Path('/sgl-workspace/sglang')
manifest = json.loads((context / 'source-manifest.json').read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


for name, entry in manifest.items():
    if digest(root / name) != entry['before']:
        raise RuntimeError(f'Unrecognized base source: {name}')

subprocess.run(['git', 'apply', '--check', str(context / 'chunked-prefill-pool-clamp.diff')], cwd=root, check=True)
subprocess.run(['git', 'apply', str(context / 'chunked-prefill-pool-clamp.diff')], cwd=root, check=True)

for name, entry in manifest.items():
    if digest(root / name) != entry['after']:
        raise RuntimeError(f'Patched source checksum mismatch: {name}')
    ast.parse((root / name).read_text(), filename=name)

print(f'Verified and applied {len(manifest)} source files')
