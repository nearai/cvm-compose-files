"""CPU check for kda-zero-sync.diff: KDAAttnBackend.forward_extend takes its logical token count from host
metadata instead of reading query_start_loc[-1] back from the GPU.

The kernels need a GPU, so the token-count block is extracted from the installed kda_backend.py by ast
and run on stubs. query_start_loc is a stub that records every read, which is what a GPU->CPU sync
would be. Fails on the unpatched source (no host path, every call reads the device value).
"""

import argparse
import ast
import os
import types

parser = argparse.ArgumentParser()
parser.add_argument("source")
args = parser.parse_args()

text = open(args.source).read()
tree = ast.parse(text, filename=args.source)
fn = next(
    node
    for cls in tree.body
    if isinstance(cls, ast.ClassDef) and cls.name == "KDAAttnBackend"
    for node in cls.body
    if isinstance(node, ast.FunctionDef) and node.name == "forward_extend"
)
start = next(
    i
    for i, stmt in enumerate(fn.body)
    if isinstance(stmt, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "physical_num_tokens" for t in stmt.targets)
)
# Everything from physical_num_tokens up to (not including) the first use of logical_num_tokens.
block = []
for stmt in fn.body[start:]:
    if block and "logical_num_tokens <" in ast.unparse(stmt):
        break
    block.append(stmt)
src = "\n".join(ast.unparse(s) for s in block)
assert "extend_seq_lens_cpu" in src, "host token count missing: kda-zero-sync.diff not applied"
code = compile(ast.Module(body=block, type_ignores=[]), "<forward_extend token count>", "exec")


class DeviceLoc:
    """Stands in for the device tensor query_start_loc; each [-1] read would be a GPU->CPU sync."""

    def __init__(self, total):
        self.total, self.reads = total, 0

    def __getitem__(self, idx):
        assert idx == -1
        self.reads += 1
        return self.total


def run(seq_cpu, device_total, draft_v2=False, physical=None):
    loc = DeviceLoc(device_total)
    fb = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_draft_extend_v2=lambda: draft_v2)
    )
    if seq_cpu is not None:
        fb.extend_seq_lens_cpu = seq_cpu
    ns = {
        "forward_batch": fb,
        "query_start_loc": loc,
        "mixed_qkv": types.SimpleNamespace(shape=(physical or device_total,)),
        "os": os,
        "int": int,
        "sum": sum,
        "getattr": getattr,
    }
    exec(code, ns)
    return ns["logical_num_tokens"], loc.reads


os.environ.pop("SGLANG_KDA_HOST_COUNT_CHECK", None)
checks = 0
# Host path: same value as the device read, and no device read at all.
for lens in ([4096], [1, 2, 3], [8192, 64, 1], [0, 7]):
    n, reads = run(lens, sum(lens), physical=sum(lens) + 64)
    assert (n, reads) == (sum(lens), 0), (lens, n, reads)
    checks += 1
# Fallbacks keep the old device read: host list missing, or draft-extend-v2.
assert run(None, 77) == (77, 1)
assert run([5, 5], 99, draft_v2=True) == (99, 1)
checks += 2
# Debug check: re-adds the read, passes when equal, raises on a mismatch.
os.environ["SGLANG_KDA_HOST_COUNT_CHECK"] = "1"
assert run([10, 20], 30) == (30, 1)
try:
    run([10, 20], 31)
except RuntimeError as exc:
    assert "KDA host token count 30 != device 31" in str(exc)
else:
    raise AssertionError("mismatch not detected with SGLANG_KDA_HOST_COUNT_CHECK=1")
os.environ["SGLANG_KDA_HOST_COUNT_CHECK"] = "0"
assert run([10, 20], 31) == (30, 0), "the check must be off unless the env var is exactly 1"
os.environ.pop("SGLANG_KDA_HOST_COUNT_CHECK")
checks += 3
print(f"kda zero-sync: {checks}/{checks} checks passed")
