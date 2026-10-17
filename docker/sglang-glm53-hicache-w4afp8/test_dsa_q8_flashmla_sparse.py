"""CPU check for dsa-q8-flashmla-sparse.diff: the flashmla_sparse_q8 DSA prefill backend works with the v7
FP8 KV layout (656-byte NoPE row, KPool index > 1).

The kernels need a GPU, so the three changed pieces are checked by ast on the installed sources:
1. dsa_backend.py forward_extend: under flashmla_sparse_q8 with KPool > 1, the page table's
   2048 + (kpool - 1) top-k IDs are padded with -1 to the flashmla_kv width (the Q8 SM90 kernel needs a
   multiple of 128). The block is extracted and run on CPU tensors.
2. dsa_backend.py _forward_flashmla_sparse_q8kv8: the kernel head width includes the reserved RoPE pad
   (kernel_head_dim, flashmla_kv_pad_rope_dim) while the model's query width is unchanged.
3. dsa_backend_kpool.py: flashmla_sparse_q8 is on the KPool allowlist next to flashmla_kv.
Fails on the unpatched source (no pad block, no kernel_head_dim, flashmla_sparse_q8 rejected by KPool).
"""

import argparse
import ast
import types

import torch

parser = argparse.ArgumentParser()
parser.add_argument("dsa_backend")
parser.add_argument("dsa_backend_kpool")
args = parser.parse_args()


def methods(path):
    tree = ast.parse(open(path).read(), filename=path)
    return {
        node.name: node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }


backend = methods(args.dsa_backend)

# 1. KPool top-k pad block in forward_extend.
pad_if = next(
    (
        node
        for node in ast.walk(backend["forward_extend"])
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.dsa_index_kpool > 1"
        and "q8_topk_pad" in ast.unparse(node)
    ),
    None,
)
assert pad_if is not None, "Q8 KPool top-k pad missing: dsa-q8-flashmla-sparse.diff not applied"
code = compile(ast.Module(body=[pad_if], type_ignores=[]), "<q8 kpool pad>", "exec")


def run_pad(width, kpool=4, topk=2176):
    ns = {
        "self": types.SimpleNamespace(dsa_index_kpool=kpool, flashmla_kv_topk=topk),
        "page_table_1": torch.arange(width, dtype=torch.int32).repeat(3, 1),
        "torch": torch,
    }
    exec(code, ns)
    return ns["page_table_1"]


out = run_pad(2051)
assert out.shape == (3, 2176), out.shape
assert torch.equal(out[:, :2051], torch.arange(2051, dtype=torch.int32).repeat(3, 1)), "valid IDs changed"
assert bool((out[:, 2051:] == -1).all()), "pad slots must be masked (-1)"
assert run_pad(2176).shape == (3, 2176), "already-aligned width must be unchanged"
assert run_pad(2051, kpool=1).shape == (3, 2051), "KPool 1 must not pad"
try:
    run_pad(2200)
except ValueError:
    pass
else:
    raise AssertionError("a top-k width above the padded width must raise")

# 2. Q8 kernel width keeps the reserved RoPE pad.
q8 = ast.unparse(backend["_forward_flashmla_sparse_q8kv8"])
for name in ("kernel_head_dim", "input_head_dim", "flashmla_kv_pad_rope_dim"):
    assert name in q8, f"{name} missing from _forward_flashmla_sparse_q8kv8"

# 3. KPool allowlist.
kpool_tree = ast.parse(open(args.dsa_backend_kpool).read(), filename=args.dsa_backend_kpool)
allowlists = [
    {elt.value for elt in node.elts if isinstance(elt, ast.Constant)}
    for node in ast.walk(kpool_tree)
    if isinstance(node, ast.Tuple)
    and any(isinstance(e, ast.Constant) and e.value == "flashmla_kv" for e in node.elts)
]
assert any("flashmla_sparse_q8" in a for a in allowlists), "flashmla_sparse_q8 not on the KPool allowlist"

print("dsa-q8-flashmla-sparse checks passed")
