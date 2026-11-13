"""CPU check for dsa-indexer-qsplit.diff as ported to SGLang v0.5.23.

IndexerKPool._get_topk_ragged_kpool_plan_qsplit splits the K-pool prefill query rows across the
attention-TP ranks, scores each rank's slice through upstream's free-memory row chunking
(_kpool_topk_by_row_chunks, #40854) and all-gathers the int32 top-k rows. The methods are extracted
by ast and run on CPU tensors with a stub logits kernel and stub pooled top-k; the gathered result
must equal the unsplit single-rank result, for world 2 and 3, chunked and unchunked, with and without
padding rows (total_q > n_real), page-table row index and topk offsets. Also checks the gate
(_qsplit_group) is off by default and below SGLANG_DSA_INDEXER_QSPLIT_MIN_ROWS.

Usage: python3 -I test_dsa_indexer_qsplit.py .../layers/attention/dsa/dsa_indexer_kpool.py
"""

import argparse
import ast
import types
from typing import Optional, Tuple

import torch

parser = argparse.ArgumentParser()
parser.add_argument("source")
args = parser.parse_args()
tree = ast.parse(open(args.source).read(), filename=args.source)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IndexerKPool")
methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
mod_fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
for name in ("_get_topk_ragged_kpool_plan_qsplit", "_kpool_topk_by_row_chunks"):
    assert name in methods, f"{name} missing"
assert "_qsplit_group" in mod_fns
plan_src = ast.unparse(methods["_get_topk_ragged_kpool_plan"])
assert "_qsplit_group(n_real)" in plan_src and "_get_topk_ragged_kpool_plan_qsplit(" in plan_src

TOPK = 3


def fp8_mqa_logits(q_fp8, k_fp8, k_scale, weights, starts, ends, *, clean_logits):
    logits = (q_fp8.float() * weights.float()[:, None]) @ k_fp8.float().T
    cols = torch.arange(k_fp8.shape[0])[None, :]
    return logits.masked_fill((cols < starts[:, None]) | (cols >= ends[:, None]), float("-inf"))


def topk_from_kpool_logits(self, logits, pool_lens, seq_lens=None, page_table=None, topk_offsets=None,
                           row_starts=None, out_rows=None, page_table_row_index=None, topk_v2_plan=None):
    idx = logits.topk(TOPK, dim=-1).indices.to(torch.int32)
    # Encode per-row inputs so a wrong slice shows up in the output.
    extra = pool_lens[:, None].to(torch.int32) * 1000
    if page_table_row_index is not None:
        extra = extra + page_table[page_table_row_index.long(), :1].to(torch.int32) * 100000
    if topk_offsets is not None:
        extra = extra + topk_offsets[:, None].to(torch.int32) * 10
    res = idx + extra
    if out_rows is not None and out_rows > res.shape[0]:
        res = torch.cat([res, res.new_full((out_rows - res.shape[0], TOPK), -1)])
    return res


chunk_rows = {"n": None}


def mqa_logits_row_chunks(*, num_rows, num_cols, device):
    n = chunk_rows["n"]
    if n is None:
        return (slice(0, num_rows),)
    return tuple(slice(s, min(s + n, num_rows)) for s in range(0, num_rows, n))


ns = {
    "torch": torch, "Optional": Optional, "Tuple": Tuple,
    "_mqa_logits_row_chunks": mqa_logits_row_chunks,
}
exec(compile(ast.Module(body=[mod_fns["_slice_rows"]], type_ignores=[]), "slice", "exec"), ns)
for name in ("_get_topk_ragged_kpool_plan_qsplit", "_kpool_topk_by_row_chunks"):
    fn = methods[name]
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), name, "exec"), ns)


def make_self():
    s = types.SimpleNamespace()
    s._fp8_mqa_logits = fp8_mqa_logits
    s._topk_from_kpool_logits = types.MethodType(topk_from_kpool_logits, s)
    s._kpool_topk_by_row_chunks = types.MethodType(ns["_kpool_topk_by_row_chunks"], s)
    return s


class Group:
    def __init__(self, world, rank, sink):
        self.world_size, self.rank_in_group, self.sink = world, rank, sink

    def all_gather_into_tensor(self, out, local):
        per = local.shape[0]
        assert out.shape[0] == per * self.world_size
        self.sink[self.rank_in_group] = local.clone()
        out.zero_()
        out[self.rank_in_group * per:(self.rank_in_group + 1) * per] = local


def run_case(n_real, total_q, world, chunk, use_row_index, use_offsets):
    torch.manual_seed(n_real * 7 + world)
    k_rows, d = 40, 8
    q = torch.randn(total_q, d)
    w = torch.rand(total_q)
    k = torch.randn(k_rows, d)
    ks = torch.randint(0, 10, (n_real,), dtype=torch.int32)
    ke = ks + torch.randint(5, 30, (n_real,), dtype=torch.int32)
    pool_lens = torch.randint(1, 50, (n_real,), dtype=torch.int32)
    seq_lens = pool_lens * 4
    page_table = torch.arange(5 * 2, dtype=torch.int32).view(5, 2) if use_row_index else None
    row_index = torch.randint(0, 5, (n_real,), dtype=torch.int32) if use_row_index else None
    offsets = torch.randint(0, 9, (n_real,), dtype=torch.int32) if use_offsets else None
    self_ = make_self()
    chunk_rows["n"] = None
    ref = self_._kpool_topk_by_row_chunks(
        q_fp8=q, weights=w, kv_fp8=(k, None), logits_starts=ks, logits_ends=ke, pool_lens=pool_lens,
        seq_lens=seq_lens, page_table=page_table, page_table_row_index=row_index, topk_offsets=offsets,
        topk_row_starts=ks, row_chunks=(slice(0, n_real),), out_rows=total_q,
    )
    chunk_rows["n"] = chunk
    sink = {}
    outs = []
    for rank in range(world):
        outs.append(ns["_get_topk_ragged_kpool_plan_qsplit"](
            self_, group=Group(world, rank, sink), q_fp8=q, weights=w, kv_fp8=(k, None),
            ks_per_q=ks, ke_per_q=ke, pool_lens=pool_lens, seq_lens_expanded=seq_lens,
            page_table_all=page_table, page_table_row_index_all=row_index, topk_offsets_all=offsets,
            n_real=n_real, total_q=total_q, total_k_rows=k_rows,
        ))
    # Every rank sees the full gather; rebuild it from the per-rank locals.
    full = torch.cat([sink[r] for r in range(world)])[:n_real]
    case = f"n_real={n_real} total_q={total_q} world={world} chunk={chunk} ri={use_row_index} off={use_offsets}"
    assert outs[0].shape == (total_q, TOPK), (case, outs[0].shape)
    assert torch.equal(full, ref[:n_real]), case
    for r, o in enumerate(outs):
        per = -(-n_real // world)
        lo, hi = min(r * per, n_real), min((r + 1) * per, n_real)
        assert torch.equal(o[lo:hi], ref[lo:hi]), (case, r)
        assert bool((o[n_real:] == -1).all()), (case, "padding rows must be -1")


n = 0
for n_real, total_q in ((17, 17), (17, 20), (16, 16), (1031, 1040)):
    for world in (2, 3):
        for chunk in (None, 4, 5):
            for ri in (False, True):
                for off in (False, True):
                    run_case(n_real, total_q, world, chunk, ri, off)
                    n += 1

# Gate: off by default, row threshold, world size 1.
gate = mod_fns["_qsplit_group"]
attn = types.SimpleNamespace(world_size=2)
dist = types.ModuleType("sglang.srt.distributed")
dist.get_attn_tp_group = lambda: attn
import sys
for name in ("sglang", "sglang.srt"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["sglang.srt.distributed"] = dist
for on, min_rows, rows, world, want in (
    (False, 1024, 5000, 2, False), (True, 1024, 1023, 2, False), (True, 1024, 1024, 2, True),
    (True, 1024, 5000, 1, False),
):
    attn.world_size = world
    gns = {"_QSPLIT": on, "_QSPLIT_MIN_ROWS": min_rows}
    exec(compile(ast.Module(body=[gate], type_ignores=[]), "gate", "exec"), gns)
    assert (gns["_qsplit_group"](rows) is not None) == want, (on, rows, world)
default = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_QSPLIT" for t in n.targets)]
assert default and "'false'" in ast.unparse(default[0]) and "SGLANG_DSA_INDEXER_QSPLIT" in ast.unparse(default[0])
print(f"dsa-indexer-qsplit checks passed ({n} split cases)")
