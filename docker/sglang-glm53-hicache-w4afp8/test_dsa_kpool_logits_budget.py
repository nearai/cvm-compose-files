"""CPU check for dsa-indexer-kpool-logits-budget.diff: the K-pool prefill indexer scores query rows in blocks
whose fp32 logits scratch fits the upstream Indexer budget, instead of one rows x key_rows allocation.

deep_gemm needs a GPU, so IndexerKPool._kpool_logits_topk_rows and the module-level budget helper are
extracted from the installed dsa_indexer_kpool.py by ast and run on CPU tensors. fp8_mqa_logits is a stub
that computes masked dot-product logits and records every scratch it would allocate; the pooled top-k is a
stub that keeps each row's top indices and encodes the per-row page table / offset it was handed, so a wrong
slice shows up in the output. Fails on the unpatched source (no helper; both callers allocate the full scratch).
"""

import argparse
import ast
import os
import types

import torch

parser = argparse.ArgumentParser()
parser.add_argument("source")
args = parser.parse_args()

text = open(args.source).read()
tree = ast.parse(text, filename=args.source)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IndexerKPool")
methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
assert "_kpool_logits_topk_rows" in methods, "blocked K-pool logits helper missing: diff not applied"
module_fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
assert "_kpool_logits_budget_bytes" in module_fns, "K-pool logits budget missing: diff not applied"

# Both K-pool prefill paths go through the helper; neither calls fp8_mqa_logits itself any more.
for name in ("_get_topk_ragged_kpool_plan", "_get_topk_ragged_kpool_plan_qsplit"):
    src = ast.unparse(methods[name])
    assert "_kpool_logits_topk_rows(" in src, f"{name} does not use the blocked helper"
    assert "fp8_mqa_logits(" not in src, f"{name} still allocates the full logits scratch"
checks = 1

consts = [
    n for n in tree.body
    if isinstance(n, (ast.Assign, ast.AnnAssign))
    and any(isinstance(t, ast.Name) and t.id.startswith("_KPOOL_LOGITS")
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))
]


class FakeKernel:
    """fp8_mqa_logits on CPU: logits[i, j] = (q[i] * w[i]) . k[j] inside [ks[i], ke[i]), -inf outside."""

    def __init__(self):
        self.allocs = []

    def fp8_mqa_logits(self, q, kv, weights, ks, ke, clean_logits=True):
        assert clean_logits
        k, _scale = kv
        logits = (q.float() * weights.float()[:, None]) @ k.float().T
        self.allocs.append(logits.numel() * 4)
        cols = torch.arange(k.shape[0])[None, :]
        return logits.masked_fill((cols < ks[:, None]) | (cols >= ke[:, None]), float("-inf"))


TOPK = 4


def fake_topk(self, logits, pool_lens, seq_lens=None, page_table=None, topk_offsets=None, row_starts=None,
              out_rows=None, page_table_row_index=None):
    n = logits.shape[0]
    assert pool_lens.shape[0] == n and seq_lens.shape[0] == n and row_starts.shape[0] == n
    idx = torch.topk(logits, TOPK, dim=1).indices.to(torch.int32)
    tag = torch.zeros(n, dtype=torch.int32)
    if page_table_row_index is not None:  # per-request table, per-row index into it
        assert page_table_row_index.shape[0] == n
        tag += page_table[page_table_row_index.long(), 0]
    elif page_table is not None:  # per-row table
        tag += page_table[:n, 0]
    if topk_offsets is not None:
        tag += topk_offsets[:n]
    out = torch.cat([idx, tag[:, None]], dim=1)
    rows = out_rows or n
    if rows > n:
        out = torch.cat([out, torch.full((rows - n, out.shape[1]), -1, dtype=out.dtype)])
    return out


def load(budget_bytes):
    kernel = FakeKernel()
    ns = {"torch": torch, "deep_gemm": kernel, "os": os, "Dict": dict, "Optional": object,
          "_kpool_logits_budget_bytes": lambda device_index: budget_bytes}
    exec(compile(ast.Module(body=consts, type_ignores=[]), "<consts>", "exec"), ns)
    exec(compile(ast.Module(body=[methods["_kpool_logits_topk_rows"]], type_ignores=[]), "<helper>", "exec"), ns)
    obj = types.SimpleNamespace(_topk_from_kpool_logits=types.MethodType(fake_topk, types.SimpleNamespace()))
    helper = types.MethodType(ns["_kpool_logits_topk_rows"], obj)
    return helper, kernel


def case(n_rows, k_rows, budget_bytes, per_row_table=False, row_index=False, offsets=False, out_rows=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n_rows, 16, generator=g)
    w = torch.rand(n_rows, generator=g) + 0.5
    k = torch.randn(k_rows, 16, generator=g)
    scale = torch.ones(k_rows)
    ks = torch.randint(0, max(1, k_rows // 4), (n_rows,), generator=g, dtype=torch.int32)
    ke = torch.clamp(ks + torch.randint(TOPK, k_rows, (n_rows,), generator=g, dtype=torch.int32), max=k_rows)
    pool_lens = ke - ks
    seq_lens = pool_lens * 4
    page_table = row_idx = topk_off = None
    if row_index:
        page_table = torch.arange(3, dtype=torch.int32)[:, None] * 1000 + 1000
        row_idx = torch.randint(0, 3, (n_rows,), generator=g, dtype=torch.int32)
    elif per_row_table:
        page_table = torch.arange(n_rows, dtype=torch.int32)[:, None] * 10
    if offsets:
        topk_off = torch.arange(n_rows, dtype=torch.int32) * 100000
    args_ = (q, w, k, scale, ks, ke, pool_lens, seq_lens, page_table, row_idx, topk_off, out_rows)
    ref, ref_kernel = load(10 ** 15)
    want = ref(*args_)
    assert len(ref_kernel.allocs) == 1
    got_fn, kernel = load(budget_bytes)
    got = got_fn(*args_)
    return want, got, kernel


# 1. Budget big enough: one call, same as before the patch.
want, got, kernel = case(64, 200_000, 10 ** 15)
assert torch.equal(want, got) and len(kernel.allocs) == 1
checks += 1

# 2. Scratch over budget: row blocks, each block's scratch within budget, identical selection.
for n, kr, budget, kw in [
    (64, 200_000, 8 * 200_000 * 4, {}),
    (100, 160_000, 7 * 160_000 * 4, {"offsets": True}),
    (90, 150_000, 13 * 150_000 * 4, {"per_row_table": True}),
    (77, 140_000, 5 * 140_000 * 4, {"row_index": True}),
    (50, 170_000, 9 * 170_000 * 4, {"offsets": True, "row_index": True}),
]:
    want, got, kernel = case(n, kr, budget, **kw)
    assert torch.equal(want, got), (n, kr, kw)
    assert len(kernel.allocs) > 1 and max(kernel.allocs) <= budget, (kernel.allocs, budget)
    checks += 1

# 3. out_rows pads rows past the real ones with -1 on the blocked path, as the single call does.
want, got, kernel = case(40, 210_000, 6 * 210_000 * 4, offsets=True, out_rows=48)
assert torch.equal(want, got) and got.shape[0] == 48 and bool((got[40:] == -1).all())
assert len(kernel.allocs) > 1
checks += 1

# 4. A budget smaller than one row still makes progress one row at a time.
want, got, kernel = case(12, 700_000, 1, offsets=True)
assert torch.equal(want, got) and len(kernel.allocs) == 12
checks += 1

# 5. Small batches skip the budget lookup entirely (no host sync on ordinary prefills).
want, got, kernel = case(16, 1_000, 1)
assert torch.equal(want, got) and len(kernel.allocs) == 1
checks += 1


# 6. Budget: min(free * F, (1 - mem_fraction_static) * total * F, 0.3 * total), measured once per device;
#    capture mode is not cached; the MiB override wins.
def budget_ns(free, total, mem_static, capture=False, override_mb=0):
    calls = {"mem_get_info": 0}

    def mem_get_info(i):
        calls["mem_get_info"] += 1
        return free, total

    cuda = types.SimpleNamespace(mem_get_info=mem_get_info,
                                 get_device_properties=lambda i: types.SimpleNamespace(total_memory=total))
    fake_torch = types.SimpleNamespace(cuda=cuda)
    rt = types.ModuleType("sglang.srt.runtime_context")
    rt.get_schedule = lambda: types.SimpleNamespace(mem_fraction_static=mem_static)
    import sys
    sys.modules.setdefault("sglang", types.ModuleType("sglang"))
    sys.modules.setdefault("sglang.srt", types.ModuleType("sglang.srt"))
    sys.modules["sglang.srt.runtime_context"] = rt
    envs = types.SimpleNamespace(SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=types.SimpleNamespace(get=lambda: 0.2))
    ns = {"torch": fake_torch, "os": os, "Dict": dict, "envs": envs, "get_is_capture_mode": lambda: capture}
    exec(compile(ast.Module(body=consts, type_ignores=[]), "<consts>", "exec"), ns)
    ns["_KPOOL_LOGITS_BUDGET_MB"] = override_mb
    ns["_KPOOL_LOGITS_BUDGET_BYTES"] = {}
    exec(compile(ast.Module(body=[module_fns["_kpool_logits_budget_bytes"]], type_ignores=[]), "<budget>", "exec"), ns)
    return ns["_kpool_logits_budget_bytes"], calls


GiB = 1 << 30
total = int(139.35 * GiB)
fn, calls = budget_ns(free=int(18 * GiB), total=total, mem_static=0.86)
want = min(int(18 * GiB * 0.2), int(total * (1 - 0.86) * 0.2), int(total * 0.3))
assert abs(fn(0) - want) <= 1 and fn(0) == fn(0) and calls["mem_get_info"] == 1  # cached after the first call
fn, _ = budget_ns(free=int(100 * GiB), total=total, mem_static=0.86)
assert abs(fn(0) - int(total * 0.14 * 0.2)) <= 1  # the static headroom caps it when free memory is large
fn, calls = budget_ns(free=int(18 * GiB), total=total, mem_static=0.86, capture=True)
fn(0), fn(0)
assert calls["mem_get_info"] == 0  # capture mode: static budget, no sync, not cached
fn, calls = budget_ns(free=int(18 * GiB), total=total, mem_static=0.86, override_mb=256)
assert fn(0) == 256 << 20 and calls["mem_get_info"] == 0
checks += 4

print(f"dsa kpool logits budget: {checks} checks OK")
