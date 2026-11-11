"""(Ported to SGLang v0.5.23.) CPU check for dsa-q8-flashmla-sparse.diff: the flashmla_sparse_q8 DSA prefill backend works with the v7
FP8 KV layout (656-byte NoPE row, KPool index > 1).

The kernels need a GPU, so the three changed pieces are checked by ast on the installed sources:
1. dsa_backend.py forward_extend: under flashmla_sparse_q8 with KPool > 1, the page table's
   2048 + (kpool - 1) top-k IDs are padded with -1 to the flashmla_kv width (the Q8 SM90 kernel needs a
   multiple of 128). The block is extracted and run on CPU tensors.
2. dsa_backend.py _forward_flashmla_sparse_q8kv8 is extracted and executed on CPU tensors with the SM90
   kernel replaced by a capturing stub. For Flash NoPE (512-wide Q/K, no model RoPE), the kernel gets
   576-wide Q and KV, with the first 512 dimensions preserved and the reserved RoPE dimensions, padded
   heads and KV tail rows zero, and d_v unchanged. Covered: power-of-two, padded and generic head counts;
   born-fp8 Q at 512 and 576 wide; bf16, kv_k and paged-prefix KV; stale persistent Q and KV buffers. Standard
   576-wide MLA keeps its width.
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

# 1. KPool top-k pad. v0.5.23 pads inside _forward_flashmla_sparse_q8kv8 via the upstream helper
# _pad_topk_width_buffered(multiple=128) (so the NEAR forward_extend pad block was dropped). Run the
# installed helper on CPU tensors: 2051 -> 2176 with -1 slots, aligned widths unchanged.
pad_fn = backend["_pad_topk_width_buffered"]
pad_fn.decorator_list = []
_pns = {"torch": torch}
exec(compile(ast.Module(body=[pad_fn], type_ignores=[]), "<pad helper>", "exec"), _pns)
_pad = _pns["_pad_topk_width_buffered"]


def run_pad(width):
    s = types.SimpleNamespace(_q8kv8_topk_pad_buf=None, _q8kv8_topk_pad_width=None)
    return _pad(s, torch.arange(width, dtype=torch.int32).repeat(3, 1), multiple=128)


out = run_pad(2051)
assert out.shape == (3, 2176), out.shape
assert torch.equal(out[:, :2051], torch.arange(2051, dtype=torch.int32).repeat(3, 1)), "valid IDs changed"
assert bool((out[:, 2051:] == -1).all()), "pad slots must be masked (-1)"
assert run_pad(2176).shape == (3, 2176), "already-aligned width must be unchanged"
assert any(
    "_pad_topk_width_buffered" in ast.unparse(n)
    for n in ast.walk(backend["_forward_flashmla_sparse_q8kv8"])
    if isinstance(n, ast.Call)
), "q8 forward no longer pads top-k"

# 2. Q8 kernel layout: run the installed _forward_flashmla_sparse_q8kv8 on CPU tensors with the SM90 kernel
# replaced by a capturing stub, and check the Q/KV tensors it would launch with.
def load_q8_forward():
    fn = backend["_forward_flashmla_sparse_q8kv8"]
    # Drop the in-function kernel import; the stub is injected as a global instead.
    fn.body = [
        node
        for node in fn.body
        if not (isinstance(node, ast.ImportFrom) and "sparse_mla_q8kv8_prefill" in (node.module or ""))
    ]
    fn.decorator_list = []
    captured = {}

    def sparse_mla_q8kv8_prefill_fwd(**kw):
        captured.update(kw)
        return kw["out"], kw["max_logits"], kw["lse"]

    def gather_dequant_requant_fp8_paged_dsa(paged, flat_idx, extra_rows, out):
        assert out.shape == (flat_idx.shape[0] + extra_rows, paged.shape[-1]), (out.shape, paged.shape)
        # The real kernel zeroes the pad rows itself; copy bytes through uint8 views (CPU fp8 lacks index ops).
        out.view(torch.uint8).zero_()
        out.view(torch.uint8)[: flat_idx.shape[0]] = paged.view(torch.uint8)[flat_idx]
        return out

    def concat_and_cast_q_fp8_pad(q_fp8, q_nope, q_rope, num_heads):
        q_fp8[:, :num_heads].copy_(torch.cat([q_nope, q_rope], dim=-1))

    def concat_cast_kv_fp8_pad(out, k, k_rope, n):
        out[:n].copy_(torch.cat([k.view(n, -1), k_rope.view(n, -1)], dim=-1))
        out[n:].zero_()
        return out

    def concat_mla_absorb_q_general(q_nope, q_rope):
        return torch.cat([q_nope, q_rope], dim=-1)

    ns = {
        "torch": torch,
        "Optional": __import__("typing").Optional,
        "sparse_mla_q8kv8_prefill_fwd": sparse_mla_q8kv8_prefill_fwd,
        "gather_dequant_requant_fp8_paged_dsa": gather_dequant_requant_fp8_paged_dsa,
        "gather_cast_kv_fp8_pad_paged": None,
        "concat_and_cast_q_fp8_pad": concat_and_cast_q_fp8_pad,
        "concat_cast_kv_fp8_pad": concat_cast_kv_fp8_pad,
        "concat_mla_absorb_q_general": concat_mla_absorb_q_general,
        "q8kv8_topk_length_from_indices": None,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<q8 forward>", "exec"), ns)
    return ns[fn.name], captured


FP8 = torch.float8_e4m3fn
D_NOPE, PAD_ROPE, TOPK, N_KV = 512, 64, 128, 7


def fp8_ref(x):
    return x.to(FP8).float()


def make_self(born_q=None, stale_kv_rows=0, stale_q=None):
    s = types.SimpleNamespace(
        kv_lora_rank=D_NOPE,
        flashmla_kv_pad_rope_dim=PAD_ROPE,
        _q8kv8_born_q_stash=None,
        _q8kv8_born_q_buf=None,
        _q8kv8_qpad_buf=None,
        _q8kv8_identity_scale=None,
        _q8kv8_kv_buf=None,
        _q8kv8_topk_length_enabled=False,
        _q8kv8_out_bufs=None,
        _q8kv8_topk_pad_buf=None,
        _q8kv8_topk_pad_width=None,
        dsa_kv_cache_store_fp8=True,
    )
    s._pad_topk_width_buffered = types.MethodType(_pad, s)
    if born_q is not None:
        s._q8kv8_born_q_buf = born_q
        s._q8kv8_born_q_stash = (born_q.shape[0], 3)
    if stale_kv_rows:
        # A previous, larger call left non-zero data in the persistent KV buffer.
        s._q8kv8_kv_buf = torch.ones(stale_kv_rows, D_NOPE + PAD_ROPE).to(FP8)
    if stale_q is not None:
        # Persistent padded-Q buffer from a previous call: active head rows hold old data, pad heads are zero.
        tokens, heads = stale_q
        buf = torch.zeros(tokens, 64, D_NOPE + PAD_ROPE, dtype=FP8)
        buf[:, :heads] = torch.ones(tokens, heads, D_NOPE + PAD_ROPE).to(FP8)
        s._q8kv8_qpad_buf = buf
    return s


def test_nope_q8_kernel_layout():
    fwd, cap = load_q8_forward()
    torch.manual_seed(0)
    width = D_NOPE + PAD_ROPE
    for num_heads in (64, 32, 192):  # power-of-two, padded to 64, generic fallback
        for kv_path in ("bf16", "kv_k", "prefix"):
            for born in (None, D_NOPE, width):
                if born is not None and num_heads != 64:
                    continue
                cap.clear()
                t = 5
                q_nope = torch.randn(t, num_heads, D_NOPE, dtype=torch.bfloat16)
                q_rope = q_nope.new_empty(t, num_heads, 0)
                born_q = None
                if born is not None:
                    born_q = torch.randn(t, num_heads, born, dtype=torch.bfloat16).to(FP8)
                self_ = make_self(
                    born_q,
                    stale_kv_rows=N_KV + TOPK + 3,
                    stale_q=(t + 2, num_heads) if num_heads < 64 else None,
                )
                kv = torch.randn(N_KV, D_NOPE, dtype=torch.bfloat16)
                page_table = torch.full((t, TOPK), -1, dtype=torch.int32)
                kwargs = {}
                if kv_path == "bf16":
                    kwargs["kv_bf16"] = kv.view(N_KV, 1, D_NOPE)
                elif kv_path == "kv_k":
                    kwargs.update(kv_bf16=None, kv_k=kv.view(N_KV, 1, D_NOPE), kv_k_rope=None)
                else:
                    # The 656-byte cache row dequantizes to 576 dims whose last 64 are zero.
                    paged = torch.zeros(N_KV + 4, width, dtype=FP8)
                    paged[:N_KV, :D_NOPE] = kv.to(FP8)
                    kwargs.update(
                        kv_bf16=None,
                        paged_kv_cache=paged,
                        page_table_1_flattened=torch.arange(N_KV),
                    )
                kwargs.setdefault("kv_bf16", None)
                o = fwd(self_, q_nope, q_rope, v_head_dim=D_NOPE, page_table_1=page_table,
                        sm_scale=0.1, layer_id=3, **kwargs)
                case = f"heads={num_heads} kv={kv_path} born={born}"
                q, k = cap["q"].float(), cap["kv"].float()
                assert q.shape[-1] == width, f"{case}: Q width {q.shape[-1]} != {width}"
                assert k.shape == (N_KV + TOPK, 1, width), f"{case}: KV shape {tuple(k.shape)}"
                q_src = born_q.float()[:, :, :D_NOPE] if born is not None else fp8_ref(q_nope)
                assert torch.equal(q[:, :num_heads, :D_NOPE], q_src), f"{case}: Q nope dims changed"
                if born != width:
                    assert bool((q[:, :, D_NOPE:] == 0).all()), f"{case}: Q reserved dims not zero"
                if num_heads < 64:
                    assert q.shape[1] == 64 and bool((q[:, num_heads:] == 0).all()), f"{case}: pad heads"
                assert torch.equal(k[:N_KV, 0, :D_NOPE], fp8_ref(kv)), f"{case}: KV nope dims changed"
                assert bool((k[:N_KV, 0, D_NOPE:] == 0).all()), f"{case}: KV reserved dims not zero"
                assert bool((k[N_KV:] == 0).all()), f"{case}: KV tail rows not zero"
                assert cap["d_v"] == D_NOPE and cap["out"].shape[-1] == D_NOPE, f"{case}: d_v changed"
                assert o.shape == (t, num_heads, D_NOPE), f"{case}: output shape {tuple(o.shape)}"

    # Standard 576-wide MLA (real RoPE) must not gain a second pad.
    cap.clear()
    q_nope = torch.randn(3, 64, D_NOPE, dtype=torch.bfloat16)
    q_rope = torch.randn(3, 64, PAD_ROPE, dtype=torch.bfloat16)
    kv = torch.randn(N_KV, 1, width, dtype=torch.bfloat16)
    fwd(make_self(), q_nope, q_rope, kv_bf16=kv, v_head_dim=D_NOPE,
        page_table_1=torch.full((3, TOPK), -1, dtype=torch.int32), sm_scale=0.1)
    assert cap["q"].shape[-1] == width and cap["kv"].shape[-1] == width, "standard MLA width changed"
    assert torch.equal(cap["q"].float(), fp8_ref(torch.cat([q_nope, q_rope], dim=-1))), "standard MLA Q changed"


test_nope_q8_kernel_layout()

# 3. KPool allowlist.
kpool_tree = ast.parse(open(args.dsa_backend_kpool).read(), filename=args.dsa_backend_kpool)
allowlists = [
    {elt.value for elt in node.elts if isinstance(elt, ast.Constant)}
    for node in ast.walk(kpool_tree)
    if isinstance(node, ast.Tuple)
    and any(isinstance(e, ast.Constant) and e.value == "flashmla_kv" for e in node.elts)
]
assert any("flashmla_sparse_q8" in a for a in allowlists), "flashmla_sparse_q8 not on the KPool allowlist"

# 4. Launch validation (v0.5.23 rejects fp8 + NoPE for q8): accepted with the 656 B NoPE row, still
# rejected for the upstream 528 B row.
mod = ast.parse(open(args.dsa_backend).read(), filename=args.dsa_backend)
vfn = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_flashmla_sparse_q8_backend")
_vns = {"torch": torch}
exec(compile(ast.Module(body=[vfn], type_ignores=[]), "<validate>", "exec"), _vns)
_validate = _vns["_validate_flashmla_sparse_q8_backend"]
_kw = dict(prefill_impl="flashmla_sparse_q8", decode_impl="flashmla_kv", kv_cache_dtype=torch.float8_e4m3fn,
           kv_cache_store_fp8=True, qk_rope_head_dim=0, device_sm_major=9)
_validate(**_kw, nope_fp8_v32_row=True)
try:
    _validate(**_kw)
except ValueError:
    pass
else:
    raise AssertionError("fp8 NoPE q8 must still be rejected without the 656 B row")

print("dsa-q8-flashmla-sparse checks passed")
