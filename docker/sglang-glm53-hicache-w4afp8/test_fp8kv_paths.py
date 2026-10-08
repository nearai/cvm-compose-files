"""CPU-only tests for the FP8 KV cache runtime paths added by fp8kv-flashmla.diff (no GPU, no torch).

Usage: python3 test_fp8kv_paths.py --root .../python/sglang/srt

The patched files are parsed, never imported: the real functions (calculate_mla_kv_cache_dim,
the DSA backend __init__ gate, _forward_flashmla_kv, _compute_flashmla_metadata,
_check_kpool_tail_backend, _get_mla_kv_buffer_from_fp8_for_dsa) are extracted by ast and executed
against stubs, once with the NoPE/FlashMLA gate enabled and once with it disabled, so a regression
in a changed branch fails here. The FlashMLA kernel itself is covered by the GPU runs in the README.
"""

import argparse
import ast
import pathlib
import sys
import types
import typing

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
root = pathlib.Path(parser.parse_args().root)
NS = types.SimpleNamespace


def parse(rel):
    return ast.parse((root / rel).read_text(), filename=rel)


def find_func(tree, name):
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(hits) == 1, (name, len(hits))
    return hits[0]


def run_func(tree, name, env):
    """Exec the named function from the parsed module into env and return it."""
    fn = find_func(tree, name)
    fn.decorator_list = []
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, name, "exec"), env)
    return env[name]


def stmts_mentioning(fn, needle):
    return [s for s in fn.body if needle in ast.unparse(s)]


# 1. KV pool row width (kv_cache_configurator.calculate_mla_kv_cache_dim).
FP8 = object()
BF16 = object()


def row_width(*, rope, dtype=FP8, prefill="tilelang", decode="tilelang", hip=False, dsa=True):
    env = {
        "ModelConfig": object,
        "torch": NS(dtype=object, float8_e4m3fn=FP8),
        "is_deepseek_dsa": lambda cfg: dsa,
        "get_exec": lambda: NS(kernel=NS(dsa_prefill_backend=prefill, dsa_decode_backend=decode)),
        "get_disagg": lambda: NS(disaggregation_mode=None),
        "DSATokenToKVPool": NS(quant_block_size=128, rope_storage_dtype=NS(itemsize=2)),
        "_is_hip": hip,
    }
    fn = run_func(parse("mem_cache/kv_cache_configurator.py"), "calculate_mla_kv_cache_dim", env)
    cfg = NS(kv_lora_rank=512, qk_rope_head_dim=rope, hf_config=None)
    return fn(model_config=cfg, kv_cache_dtype=dtype)


assert row_width(rope=0, prefill="flashmla_kv", decode="flashmla_kv") == 656
assert row_width(rope=0, prefill="tilelang", decode="flashmla_kv") == 656  # either role selects it
assert row_width(rope=0, prefill="flashmla_kv", decode="tilelang") == 656
assert row_width(rope=0, prefill="tilelang", decode="tilelang") == 528  # gate off: no padding
assert row_width(rope=0, prefill="flashmla_kv", decode="flashmla_kv", hip=True) == 528
assert row_width(rope=64, prefill="flashmla_kv", decode="flashmla_kv") == 656  # real rope: unchanged
assert row_width(rope=64) == 656
assert row_width(rope=0, dtype=BF16, prefill="flashmla_kv", decode="flashmla_kv") == 512
assert row_width(rope=0, prefill="trtllm", decode="trtllm") == 512
assert row_width(rope=64, dsa=False, prefill="flashmla_kv", decode="flashmla_kv") == 576

# 2. DSA backend __init__ gate: q rope padding and the FlashMLA topk (index table width).
backend_tree = parse("layers/attention/dsa_backend.py")
init_fns = [
    n
    for n in ast.walk(backend_tree)
    if isinstance(n, ast.FunctionDef) and n.name == "__init__" and stmts_mentioning(n, "flashmla_kv_pad_rope_dim")
]
assert len(init_fns) == 1
gate_stmts = stmts_mentioning(init_fns[0], "flashmla_kv_pad_rope_dim") + stmts_mentioning(init_fns[0], "flashmla_kv_topk")
assert len(gate_stmts) == 2, len(gate_stmts)
gate_code = compile(ast.fix_missing_locations(ast.Module(body=gate_stmts, type_ignores=[])), "gate", "exec")


def backend(*, rope=0, fp8=True, dim=656, kpool=1, topk=2048):
    self = NS(
        qk_rope_head_dim=rope,
        dsa_kv_cache_store_fp8=fp8,
        kv_cache_dim=dim,
        dsa_index_kpool=kpool,
        dsa_index_topk=topk,
    )
    exec(gate_code, {"self": self})
    return self


assert backend().flashmla_kv_pad_rope_dim == 64
for off in (dict(rope=64), dict(fp8=False), dict(dim=528), dict(dim=1024)):
    assert backend(**off).flashmla_kv_pad_rope_dim == 0, off
for kpool, want in ((1, 2048), (2, 2176), (4, 2176), (129, 2176), (130, 2304)):
    got = backend(kpool=kpool).flashmla_kv_topk
    assert got == want and got % 128 == 0, (kpool, got)
assert backend(kpool=1, topk=2000).flashmla_kv_topk == 2000  # kpool off: untouched, not rounded


# 3. _forward_flashmla_kv (q pad, -1 padded index table) and _compute_flashmla_metadata (topk).
class Fake:
    def __init__(self, *shape):
        self.shape = tuple(shape)
        self.device = "cpu"

    def view(self, *shape):
        total = 1
        for d in self.shape:
            total *= d
        known = 1
        for d in shape:
            known *= d if d != -1 else 1
        return Fake(*[total // known if d == -1 else d for d in shape])

    def new_zeros(self, *shape):
        return Fake(*shape)

    def unsqueeze(self, dim):
        s = list(self.shape)
        s.insert(dim, 1)
        return Fake(*s)

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return Fake(*self.shape)


pads = []


def fake_pad(x, widths, value=0):
    pads.append((x.shape, widths, value))
    return Fake(*x.shape[:-1], x.shape[-1] + sum(widths))


calls = []
kernel = types.ModuleType("sgl_kernel.flash_mla")
kernel.flash_mla_with_kvcache = lambda **kw: (calls.append(kw) or (Fake(*kw["q"].shape), None))
kernel.get_mla_metadata = lambda **kw: (calls.append(kw) or ("meta", "splits"))
sys.modules["sgl_kernel"] = types.ModuleType("sgl_kernel")
sys.modules["sgl_kernel.flash_mla"] = kernel
env = {
    "torch": NS(
        Tensor=object,
        int32="int32",
        nn=NS(functional=NS(pad=fake_pad)),
        empty=lambda shape, dtype=None, device=None: Fake(*shape),
    ),
    "quantize_k_cache": lambda kv: kv,
    "DSAMetadata": object,
    "DSAFlashMLAMetadata": lambda **kw: NS(**kw),
}
forward = run_func(backend_tree, "_forward_flashmla_kv", env)
compute = run_func(backend_tree, "_compute_flashmla_metadata", env)


def decode(be, head_dim, table_width):
    be.real_page_size = 64
    be.flashmla_kv_num_q_heads = 64
    pads.clear()
    calls.clear()
    meta = NS(
        dsa_cache_seqlens_int32=Fake(2),
        flashmla_metadata=NS(flashmla_metadata="m", num_splits="s"),
    )
    layer = NS(tp_q_head_num=64, head_dim=head_dim)
    forward(be, Fake(2, 64, head_dim), Fake(1, 64, 1, be.kv_cache_dim), 512, 0.1, layer, meta, Fake(2, table_width))
    return calls[-1]


nope = backend(kpool=4)
kw = decode(nope, 512, 2048 + 3)
assert kw["q"].shape == (2, 1, 64, 576), kw["q"].shape  # zero-padded to the d_qk=576 the kernel asserts
assert kw["indices"].shape == (2, 1, 2176), kw["indices"].shape
assert pads == [((2, 1, 64, 512), (0, 64), 0), ((2, 1, 2051), (0, 125), -1)], pads
try:
    decode(nope, 512, 2177)  # wider than the padded table must trip the assert, not truncate
except AssertionError:
    pass
else:
    raise SystemExit("an index table wider than flashmla_kv_topk was accepted")

plain = backend(rope=64, kpool=1)  # gate off, kpool off: the original pass-through behaviour
kw = decode(plain, 576, 2048)
assert pads == [] and kw["q"].shape == (2, 1, 64, 576) and kw["indices"].shape == (2, 1, 2048), (pads, kw)

compute(nope, Fake(2), 1)
assert calls[-1]["topk"] == 2176, calls[-1]
compute(plain, Fake(2), 1)
assert calls[-1]["topk"] == 2048, calls[-1]

# 4. Tail tokens are allowed on the flashmla_kv implementation, and still refused elsewhere.
check = run_func(
    parse("layers/attention/dsa/dsa_backend_kpool.py"),
    "_check_kpool_tail_backend",
    {"Optional": typing.Optional, "torch": NS(Tensor=object), "_DSA_IMPL_T": object},
)
tail = NS(dsa_index_kpool=4)
for impl in ("fa3", "tilelang", "trtllm", "flashmla_kv"):
    check(tail, object(), impl, "decode")
for impl in ("flashmla_sparse", "aiter"):
    try:
        check(tail, object(), impl, "decode")
    except NotImplementedError as e:
        assert "FlashMLA-KV" in str(e)
    else:
        raise SystemExit(f"index_kpool > 1 accepted on {impl}")
check(NS(dsa_index_kpool=1), object(), "flashmla_sparse", "decode")  # kpool off: any backend

# 5. Hybrid-backend metadata selection and NoPE k_pe in the FP8 MHA dequant path.
mha = find_func(parse("models/deepseek_common/attention_forward_methods/forward_mha.py"), "_get_mla_kv_buffer_from_fp8_for_dsa")
pick = [s for s in mha.body if "full_attn_backend" in ast.unparse(s)]
nope_k_pe = [s for s in mha.body if isinstance(s, ast.If) and "qk_rope_head_dim == 0" in ast.unparse(s.test)]
assert len(pick) == 1 and len(nope_k_pe) == 1
pick_code = compile(ast.fix_missing_locations(ast.Module(body=pick, type_ignores=[])), "pick", "exec")
k_pe_code = compile(ast.fix_missing_locations(ast.Module(body=nope_k_pe, type_ignores=[])), "k_pe", "exec")


def select(backend_obj):
    scope = {"backend": backend_obj}
    exec(pick_code, scope)
    return scope["backend"]


inner = NS(forward_metadata="full")
assert select(NS(full_attn_backend=inner)) is inner  # hybrid wrapper -> wrapped DSA backend
assert select(inner) is inner  # plain DSA backend is used as is


def k_pe_for(rope):
    scope = {"self": NS(qk_rope_head_dim=rope), "k_pe": "rope-slots"}
    exec(k_pe_code, scope)
    return scope["k_pe"]


assert k_pe_for(0) is None
assert k_pe_for(64) == "rope-slots"
print("test_fp8kv_paths OK: row width, q pad, kpool topk padding, tail backend and hybrid selection, gate on and off")
