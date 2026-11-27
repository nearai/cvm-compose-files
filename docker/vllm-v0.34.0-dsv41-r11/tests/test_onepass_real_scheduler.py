"""One-pass warm prefill for GLM-5.3 Flash geometry (CPU only, real Scheduler).

Geometry as E3 prefill engines: attention + Mamba (KDA) block 4736,
`--prefix-match-unit 64` (hash block 64), Mamba "align", FlashKDA prefill
checkpoints (1 checkpoint block, alignment 16), MNBT 8192, long-prefill
threshold 4736.

Arms (all default-off, set per test):
  stock     : overlay with both flags off (== r10 scheduler/manager)
  ckpt      : VLLM_MAMBA_ONEPASS_CKPT=1 (backport of vllm#66938 + #60888)
  tailskip  : VLLM_MAMBA_ONEPASS_TAIL_SKIP_TOKENS=1024 (drop the tail stop)
  both      : ckpt + tailskip

Oracle (`H`): emulates the worker's Mamba state writes from the real
SchedulerOutput (append-only worker block table, CoW copies before the
forward, align pre-copy read from column (start-1)//B, final state written to
column cdiv(end,B)-1, internal checkpoint written to column cdiv(end,B)-2
when the shared validity rule allows, exactly as
`compute_mamba_prefill_checkpoints` does). After every step it checks:
  * every chunk reads the state of exactly its start (no wrong initial state);
  * every Mamba block that carries a prefix-cache key holds the state of the
    prefix that key names (no poisoned cache entry).
"""

import json
import os
import random
from collections import defaultdict

import pytest

from vllm.platforms import current_platform

if not current_platform.device_type:
    current_platform.device_type = "cuda"

import torch

import vllm.v1.kv_cache_interface as kvi
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

MODEL = os.environ.get("ONEPASS_MODEL", "/models/qwen36")
B = 4736
UNIT = 64
ALIGN = 16
ATTN, MAMBA = 0, 1
EOS = 50256
init_none_hash(sha256)
HASHER = get_request_block_hasher(UNIT, sha256)

ARMS = {
    "stock": (False, 0),
    "ckpt": (True, 0),
    "tailskip": (False, 1024),
    "both": (True, 1024),
}


def build(monkeypatch, arm, spec_blocks=0, mnbt=8192, threshold=4736, nblocks=600):
    ckpt, skip = ARMS[arm]
    monkeypatch.setattr(kvi, "MAMBA_ONEPASS_CKPT", ckpt, raising=False)
    monkeypatch.setenv("VLLM_MAMBA_ONEPASS_TAIL_SKIP_TOKENS", str(skip))
    model_config = ModelConfig(
        model=MODEL, trust_remote_code=True, dtype="float16", seed=42,
        max_model_len=65536,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=64, max_num_batched_tokens=mnbt, max_model_len=65536,
        long_prefill_token_threshold=threshold, enable_chunked_prefill=True,
        is_encoder_decoder=False, watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=B, gpu_memory_utilization=0.9, cache_dtype="auto",
        enable_prefix_caching=True, mamba_cache_mode="align",
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config, model_config=model_config,
        cache_config=cache_config, parallel_config=ParallelConfig(),
    )
    groups = [
        KVCacheGroupSpec(["attn"], MLAAttentionSpec(
            block_size=B, num_kv_heads=1, max_tp_shards=1, head_size=576,
            dtype=torch.float32)),
        KVCacheGroupSpec(["kda"], MambaSpec(
            block_size=B, shapes=(1, 1), dtypes=(torch.float32,),
            mamba_cache_mode="align", num_speculative_blocks=spec_blocks,
            num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=ALIGN)),
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=nblocks, kv_cache_tensors=[], kv_cache_groups=groups)
    for attr, val in (("hash_block_size", UNIT), ("cache_hit_alignment_tokens", UNIT)):
        if hasattr(kv_cache_config, attr):
            setattr(kv_cache_config, attr, val)
    cache_config.num_gpu_blocks = nblocks
    register_all_kvcache_specs(vllm_config)
    s = Scheduler(
        vllm_config=vllm_config, kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=B, hash_block_size=UNIT, log_stats=True,
    )
    assert s.mamba_partial_cache_hit and s.mamba_has_prefill_checkpoint_blocks
    assert s.kv_cache_manager.coordinator.enable_partial_hash_hits
    assert getattr(s, "mamba_onepass_tail_skip_tokens", 0) == skip
    return s


def tokens(conv, n):
    return list(range(conv * 100_000 + 1000, conv * 100_000 + 1000 + n))


def mkreq(rid, conv, n, max_tokens=1):
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sp.update_from_generation_config({}, EOS)
    return Request(request_id=rid, prompt_token_ids=tokens(conv, n),
                   sampling_params=sp, pooling_params=None, block_hasher=HASHER)


class H:
    """Real Scheduler driver + worker-state oracle."""

    def __init__(self, s, loaded=True):
        self.s = s
        self.pool = s.kv_cache_manager.block_pool
        self.null = self.pool.null_block.block_id
        self.wt = {}  # worker block table rows (append-only)
        self.state = {}  # Mamba block id -> (conv, tokens)
        self.chunks = defaultdict(list)
        self.hits = {}
        self.conv = {}
        self.hash_owner = {}  # BlockHash -> (conv, tokens)
        self.violations = []
        self.steps = []
        self.export = True  # negative control: a worker that never exports
        if loaded:
            # A decoding neighbour (1 token/step) keeps >1 request eligible,
            # so the long-prefill threshold (4736) applies as under load.
            self.add("bg", 99, 100, max_tokens=100_000)
            while self.s.requests["bg"].num_computed_tokens < 100:
                self.step()

    def add(self, rid, conv, n, max_tokens=1):
        r = mkreq(rid, conv, n, max_tokens)
        self.conv[rid] = conv
        for i, h in enumerate(list(r.block_hashes)):
            self.hash_owner[bytes(h)] = (conv, (i + 1) * UNIT)
        self.s.add_request(r)
        return r

    def step(self):
        out = self.s.schedule()
        # 1. CoW copies run before the forward.
        for c in out.kv_cache_block_copies or []:
            if c.src_block_id in self.state:
                self.state[c.dst_block_id] = self.state[c.src_block_id]
            else:
                self.state.pop(c.dst_block_id, None)
        starts = {}
        for nr in out.scheduled_new_reqs:
            self.wt[nr.req_id] = [list(x) for x in nr.block_ids]
            starts[nr.req_id] = nr.num_computed_tokens
            self.hits[nr.req_id] = nr.num_computed_tokens
        cr = out.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            nb = cr.new_block_ids[i]
            if nb is not None:
                for g, ids in enumerate(nb):
                    self.wt[rid][g].extend(ids)
            starts[rid] = cr.num_computed_tokens[i]
        writes = []
        finishing = []
        for rid, n in out.num_scheduled_tokens.items():
            req = self.s.requests[rid]
            s0, e = starts[rid], starts[rid] + n
            if e >= req.num_prompt_tokens:
                finishing.append(rid)
            if s0 >= req.num_prompt_tokens:
                continue
            conv = self.conv[rid]
            self.chunks[rid].append((s0, n))
            row = self.wt[rid][MAMBA]
            if s0 > 0:
                got = self.state.get(row[(s0 - 1) // B])
                if got != (conv, s0):
                    self.violations.append(("read", rid, s0, got))
            pos = kvi.get_mamba_prefill_checkpoint_position(e, UNIT, False, 0)
            if self.export and kvi.is_mamba_prefill_checkpoint_valid(
                s0, e, pos, UNIT, B, ALIGN
            ):
                bid = row[cdiv(e, B) - 2]
                if bid != self.null:
                    writes.append((bid, (conv, pos)))
            writes.append((row[cdiv(e, B) - 1], (conv, e)))
        for bid, st in writes:  # forward: checkpoint export, then final state
            self.state[bid] = st
        ids = list(out.num_scheduled_tokens)
        toks = [[7] if rid in finishing else [] for rid in ids]
        self.s.update_from_output(
            out, ModelRunnerOutput(req_ids=ids, sampled_token_ids=toks))
        self.steps.append(dict(out.num_scheduled_tokens))
        self.check_cache()
        return out

    def check_cache(self):
        for blk in self.pool.blocks:
            key = blk.block_hash
            if key is None or get_group_id(key) != MAMBA:
                continue
            owner = self.hash_owner.get(bytes(get_block_hash(key)))
            if owner is None or owner[1] != blk.block_hash_num_tokens:
                self.violations.append(("key", blk.block_id, owner,
                                        blk.block_hash_num_tokens))
            elif self.state.get(blk.block_id) != owner:
                self.violations.append(("poison", blk.block_id, owner,
                                        self.state.get(blk.block_id)))

    def run(self, n=200):
        for _ in range(n):
            if not set(self.s.requests) - {"bg"}:
                return
            self.step()
        raise AssertionError("did not drain")

    def keys(self, group):
        out = set()
        for blk in self.pool.blocks:
            key = blk.block_hash
            if key is not None and get_group_id(key) == group:
                out.add(self.hash_owner[bytes(get_block_hash(key))])
        return out

    def probe(self, conv, n):
        r = mkreq("probe", conv, n)
        _, hit, _ = self.s.kv_cache_manager.get_computed_blocks(r)
        return hit


def turn(h, rid, conv, n):
    h.add(rid, conv, n)
    h.run()
    return h.hits[rid], h.chunks[rid]


# ---------------------------------------------------------------- walk 7800

@pytest.mark.parametrize("spec_blocks", [0, 3])
@pytest.mark.parametrize("arm", list(ARMS))
def test_warm_7800_walk(monkeypatch, arm, spec_blocks):
    h = H(build(monkeypatch, arm, spec_blocks))
    seed_hit, seed_chunks = turn(h, "seed", 0, 7041)  # leaves a Mamba state at 7040
    assert seed_hit == 0
    assert seed_chunks == [(0, 4736), (4736, 2304), (7040, 1)]
    hit, chunks = turn(h, "warm", 0, 7800)
    assert hit == 7040
    if arm == "stock":
        assert chunks == [(7040, 704), (7744, 56)]  # tail stop at 7744
    else:
        assert chunks == [(7040, 760)]
    mamba = h.keys(MAMBA)
    # The prompt checkpoint 7744 is cached unless only the tail stop was dropped.
    assert ((0, 7744) in mamba) == (arm != "tailskip")
    assert (0, 7040) in mamba and (0, 4736) in mamba
    assert (0, 7744) in h.keys(ATTN)
    next_hit = h.probe(0, 8100)
    assert next_hit == (7040 if arm == "tailskip" else 7744)
    assert not h.violations, h.violations[:5]


def test_next_turn_state_equal_stock_vs_ckpt(monkeypatch):
    """Cached keys after the warm turn: ckpt writes exactly what stock writes."""
    res = {}
    for arm in ("stock", "ckpt", "tailskip", "both"):
        h = H(build(monkeypatch, arm))
        turn(h, "seed", 0, 7041)
        turn(h, "warm", 0, 7800)
        # Conversation 0 only: under tailskip the 100-token neighbour (tail
        # 64 <= N) also runs in one pass and caches no 64-token Mamba state.
        res[arm] = tuple({k for k in h.keys(g) if k[0] == 0} for g in (ATTN, MAMBA))
        assert not h.violations, (arm, h.violations[:5])
    assert res["ckpt"] == res["stock"]
    assert res["both"] == res["stock"]
    attn_s, mamba_s = res["stock"]
    attn_t, mamba_t = res["tailskip"]
    assert attn_t == attn_s
    assert mamba_s - mamba_t == {(0, 7744)} and not (mamba_t - mamba_s)


# ------------------------------------------------------------ prompt bands

# (seed prompt len -> hit, warm prompt len, expected passes per arm)
BANDS = [
    # hit mid-block 1, end same block (the 7800 case): #60888
    ("hit7040_end7800", 7041, 7800, dict(stock=2, ckpt=1, tailskip=1, both=1)),
    # hit mid-block 1, end in next block: #66938
    ("hit7040_end9800", 7041, 9800, dict(stock=3, ckpt=1, tailskip=2, both=1)),
    # hit mid-block 0, end block 1: #66938 (col == initial)
    ("hit2944_end6000", 2945, 6000, dict(stock=3, ckpt=1, tailskip=3, both=1)),
    # whole prompt inside block 0: no column for a checkpoint
    ("hit2944_end3300", 2945, 3300, dict(stock=2, ckpt=2, tailskip=1, both=1)),
    # block-aligned hit, end same block: col == initial is the shared block
    ("hit4736_end7800", 4737, 7800, dict(stock=2, ckpt=2, tailskip=2, both=2)),
    # follow-up with a tiny tail (<=100 uncached band)
    ("hit7744_end7840", 7745, 7840, dict(stock=2, ckpt=1, tailskip=1, both=1)),
]
COLD = [
    ("cold3000", 3000, dict(stock=2, ckpt=2, tailskip=2, both=2)),
    ("cold7800", 7800, dict(stock=3, ckpt=3, tailskip=3, both=3)),
    ("cold4000", 4000, dict(stock=2, ckpt=2, tailskip=2, both=2)),
]


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("band", BANDS, ids=[b[0] for b in BANDS])
def test_warm_bands(monkeypatch, arm, band):
    _, seed, warm, expect = band
    h = H(build(monkeypatch, arm))
    turn(h, "seed", 0, seed)
    hit, chunks = turn(h, "warm", 0, warm)
    assert hit == (seed - 1) // UNIT * UNIT
    assert len(chunks) == expect[arm], chunks
    assert not h.violations, h.violations[:5]


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("band", COLD, ids=[b[0] for b in COLD])
def test_cold_bands(monkeypatch, arm, band):
    _, n, expect = band
    h = H(build(monkeypatch, arm))
    hit, chunks = turn(h, "cold", 0, n)
    assert hit == 0
    assert len(chunks) == expect[arm], chunks
    assert not h.violations, h.violations[:5]


def test_tailskip_loss_is_bounded(monkeypatch):
    """Skipped tails compound only until the gap exceeds N, then a stop saves."""
    h = H(build(monkeypatch, "tailskip"))
    turn(h, "t0", 0, 7041)
    hits = []
    for i, n in enumerate((7800, 8100, 9000, 9100)):
        hit, chunks = turn(h, f"t{i + 1}", 0, n)
        hits.append((hit, len(chunks)))
    # 7800: skip (gap 704); 8100: skip (gap 1024); 9000: gap 1920 > N -> stop.
    assert hits == [(7040, 1), (7040, 1), (7040, 2), (8960, 1)]
    assert not h.violations


def test_guard_rejects_unsafe_geometry(monkeypatch):
    with pytest.raises(ValueError, match="VLLM_MAMBA_ONEPASS_CKPT"):
        build(monkeypatch, "ckpt", threshold=2048)
    build(monkeypatch, "stock", threshold=2048)  # flags off: no guard


# ---------------------------------------------------- random workload oracle

def workload(seed, nconv=6, turns=6):
    rng = random.Random(seed)
    plan = []  # (arrival_step, rid, conv, n)
    for c in range(nconv):
        n = rng.choice([1500, 3000, 5200, 7000, 12000])
        t = rng.randrange(0, 6)
        for k in range(turns):
            plan.append((t, f"c{c}t{k}", c, n))
            n += rng.choice([20, 60, 90, 400, 900, 1500, 2600, 4000])
            t += rng.randrange(4, 14)
    return sorted(plan)


def drive(h, plan, max_steps=3000):
    plan = list(plan)
    step = 0
    while plan or set(h.s.requests) - {"bg"}:
        while plan and plan[0][0] <= step:
            _, rid, conv, n = plan.pop(0)
            h.add(rid, conv, n)
        if h.s.requests:
            h.step()
        step += 1
        assert step < max_steps
    ch = {k: v for k, v in h.chunks.items() if k != "bg"}
    passes = sum(len(v) for v in ch.values())
    computed = sum(n for v in ch.values() for _, n in v)
    return passes, computed, len(ch)


@pytest.mark.parametrize("spec_blocks", [0, 3])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_workload_oracle(monkeypatch, seed, spec_blocks):
    plan = workload(seed)
    res = {}
    for arm in ARMS:
        h = H(build(monkeypatch, arm, spec_blocks))
        res[arm] = drive(h, plan)
        assert not h.violations, (arm, h.violations[:5])
    print(f"\nRANDOM seed={seed} spec={spec_blocks} "
          + " ".join(f"{a}:passes={p},tokens={t},reqs={r}"
                     for a, (p, t, r) in res.items()))
    assert res["ckpt"][0] <= res["stock"][0]
    assert res["ckpt"][1] == res["stock"][1]  # same recompute: same hits


def test_dump_flags_off(monkeypatch):
    """Writes the stock-arm trace; run once with and once without the overlay
    and compare the files (flags off must equal the shipped code)."""
    path = os.environ.get("ONEPASS_DUMP")
    if not path:
        pytest.skip("ONEPASS_DUMP not set")
    out = {}
    for seed in (1, 2, 3):
        for spec in (0, 3):
            h = H(build(monkeypatch, "stock", spec))
            drive(h, workload(seed))
            out[f"{seed}/{spec}"] = dict(
                steps=h.steps,
                attn=sorted(map(list, h.keys(ATTN))),
                mamba=sorted(map(list, h.keys(MAMBA))),
                violations=len(h.violations),
            )
    with open(path, "w") as f:
        json.dump(out, f, sort_keys=True)



def test_oracle_negative_control(monkeypatch):
    """The oracle flags a keyed checkpoint the worker did not write."""
    h = H(build(monkeypatch, "ckpt"))
    turn(h, "seed", 0, 7041)
    h.export = False
    turn(h, "warm", 0, 7800)
    assert any(v[0] == "poison" and v[2] == (0, 7744) for v in h.violations)
