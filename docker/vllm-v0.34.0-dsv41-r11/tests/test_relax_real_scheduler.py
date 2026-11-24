"""Relaxed align-mode chunk ends for GLM-5.3 Flash geometry (CPU, real Scheduler).

Geometry as E3 prefill engines: attention + Mamba (KDA) block 4736,
`--prefix-match-unit 64`, Mamba "align", FlashKDA prefill checkpoints
(1 checkpoint block, alignment 16), MNBT 8192, long-prefill threshold 4736.

Arms:
  stock : both flags off (== r10 scheduler/manager/worker rule)
  ckpt  : VLLM_MAMBA_ONEPASS_CKPT=1 (the onepass overlay)
  relax : VLLM_MAMBA_ONEPASS_CKPT=1 + VLLM_MAMBA_RELAX_ALIGN=1

Oracle (`H`): emulates the worker's Mamba state writes from the real
SchedulerOutput (append-only worker block table, CoW copies before the
forward, pre-copy read from column (start-1)//B, final state to column
cdiv(end,B)-1) and the in-forward checkpoint exactly where the overlay's own
worker function `compute_mamba_prefill_checkpoints` puts it. After every step:
  * every chunk reads the state of exactly its start;
  * every Mamba block carrying a prefix-cache key holds the state its key
    names (no poisoned entry), including blocks parked in the free queue.
"""

import json
import os
import random
import statistics
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
from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
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
MNBT = 8192
ATTN, MAMBA = 0, 1
EOS = 50256
init_none_hash(sha256)
HASHER = get_request_block_hasher(UNIT, sha256)

ARMS = {"stock": (False, False), "ckpt": (True, False), "relax": (True, True)}


def build(monkeypatch, arm, spec_blocks=0, mnbt=MNBT, threshold=4736,
          nblocks=600, max_num_seqs=64):
    ckpt, relax = ARMS[arm]
    monkeypatch.setattr(kvi, "MAMBA_ONEPASS_CKPT", ckpt, raising=False)
    monkeypatch.setattr(kvi, "MAMBA_RELAX_ALIGN", relax, raising=False)
    monkeypatch.setenv("VLLM_MAMBA_ONEPASS_TAIL_SKIP_TOKENS", "0")
    model_config = ModelConfig(
        model=MODEL, trust_remote_code=True, dtype="float16", seed=42,
        max_model_len=65536,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=mnbt,
        max_model_len=65536, long_prefill_token_threshold=threshold,
        enable_chunked_prefill=True, is_encoder_decoder=False, watermark=0.0,
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
    assert getattr(s, "mamba_relax_align", False) == relax
    return s


def tokens(conv, n):
    return list(range(conv * 100_000 + 1000, conv * 100_000 + 1000 + n))


def mkreq(rid, conv, n, max_tokens=1):
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sp.update_from_generation_config({}, EOS)
    return Request(request_id=rid, prompt_token_ids=tokens(conv, n),
                   sampling_params=sp, pooling_params=None, block_hasher=HASHER)


class H:
    """Real Scheduler driver + worker-state oracle + step-fill recorder."""

    def __init__(self, s, loaded=True, worker="ok"):
        self.s = s
        self.pool = s.kv_cache_manager.block_pool
        self.null = self.pool.null_block.block_id
        self.wt = {}
        self.state = {}  # Mamba block id -> (conv, tokens)
        self.chunks = defaultdict(list)
        self.hits = {}
        self.conv = {}
        self.prompt = {}
        self.hash_owner = {}
        self.violations = []
        self.steps = []
        self.fill = []  # per step: (prefill rows, prefill tokens, waiting after)
        self.lost_boundaries = 0  # crossed strictly inside a chunk, not written
        self.relax_exports = 0
        self.worker = worker  # "ok" | "no_relax_export" | "no_export"
        if loaded:
            # A decoding neighbour keeps >1 request eligible, so the 4736
            # threshold applies as under load.
            self.add("bg", 99, 100, max_tokens=100_000)
            while self.s.requests["bg"].num_computed_tokens < 100:
                self.step()

    def add(self, rid, conv, n, max_tokens=1):
        r = mkreq(rid, conv, n, max_tokens)
        self.conv[rid] = conv
        self.prompt[rid] = n
        for i, h in enumerate(list(r.block_hashes)):
            self.hash_owner[bytes(h)] = (conv, (i + 1) * UNIT)
        self.s.add_request(r)
        return r

    def export(self, s0, e):
        offsets, cols = compute_mamba_prefill_checkpoints(
            [e], [e - s0], hash_block_size=UNIT, mamba_block_size=B,
            checkpoint_alignment=ALIGN, drop_eagle_block=False,
            cache_hit_alignment_tokens=UNIT, num_hash_lookahead_tokens=0)
        if not offsets[0]:
            return None
        relax = s0 % UNIT == ALIGN and kvi.MAMBA_RELAX_ALIGN
        if self.worker == "no_export" or (relax and self.worker == "no_relax_export"):
            return None
        return cols[0], s0 + offsets[0], relax

    def step(self):
        out = self.s.schedule()
        for c in out.kv_cache_block_copies or []:  # CoW before the forward
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
        writes, finishing = [], []
        prows = ptoks = 0
        for rid, n in out.num_scheduled_tokens.items():
            req = self.s.requests[rid]
            s0, e = starts[rid], starts[rid] + n
            if e >= req.num_prompt_tokens:
                finishing.append(rid)
            if s0 >= req.num_prompt_tokens:
                continue
            prows += 1
            ptoks += n
            conv = self.conv[rid]
            self.chunks[rid].append((s0, n))
            row = self.wt[rid][MAMBA]
            if s0 > 0:
                got = self.state.get(row[(s0 - 1) // B])
                if got != (conv, s0):
                    self.violations.append(("read", rid, s0, got))
            if kvi.MAMBA_RELAX_ALIGN and e < req.num_prompt_tokens:
                # Non-final ends: block boundary, relaxed residue, or a unit stop.
                if e % B and e % UNIT not in (0, ALIGN):
                    self.violations.append(("end", rid, s0, e))
            crossed = [b for b in range(cdiv(s0 + 1, B) * B, e, B) if s0 < b < e]
            if s0 % UNIT == ALIGN and len(crossed) > 1:
                self.violations.append(("cross2", rid, s0, e))
            ex = self.export(s0, e)
            if ex is not None:
                col, pos, relax = ex
                self.relax_exports += relax
                bid = row[col]
                if bid != self.null:
                    writes.append((bid, (conv, pos)))
                crossed = [b for b in crossed if b != pos]
            self.lost_boundaries += len(crossed)
            writes.append((row[cdiv(e, B) - 1], (conv, e)))
        for bid, st in writes:  # forward: checkpoint export, then final state
            self.state[bid] = st
        ids = list(out.num_scheduled_tokens)
        toks = [[7] if rid in finishing else [] for rid in ids]
        self.s.update_from_output(
            out, ModelRunnerOutput(req_ids=ids, sampled_token_ids=toks))
        waiting = len([r for r in self.s.requests if r != "bg"
                       and r not in out.num_scheduled_tokens])
        self.fill.append((prows, ptoks, waiting))
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

    def run(self, n=400):
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


# ------------------------------------------------------------ helpers

def test_relax_helpers(monkeypatch):
    monkeypatch.setattr(kvi, "MAMBA_ONEPASS_CKPT", True)
    monkeypatch.setattr(kvi, "MAMBA_RELAX_ALIGN", True)
    assert kvi.mamba_relax_residue(UNIT, ALIGN) == ALIGN
    assert kvi.mamba_relax_residue(UNIT, UNIT) is None
    assert kvi.mamba_relax_residue(UNIT, None) is None
    assert kvi.mamba_relax_floor(3456, UNIT, ALIGN) == 3408
    assert kvi.mamba_relax_floor(3408, UNIT, ALIGN) == 3408
    assert kvi.mamba_relax_floor(10, UNIT, ALIGN) == -1
    f = kvi.mamba_relax_boundary_checkpoint
    assert f(3408, 8000, B, UNIT, ALIGN) == 4736
    assert f(3408, 4736, B, UNIT, ALIGN) == 0  # ends on the boundary
    assert f(3408, 4000, B, UNIT, ALIGN) == 0  # no crossing
    assert f(3392, 8000, B, UNIT, ALIGN) == 0  # unit-aligned start: tail rule
    assert f(4736, 8000, B, UNIT, ALIGN) == 0
    monkeypatch.setattr(kvi, "MAMBA_RELAX_ALIGN", False)
    assert f(3408, 8000, B, UNIT, ALIGN) == 0
    assert kvi.mamba_relax_floor(3456, UNIT, ALIGN) == -1


# ------------------------------------------------- the `break` it removes

@pytest.mark.parametrize("arm", list(ARMS))
def test_two_cold_requests_fill_the_step(monkeypatch, arm):
    """Two cold 7800-token prompts: stock schedules 4736 + 0 (break)."""
    h = H(build(monkeypatch, arm), loaded=False)
    h.add("a", 0, 7800)
    h.add("b", 1, 7800)
    out = h.step()
    got = dict(out.num_scheduled_tokens)
    if arm == "relax":
        assert got == {"a": 4736, "b": 3408}, got  # floor of 3456 to 64k+16
    else:
        assert got == {"a": 4736}, got
    h.run()
    assert not h.violations, h.violations[:5]
    if arm == "relax":
        # b resumes at 3408 and crosses 4736 with an in-forward checkpoint.
        assert h.chunks["b"][1][0] == 3408
        assert h.relax_exports >= 1
        assert (1, 4736) in h.keys(MAMBA)
    # Same cached states for the next turn in every arm.
    assert {k for k in h.keys(MAMBA) if k[0] == 1} >= {(1, 4736), (1, 7744)}


@pytest.mark.parametrize("arm", list(ARMS))
def test_small_request_behind_unfit_head(monkeypatch, arm):
    """Head can't use the leftover budget; a 60-token request behind it can."""
    h = H(build(monkeypatch, arm), loaded=False)
    h.add("a", 0, 7800)
    h.add("b", 1, 7800)
    h.add("c", 2, 60)
    out = h.step()
    got = dict(out.num_scheduled_tokens)
    if arm == "relax":
        assert got["a"] == 4736 and got["b"] == 3408, got
    else:
        assert got == {"a": 4736}, got  # break: c waits behind b
    h.run()
    assert not h.violations, h.violations[:5]


# ------------------------------------------------- single-request bands

BANDS = [
    ("hit7040_end7800", 7041, 7800),
    ("hit7040_end9800", 7041, 9800),
    ("hit2944_end6000", 2945, 6000),
    ("hit2944_end3300", 2945, 3300),
    ("hit4736_end7800", 4737, 7800),
    ("hit7744_end7840", 7745, 7840),
    ("hit2944_end20000", 2945, 20000),
]


@pytest.mark.parametrize("band", BANDS, ids=[b[0] for b in BANDS])
def test_single_request_bands_match_onepass(monkeypatch, band):
    """Alone in the step (budget >= threshold) relax == onepass chunking."""
    _, seed, warm = band
    res = {}
    for arm in ARMS:
        h = H(build(monkeypatch, arm))
        turn(h, "seed", 0, seed)
        hit, chunks = turn(h, "warm", 0, warm)
        assert not h.violations, (arm, h.violations[:5])
        res[arm] = (hit, chunks, {k for k in h.keys(MAMBA) if k[0] == 0},
                    h.probe(0, warm + 500))
    assert res["relax"] == res["ckpt"], (res["relax"], res["ckpt"])
    assert res["relax"][3] == res["stock"][3]  # next-turn hit


# ------------------------------------------------- random multi-turn oracle

def workload(seed, nconv=6, turns=6):
    rng = random.Random(seed)
    plan = []
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


def next_turn_hits(h, plan):
    last = {}
    for _, _, conv, n in plan:
        last[conv] = max(last.get(conv, 0), n)
    return {c: h.probe(c, n + 3000) for c, n in sorted(last.items())}


@pytest.mark.parametrize("lookahead", [0, 4])
@pytest.mark.parametrize("spec_blocks", [0, 3])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_workload_oracle(monkeypatch, seed, spec_blocks, lookahead):
    plan = workload(seed)
    res = {}
    # lookahead > 0 without the matching hash lookahead is a synthetic config:
    # stock can livelock there (tail stop vs `_reserve_prefill_lookahead`,
    # seed 3), so that variant compares the two one-pass arms only.
    arms = list(ARMS) if not lookahead else ["ckpt", "relax"]
    for arm in arms:
        s = build(monkeypatch, arm, spec_blocks)
        s.num_prefill_lookahead = lookahead
        h = H(s, loaded=False)
        passes, computed, nreq = drive(h, plan)
        assert not h.violations, (arm, h.violations[:5])
        res[arm] = (passes, computed, nreq, h.lost_boundaries,
                    next_turn_hits(h, plan), len(h.steps), h.relax_exports)
    print(f"\nRANDOM seed={seed} spec={spec_blocks} la={lookahead} " + " ".join(
        f"{a}:steps={r[5]},passes={r[0]},tokens={r[1]},lostB={r[3]},relaxX={r[6]}"
        for a, r in res.items()))
    assert all(r[4] == res["relax"][4] for r in res.values())


# ------------------------------------------------- step fill under load

def load_plan(seed, nconv=40, turns=4, horizon=30.0):
    """Open-loop E3-like arrivals: multi-turn conversations, prompts grow by a
    bimodal delta (tool-call follow-ups vs new content)."""
    rng = random.Random(seed)
    plan = []
    for c in range(nconv):
        n = rng.randrange(2000, 14000)
        t = rng.uniform(0, horizon)
        for k in range(turns):
            plan.append((t, f"c{c}t{k}", c, n))
            kind = rng.choices([0, 1, 2], weights=[4, 4, 2])[0]
            n += (rng.randrange(20, 100), rng.randrange(1000, 5000),
                  rng.randrange(5000, 10000))[kind]
            n = min(n, 60000)
            t += rng.uniform(6.0, 12.0)
    return sorted(plan)


def drive_timed(h, plan, fixed=0.300, per_tok=35e-6, max_steps=20000):
    plan = list(plan)
    now, arrive, done = 0.0, {}, {}
    n = 0
    while plan or h.s.requests:
        while plan and plan[0][0] <= now:
            t, rid, conv, ntok = plan.pop(0)
            arrive[rid] = t
            h.add(rid, conv, ntok)
        if not h.s.requests:
            now = plan[0][0]
            continue
        before = set(h.s.requests)
        h.step()
        _, ptoks, _ = h.fill[-1]
        now += fixed + per_tok * ptoks
        for rid in before - set(h.s.requests):
            done[rid] = now
        n += 1
        assert n < max_steps
    lat = [done[r] - arrive[r] for r in done]
    busy = [f for f in h.fill if f[0] > 0]
    with_wait = [f for f in busy if f[2] > 0]
    passes = sum(len(v) for v in h.chunks.values())
    return dict(
        steps=len(busy),
        tok_mean=round(statistics.mean(f[1] for f in busy)),
        tok_p50=round(statistics.median(f[1] for f in busy)),
        tok_mean_waiting=round(statistics.mean(f[1] for f in with_wait)),
        reqs_mean=round(statistics.mean(f[0] for f in busy), 2),
        one_req_share=round(sum(f[0] == 1 for f in with_wait) / len(with_wait), 3),
        underfilled_share=round(
            sum(f[1] < MNBT - UNIT for f in with_wait) / len(with_wait), 3),
        passes_per_req=round(passes / len(h.chunks), 2),
        tokens=sum(nn for v in h.chunks.values() for _, nn in v),
        lat_mean=round(statistics.mean(lat), 2),
        lat_p50=round(statistics.median(lat), 2),
        lat_p90=round(sorted(lat)[int(0.9 * len(lat))], 2),
        makespan=round(now, 1),
        lostB=h.lost_boundaries,
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_step_fill_under_load(monkeypatch, seed):
    plan = load_plan(seed)
    res = {}
    for arm in ARMS:
        h = H(build(monkeypatch, arm, nblocks=6000, max_num_seqs=128),
              loaded=False)
        res[arm] = drive_timed(h, plan)
        assert not h.violations, (arm, h.violations[:5])
        res[arm]["next_hits"] = next_turn_hits(h, [(0, 0, c, n) for _, _, c, n in plan])
    for arm, r in res.items():
        print(f"\nFILL seed={seed} {arm}: " + " ".join(
            f"{k}={v}" for k, v in r.items() if k != "next_hits"))
    assert res["relax"]["next_hits"] == res["ckpt"]["next_hits"] == res["stock"]["next_hits"]
    assert res["relax"]["tok_mean_waiting"] > res["ckpt"]["tok_mean_waiting"]
    assert res["relax"]["one_req_share"] < res["ckpt"]["one_req_share"]


# ------------------------------------------------- negative controls

def test_broken_worker_is_caught(monkeypatch):
    """A worker that skips the relaxed boundary export poisons the cache."""
    h = H(build(monkeypatch, "relax"), loaded=False, worker="no_relax_export")
    h.add("a", 0, 7800)
    h.add("b", 1, 7800)
    h.run()
    assert any(v[0] == "poison" and v[2] == (1, 4736) for v in h.violations), \
        h.violations[:5]


def test_relax_flag_needs_onepass_flag():
    """VLLM_MAMBA_RELAX_ALIGN=1 without VLLM_MAMBA_ONEPASS_CKPT=1 is inert."""
    import subprocess
    import sys
    code = ("import vllm.v1.kv_cache_interface as k;"
            "print(k.MAMBA_ONEPASS_CKPT, k.MAMBA_RELAX_ALIGN)")
    for ckpt, want in (("0", "False False"), ("1", "True True")):
        env = dict(os.environ, VLLM_MAMBA_RELAX_ALIGN="1",
                   VLLM_MAMBA_ONEPASS_CKPT=ckpt)
        got = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True, check=True)
        assert got.stdout.strip().splitlines()[-1] == want


def test_guard_rejects_unsafe_geometry(monkeypatch):
    with pytest.raises(ValueError, match="VLLM_MAMBA_ONEPASS_CKPT"):
        build(monkeypatch, "relax", threshold=2048)


# ------------------------------------------------- flags-off trace dump

def test_dump(monkeypatch):
    """RELAX_DUMP=<path> RELAX_DUMP_ARM=stock|ckpt: write the trace; compare
    across overlays (relax flag off must equal the shipped code)."""
    path = os.environ.get("RELAX_DUMP")
    if not path:
        pytest.skip("RELAX_DUMP not set")
    arm = os.environ.get("RELAX_DUMP_ARM", "stock")
    out = {}
    for seed in (1, 2, 3):
        for spec in (0, 3):
            h = H(build(monkeypatch, arm, spec))
            drive(h, workload(seed))
            out[f"{seed}/{spec}"] = dict(
                steps=h.steps,
                attn=sorted(map(list, h.keys(ATTN))),
                mamba=sorted(map(list, h.keys(MAMBA))),
                violations=len(h.violations),
            )
        plan = load_plan(seed, nconv=16)
        h = H(build(monkeypatch, arm, nblocks=6000, max_num_seqs=128), loaded=False)
        drive_timed(h, plan)
        out[f"load/{seed}"] = dict(steps=h.steps,
                                   mamba=sorted(map(list, h.keys(MAMBA))))
    with open(path, "w") as f:
        json.dump(out, f, sort_keys=True)
