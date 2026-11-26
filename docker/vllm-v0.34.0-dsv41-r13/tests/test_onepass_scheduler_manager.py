# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the one-pass Mamba checkpoint (VLLM_NEAR_ONEPASS_CKPT).

GLM-5.3 Flash geometry: attention and KDA block 4736, --prefix-match-unit 64,
align mode, FlashKDA prefill checkpoints (alignment 16), MTP 3 (3 speculative
scratch blocks). The scheduler split is the real
`Scheduler._mamba_block_aligned_split` on a stub, as in the r12
test_pmu64_block4736.py; the manager is the real KVCacheManager.
"""

from math import lcm
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.prefix_cache.test_partial_prefix_cache_hits import (
    drain_boundary_state_offloads,
)
from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    get_mamba_onepass_checkpoint_col,
    get_mamba_prefill_checkpoint_position,
)

BLOCK = 4736
UNIT = 64
ALIGN = 16
MAMBA_GROUP = 1
SPEC = 3


def _manager(onepass, num_speculative_blocks=SPEC):
    init_none_hash(sha256)
    mamba = MambaSpec(
        block_size=BLOCK,
        shapes=(1, 1),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=ALIGN,
        onepass_prefill_checkpoint=onepass,
    )
    config = KVCacheConfig(
        num_blocks=96,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(["mamba"], mamba),
        ],
    )
    return make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=16384,
        enable_caching=True,
        dcp_world_size=1,
        scheduler_block_size=lcm(BLOCK, BLOCK),
        hash_block_size=UNIT,
        use_eagle=False,
        num_prefill_lookahead=0,
        num_hash_lookahead_tokens=0,
    )


def _scheduler(onepass_blocks):
    """Stub with the attributes `_mamba_block_aligned_split` reads.
    `mamba_onepass_num_speculative_blocks` is what Scheduler.__init__ derives
    from the MambaSpec (test_scheduler_init_reads_spec covers that)."""
    return SimpleNamespace(
        block_size=BLOCK,
        cache_config=SimpleNamespace(block_size=BLOCK),
        hash_block_size=UNIT,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=0,
        mamba_partial_cache_hit=True,
        mamba_shared_prefix_checkpoint=False,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=ALIGN,
        mamba_onepass_num_speculative_blocks=onepass_blocks,
    )


def _prefill(manager, scheduler, request_id, prompt):
    """Schedule a prompt chunk by chunk; returns (request, hit, chunks)."""
    request = make_request(request_id, prompt, UNIT, sha256)
    computed, num_hit, junction = manager.get_computed_blocks(request)
    request.shared_prefix_boundary = junction
    start = num_hit
    num_new = Scheduler._mamba_block_aligned_split(
        scheduler, request, len(prompt) - start, num_hit, 0
    )
    chunks = [(start, num_new)]
    assert (
        manager.allocate_slots(
            request, num_new, num_hit, computed, num_lookahead_tokens=SPEC
        )
        is not None
    )
    request.num_computed_tokens = start + num_new
    while request.num_computed_tokens < len(prompt):
        manager.new_step_starts()
        start = request.num_computed_tokens
        num_new = Scheduler._mamba_block_aligned_split(
            scheduler, request, len(prompt) - start
        )
        chunks.append((start, num_new))
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=SPEC)
            is not None
        )
        request.num_computed_tokens = start + num_new
    return request, num_hit, chunks


def _mamba_mgr(manager):
    return manager.coordinator.single_type_managers[MAMBA_GROUP]


def _hit(manager, tokens, rid="probe"):
    probe = make_request(rid, tokens, UNIT, sha256)
    _, hit, _ = manager.get_computed_blocks(probe)
    return hit


# --- the shared predicate -------------------------------------------------


def test_predicate_same_block_tail_gets_first_scratch_column():
    # cold 168: checkpoint 128 in block 0 -> scratch column 1
    assert get_mamba_onepass_checkpoint_col(0, 168, 128, UNIT, BLOCK, ALIGN, 3) == 1
    # warm tail [2944, 3457): checkpoint 3456 (the 448 + 9 case scaled)
    assert get_mamba_onepass_checkpoint_col(2944, 3457, 3456, UNIT, BLOCK, ALIGN, 3) == 1
    # tail in block 1, start on the block boundary
    assert (
        get_mamba_onepass_checkpoint_col(BLOCK, BLOCK + 200, BLOCK + 192, UNIT, BLOCK, ALIGN, 3)
        == 2
    )


@pytest.mark.parametrize(
    "args",
    [
        (0, 168, 128, UNIT, BLOCK, None, 3),  # Triton backend: no checkpoint output
        (0, 168, 128, UNIT, BLOCK, ALIGN, 0),  # no MTP: no scratch column
        (0, 100, 64, UNIT, BLOCK, ALIGN, 3),  # fine: 64 is reachable
        (0, 64, 0, UNIT, BLOCK, ALIGN, 3),  # checkpoint == start
        (64, 120, 64, UNIT, BLOCK, ALIGN, 3),  # checkpoint == start
        (32, 300, 256, UNIT, BLOCK, ALIGN, 3),  # unaligned start
        (128, 160, 128, UNIT, BLOCK, ALIGN, 3),  # less than one unit in
        (0, 4800, 4736, UNIT, BLOCK, ALIGN, 3),  # crosses a block
        (0, 4734, 4672, UNIT, BLOCK, ALIGN, 3),  # next step (1 + 3) crosses
        (2944, 3000, 2944, UNIT, BLOCK, ALIGN, 3),  # nothing new to keep
    ],
)
def test_predicate_declines(args):
    if args == (0, 100, 64, UNIT, BLOCK, ALIGN, 3):
        assert get_mamba_onepass_checkpoint_col(*args) == 1
        return
    assert get_mamba_onepass_checkpoint_col(*args) == -1


def test_predicate_next_step_guard_edge():
    # end + S stays in block 0: allowed; one token more: refused.
    assert get_mamba_onepass_checkpoint_col(0, 4732, 4672, UNIT, BLOCK, ALIGN, 3) == 1
    assert get_mamba_onepass_checkpoint_col(0, 4733, 4672, UNIT, BLOCK, ALIGN, 3) == -1


# --- scheduler split ------------------------------------------------------


@pytest.mark.parametrize("onepass", [False, True])
def test_cold_168_split_vs_onepass(onepass):
    manager = _manager(onepass)
    sched = _scheduler(SPEC if onepass else 0)
    tokens = list(range(1000, 1168))
    req, hit, chunks = _prefill(manager, sched, "r", tokens)
    assert hit == 0
    if onepass:
        assert chunks == [(0, 168)]
    else:
        assert chunks == [(0, 128), (128, 40)]
    # Either way the follow-up resumes at 128.
    manager.new_step_starts()
    assert _hit(manager, tokens + [7] * 50) == 128


@pytest.mark.parametrize("onepass", [False, True])
def test_warm_tail_448_plus_9_runs_one_pass(onepass):
    manager = _manager(onepass)
    sched = _scheduler(SPEC if onepass else 0)
    tokens = list(range(5000, 5000 + 3457))
    owner, _, _ = _prefill(manager, sched, "owner", tokens[:3000])
    manager.free(owner)
    manager.new_step_starts()
    drain_boundary_state_offloads(manager)
    req, hit, chunks = _prefill(manager, sched, "warm", tokens)
    assert hit == 2944
    if onepass:
        assert chunks == [(2944, 513)]
    else:
        assert chunks == [(2944, 512), (3456, 1)]
    manager.new_step_starts()
    assert _hit(manager, tokens + [3] * 64) == 3456


def test_flag_off_spec_is_default_and_split_unchanged():
    """Spec flag off: same split as r12 for every prompt length."""
    manager = _manager(False)
    sched_off = _scheduler(0)
    for n in (65, 168, 457, 2000, 4700, 4735, 6000):
        req = make_request(f"r{n}", list(range(n)), UNIT, sha256)
        stock = Scheduler._mamba_block_aligned_split(
            SimpleNamespace(**{k: v for k, v in vars(sched_off).items()
                               if k != "mamba_onepass_num_speculative_blocks"}),
            req, n, 0, 0,
        )
        assert Scheduler._mamba_block_aligned_split(sched_off, req, n, 0, 0) == stock
    assert MambaSpec(block_size=1, shapes=(1,), dtypes=(torch.float32,)).onepass_prefill_checkpoint is False


def test_guard_falls_back_to_split():
    manager = _manager(True)
    sched = _scheduler(SPEC)
    tokens = list(range(4735))
    _, _, chunks = _prefill(manager, sched, "edge", tokens)
    assert chunks == [(0, 4672), (4672, 63)]


def test_no_spec_blocks_falls_back_to_split():
    manager = _manager(True, num_speculative_blocks=0)
    assert not _mamba_mgr(manager).onepass_prefill_checkpoint
    sched = _scheduler(0)  # what Scheduler.__init__ derives for S = 0
    _, _, chunks = _prefill(manager, sched, "nospec", list(range(168)))
    assert chunks == [(0, 128), (128, 40)]


# --- manager: key on the scratch block, CoW before the next forward -------


def test_onepass_keys_scratch_then_cows_before_next_step():
    manager = _manager(True)
    sched = _scheduler(SPEC)
    tokens = list(range(1000, 1168))
    req, _, chunks = _prefill(manager, sched, "r", tokens)
    assert chunks == [(0, 168)]
    mgr = _mamba_mgr(manager)
    blocks = manager.get_blocks("r").blocks[MAMBA_GROUP]
    # running column 0, scratch columns 1..3
    assert len(blocks) == 1 + SPEC
    running, scratch = blocks[0], blocks[1]
    assert scratch.block_hash_num_tokens == 128
    assert running.block_hash is None
    assert mgr._partial_hit_reqs["r"] == (1, scratch)
    # No extra block was taken for the checkpoint, and no CoW yet.
    copies, _ = manager.take_kv_cache_block_copies()
    assert not copies
    # The worker writes the column the manager keyed.
    offsets, cols = compute_mamba_prefill_checkpoints(
        [168], [168], hash_block_size=UNIT, mamba_block_size=BLOCK,
        checkpoint_alignment=ALIGN, drop_eagle_block=False,
        cache_hit_alignment_tokens=UNIT, onepass=True, num_speculative_blocks=SPEC,
    )
    assert (offsets, cols) == ([128], [1])

    # Next step: first decode (1 token + 3 lookahead). The key moves to a
    # fresh block and the copy is queued for before that step's forward.
    manager.new_step_starts()
    req.append_output_token_ids(7)
    assert manager.allocate_slots(req, 1, num_lookahead_tokens=SPEC) is not None
    copies, _ = manager.take_kv_cache_block_copies()
    mamba_copies = [c for c in copies if c[0] == scratch.block_id]
    assert len(mamba_copies) == 1
    cow_id = mamba_copies[0][1]
    assert scratch.block_hash is None  # scratch is free to be overwritten
    assert manager.get_blocks("r").blocks[MAMBA_GROUP][1] is scratch
    assert manager.block_pool.blocks[cow_id].block_hash_num_tokens == 128
    offloads = drain_boundary_state_offloads(manager)
    assert (MAMBA_GROUP, cow_id, 128) in offloads.get("r", [])
    manager.new_step_starts()
    assert _hit(manager, tokens + [5] * 40) == 128


def test_request_finishing_in_prefill_step_keeps_scratch_key():
    """A P leg (max_tokens 1) ends after the prefill step: the scratch block
    is freed with its key and nothing writes it again."""
    manager = _manager(True)
    sched = _scheduler(SPEC)
    tokens = list(range(9, 9 + 457))
    req, _, chunks = _prefill(manager, sched, "p", tokens)
    assert chunks == [(0, 457)]
    scratch = manager.get_blocks("p").blocks[MAMBA_GROUP][1]
    manager.free(req)
    manager.new_step_starts()
    assert scratch.block_hash_num_tokens == 448
    assert _hit(manager, tokens + [1] * 20) == 448


def test_worker_and_manager_agree_on_offsets():
    """compute_mamba_prefill_checkpoints (worker) vs the manager's record,
    over many (start, end) pairs, with the flag on; flag off = stock."""
    for start in range(0, 2 * BLOCK, 448):
        for qlen in (1, 4, 63, 64, 65, 168, 457, 2000, 4800):
            end = start + qlen
            x = get_mamba_prefill_checkpoint_position(end, UNIT, False)
            stock = compute_mamba_prefill_checkpoints(
                [end], [qlen], UNIT, BLOCK, ALIGN, False, UNIT
            )
            off = compute_mamba_prefill_checkpoints(
                [end], [qlen], UNIT, BLOCK, ALIGN, False, UNIT,
                onepass=False, num_speculative_blocks=SPEC,
            )
            on = compute_mamba_prefill_checkpoints(
                [end], [qlen], UNIT, BLOCK, ALIGN, False, UNIT,
                onepass=True, num_speculative_blocks=SPEC,
            )
            assert off == stock
            col = get_mamba_onepass_checkpoint_col(start, end, x, UNIT, BLOCK, ALIGN, SPEC)
            if stock[1][0] >= 0:
                assert on == stock  # upstream column wins; predicates disjoint
                assert col == -1
            elif col >= 0:
                assert on == ([x - start], [col])
                assert col == (end - 1) // BLOCK + 1
            else:
                assert on == stock


def test_scheduler_init_reads_spec():
    """Scheduler.__init__'s derivation, run on the attributes it reads."""
    import inspect

    src = inspect.getsource(Scheduler.__init__)
    assert "mamba_onepass_num_speculative_blocks" in src
    assert "onepass_prefill_checkpoint" in src
