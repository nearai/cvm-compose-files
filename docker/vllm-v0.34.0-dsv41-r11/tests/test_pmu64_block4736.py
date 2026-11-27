# SPDX-License-Identifier: Apache-2.0
"""Local regression tests for patches 26/27 (backports of vllm#67717 and
vllm#67451) at the GLM-5.3 Flash FP8 geometry: attention and Mamba (KDA)
block 4736, `--prefix-match-unit 64` (hash block 64), align mode, prefill
checkpoints on. Same helpers and flow as the upstream tests, larger sizes."""

from types import SimpleNamespace

import pytest

from tests.v1.core.prefix_cache.test_partial_prefix_cache_hits import (
    drain_boundary_state_offloads,
    make_full_mamba_manager,
)
from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler

BLOCK = 4736
UNIT = 64
MAMBA_GROUP = 1


def _manager(num_prefill_checkpoint_blocks=1):
    init_none_hash(sha256)
    return make_full_mamba_manager(
        dcp_world_size=1,
        hash_block_size=UNIT,
        full_block_size=BLOCK,
        mamba_block_size=BLOCK,
        num_blocks=64,
        num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
    )


def _scheduler(has_prefill_checkpoint_blocks=True):
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
        mamba_has_prefill_checkpoint_blocks=has_prefill_checkpoint_blocks,
        mamba_prefill_checkpoint_alignment=16,
    )


def _prefill(manager, scheduler, request_id, prompt):
    request = make_request(request_id, prompt, UNIT, sha256)
    computed, num_hit, junction = manager.get_computed_blocks(request)
    request.shared_prefix_boundary = junction
    start = num_hit
    num_new = Scheduler._mamba_block_aligned_split(
        scheduler, request, len(prompt) - start, num_hit, 0
    )
    chunks = [(start, num_new)]
    assert manager.allocate_slots(request, num_new, num_hit, computed) is not None
    request.num_computed_tokens = start + num_new
    while request.num_computed_tokens < len(prompt):
        manager.new_step_starts()
        start = request.num_computed_tokens
        num_new = Scheduler._mamba_block_aligned_split(
            scheduler, request, len(prompt) - start
        )
        chunks.append((start, num_new))
        assert manager.allocate_slots(request, num_new) is not None
        request.num_computed_tokens = start + num_new
    manager.new_step_starts()
    return request, num_hit, chunks


def test_block4736_unit64_mid_block_resume_is_not_cached_as_block_end():
    """Patch 26 (#67717): a follow-up turn resumes from a 64-aligned partial
    hit inside block 0 and its chunk runs past 4736. Block 0 then holds the
    resume-point Mamba state, so it must not be published under the
    4736-token full-block hash; a later request must hit the true resume
    point, not a stale "4736" state."""
    manager, scheduler = _manager(), _scheduler()
    tokens = list(range(1000, 1000 + 11000))

    # Turn 1: a 3000-token prompt leaves a Mamba state at a 64-aligned point.
    owner, owner_hit, _ = _prefill(manager, scheduler, "owner", tokens[:3000])
    assert owner_hit == 0
    manager.free(owner)
    drain_boundary_state_offloads(manager)

    # Turn 2 shares the 3000-token prefix and reaches block 2, so the
    # internal-checkpoint column (block 1) is past the resume block and the
    # split runs [hit, 10000) in one step (cf. upstream's (96, 134) chunk).
    resume, num_hit, chunks = _prefill(manager, scheduler, "resume", tokens[:10000])
    assert 0 < num_hit < BLOCK and num_hit % UNIT == 0  # partial hit in block 0
    first_start, first_len = chunks[0]
    assert first_start == num_hit
    # Precondition for the bug: the first chunk runs past the block end.
    assert first_start + first_len > BLOCK
    resumed_block = manager.get_blocks("resume").blocks[MAMBA_GROUP][0]
    offloads = drain_boundary_state_offloads(manager).get("resume", [])
    manager.free(resume)
    manager.new_step_starts()

    assert resumed_block.block_hash_num_tokens != BLOCK
    assert (MAMBA_GROUP, resumed_block.block_id, BLOCK) not in offloads
    # A probe sharing the first full block must not get a 4736-token hit
    # backed by block 0's resume-point state.
    probe = make_request("probe", tokens[:BLOCK] + [9] * 200, UNIT, sha256)
    _, probe_hit, _ = manager.get_computed_blocks(probe)
    assert probe_hit == num_hit


@pytest.mark.parametrize("num_prefill_checkpoint_blocks", [0, 1])
def test_block4736_unit64_partial_hit_cow_source_keyed_this_step_is_deferred(
    num_prefill_checkpoint_blocks,
):
    """Patch 27 (#67451): a partial hit whose attention CoW source block was
    keyed by another request in the same step is deferred one step, then
    scheduled with the copy from that (now written) block."""
    manager = _manager(num_prefill_checkpoint_blocks)
    prompt = list(range(100, 100 + 4 * BLOCK))
    hit = BLOCK + 640  # 5376, a multiple of the 64-token unit, mid-block 1

    # Earlier steps: a (hit + 1)-token prompt, split as the scheduler does,
    # leaves a Mamba state key at `hit` tokens.
    scheduler = _scheduler(bool(num_prefill_checkpoint_blocks))
    first, first_hit, _ = _prefill(manager, scheduler, "first", prompt[: hit + 1])
    assert first_hit == 0
    manager.free(first)
    manager.new_step_starts()

    # This step: a writer computes [0, 3 * BLOCK) without a lookup and keys
    # its full attention blocks; its only Mamba state is at 3 * BLOCK, past
    # the reader (as upstream's writer state at 12 is past its 9-token reader).
    writer = make_request("writer", prompt[: 3 * BLOCK + 1], UNIT, sha256)
    assert manager.allocate_slots(writer, 3 * BLOCK, 0, manager.empty_kv_cache_blocks)
    source_block_id = manager.get_blocks("writer").get_block_ids()[0][1]

    # The reader covers block 1 fully, so its attention lookup matches the
    # writer's full block 1 (as upstream's reader matches the writer's block);
    # the Mamba state caps the hit at `hit`, inside that block.
    reader_len = 2 * BLOCK + 100
    reader = make_request("reader", prompt[:reader_len], UNIT, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(reader)
    assert num_computed == hit
    assert computed.blocks[0][-1].block_id == source_block_id
    num_new = reader_len - hit
    assert manager.allocate_slots(reader, num_new, num_computed, computed) is None

    manager.new_step_starts()
    computed, num_computed, _ = manager.get_computed_blocks(reader)
    assert num_computed == hit
    assert manager.allocate_slots(reader, num_new, num_computed, computed)
    copies, _ = manager.take_kv_cache_block_copies()
    assert any(copy.src_block_id == source_block_id for copy in copies)
