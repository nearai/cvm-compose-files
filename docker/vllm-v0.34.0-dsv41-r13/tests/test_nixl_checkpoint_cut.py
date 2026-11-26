"""CPU tests for patches 28 and 29 (NIXL P/D transfers stop at the Mamba checkpoint).

Run in the built image (no GPU): python3 -m pytest -q -p no:cacheprovider tests/
Cases from upstream vllm#67064 (test_nixl_connector_hma.py), with a local
scheduler builder, plus block-size-2432 cases and the patch 29 re-query.
"""
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import metadata as nixl_md
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position
from vllm.v1.request import Request


def make_sched(push=False, has_mamba=True, lookahead=0, hash_lookahead=0, bs=16):
    cls = NixlPushConnectorScheduler if push else NixlPullConnectorScheduler
    s = object.__new__(cls)
    s._has_mamba = has_mamba
    s._bounded_replay = False
    s._kv_cache_manager = None
    s.kv_recompute_threshold = 0
    s.vllm_config = SimpleNamespace(
        num_prefill_lookahead_tokens=lookahead,
        num_hash_lookahead_tokens=hash_lookahead,
    )
    s.kv_cache_config = SimpleNamespace(hash_block_size=bs)
    return s


def bind(s, bs=16, drop_eagle_block=False, hash_lookahead=0):
    """Stand-in for a bound KVCacheManager with one Mamba align group."""
    s._kv_cache_manager = SimpleNamespace(
        get_state_checkpoint_tokens=lambda n: get_mamba_prefill_checkpoint_position(
            n, bs, drop_eagle_block, num_hash_lookahead_tokens=hash_lookahead
        )
    )


def make_req(n, **params):
    r = Request("r", list(range(n)), SamplingParams(max_tokens=16), None)
    r.kv_transfer_params = dict(params)
    return r


def p_cut(s, n):
    r = make_req(n, do_remote_decode=True)
    s._truncate_request_for_prefill(r)
    return r


# Upstream vllm#67064 cases (block 16, no MTP: backoff 1).
@pytest.mark.parametrize("push", [False, True], ids=["pull", "push"])
@pytest.mark.parametrize(
    "n,drop,expected",
    [(50, False, 48), (49, False, 48), (10, False, 9), (50, True, 32)],
    ids=["cut_to_checkpoint", "n1_is_checkpoint", "no_checkpoint", "eagle_drop"],
)
def test_mamba_checkpoint_cut(n, drop, expected, push):
    p, d = make_sched(push), make_sched(push)
    bind(p, drop_eagle_block=drop)
    bind(d, drop_eagle_block=drop)
    r = p_cut(p, n)
    assert r.num_prompt_tokens == expected
    assert len(r.prompt_token_ids) == expected
    assert r.max_tokens == 1
    assert r.kv_transfer_params["_p_side_truncated"]
    # Preemption/reschedule must not cut again.
    p._truncate_request_for_prefill(r)
    assert r.num_prompt_tokens == expected
    assert d.get_num_new_matched_tokens(make_req(n, do_remote_prefill=True), 0) == (
        expected,
        True,
    )


@pytest.mark.parametrize(
    "remote,expected",
    [(48, 48), (32, 32), (49, 49), (50, 0)],
    ids=["agree", "p_cut_lower", "p_without_cut", "p_past_backoff"],
)
def test_mamba_checkpoint_cut_d_follows_prefiller(remote, expected):
    s = make_sched()
    bind(s)
    r = make_req(50, do_remote_prefill=True, remote_num_tokens=remote)
    count, _ = s.get_num_new_matched_tokens(r, num_computed_tokens=0)
    assert count == expected


def test_unbound_keeps_backoff_cut():
    """No align group bound (e.g. prefix caching off): the N-backoff cut."""
    s = make_sched()
    assert p_cut(s, 50).num_prompt_tokens == 49
    assert s.get_num_new_matched_tokens(make_req(50, do_remote_prefill=True), 0) == (
        49,
        True,
    )


def _sched_stub(connector, **kw):
    """Stub `self` for the real Scheduler._requery_connector_after_reconcile."""
    attrs = dict(
        connector=connector, prefix_replay_tokens=0, aux_output_connector=None,
        block_size=16,
    )
    attrs.update(kw)
    return SimpleNamespace(**attrs)


def _requery(stub, req, num_local):
    from vllm.v1.core.sched.scheduler import Scheduler

    return Scheduler._requery_connector_after_reconcile(stub, req, num_local)


@pytest.mark.parametrize("push", [False, True], ids=["pull", "push"])
def test_diverged_hit_at_cut_requeries_from_reconciled_hit(push):
    """Patch 29 (block 16). D's full-attention hit already reaches the
    cut (48) while its Mamba state lags (32): the first query finds nothing past
    the attention hit, so the scheduler reconciles to 32 and asks again; D must
    then load [32, 48) incl. the state at 48 instead of recomputing from 32."""
    s = make_sched(push)
    bind(s)
    r = make_req(50, do_remote_prefill=True)
    assert s.get_num_new_matched_tokens(r, 48) == (0, False)
    assert _requery(_sched_stub(s), r, 32) == (16, True)
    # Mamba state evicted while attention survived: load the whole cut.
    assert _requery(_sched_stub(s), r, 0) == (48, True)


# Block 2432: turn 1 is 9700 + 300 reply tokens, turn 2 adds 100.
def test_diverged_hit_requery_block2432():
    B_ = 2432
    n2 = 9700 + 300 + 100
    s = make_sched(lookahead=1, hash_lookahead=1, bs=B_)
    bind(s, bs=B_, hash_lookahead=1)
    c2 = (n2 - 1) // B_ * B_
    assert c2 == 9728
    # P's cut on turn 2 (what P reports in remote_num_tokens).
    p = make_sched(lookahead=1, hash_lookahead=1, bs=B_)
    bind(p, bs=B_, hash_lookahead=1)
    assert p_cut(p, n2).num_prompt_tokens == c2
    r = make_req(n2, do_remote_prefill=True, remote_num_tokens=c2)
    # D's attention hit (reply blocks hashed through 9728) == the cut: no pull.
    assert s.get_num_new_matched_tokens(r, c2) == (0, False)
    # Reconciled to the turn-1 checkpoint C1 = 7296: pull one block + h(9728).
    c1 = (9700 - 1) // B_ * B_
    assert c1 == 7296
    stub = _sched_stub(s, block_size=B_)
    assert _requery(stub, r, c1) == (c2 - c1, True)
    # D then recomputes only the 372-token tail, not [7296, 10100).
    assert n2 - c2 == 372


def test_real_get_computed_blocks_for_connector_reports_divergence():
    """The real lookup reports the attention-only hit and hit_diverged for
    a diverged-hit shape, which is what sends the scheduler into the reconcile path."""
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    coord = object.__new__(HybridKVCacheCoordinator)
    coord.full_attention_group_id = 0
    coord.find_longest_cache_hit_per_group = lambda hashes, n: (
        (["fa"] * 4, ["m"] * 3), (9728, 7296)
    )
    fake = SimpleNamespace(
        kv_cache_config=SimpleNamespace(has_mamba_layers=True),
        coordinator=coord,
        prefix_cache_lookup_enabled=lambda req: True,
        create_kv_cache_blocks=lambda blocks: blocks,
        empty_kv_cache_blocks=None,
    )
    r = make_req(10100, do_remote_prefill=True)
    _, num_local, _, diverged = KVCacheManager.get_computed_blocks_for_connector(
        fake, r
    )
    assert (num_local, diverged) == (9728, True)
    # No Mamba layers: plain lookup, never diverged (the re-query never runs).
    fake.kv_cache_config.has_mamba_layers = False
    fake.get_computed_blocks = lambda req, m=None: ("b", 9728, 0)
    assert KVCacheManager.get_computed_blocks_for_connector(fake, r)[3] is False


def test_requery_no_change_cases():
    class Boom:
        def get_num_new_matched_tokens(self, *a):
            raise AssertionError("must not query")

    req = make_req(50, do_remote_prefill=True)
    # Paths that post-process the first query keep reconcile-only behaviour.
    assert _requery(_sched_stub(Boom(), prefix_replay_tokens=128), req, 32) == (
        0, False)
    assert _requery(_sched_stub(Boom(), aux_output_connector=object()), req, 32) == (
        0, False)
    # Sub-block reconciled hit: the connector is only ever shown aligned hits.
    assert _requery(_sched_stub(Boom()), req, 33) == (0, False)
    skip = SimpleNamespace(skip_reading_external_kv=True)
    assert _requery(_sched_stub(Boom()), skip, 32) == (0, False)
    # Local-only request (no kv_transfer_params): nothing to load, as before.
    s = make_sched()
    bind(s)
    assert _requery(_sched_stub(s), make_req(50), 32) == (0, False)
    # A connector lookup still pending (None) falls back to recompute.
    pending = SimpleNamespace(get_num_new_matched_tokens=lambda *a: (None, False))
    assert _requery(_sched_stub(pending), req, 32) == (0, False)


# P's next-turn prefix hit after a cut.
@pytest.mark.parametrize("n", [9700, 10100, 50000])
def test_prefill_cache_unit_loss_after_cut(n):
    """P truncates to C and keeps (C - L)//B hashes, so its next-turn prefix hit
    tops out at C - B; a cut at N-1 keeps floor((N-1-L)/B) blocks >= C/B.
    P keeps one block of prefix cache less per P/D turn."""
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    B_, L = 2432, 1
    init_none_hash(sha256)

    def req():
        r = Request(
            "p", list(range(n)), SamplingParams(max_tokens=16), None,
            block_hasher=get_request_block_hasher(B_, sha256, L),
        )
        r.kv_transfer_params = {"do_remote_decode": True}
        return r

    cut = make_sched(lookahead=L, hash_lookahead=L, bs=B_)
    bind(cut, bs=B_, hash_lookahead=L)
    r = req()
    cut._truncate_request_for_prefill(r)
    c = r.num_prompt_tokens
    assert c == (n - 1) // B_ * B_
    assert len(r.block_hashes) * B_ == c - B_

    old = req()
    make_sched(lookahead=L, hash_lookahead=L, bs=B_)._truncate_request_for_prefill(old)
    assert old.num_prompt_tokens == n - 1
    # For these n ((n-1) % B != 0) a cut at N-1 keeps the block ending at C.
    assert len(old.block_hashes) * B_ == (n - 1 - L) // B_ * B_ == c


# Block 2432, single-module MTP (L=1, backoff 1) or L=3.
B = 2432


@pytest.mark.parametrize("L", [1, 3], ids=["mtp1", "mtp3"])
@pytest.mark.parametrize("n", [100, 2432, 2433, 2434, 2435, 2436, 4866, 4867, 9600, 50000])
def test_block2432_cut_and_tail(n, L):
    backoff = max(1, L - 1)
    p = make_sched(lookahead=L, hash_lookahead=L, bs=B)
    d = make_sched(lookahead=L, hash_lookahead=L, bs=B)
    bind(p, bs=B, hash_lookahead=L)
    bind(d, bs=B, hash_lookahead=L)
    c = (n - backoff + 1 - L) // B * B
    expected = c if c > 0 else n - backoff
    assert p_cut(p, n).num_prompt_tokens == expected
    assert d.get_num_new_matched_tokens(make_req(n, do_remote_prefill=True), 0) == (
        expected,
        True,
    )
    tail = n - expected
    # D recomputes ((n - backoff + 1 - L) mod B) + backoff - 1 + L tokens:
    # [1, B] for L=1, [4, B+3] for L=3; just the backoff below one block.
    assert backoff <= tail <= B + backoff + L - 2 if c > 0 else tail == backoff
    if c > 0:
        assert expected % B == 0
        # The cut is a boundary a later local prefill of >= n tokens can hit:
        # the block hash for the block ending at c needs tokens [c, c+L) to exist.
        assert expected + L <= n


# The real KVCacheManager.get_state_checkpoint_tokens.
def _mgr(cls, **attrs):
    m = object.__new__(cls)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("L", [0, 1, 3])
def test_kv_cache_manager_checkpoint_tokens(drop, L):
    fa = _mgr(FullAttentionManager)
    mamba = _mgr(
        MambaManager,
        mamba_cache_mode="align",
        cache_hit_alignment_tokens=B,
        drop_eagle_checkpoint_block=drop,
    )
    fake = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=(fa, mamba)),
        block_pool=SimpleNamespace(num_hash_lookahead_tokens=L),
    )
    for n in (1, B, B + 1, 3 * B + 7):
        assert KVCacheManager.get_state_checkpoint_tokens(
            fake, n
        ) == get_mamba_prefill_checkpoint_position(n, B, drop, L)
    mamba.mamba_cache_mode = "all"
    assert KVCacheManager.get_state_checkpoint_tokens(fake, 3 * B + 7) == 0


# Connector version: a peer without patch 28 reports version 16.
def _hash_at_version(cfg, version):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nixl_md, "NIXL_CONNECTOR_VERSION", version)
        return nixl_md.compute_nixl_compatibility_hash(cfg, "FLASHMLA_SPARSE")


def _fake_vllm_config():
    mc = SimpleNamespace(
        model="glm",
        dtype="bfloat16",
        get_total_num_kv_heads=lambda: 1,
        get_head_size=lambda: 576,
        get_total_num_hidden_layers=lambda: 45,
    )
    return SimpleNamespace(
        model_config=mc,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )


def test_version_bumped_and_mixed_pair_hash_differs():
    assert nixl_md.NIXL_CONNECTOR_VERSION == 1017
    cfg = _fake_vllm_config()
    new_hash = nixl_md.compute_nixl_compatibility_hash(cfg, "FLASHMLA_SPARSE")
    old_hash = _hash_at_version(cfg, 16)
    # Only the connector version differs, so a mixed P/D pair fails the
    # handshake compatibility check (enforce_handshake_compat, default True).
    assert new_hash != old_hash
    assert new_hash == nixl_md.compute_nixl_compatibility_hash(cfg, "FLASHMLA_SPARSE")


def test_bind_only_with_align_prefix_caching():
    from vllm.v1.kv_cache_interface import MambaSpec

    spec = object.__new__(MambaSpec)
    object.__setattr__(spec, "mamba_cache_mode", "align")
    for enable, mode, bound in [(True, "align", True), (False, "align", False),
                                (True, "all", False)]:
        object.__setattr__(spec, "mamba_cache_mode", mode)
        s = make_sched()
        s.vllm_config.cache_config = SimpleNamespace(enable_prefix_caching=enable)
        s.kv_cache_config.kv_cache_groups = [SimpleNamespace(kv_cache_spec=spec)]
        mgr = object()
        s.bind_kv_cache_manager(mgr)
        assert (s._kv_cache_manager is mgr) is bound


# Mixed-pair refusal through the real handshake path.
def _free_port():
    import socket

    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def _handshake_against(remote_hash):
    """Real P-side listener serving `remote_hash`; real D-side
    `_nixl_handshake` (stub self) holding this build's hash."""
    import threading

    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
        NixlBaseConnectorScheduler,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )

    payload = msgspec.msgpack.Encoder().encode(
        nixl_md.NixlHandshakePayload(
            compatibility_hash=remote_hash, agent_metadata_bytes=b"\xc1"
        )
    )
    port, ready, stop = _free_port(), threading.Event(), threading.Event()
    t = threading.Thread(
        target=NixlBaseConnectorScheduler._nixl_handshake_listener,
        args=({(0, 0): payload}, ready, stop, "127.0.0.1", port),
        daemon=True,
    )
    t.start()
    ready.wait(5)
    worker = SimpleNamespace(
        _is_csa_linear=False,
        use_host_buffer=True,
        transfer_topo=SimpleNamespace(handshake_target_ranks=lambda *a: [0]),
        compat_hash=nixl_md.compute_nixl_compatibility_hash(
            _fake_vllm_config(), "FLASHMLA_SPARSE"
        ),
        enforce_compat_hash=True,
    )
    try:
        with pytest.raises(RuntimeError) as e:
            NixlBaseConnectorWorker._nixl_handshake(
                worker, "127.0.0.1", port, 1, "p-engine"
            )
        return str(e.value)
    finally:
        stop.set()
        t.join(5)


def test_handshake_refuses_version_16_peer():
    cfg = _fake_vllm_config()
    assert "compatibility hash mismatch" in _handshake_against(_hash_at_version(cfg, 16))
    # Control: a matching peer passes the hash check and only then fails on
    # the (deliberately invalid) agent metadata.
    same = nixl_md.compute_nixl_compatibility_hash(cfg, "FLASHMLA_SPARSE")
    msg = _handshake_against(same)
    assert "compatibility hash mismatch" not in msg
    assert "Failed to decode NixlAgentMetadata" in msg
