# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch reference for the one-pass checkpoint (no vLLM import).

KDA recurrence per token t (state S is K x V, decay g_t <= 0 per key dim):
    S_t = (I - beta_t k_t k_t^T) diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t
The chunked form here computes each 64-token chunk's affine map
S -> M S + N by a tree reduction (a different evaluation order from the
recurrence, as a chunked kernel is), and carries the state only across
chunk boundaries. The one-pass checkpoint is the boundary state at offset k
(k a multiple of 64 from the row start). Checked: it equals the recurrent
state after k tokens, and resuming from it reproduces the single pass
(outputs and final state), which is what the two-pass split computed.
The conv part checks the exporter's window math (store_cache_checkpoints_kernel:
token = cu_start + offset - STATE_LEN + h) against a two-pass causal conv.
"""

import pytest
import torch

CHUNK = 64
K = 16
V = 16
CONV_W = 4  # GLM-5.3 linear_conv_kernel_dim; state_len = 3


def _inputs(T, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, K, generator=g, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(T, K, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(T, V, generator=g, dtype=dtype)
    beta = torch.sigmoid(torch.randn(T, generator=g, dtype=dtype))
    # bounded safe gate, like lower_bound * sigmoid(.)
    gate = -5.0 * torch.sigmoid(torch.randn(T, K, generator=g, dtype=dtype))
    s0 = torch.randn(K, V, generator=g, dtype=dtype) * 0.1
    return q, k, v, beta, gate, s0


def _step_map(k, beta, gate, v):
    A = (torch.eye(K, dtype=k.dtype) - beta * torch.outer(k, k)) @ torch.diag(gate.exp())
    return A, beta * torch.outer(k, v)


def recurrent(q, k, v, beta, gate, s0, snapshot_at=()):
    S = s0.clone()
    outs, snaps = [], {}
    for t in range(q.shape[0]):
        if t in snapshot_at:
            snaps[t] = S.clone()
        A, B = _step_map(k[t], beta[t], gate[t], v[t])
        S = A @ S + B
        outs.append(S.T @ q[t])
    if q.shape[0] in snapshot_at:
        snaps[q.shape[0]] = S.clone()
    return torch.stack(outs), S, snaps


def _chunk_map(k, beta, gate, v):
    maps = [_step_map(k[t], beta[t], gate[t], v[t]) for t in range(k.shape[0])]
    while len(maps) > 1:  # tree reduction: (A2,B2) o (A1,B1) = (A2 A1, A2 B1 + B2)
        nxt = []
        for i in range(0, len(maps) - 1, 2):
            (A1, B1), (A2, B2) = maps[i], maps[i + 1]
            nxt.append((A2 @ A1, A2 @ B1 + B2))
        if len(maps) % 2:
            nxt.append(maps[-1])
        maps = nxt
    return maps[0]


def chunked(q, k, v, beta, gate, s0, checkpoint_offset=0):
    """One pass over a row, chunks aligned to the row start; returns outputs,
    final state and the state at `checkpoint_offset` (None when 0)."""
    T = q.shape[0]
    S = s0.clone()
    outs = []
    ckpt = None
    for c0 in range(0, T, CHUNK):
        if checkpoint_offset and c0 == checkpoint_offset:
            ckpt = S.clone()
        c1 = min(c0 + CHUNK, T)
        # outputs inside the chunk from the chunk-start state
        o, _, _ = recurrent(q[c0:c1], k[c0:c1], v[c0:c1], beta[c0:c1], gate[c0:c1], S)
        outs.append(o)
        M, N = _chunk_map(k[c0:c1], beta[c0:c1], gate[c0:c1], v[c0:c1])
        S = M @ S + N
    if checkpoint_offset == T:
        ckpt = S.clone()
    assert not checkpoint_offset or ckpt is not None, "offset not on a chunk boundary"
    return torch.cat(outs), S, ckpt


@pytest.mark.parametrize("T,offset", [(168, 128), (457, 448), (513, 512), (200, 64)])
def test_checkpoint_equals_recurrent_state_at_offset(T, offset):
    q, k, v, beta, gate, s0 = _inputs(T, seed=T)
    _, _, snaps = recurrent(q, k, v, beta, gate, s0, snapshot_at={offset})
    _, _, ckpt = chunked(q, k, v, beta, gate, s0, checkpoint_offset=offset)
    torch.testing.assert_close(ckpt, snaps[offset], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("T,offset", [(168, 128), (457, 448), (513, 512), (300, 192)])
def test_one_pass_equals_two_pass(T, offset):
    q, k, v, beta, gate, s0 = _inputs(T, seed=7 * T)
    o1, s1, ckpt = chunked(q, k, v, beta, gate, s0, checkpoint_offset=offset)
    # two-pass: [0, offset) then [offset, T) from the state pass 1 left
    oa, sa, _ = chunked(q[:offset], k[:offset], v[:offset], beta[:offset], gate[:offset], s0)
    ob, sb, _ = chunked(q[offset:], k[offset:], v[offset:], beta[offset:], gate[offset:], sa)
    torch.testing.assert_close(ckpt, sa, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(o1, torch.cat([oa, ob]), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(s1, sb, rtol=1e-10, atol=1e-12)
    # A later request resuming from the checkpoint matches the single pass.
    oc, sc, _ = chunked(q[offset:], k[offset:], v[offset:], beta[offset:], gate[offset:], ckpt)
    torch.testing.assert_close(oc, o1[offset:], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(sc, s1, rtol=1e-10, atol=1e-12)


def test_fp32_one_pass_close_to_two_pass():
    """fp32: same math, different rounding order; differences stay at fp32
    noise. (GPU check settles the real kernel's tolerance.)"""
    q, k, v, beta, gate, s0 = _inputs(457, seed=3, dtype=torch.float32)
    o1, s1, ckpt = chunked(q, k, v, beta, gate, s0, checkpoint_offset=448)
    _, sa, _ = chunked(q[:448], k[:448], v[:448], beta[:448], gate[:448], s0)
    ob, sb, _ = chunked(q[448:], k[448:], v[448:], beta[448:], gate[448:], sa)
    torch.testing.assert_close(ckpt, sa, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(o1[448:], ob, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s1, sb, rtol=1e-4, atol=1e-5)


def test_varlen_rows_checkpoint_by_row_offset():
    """Packed rows (cu_seqlens): a warm row resumes at 2944 with its own
    initial state; its checkpoint offset counts from its own row start."""
    rows = [(0, 168, 128), (2944, 3457, 3456 - 2944)]  # (start, end, offset)
    for i, (start, end, offset) in enumerate(rows):
        n = end - start
        q, k, v, beta, gate, s0 = _inputs(n, seed=100 + i)
        _, _, snaps = recurrent(q, k, v, beta, gate, s0, snapshot_at={offset})
        _, _, ckpt = chunked(q, k, v, beta, gate, s0, checkpoint_offset=offset)
        torch.testing.assert_close(ckpt, snaps[offset], rtol=1e-10, atol=1e-12)


# --- conv window ------------------------------------------------------------


def causal_conv(x, w, window):
    """x: [T, C]; w: [C, W]; window: [C, W-1] (oldest first). Returns y, new window."""
    full = torch.cat([window.T, x], dim=0)  # [W-1+T, C]
    W = w.shape[1]
    y = torch.stack([(full[t : t + W] * w.T).sum(0) for t in range(x.shape[0])])
    return y, full[-(W - 1) :].T.clone()


def export_conv_window(raw, cu_seqlens, offsets, state_len):
    """Mirror of store_cache_checkpoints_kernel's conv part: per row with a
    checkpoint, window[h] = raw[cu_start + offset - state_len + h]."""
    out = []
    for row, off in enumerate(offsets):
        if off == 0 or off < state_len:
            out.append(None)
            continue
        end = int(cu_seqlens[row]) + off
        out.append(raw[end - state_len : end].T.clone())
    return out


def test_conv_window_export_matches_two_pass():
    g = torch.Generator().manual_seed(1)
    C = 8
    w = torch.randn(C, CONV_W, generator=g, dtype=torch.float64)
    lens = [168, 513]
    offsets = [128, 512]
    raws = [torch.randn(n, C, generator=g, dtype=torch.float64) for n in lens]
    windows0 = [torch.randn(C, CONV_W - 1, generator=g, dtype=torch.float64) for _ in lens]
    packed = torch.cat(raws)
    cu = torch.tensor([0, lens[0], lens[0] + lens[1]])
    exported = export_conv_window(packed, cu, offsets, CONV_W - 1)
    for raw, w0, off, win in zip(raws, windows0, offsets, exported):
        y1, _ = causal_conv(raw, w, w0)
        _, win_two_pass = causal_conv(raw[:off], w, w0)
        torch.testing.assert_close(win, win_two_pass, rtol=0, atol=0)
        y2, _ = causal_conv(raw[off:], w, win)
        torch.testing.assert_close(y2, y1[off:], rtol=0, atol=0)
