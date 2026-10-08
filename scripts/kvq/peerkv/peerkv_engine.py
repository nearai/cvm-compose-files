"""In-host GPU-to-GPU KV prefix sharing between two GLM-5.3 TP2 replicas (HiCache off).

Replica B misses a prefix that its sibling A holds in VRAM: B asks A to lend it, A pins the
matching tree node and returns its KV token indices and KDA (mamba) state slot, B copies those
rows/pages/slot straight out of A's GPU memory (CUDA IPC, NVLink/PPCIe peer copies), inserts
them into its own radix tree, and admits the request as a normal prefix hit. Any failure falls
back to recompute; a fetch never aborts a request.

Invariants:
- TP ranks keep identical radix trees and allocators. Every tree/allocator mutation here runs on
  every rank at the same scheduler iteration:
  * lend/release/expire messages are injected on rank 0 into the received-request list and
    broadcast in-band with ordinary requests (`inject` -> `consume`);
  * the receiver path runs inside `_add_request_to_queue`, which every rank executes in the same
    order; rank 0 does the RPC and broadcasts its result; a MIN all-reduce decides hit/miss.
- Rank i reads only peer rank i (KDA state is head-sharded across TP ranks).
- Both containers see the same GPUs in the same order (device_ids = the pair's GPUs), so a device
  index means the same physical GPU in both processes.

Env: SGLANG_PEERKV=1, SGLANG_PEERKV_SELF=<id>, SGLANG_PEERKV_PEER=<id>, SGLANG_PEERKV_DIR=/peerkv,
SGLANG_PEERKV_MIN_TOKENS (4096), SGLANG_PEERKV_RPC_TIMEOUT_S (2), SGLANG_PEERKV_LEASE_TTL_S (30).
"""
from __future__ import annotations

import ctypes
import logging
import os
import pickle
import queue
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("peerkv")
TAG = "[peerkv]"

ENABLED = os.environ.get("SGLANG_PEERKV", "0") == "1"
SELF = os.environ.get("SGLANG_PEERKV_SELF", "")
PEER = os.environ.get("SGLANG_PEERKV_PEER", "")
DIR = os.environ.get("SGLANG_PEERKV_DIR", "/peerkv")
# Each lend RPC stalls this scheduler for about one holder iteration (~60 ms on gpu04), hit or
# not, so only ask when the missing span is long enough that a hit clearly beats prefilling it.
MIN_TOKENS = int(os.environ.get("SGLANG_PEERKV_MIN_TOKENS", "16384"))
RPC_TIMEOUT_S = float(os.environ.get("SGLANG_PEERKV_RPC_TIMEOUT_S", "2"))
LEASE_TTL_S = float(os.environ.get("SGLANG_PEERKV_LEASE_TTL_S", "30"))
GATHER_CHUNK = int(os.environ.get("SGLANG_PEERKV_GATHER_CHUNK_ROWS", "32768"))
LOG_FETCHES = os.environ.get("SGLANG_PEERKV_LOG_FETCHES", "1") == "1"
# Lab only: re-exec this file when its mtime changes (bind-mounted dir), keeping _S/_M, so the
# fetch/lend logic can be iterated without restarting the engines. Never set in prod.
HOTRELOAD = os.environ.get("SGLANG_PEERKV_HOTRELOAD", "0") == "1"
_HOT: Dict[str, Any] = {"mtime": None, "mod": None}
_HOT_CHILD = False


# --------------------------------------------------------------------------------------------
# In-band messages (picklable; carried by the scheduler's own request broadcast)
# --------------------------------------------------------------------------------------------
@dataclass
class PeerKVMsg:
    op: str  # "lend" | "release" | "expire"
    lease: str
    tokens: Any = None
    extra_key: Any = None
    limit: Optional[int] = None
    p_b: int = 0


# --------------------------------------------------------------------------------------------
# Framed pickle over a unix stream socket
# --------------------------------------------------------------------------------------------
def _send(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def _recv(sock: socket.socket):
    (n,) = struct.unpack("!Q", _recv_exact(sock, 8))
    return pickle.loads(_recv_exact(sock, n))


def sock_path(replica: str) -> str:
    return os.path.join(DIR, f"{replica}.sock")


def handles_path(replica: str, rank: int) -> str:
    return os.path.join(DIR, f"{replica}.tp{rank}.handles")


# --------------------------------------------------------------------------------------------
# CUDA IPC (driver API via ctypes; no torch shared-file ref counting across containers)
# --------------------------------------------------------------------------------------------
class _IpcHandle(ctypes.Structure):
    """cudaIpcMemHandle_t: a 64-byte struct passed BY VALUE to cudaIpcOpenMemHandle."""
    _fields_ = [("reserved", ctypes.c_char * 64)]


class _Cuda:
    def __init__(self):
        self.rt = ctypes.CDLL("libcudart.so") if _have("libcudart.so") else _find_cudart()
        self.drv = ctypes.CDLL("libcuda.so.1")
        self.rt.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
        self.rt.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
        self.opened: Dict[bytes, int] = {}  # one mapping per peer allocation (many tensors share one)

    def mem_base(self, ptr: int):
        base, size = ctypes.c_uint64(), ctypes.c_size_t()
        r = self.drv.cuMemGetAddressRange_v2(ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(ptr))
        if r != 0:
            raise RuntimeError(f"cuMemGetAddressRange failed: {r}")
        return base.value, size.value

    def ipc_handle(self, base: int) -> bytes:
        h = _IpcHandle()
        r = self.rt.cudaIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(base))
        if r != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed: {r} (expandable_segments must be False)")
        return ctypes.string_at(ctypes.addressof(h), 64)

    def ipc_open(self, handle: bytes) -> int:
        if handle in self.opened:
            return self.opened[handle]
        h = _IpcHandle.from_buffer_copy(handle)
        ptr = ctypes.c_void_p()
        # cudaIpcMemLazyEnablePeerAccess = 1
        r = self.rt.cudaIpcOpenMemHandle(ctypes.byref(ptr), h, ctypes.c_uint(1))
        if r != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed: {r}")
        self.opened[handle] = ptr.value
        return ptr.value

    def close_all(self) -> int:
        """Unmap every peer allocation. Required when the peer dies: while a mapping is open the
        driver keeps the dead peer's memory allocated, and its restart then finds no free VRAM."""
        n = 0
        for _h, ptr in list(self.opened.items()):
            if self.rt.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr)) == 0:
                n += 1
        self.opened.clear()
        return n


def _have(name: str) -> bool:
    try:
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


def _find_cudart():
    import glob
    for pat in ("/usr/local/cuda*/lib64/libcudart.so*", "/usr/local/lib/python3*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*",
                "/usr/lib/python3*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"):
        for p in sorted(glob.glob(pat)):
            try:
                return ctypes.CDLL(p)
            except OSError:
                pass
    raise OSError("libcudart not found")


class _CAI:
    """Minimal __cuda_array_interface__ holder to wrap a raw device pointer as a uint8 tensor."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3, "strides": None}


def _wrap(ptr: int, nbytes: int, dtype: torch.dtype, shape, device: int) -> torch.Tensor:
    with torch.cuda.device(device):
        raw = torch.as_tensor(_CAI(ptr, nbytes), device=f"cuda:{device}")
    return raw.view(dtype).view(shape)


# --------------------------------------------------------------------------------------------
# The tensors that make up a cached prefix, per TP rank
# --------------------------------------------------------------------------------------------
def collect_tensors(scheduler) -> List[Dict[str, Any]]:
    """[{name, kind: token|page|slot, tensor}] in a fixed order (identical across replicas built
    from the same argv and image)."""
    out = []
    kvp = scheduler.token_to_kv_pool_allocator.get_kvcache()
    full = getattr(kvp, "full_kv_pool", kvp)

    def add_dsa_pool(prefix, pool):
        for i, t in enumerate(pool.kv_buffer):
            out.append({"name": f"{prefix}.kv.{i}", "kind": "token", "tensor": t})
        ikc = getattr(pool, "index_key_cache", None)
        if ikc is not None:
            for i, t in enumerate(ikc.buffer):
                if t.numel() > 0:
                    out.append({"name": f"{prefix}.idx.{i}", "kind": "page", "tensor": t})

    add_dsa_pool("target", full)
    dw = getattr(scheduler, "draft_worker", None)
    dpool = getattr(dw, "primary_draft_kv_pool", None) if dw is not None else None
    if dpool is not None:
        add_dsa_pool("draft", dpool)
    mpool = scheduler.req_to_token_pool.mamba_pool
    for j, (fname, st, _axis) in enumerate(mpool._iter_transfer_state_tensors()):
        out.append({"name": f"mamba.{fname}.{j}", "kind": "slot", "tensor": st})
    return out


def describe(tensors) -> List[tuple]:
    return [(d["name"], d["kind"], tuple(d["tensor"].shape), str(d["tensor"].dtype)) for d in tensors]


# --------------------------------------------------------------------------------------------
# Per-process state
# --------------------------------------------------------------------------------------------
@dataclass
class _Lease:
    node: Any
    t0: float


@dataclass
class _State:
    scheduler: Any
    rank: int
    tp_size: int
    page_size: int
    tensors: List[Dict[str, Any]]
    epoch: str
    cuda: Any = None
    inbox: "queue.Queue" = field(default_factory=queue.Queue)
    replies: Dict[str, Any] = field(default_factory=dict)
    reply_cv: threading.Condition = field(default_factory=threading.Condition)
    leases: Dict[str, _Lease] = field(default_factory=dict)
    lease_t0: Dict[str, float] = field(default_factory=dict)  # rank-0 clock for TTL injection
    peer_epoch: Optional[str] = None
    peer_tensors: Optional[List[torch.Tensor]] = None
    seq: int = 0
    peer_gbps: float = 0.0
    map_lock: threading.Lock = field(default_factory=threading.Lock)
    stats: Dict[str, float] = field(default_factory=lambda: {
        "fetch_tries": 0, "fetch_hits": 0, "hit_tokens": 0, "fetch_s": 0.0, "fallback": 0,
        "lend_granted": 0, "lend_declined": 0, "lend_expired": 0})


_S: Optional[_State] = None


def _metrics_init():
    try:
        from prometheus_client import Counter, Histogram
        return {
            "req": Counter("sglang:peerkv_fetch_requests_total", "peer fetch attempts by outcome", ["outcome"]),
            "tok": Counter("sglang:peerkv_hit_tokens_total", "prompt tokens served from a sibling replica's VRAM"),
            "sec": Histogram("sglang:peerkv_fetch_seconds", "peer fetch time (lend RPC + copy + insert)",
                             buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)),
            "lend": Counter("sglang:peerkv_lend_total", "lend requests served by this replica", ["outcome"]),
        }
    except Exception:  # pragma: no cover
        return None


_M = None


def _impl():
    """The hot-reloaded module to delegate to, or None to run this one."""
    if not HOTRELOAD or _HOT_CHILD:
        return None
    try:
        mt = os.stat(__file__).st_mtime
    except OSError:
        return _HOT["mod"]
    if _HOT["mtime"] is None:
        _HOT["mtime"] = mt
    elif mt != _HOT["mtime"]:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("peerkv_engine_hot", __file__)
            mod = importlib.util.module_from_spec(spec)
            mod._HOT_CHILD = True
            sys.modules["peerkv_engine_hot"] = mod  # dataclasses resolve their module via sys.modules
            spec.loader.exec_module(mod)
            mod._HOT_CHILD = True
            # In-band messages are pickled across TP ranks, which may reload an iteration apart:
            # keep the original class so every rank can unpickle them.
            mod.PeerKVMsg = PeerKVMsg
            mod._S, mod._M = _S, _M
            if _S is not None:
                opened = getattr(_S.cuda, "opened", {})
                _S.cuda = mod._Cuda()
                _S.cuda.opened = opened
                _S.peer_epoch, _S.peer_tensors = None, None
            _HOT["mod"] = mod
            logger.info(f"{TAG} rank {getattr(_S, 'rank', '?')}: hot-reloaded {__file__}")
        except Exception as e:
            logger.warning(f"{TAG} hot reload failed, keeping previous code: {e!r}")
        _HOT["mtime"] = mt
    return _HOT["mod"]


def _m(name, *labels, inc=1.0, observe=None):
    if _M is None or _S is None or _S.rank != 0:
        return
    try:
        c = _M[name].labels(*labels) if labels else _M[name]
        c.observe(observe) if observe is not None else c.inc(inc)
    except Exception:
        pass


# --------------------------------------------------------------------------------------------
# Init (every rank), called from Scheduler.__init__ after the pools and tree exist
# --------------------------------------------------------------------------------------------
def peerkv_init(scheduler) -> None:
    global _S, _M
    if not ENABLED:
        return
    try:
        rank = scheduler.tp_group.rank_in_group if hasattr(scheduler.tp_group, "rank_in_group") else scheduler.tp_rank
    except Exception:
        rank = getattr(scheduler, "tp_rank", 0)
    tensors = collect_tensors(scheduler)
    page = int(getattr(scheduler.tree_cache, "page_size", 64))
    # One epoch per replica instance, identical on every TP rank (each rank is its own process, so
    # a per-process value would never match the holder rank-0 epoch carried in a grant).
    epoch = f"{SELF}-{os.getpid()}-{int(time.time())}"
    tp_size = scheduler.ps.tp_size
    if tp_size > 1:
        from sglang.srt.utils.common import broadcast_pyobj
        epoch = broadcast_pyobj([epoch], scheduler.tp_group.rank, scheduler.tp_cpu_group, src=scheduler.tp_group.ranks[0])[0]
    _S = _State(scheduler=scheduler, rank=rank, tp_size=tp_size, page_size=page, tensors=tensors, epoch=epoch)
    _S.cuda = _Cuda()
    os.makedirs(DIR, exist_ok=True)
    recs = []
    for d in tensors:
        t = d["tensor"]
        base, size = _S.cuda.mem_base(t.data_ptr())
        recs.append({"name": d["name"], "kind": d["kind"], "shape": tuple(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                     "nbytes": t.numel() * t.element_size(), "offset": t.data_ptr() - base,
                     "handle": _S.cuda.ipc_handle(base), "device": t.device.index})
    tmp = handles_path(SELF, rank) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"epoch": _S.epoch, "tensors": recs}, f)
    os.replace(tmp, handles_path(SELF, rank))
    if rank == 0:
        _M = _metrics_init()
        threading.Thread(target=_serve, daemon=True, name="peerkv-lend").start()
    dev = torch.cuda.current_device()
    p2p = {j: torch.cuda.can_device_access_peer(dev, j) for j in range(torch.cuda.device_count()) if j != dev}
    logger.info(f"{TAG} rank {rank}: {len(tensors)} tensors exported for {SELF} (peer {PEER}); page={page} dev={dev} p2p={p2p}")


# --------------------------------------------------------------------------------------------
# Holder: rank-0 lend server thread (never touches the tree; only queues and waits)
# --------------------------------------------------------------------------------------------
def _serve():
    path = sock_path(SELF)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(16)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=_handle_conn, args=(conn,), daemon=True).start()


def _handle_conn(conn: socket.socket):
    with conn:
        try:
            msg = _recv(conn)
            if msg.op == "release":
                _S.inbox.put(msg)
                _send(conn, {"ok": True})
                return
            _S.inbox.put(msg)
            deadline = time.monotonic() + RPC_TIMEOUT_S
            with _S.reply_cv:
                while msg.lease not in _S.replies:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    _S.reply_cv.wait(left)
                reply = _S.replies.pop(msg.lease, None)
            if reply is None:
                # Too late: make sure a grant produced later is released in-band.
                _S.inbox.put(PeerKVMsg(op="release", lease=msg.lease))
                reply = {"ok": False, "why": "holder_timeout"}
            _send(conn, reply)
        except Exception as e:  # pragma: no cover
            logger.warning(f"{TAG} lend conn error: {e!r}")


def inject(recv_reqs):
    """Rank 0 (list) only: append queued lend/release messages and TTL expiries."""
    if _S is None or recv_reqs is None or _S.rank != 0:
        return recv_reqs
    h = _impl()
    if h is not None:
        return h.inject(recv_reqs)
    now = time.monotonic()
    while True:
        try:
            m = _S.inbox.get_nowait()
        except queue.Empty:
            break
        if m.op == "lend":
            _S.lease_t0[m.lease] = now
        recv_reqs.append(m)
    for lease, t0 in list(_S.lease_t0.items()):
        if now - t0 > LEASE_TTL_S:
            recv_reqs.append(PeerKVMsg(op="expire", lease=lease))
            _S.lease_t0.pop(lease, None)
    return recv_reqs


def consume(scheduler, recv_reqs):
    """Every rank, same iteration: handle and drop PeerKVMsg items."""
    if _S is not None and not _HOT_CHILD:
        _ensure_prewarm()  # once per process; the hot-reload parent owns the thread
    if _S is None or not recv_reqs:
        return recv_reqs
    h = _impl()
    if h is not None:
        return h.consume(scheduler, recv_reqs)
    keep = []
    for r in recv_reqs:
        if isinstance(r, PeerKVMsg):
            try:
                _handle_msg(scheduler, r)
            except Exception as e:
                logger.warning(f"{TAG} msg {r.op} {r.lease} failed: {e!r}")
        else:
            keep.append(r)
    return keep


def _handle_msg(scheduler, m: PeerKVMsg):
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    tree = scheduler.tree_cache
    if m.op in ("release", "expire"):
        lease = _S.leases.pop(m.lease, None)
        _S.lease_t0.pop(m.lease, None)
        if lease is not None:
            tree.dec_lock_ref(lease.node)
            if m.op == "expire":
                _S.stats["lend_expired"] += 1
                _m("lend", "expired")
        if LOG_FETCHES and _S.rank == 0:
            logger.info(f"{TAG} {m.op} {m.lease} (held={lease is not None}); open leases={len(_S.leases)}")
        return
    # lend
    res = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=m.tokens, extra_key=m.extra_key, limit=m.limit)))
    L = len(res.device_indices)
    reply = {"ok": False, "why": "short", "L": L}
    node = res.last_device_node
    if L - m.p_b >= MIN_TOKENS and L % _S.page_size == 0 and m.p_b % _S.page_size == 0:
        n = tree.resolve_node_handle(node)
        cd = getattr(n, "component_data", None)
        mv = None
        if cd is not None:
            try:
                mv = cd[ComponentType.MAMBA].value
            except Exception:
                mv = None
        if mv is None:
            reply = {"ok": False, "why": "no_mamba_state", "L": L}
        else:
            tree.inc_lock_ref(node)
            _S.leases[m.lease] = _Lease(node=node, t0=time.monotonic())
            reply = {"ok": True, "L": L, "p_b": m.p_b, "epoch": _S.epoch,
                     "kv_idx": res.device_indices[m.p_b:L].to("cpu", dtype=torch.int64),
                     "mamba_slot": int(mv.view(-1)[0].item())}
    if not reply["ok"]:
        _S.lease_t0.pop(m.lease, None)  # nothing pinned, nothing to expire
    _S.stats["lend_granted" if reply["ok"] else "lend_declined"] += 1
    _m("lend", "granted" if reply["ok"] else reply["why"])
    if _S.rank == 0:
        with _S.reply_cv:
            _S.replies[m.lease] = reply
            _S.reply_cv.notify_all()


# --------------------------------------------------------------------------------------------
# Receiver: inside _add_request_to_queue (every rank, same order)
# --------------------------------------------------------------------------------------------
def _rpc(replica: str, msg, timeout: float):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path(replica))
        _send(s, msg)
        return _recv(s)
    finally:
        s.close()


def _layout_key(name, kind, shape, dtype):
    """Layout identity without the pool-length axis (rows for token/page tensors, slots for state)."""
    axis = 1 if kind == "slot" else 0
    return (name, kind, tuple(shape[:axis]) + tuple(shape[axis + 1:]), dtype)


PREWARM_POLL_S = float(os.environ.get("SGLANG_PEERKV_PREWARM_POLL_S", "5"))
# A fetch copies inside the scheduler loop, so it stalls this replica for its duration. Skip any
# fetch whose copy is estimated to take longer than this (estimate = bytes / measured GB/s).
MAX_COPY_S = float(os.environ.get("SGLANG_PEERKV_MAX_COPY_S", "0.5"))
REBENCH_S = float(os.environ.get("SGLANG_PEERKV_REBENCH_S", "300"))
_PREWARM = {"thread": None}


def _peer_file_epoch() -> Optional[str]:
    try:
        with open(handles_path(PEER, _S.rank), "rb") as f:
            return pickle.load(f)["epoch"]
    except Exception:
        return None


def _prewarm_loop():
    """Map the peer's pools (and run one tiny copy) off the request path; re-map on peer restart.
    A first mapping costs seconds (CUDA context on the peer GPU, lazy peer access); doing it in a
    fetch would stall this replica's scheduler."""
    torch.cuda.set_device(_S.tensors[0]["tensor"].device)  # threads start on cuda:0
    failed = None
    last_bench = time.monotonic()
    while True:
        try:
            ep = _peer_file_epoch()
            # Re-measure now and then: one sample taken under load can read far too low; keep the best.
            if _S.peer_tensors is not None and time.monotonic() - last_bench > REBENCH_S:
                last_bench = time.monotonic()
                prev = _S.peer_gbps
                with _S.map_lock:
                    if _S.peer_tensors is not None:
                        _bench_copy(_S.peer_tensors)
                _S.peer_gbps = max(prev, _S.peer_gbps)
            if _S.peer_epoch is not None and (ep != _S.peer_epoch or not _epoch_alive(_S.peer_epoch)):
                _unmap_peer(f"peer epoch {_S.peer_epoch} gone (file now {ep})")
            if ep is not None and ep != _S.peer_epoch and ep != failed and _epoch_alive(ep):
                t0 = time.monotonic()
                with _S.map_lock:
                    peer = _open_peer(ep)
                    for d, p in zip(_S.tensors, peer):  # touch every mapping once (lazy peer enable, kernels)
                        t = d["tensor"]
                        _ = (p[:, :1] if d["kind"] == "slot" else p[:1]).to(t.device)
                    torch.cuda.synchronize()
                    gbps = _bench_copy(peer)
                logger.info(f"{TAG} rank {_S.rank}: peer {PEER} epoch {ep} mapped in {time.monotonic() - t0:.2f}s; "
                            f"copy {gbps:.1f} GB/s, {_bytes_per_token() / 1e3:.1f} KB/token, "
                            f"max fetch {int(MAX_COPY_S * gbps * 1e9 / max(1, _bytes_per_token()))} tokens")
        except Exception as e:
            # Usually a stale handle file from a dead or still-starting peer; retry once its file
            # changes, or after 60 s, and log once per epoch.
            if failed != ep:
                logger.warning(f"{TAG} rank {_S.rank}: peer epoch {ep} not mappable yet: {e!r}")
            failed = ep
        if failed is not None and int(time.monotonic()) % 60 < PREWARM_POLL_S:
            failed = None
        time.sleep(PREWARM_POLL_S)


def _bytes_per_token() -> int:
    n = 0
    for d in _S.tensors:
        t = d["tensor"]
        if d["kind"] == "token":
            n += t[0].numel() * t.element_size()
        elif d["kind"] == "page":
            n += t[0].numel() * t.element_size() // 64
    return n


def _bench_copy(peer) -> float:
    """Measured peer->local GB/s with the real gather path (one token tensor, ~32K rows)."""
    i = next(k for k, d in enumerate(_S.tensors) if d["kind"] == "token")
    t, p = _S.tensors[i]["tensor"], peer[i]
    rows = min(32768, p.shape[0] - 64, t.shape[0] - 64)
    src = torch.arange(64, 64 + rows, dtype=torch.int64)
    scratch = torch.empty((rows,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
    best = 0.0
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.cuda.device(p.device):
            g = p.index_select(0, src.to(p.device))
        scratch.copy_(g.to(t.device))
        torch.cuda.synchronize()
        best = max(best, scratch.numel() * scratch.element_size() / 1e9 / max(time.monotonic() - t0, 1e-6))
    _S.peer_gbps = best
    return best


def _epoch_alive(epoch: str) -> bool:
    """Epoch = <replica>-<rank0 pid>-<time>; with pid: host the sibling's pids are visible."""
    try:
        pid = int(epoch.rsplit("-", 2)[1])
    except Exception:
        return True
    return os.path.exists(f"/proc/{pid}")


def _unmap_peer(why: str):
    with _S.map_lock:
        devs = {t.device for t in (_S.peer_tensors or [])}
        _S.peer_epoch, _S.peer_tensors = None, None
        torch.cuda.synchronize()
        n = _S.cuda.close_all()
        for d in devs:  # gather buffers this process cached on the peer's GPUs
            with torch.cuda.device(d):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    logger.info(f"{TAG} rank {_S.rank}: unmapped {n} peer allocations: {why}")


def _ensure_prewarm():
    if _PREWARM["thread"] is None and _S is not None and PEER:
        th = threading.Thread(target=_prewarm_loop, daemon=True, name="peerkv-prewarm")
        _PREWARM["thread"] = th
        th.start()


def _open_peer(epoch: str):
    """Map the peer rank's tensors (same rank index) once per peer epoch."""
    if _S.peer_epoch == epoch and _S.peer_tensors is not None:
        return _S.peer_tensors
    with open(handles_path(PEER, _S.rank), "rb") as f:
        meta = pickle.load(f)
    if meta["epoch"] != epoch:
        raise RuntimeError(f"peer handles epoch {meta['epoch']} != grant epoch {epoch}")
    # Pool lengths follow each replica's free memory, so they may differ; everything else must match.
    mine = [_layout_key(n, k, s, d) for n, k, s, d in describe(_S.tensors)]
    theirs = [_layout_key(r["name"], r["kind"], tuple(r["shape"]), "torch." + r["dtype"]) for r in meta["tensors"]]
    if mine != theirs:
        diff = next(((a, b) for a, b in zip(mine, theirs) if a != b), (len(mine), len(theirs)))
        raise RuntimeError(f"peer tensor layout differs (argv/image mismatch): {diff}")
    out = []
    for r in meta["tensors"]:
        with torch.cuda.device(r["device"]):
            base = _S.cuda.ipc_open(r["handle"])
        out.append(_wrap(base + r["offset"], r["nbytes"], getattr(torch, r["dtype"]), r["shape"], r["device"]))
    _S.peer_tensors = out  # tensors before epoch: a reader that sees the new epoch sees its tensors
    _S.peer_epoch = epoch
    return out


def _copy_rows(dst: torch.Tensor, src: torch.Tensor, dst_idx: torch.Tensor, src_idx_cpu: torch.Tensor):
    """dst[dst_idx] = src[src_idx]: gather on the peer GPU in chunks, peer-copy, scatter locally."""
    src_dev, dst_dev = src.device, dst.device
    n = src_idx_cpu.numel()
    for a in range(0, n, GATHER_CHUNK):
        b = min(n, a + GATHER_CHUNK)
        si = src_idx_cpu[a:b].to(src_dev, non_blocking=True)
        with torch.cuda.device(src_dev):
            g = src.index_select(0, si)
        dst.index_copy_(0, dst_idx[a:b], g.to(dst_dev))


def _copy_all(peer: List[torch.Tensor], kv_src_cpu: torch.Tensor, kv_dst: torch.Tensor, slot_src: int, slot_dst: int):
    page = 64  # DSA KV/indexer page (asserted page_size == 64 by DSATokenToKVPool)
    pg_src = (kv_src_cpu[::page] // page).to(torch.int64)
    pg_dst = (kv_dst[::page] // page).to(torch.int64)
    for d, p in zip(_S.tensors, peer):
        t = d["tensor"]
        if d["kind"] == "token":
            _copy_rows(t, p, kv_dst, kv_src_cpu)
        elif d["kind"] == "page":
            _copy_rows(t, p, pg_dst.to(t.device), pg_src)
        else:  # slot: [L, size+1, ...]
            t[:, slot_dst].copy_(p[:, slot_src].to(t.device))
    torch.cuda.current_stream().synchronize()
    if VERIFY:
        _verify_copy(peer, kv_src_cpu, kv_dst, slot_src, slot_dst, pg_src, pg_dst)


VERIFY = os.environ.get("SGLANG_PEERKV_VERIFY", "0") == "1" or os.path.exists("/etc/glm53/peerkv.verify")  # lab only


def _verify_copy(peer, kv_src_cpu, kv_dst, slot_src, slot_dst, pg_src, pg_dst):
    bad = []
    for d, p in zip(_S.tensors, peer):
        t = d["tensor"]
        if d["kind"] == "token":
            a, b = t.index_select(0, kv_dst), p.index_select(0, kv_src_cpu.to(p.device)).to(t.device)
        elif d["kind"] == "page":
            a, b = t.index_select(0, pg_dst.to(t.device)), p.index_select(0, pg_src.to(p.device)).to(t.device)
        else:
            a, b = t[:, slot_dst], p[:, slot_src].to(t.device)
        if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
            bad.append(d["name"])
    logger.info(f"{TAG} rank {_S.rank}: verify {'OK' if not bad else 'MISMATCH ' + ','.join(bad[:6])} "
                f"({len(_S.tensors)} tensors, {kv_dst.numel()} tokens)")
    if bad:
        raise RuntimeError(f"verify mismatch: {bad[:6]}")


def maybe_fetch(scheduler, req) -> None:
    """Try to fill req's missing prefix from the peer's VRAM. Never raises; never aborts req."""
    if _S is None or not PEER:
        return
    h = _impl()
    if h is not None:
        return h.maybe_fetch(scheduler, req)
    try:
        _maybe_fetch(scheduler, req)
    except Exception as e:  # the request then prefills normally
        logger.warning(f"{TAG} rank {_S.rank}: fetch error, falling back to prefill: {e!r}")
        _m("req", f"error:{type(e).__name__}")


def _maybe_fetch(scheduler, req) -> None:
    if getattr(req, "cache_salt", None) or getattr(req, "positional_embed_overrides", None) is not None:
        return
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.utils.common import broadcast_pyobj
    import torch.distributed as dist

    tree = scheduler.tree_cache
    tokens = req.origin_input_ids
    limit = len(tokens) - 1
    if limit < MIN_TOKENS:
        return
    local = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=tokens, extra_key=req.extra_key, limit=limit)))
    p_b = len(local.device_indices)
    if limit - p_b < MIN_TOKENS:
        return
    t0 = time.monotonic()
    _S.stats["fetch_tries"] += 1
    _S.seq += 1
    lease = f"{SELF}:{_S.epoch}:{_S.seq}"
    grant = None
    # Rank 0 decides for every rank (its answer is broadcast): skip without asking the peer when
    # the peer is unmapped here or the copy would stall the scheduler longer than MAX_COPY_S.
    est_s = (limit - p_b) * _bytes_per_token() / max(_S.peer_gbps, 1e-3) / 1e9
    if _S.rank == 0 and (_S.peer_tensors is None or est_s > MAX_COPY_S):
        grant = {"ok": False, "why": "unmapped" if _S.peer_tensors is None else "too_slow"}
    elif _S.rank == 0:
        try:
            grant = _rpc(PEER, PeerKVMsg(op="lend", lease=lease, tokens=tokens, extra_key=req.extra_key, limit=limit, p_b=p_b),
                         RPC_TIMEOUT_S + 0.5)
        except Exception as e:
            grant = {"ok": False, "why": f"rpc:{type(e).__name__}"}
    if _S.tp_size > 1:
        grant = broadcast_pyobj([grant], scheduler.tp_group.rank, scheduler.tp_cpu_group, src=scheduler.tp_group.ranks[0])[0]
    if not grant or not grant.get("ok"):
        _m("req", (grant or {}).get("why", "miss"))
        return

    L, kv_src = grant["L"], grant["kv_idx"]
    n = L - p_b
    # Pin the local prefix so the eviction below cannot free pages that become part of `value`.
    local_node = local.last_device_node
    tree.inc_lock_ref(local_node)
    try:
        _fetch_insert(scheduler, req, tree, tokens, local, p_b, L, n, kv_src, grant, lease, t0)
    finally:
        tree.dec_lock_ref(local_node)


def _fetch_insert(scheduler, req, tree, tokens, local, p_b, L, n, kv_src, grant, lease, t0) -> None:
    t_lend = time.monotonic()
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    import torch.distributed as dist

    alloc = scheduler.token_to_kv_pool_allocator
    mamba_alloc = scheduler.req_to_token_pool.mamba_allocator
    kv_dst = alloc.alloc(n)
    if kv_dst is None:
        tree.evict(EvictParams(num_tokens=n))
        kv_dst = alloc.alloc(n)
    slot = mamba_alloc.alloc(1) if kv_dst is not None else None
    if kv_dst is not None and slot is None:
        tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        slot = mamba_alloc.alloc(1)
    ok = kv_dst is not None and slot is not None
    why = "no_space" if not ok else "ok"
    t_alloc = time.monotonic()
    t_open = t_alloc
    if not ok:
        logger.warning(f"{TAG} rank {_S.rank}: no space for {n} tokens (kv={kv_dst is not None} slot={slot is not None})")
    if ok:
        try:
            with _S.map_lock:
                if _S.peer_epoch != grant["epoch"] or _S.peer_tensors is None:
                    raise RuntimeError(f"peer not mapped yet (have {_S.peer_epoch}, grant {grant['epoch']})")
                peer = _S.peer_tensors
                t_open = time.monotonic()
                _copy_all(peer, kv_src, kv_dst.to(torch.int64), int(grant["mamba_slot"]), int(slot.view(-1)[0].item()))
        except Exception as e:
            ok, why = False, f"copy:{type(e).__name__}"
            logger.warning(f"{TAG} rank {_S.rank}: copy failed: {e!r}")
    t_copy = time.monotonic()
    if _S.tp_size > 1:
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=scheduler.tp_cpu_group)
        if ok and not flag.item():
            why = "peer_rank_failed"
        ok = bool(flag.item())
    if LOG_FETCHES:
        logger.info(f"{TAG} rank {_S.rank}: fetch n={n} {why} ok={ok} lend={t_lend - t0:.3f}s alloc={t_alloc - t_lend:.3f}s "
                    f"open={t_open - t_alloc:.3f}s copy={t_copy - t_open:.3f}s")
    # Release the lease (in-band on the holder) whatever happened.
    if _S.rank == 0:
        try:
            _rpc(PEER, PeerKVMsg(op="release", lease=lease), 1.0)
        except Exception:
            pass
    if not ok:
        if kv_dst is not None:
            alloc.free(kv_dst)
        if slot is not None:
            mamba_alloc.free(slot)
        _S.stats["fallback"] += 1
        _m("req", why)
        return

    key = RadixKey(token_ids=tokens, extra_key=req.extra_key, limit=L)
    value = torch.cat([local.device_indices.to(torch.int64), kv_dst.to(torch.int64)])
    res = tree.insert(InsertParams(key=key, value=value, mamba_value=slot.view(-1)[:1], prev_prefix_len=p_b,
                                   chunked=True, priority=getattr(req, "priority", 0) or 0, track_adopted_ranges=True))
    if res.mamba_exist:
        mamba_alloc.free(slot.view(-1)[:1])
    # Do NOT free non-adopted pages here: for [prev_prefix_len, L) slices that overlap nodes the
    # tree already holds (e.g. KV present but no mamba state at that depth, so the local match
    # stopped short), insert itself emits FreeDeviceKV for the duplicate slice. Freeing them again
    # double-frees (caught by the strict idle mem check on gpu04 r2, 2026-10-08 19:09Z).
    adopted = sum(e - s for s, e in (res.adopted_ranges or {}).get(ComponentType.FULL, []) if e > p_b)
    keep = torch.tensor([min(adopted, n)])
    dt = time.monotonic() - t0
    _S.stats["fetch_hits"] += 1
    _S.stats["hit_tokens"] += n
    _S.stats["fetch_s"] += dt
    _m("req", "hit")
    _m("tok", inc=n)
    _m("sec", observe=dt)
    if LOG_FETCHES and _S.rank == 0:
        logger.info(f"{TAG} hit rid={req.rid} p_b={p_b} L={L} fetched={n} adopted={int(keep.sum())} s={dt:.3f}")
