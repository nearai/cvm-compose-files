#!/usr/bin/env python3
"""peerkv copy microbench: the engine's CUDA-IPC export/open + copy path, without the engines.

Starts in seconds, so the copy path can be checked on a host (e.g. one under TDX + PPCIe) without
taking replicas down for an engine boot. Each rank allocates the tensors described by a real
engine's handle file (exact shapes and dtypes; pool lengths scaled by --frac to save memory),
exports them like `peerkv_init`, and the receiver copies N tokens with the engine's own
`_open_peer` + `_copy_all`, then checks every byte.

How to run (inside the engine image, all four GPUs visible, --ipc host --pid host):
  python3 peerkv_copybench.py --layout /peerkv/pa.tp0.handles --tokens 65536,114688,229376
Spawns 4 processes: holder ranks on cuda:0,1, receiver ranks on cuda:2,3 (rank i reads peer rank i).
Prints one JSON line per (rank, tokens): open_s, copy_s, GB, GB/s, equal, p2p.
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peerkv_engine as pk  # noqa: E402

PAGE = 64


def scaled_shape(rec, frac):
    shape = list(rec["shape"])
    axis = 1 if rec["kind"] == "slot" else 0
    shape[axis] = max(PAGE * 8, int(shape[axis] * frac) // PAGE * PAGE)
    return shape


def alloc(recs, frac, dev, fill):
    out = []
    for r in recs:
        dt = getattr(torch, r["dtype"])
        shape = scaled_shape(r, frac)
        t = torch.empty(shape, dtype=dt, device=dev)
        if fill:
            g = torch.Generator(device=dev).manual_seed(hash(r["name"]) & 0xFFFF)
            t.view(torch.uint8).copy_(torch.randint(0, 256, t.view(torch.uint8).shape, generator=g, device=dev, dtype=torch.uint8))
        else:
            t.zero_()
        out.append({"name": r["name"], "kind": r["kind"], "tensor": t})
    return out


def holder(rank, args, recs, ready, done):
    dev = rank
    pk.DIR = args.dir
    torch.cuda.set_device(dev)
    tensors = alloc(recs, args.frac, dev, fill=True)
    pk._S = pk._State(scheduler=None, rank=rank, tp_size=2, page_size=256, tensors=tensors, epoch="bench-A")
    pk._S.cuda = pk._Cuda()
    pk.SELF = "bench-A"
    out = []
    for d in tensors:
        t = d["tensor"]
        base, _ = pk._S.cuda.mem_base(t.data_ptr())
        out.append({"name": d["name"], "kind": d["kind"], "shape": tuple(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                    "nbytes": t.numel() * t.element_size(), "offset": t.data_ptr() - base,
                    "handle": pk._S.cuda.ipc_handle(base), "device": t.device.index})
    with open(os.path.join(args.dir, f"bench-A.tp{rank}.handles"), "wb") as f:
        pickle.dump({"epoch": "bench-A", "tensors": out}, f)
    ready.wait()
    done.wait()


def receiver(rank, args, recs, ready, done):
    dev = 2 + rank
    pk.DIR = args.dir
    torch.cuda.set_device(dev)
    tensors = alloc(recs, args.frac, dev, fill=False)
    pk._S = pk._State(scheduler=None, rank=rank, tp_size=2, page_size=256, tensors=tensors, epoch="bench-B")
    pk._S.cuda = pk._Cuda()
    pk.PEER = "bench-A"
    ready.wait()
    p2p = {j: torch.cuda.can_device_access_peer(dev, j) for j in range(torch.cuda.device_count()) if j != dev}
    t0 = time.monotonic()
    try:
        peer = pk._open_peer("bench-A")
    except Exception as e:
        print(json.dumps({"rank": rank, "dev": dev, "error": f"open: {e!r}", "p2p": p2p}), flush=True)
        done.wait()
        return
    open_s = time.monotonic() - t0
    rows = min(d["tensor"].shape[0] for d in tensors if d["kind"] == "token") - PAGE
    slots = min(d["tensor"].shape[1] for d in tensors if d["kind"] == "slot")
    for n in args.tokens:
        n = min(n, rows - rows % 256) // 256 * 256
        # Holder rows [0, n) -> receiver rows [n_off, n_off + n): exercise a real offset.
        src = torch.arange(PAGE, PAGE + n, dtype=torch.int64)
        pages = torch.randperm(n // PAGE, device=dev) + 1  # page-shuffled, page-aligned destination
        dst = (pages[:, None] * PAGE + torch.arange(PAGE, device=dev)).reshape(-1)
        s_src, s_dst = 3 % slots, 5 % slots
        torch.cuda.synchronize(dev)
        t1 = time.monotonic()
        err = None
        try:
            pk._copy_all(peer, src, dst, s_src, s_dst)
        except Exception as e:
            err = repr(e)
        torch.cuda.synchronize(dev)
        copy_s = time.monotonic() - t1
        nbytes, equal = 0, err is None
        if err is None:
            pg_s, pg_d = (src[::PAGE] // PAGE).to(dev), dst[::PAGE] // PAGE
            for d, p in zip(tensors, peer):
                t = d["tensor"]
                if d["kind"] == "token":
                    a, b = t.index_select(0, dst), p.index_select(0, src.to(p.device)).to(dev)
                elif d["kind"] == "page":
                    a, b = t.index_select(0, pg_d), p.index_select(0, pg_s.to(p.device)).to(dev)
                else:
                    a, b = t[:, s_dst], p[:, s_src].to(dev)
                nbytes += a.numel() * a.element_size()
                if not torch.equal(a.view(torch.uint8), b.contiguous().view(torch.uint8)):
                    equal = False
                    err = err or f"mismatch in {d['name']}"
        gb = nbytes / 1e9
        print(json.dumps({"rank": rank, "dev": dev, "tokens": n, "open_s": round(open_s, 3), "copy_s": round(copy_s, 3),
                          "GB": round(gb, 3), "GBps": round(gb / copy_s, 1) if copy_s else None, "equal": equal,
                          "error": err, "p2p": p2p}), flush=True)
        open_s = 0.0
    done.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True, help="a real engine's <replica>.tp0.handles file")
    ap.add_argument("--tokens", default="65536,114688,229376")
    ap.add_argument("--frac", type=float, default=0.3, help="fraction of each pool length to allocate")
    ap.add_argument("--dir", default="/tmp/peerkv-bench")
    args = ap.parse_args()
    args.tokens = [int(x) for x in args.tokens.split(",")]
    os.makedirs(args.dir, exist_ok=True)
    pk.DIR = args.dir
    with open(args.layout, "rb") as f:
        recs = pickle.load(f)["tensors"]
    ctx = mp.get_context("spawn")
    ready, done = ctx.Barrier(4), ctx.Barrier(4)
    procs = [ctx.Process(target=holder, args=(r, args, recs, ready, done)) for r in (0, 1)]
    procs += [ctx.Process(target=receiver, args=(r, args, recs, ready, done)) for r in (0, 1)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
