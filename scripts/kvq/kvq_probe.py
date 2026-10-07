#!/usr/bin/env python3
"""In-host KV sharing primitives under NVIDIA CC (gpu13 GPUs 4-7 visible as cuda:0-3).

Prints one `KVQ {json}` line per result, then `KVQ_DONE`. Checks:
  env        nvidia-smi topology, CC mode, host memory
  p2p        in-process peer copy cuda:a -> cuda:b (bulk 1 GiB + scattered 64 KiB pages), checksummed
  ipc        cross-process CUDA IPC (legacy cudaIpc handles, what NIXL/UCX cuda_ipc uses with
             expandable_segments:False): a producer process on cuda:a exports, a consumer on
             cuda:b imports and copies, checksummed
  host       pinned (cudaMallocHost) H2D/D2H, tmpfs file -> pinned -> GPU (the shared-tier
             restore path) with 1 and 2 concurrent processes, and cudaHostRegister (801 expected)
"""
import faulthandler, json, os, subprocess, sys, time, mmap, ctypes

faulthandler.enable()  # a native crash in CUDA init still leaves a stack in the log
print("KVQ " + json.dumps({"kind": "start", "pid": os.getpid()}), flush=True)
import torch  # noqa: E402
import torch.multiprocessing as mp

PAGE = 64 * 1024
GiB = 1 << 30


def out(kind, **kw):
    print("KVQ " + json.dumps({"kind": kind, **kw}), flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def checksum(t):
    """Order-sensitive checksum of a uint8 CUDA tensor, computed on its own device."""
    v = t.view(torch.int32).to(torch.int64)
    w = (torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 65521) + 1
    return int((v * w).sum().item()) & ((1 << 62) - 1)


def fill(n, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.randint(0, 256, (n,), dtype=torch.uint8, device=dev, generator=g)


def timed(fn, dev_list, reps=3):
    best = 1e9
    for _ in range(reps):
        for d in dev_list:
            torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        fn()
        for d in dev_list:
            torch.cuda.synchronize(d)
        best = min(best, time.perf_counter() - t0)
    return best


def p2p(a, b):
    da, db = f"cuda:{a}", f"cuda:{b}"
    res = {"src": a, "dst": b, "can_access_peer": torch.cuda.can_device_access_peer(a, b)}
    try:
        src = fill(GiB, da, 1234 + a)
        cs = checksum(src)
        dst = torch.empty_like(src, device=db)
        dt = timed(lambda: dst.copy_(src, non_blocking=True), [da, db])
        res.update(bulk_ok=checksum(dst) == cs, bulk_gbps=round(GiB / dt / 1e9, 1))
        # Scattered pages: 35K x 64 KiB is the shape of a ~189K-token GLM KV move.
        npg = 35000
        srcp = fill(npg * PAGE, da, 99 + a).view(npg, PAGE)
        dstp = torch.zeros(npg, PAGE, dtype=torch.uint8, device=db)
        perm_s = torch.randperm(npg, device=da)
        perm_d = torch.randperm(npg, device=db)
        def gather():
            dstp[perm_d] = srcp[perm_s].to(db, non_blocking=True)
        dt2 = timed(gather, [da, db], reps=2)
        ok = checksum(dstp[perm_d].contiguous().view(-1)) == checksum(srcp[perm_s].contiguous().view(-1).to(db))
        res.update(scatter_ok=ok, scatter_gbps=round(npg * PAGE / dt2 / 1e9, 1))
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)[:500]
    out("p2p", **res)


def _ipc_producer(dev, q, done, seed):
    torch.cuda.set_device(dev)
    t = fill(GiB, f"cuda:{dev}", seed)
    q.put((t, checksum(t)))  # torch shares CUDA tensors via cudaIpcGetMemHandle
    done.wait(600)


def ipc(a, b):
    res = {"src": a, "dst": b, "mode": "cross-process cudaIpc"}
    ctx = mp.get_context("spawn")
    q, done = ctx.Queue(), ctx.Event()
    p = ctx.Process(target=_ipc_producer, args=(a, q, done, 4321 + a))
    p.start()
    try:
        torch.cuda.set_device(b)
        remote, cs = q.get(timeout=300)
        res["remote_device"] = str(remote.device)
        dst = torch.empty(remote.numel(), dtype=torch.uint8, device=f"cuda:{b}")
        dt = timed(lambda: dst.copy_(remote, non_blocking=True), [remote.device.index, b])
        res.update(ok=checksum(dst) == cs, gbps=round(GiB / dt / 1e9, 1))
        del remote
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)[:500]
    finally:
        done.set()
        p.join(60)
    out("ipc", **res)


def _host_restore(dev, path, size, barrier, rq):
    torch.cuda.set_device(dev)
    pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(size, dtype=torch.uint8, device=f"cuda:{dev}")
    mv = memoryview(pinned.numpy())
    barrier.wait()
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        n = 0
        while n < size:
            r = f.readinto(mv[n:n + 64 * 1024 * 1024])
            if not r:
                break
            n += r
    t1 = time.perf_counter()
    gpu.copy_(pinned, non_blocking=True)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    rq.put({"dev": dev, "read_gbps": round(size / (t1 - t0) / 1e9, 2), "h2d_gbps": round(size / (t2 - t1) / 1e9, 1),
            "end_to_end_gbps": round(size / (t2 - t0) / 1e9, 2)})


def host():
    res = {}
    size = 4 * GiB
    try:
        pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)  # cudaHostAlloc
        g = torch.empty(size, dtype=torch.uint8, device="cuda:0")
        res["h2d_gbps"] = round(size / timed(lambda: g.copy_(pinned, non_blocking=True), ["cuda:0"]) / 1e9, 1)
        res["d2h_gbps"] = round(size / timed(lambda: pinned.copy_(g, non_blocking=True), ["cuda:0"]) / 1e9, 1)
        res["pinned_alloc_ok"] = True
        del pinned, g
    except Exception as e:  # noqa: BLE001
        res["pinned_error"] = repr(e)[:300]
    # cudaHostRegister on shared memory: the upstream HiCache path. 801 = not supported under CC.
    try:
        buf = mmap.mmap(-1, 64 * 1024 * 1024)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        rc = torch.cuda.cudart().cudaHostRegister(addr, 64 * 1024 * 1024, 0)
        res["cudaHostRegister_rc"] = int(rc.value if hasattr(rc, "value") else rc)
        if res["cudaHostRegister_rc"] == 0:
            torch.cuda.cudart().cudaHostUnregister(addr)
    except Exception as e:  # noqa: BLE001
        res["cudaHostRegister_error"] = repr(e)[:300]
    out("host", **res)
    # Shared-tier restore path: tmpfs file written by one replica, read by the other into pinned memory.
    d = "/dev/shm/kvq"
    os.makedirs(d, exist_ok=True)
    fsize = 2 * GiB
    paths = []
    for i in range(2):
        p = f"{d}/blob{i}"
        t0 = time.perf_counter()
        with open(p, "wb") as f:
            chunk = os.urandom(64 * 1024 * 1024)
            for _ in range(fsize // len(chunk)):
                f.write(chunk)
        paths.append(p)
        out("tmpfs_write", file=i, gbps=round(fsize / (time.perf_counter() - t0) / 1e9, 2))
    ctx = mp.get_context("spawn")
    for nproc in (1, 2):
        barrier, rq = ctx.Barrier(nproc), ctx.Queue()
        procs = [ctx.Process(target=_host_restore, args=(2 * i, paths[i], fsize, barrier, rq)) for i in range(nproc)]
        [p.start() for p in procs]
        results = [rq.get(timeout=600) for _ in procs]
        [p.join(60) for p in procs]
        out("tmpfs_restore", concurrent=nproc, per_proc=results)
    for p in paths:
        os.remove(p)


def main():
    print(sh("nvidia-smi -L; nvidia-smi conf-compute -f; nvidia-smi conf-compute -mgm"), flush=True)
    out("env", cuda_visible=os.environ.get("NVIDIA_VISIBLE_DEVICES"), devices=torch.cuda.device_count(),
        names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        torch=torch.__version__, cuda=torch.version.cuda,
        alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    print(sh("nvidia-smi -L; nvidia-smi topo -m; nvidia-smi conf-compute -f; nvidia-smi conf-compute -mgm; "
             "nvidia-smi conf-compute -q 2>/dev/null | head -40; free -g; nproc; df -h /dev/shm"), flush=True)
    n = torch.cuda.device_count()
    if n < 4:
        out("fatal", msg=f"expected 4 visible GPUs, got {n}")
        return
    # Same-replica pairs (0,1)/(2,3) and cross-replica pairs (0,2)/(1,3)/(2,0).
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 0), (3, 1)]:
        p2p(a, b)
        torch.cuda.empty_cache()
    for a, b in [(0, 2), (2, 0), (1, 3)]:
        ipc(a, b)
        torch.cuda.empty_cache()
    host()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        main()
    finally:
        print("KVQ_DONE", flush=True)
