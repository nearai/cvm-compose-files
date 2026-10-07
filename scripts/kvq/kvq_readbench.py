#!/usr/bin/env python3
"""Read-path microbenchmark for the shared HiCache file tier under CC (one-shot; reads only).

Reads up to KVQ_RB_FILES real page files from the shared store (/kvshared, mounted read-only)
with T threads, into (a) pageable numpy buffers, (b) a per-thread pinned scratch page, (c) one
large pinned buffer at per-page offsets (like the host pool), using Python FileIO.readinto and
os.preadv. Prints `KVQ {json}` per (method, threads)."""
import json, os, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

D = os.environ.get("KVQ_RB_DIR", "/kvshared")
N = int(os.environ.get("KVQ_RB_FILES", "1024"))
files = sorted((e.path for e in os.scandir(D) if e.is_file() and e.name.endswith(".bin") and "." not in e.name[:-4]),
               key=lambda p: os.path.getsize(p), reverse=True)[:N]
size = os.path.getsize(files[0]) if files else 0
files = [f for f in files if os.path.getsize(f) == size]
print("KVQ " + json.dumps({"kind": "readbench_setup", "files": len(files), "page_bytes": size}), flush=True)
big = torch.empty(len(files) * size, dtype=torch.uint8, pin_memory=True)
big_np = big.numpy()
tls = threading.local()

def scratch(pinned):
    k = "p" if pinned else "n"
    b = getattr(tls, k, None)
    if b is None:
        b = torch.empty(size, dtype=torch.uint8, pin_memory=True).numpy() if pinned else np.empty(size, dtype=np.uint8)
        setattr(tls, k, b)
    return b

def read_fileio(path, buf):
    with open(path, "rb", buffering=0) as f:
        return f.readinto(memoryview(buf))

def read_preadv(path, buf):
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.preadv(fd, [memoryview(buf)], 0)
    finally:
        os.close(fd)

methods = {
    "fileio_pageable": lambda i, p: read_fileio(p, scratch(False)),
    "fileio_pinned_scratch": lambda i, p: read_fileio(p, scratch(True)),
    "fileio_pinned_slot": lambda i, p: read_fileio(p, big_np[i * size:(i + 1) * size]),
    "preadv_pinned_slot": lambda i, p: read_preadv(p, big_np[i * size:(i + 1) * size]),
}
for name, fn in methods.items():
    for threads in (1, 2, 4, 8, 16):
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(lambda ip: fn(*ip), list(enumerate(files))[:32]))  # warm
            t0 = time.perf_counter()
            n = sum(ex.map(lambda ip: fn(*ip), list(enumerate(files))))
            dt = time.perf_counter() - t0
        print("KVQ " + json.dumps({"kind": "readbench", "method": name, "threads": threads,
              "MBps": round(n / dt / 1e6), "ms_per_page": round(dt / len(files) * 1e3 * threads, 2)}), flush=True)
print("KVQ_DONE", flush=True)
