#!/usr/bin/env python3
"""NIXL VRAM->VRAM microbenchmark between two processes (run each in its own container to mirror SGLang P/D).

  target:    python3 nixl_bench.py target    --dir /shared/x --gpu 0
  initiator: python3 nixl_bench.py initiator --dir /shared/x --gpu 0 [--shapes 1x1073741824,16384x65536,65536x16384]

The target registers a VRAM buffer and writes its agent metadata + buffer address to --dir; the initiator registers its
own buffer, loads the target's metadata, and times WRITE transfers of each shape (N descriptors x S bytes, same total
bytes where possible), contiguous and strided. Prints GB/s and µs/descriptor. Set UCX_PROTO_INFO=y to see UCX's choice.
--gpu is the CUDA index *inside* the container."""
import argparse, json, os, pickle, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("role", choices=["target", "initiator"])
ap.add_argument("--dir", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--buf-gib", type=float, default=2.0)
ap.add_argument("--shapes", default="1x1073741824,256x4194304,16384x65536,65536x16384")
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--backend", default="UCX")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--params", default="")
ap.add_argument("--strict", action="store_true")
ap.add_argument("--prepped", action="store_true")
a = ap.parse_args()

from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

torch.cuda.set_device(a.gpu)
nbytes = int(a.buf_gib * (1 << 30))
buf = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{a.gpu}")
buf.fill_(7 if a.role == "target" else 3)
torch.cuda.synchronize()
os.makedirs(a.dir, exist_ok=True)
name = f"{a.role}-{os.getpid()}"
if a.params:
    agent = nixl_agent(name, nixl_agent_config(backends=[], num_threads=a.threads))
    params = agent.get_plugin_params(a.backend)[0] if hasattr(agent, "get_plugin_params") else {}
    params = dict(params); params.update(dict(kv.split("=", 1) for kv in a.params.split(",")))
    print(f"[{a.role}] {a.backend} params: {params}", flush=True)
    agent.create_backend(a.backend, params)
else:
    _cfg = dict(num_threads=a.threads)
    if a.strict: _cfg["sync_mode"] = nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT
    agent = nixl_agent(name, nixl_agent_config(backends=[], **_cfg))
    agent.create_backend(a.backend, {"num_threads": str(a.threads)} if a.threads else {})
if a.role == "initiator":
    try: print(f"[initiator] plugin params: {agent.get_plugin_params(a.backend)}", flush=True)
    except Exception as e: print(f"[initiator] get_plugin_params: {e}", flush=True)
reg = agent.register_memory([(buf.data_ptr(), nbytes, a.gpu, "")], "VRAM")
meta_path = os.path.join(a.dir, "target.pkl")

if a.role == "target":
    with open(meta_path + ".tmp", "wb") as f:
        pickle.dump({"name": name, "meta": agent.get_agent_metadata(), "addr": buf.data_ptr(), "len": nbytes, "gpu": a.gpu}, f)
    os.replace(meta_path + ".tmp", meta_path)
    print(f"[target] ready name={name} addr={buf.data_ptr():#x} len={nbytes}", flush=True)
    done = os.path.join(a.dir, "done")
    while not os.path.exists(done):
        time.sleep(0.5)
    # verify a few bytes were written by the initiator (3s)
    print(f"[target] first bytes after transfers: {buf[:8].tolist()} last: {buf[-8:].tolist()}", flush=True)
    sys.exit(0)

# initiator
while not os.path.exists(meta_path):
    time.sleep(0.5)
t = pickle.load(open(meta_path, "rb"))
remote = agent.add_remote_agent(t["meta"])
if isinstance(remote, bytes):
    remote = remote.decode()
print(f"[initiator] connected to {remote}", flush=True)
rows = []
for shape in a.shapes.split(","):
    n, s = map(int, shape.split("x"))
    for layout in ("contig", "strided"):
        stride = s if layout == "contig" else min(2 * s, nbytes // max(n, 1))
        if stride < s or n * stride > nbytes:
            continue
        loc = [(buf.data_ptr() + i * stride, s, a.gpu) for i in range(n)]
        rem = [(t["addr"] + i * stride, s, t["gpu"]) for i in range(n)]
        if a.prepped:
            lp = agent.prep_xfer_dlist("", loc, "VRAM"); rp = agent.prep_xfer_dlist(remote, rem, "VRAM")
            idx = list(range(n))
        else:
            ld = agent.get_xfer_descs(loc, "VRAM")
            rd = agent.get_xfer_descs(rem, "VRAM")
        times = []
        for it in range(a.iters + 1):
            h = agent.make_prepped_xfer("WRITE", lp, idx, rp, idx, b"x") if a.prepped else agent.initialize_xfer("WRITE", ld, rd, remote, b"x")
            t0 = time.perf_counter()
            st = agent.transfer(h)
            while True:
                st = agent.check_xfer_state(h)
                if st in ("DONE", "ERR"):
                    break
            dt = time.perf_counter() - t0
            agent.release_xfer_handle(h)
            if st == "ERR":
                print(f"[initiator] {shape} {layout}: transfer ERR", flush=True); break
            if it:  # first iteration is warmup
                times.append(dt)
        if times:
            best = min(times); tot = n * s
            rows.append((shape, layout, tot, best))
            print(f"[initiator] {n:6d} x {s:>10d} B {layout:7s}: {tot/1e9:6.2f} GB in {best*1e3:8.1f} ms = {tot/best/1e9:7.2f} GB/s, {best*1e6/n:8.1f} us/desc", flush=True)
open(os.path.join(a.dir, "done"), "w").write("1")
json.dump([dict(shape=r[0], layout=r[1], bytes=r[2], sec=r[3]) for r in rows], open(os.path.join(a.dir, f"result-{os.getpid()}.json"), "w"))
