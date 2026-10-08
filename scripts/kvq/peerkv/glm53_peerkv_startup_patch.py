#!/usr/bin/env python3
"""Startup patch: in-host GPU-to-GPU KV prefix sharing (peerkv) on the GLM-5.3 HiCache W4AFP8 v7
image (docker.io/nearaidev/sglang@sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5).

Four exact-anchor hooks into the scheduler; the logic lives in /etc/glm53/peerkv_engine.py
(mounted next to this file). Every hook is inert unless SGLANG_PEERKV=1 loaded that module.
Fail closed: each file must have the pinned v7 sha256 before and the pinned patched sha256 after,
or the replica exits before serving. Idempotent: a restarted container whose files already have
the patched sha256 is verified and left alone.
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

TAG = "[peerkv-patch]"
ROOT = Path(os.environ.get("KVQ_PEERKV_PATCH_ROOT", "/sgl-workspace/sglang/python/sglang/srt"))
ENGINE_SRC = Path(os.environ.get("KVQ_PEERKV_ENGINE", "/etc/glm53/peerkv_engine.py"))

S = "managers/scheduler.py"
R = "managers/scheduler_components/request_receiver.py"
# rel -> (v7 sha256, patched sha256)
EXPECTED = {
    S: ("31d6284a3260da15d3335382af39e7a634d759e1ab6eea5662bc2468008ebd52", "8fc9b7f215f29e35a1fb0b4087f40ea6ab4c73e60942f161b22856f79d3e4f59"),
    R: ("a90fed7496d1ee07b0b6814936bc4c9076361267a5869c9cecb6657888d90e5c", "40941f215199f973f222359cda9a0d20bbc648875cb643f01511c543e4b077e8"),
}

EDITS = {
    S: [
        ("        self.init_kv_events_publisher()\n",
         "        self.init_kv_events_publisher()\n"
         "        # peerkv: export this rank's pools for the sibling replica and start the lend server.\n"
         "        if os.environ.get(\"SGLANG_PEERKV\", \"0\") == \"1\":\n"
         "            sys.path.insert(0, \"/etc/glm53\")\n"
         "            __import__(\"peerkv_engine\").peerkv_init(self)\n",
         "scheduler init hook"),
        ("            if self._abort_on_queued_limit(req):\n                return\n"
         "            self._prefetch_kvcache(req)\n            self.waiting_queue.append(req)\n",
         "            if self._abort_on_queued_limit(req):\n                return\n"
         "            # peerkv: fill a missing prefix from the sibling replica's VRAM (falls back to prefill).\n"
         "            _pk = sys.modules.get(\"peerkv_engine\")\n"
         "            if _pk is not None:\n"
         "                _pk.maybe_fetch(self, req)\n"
         "            self._prefetch_kvcache(req)\n            self.waiting_queue.append(req)\n",
         "receiver fetch hook"),
        ("        now = time.monotonic()\n        self.session_controller.maybe_reap(now)\n",
         "        now = time.monotonic()\n        self.session_controller.maybe_reap(now)\n"
         "        # peerkv: lend/release/expire messages, handled on every TP rank in the same iteration.\n"
         "        _pk = sys.modules.get(\"peerkv_engine\")\n"
         "        if _pk is not None:\n"
         "            recv_reqs = _pk.consume(self, recv_reqs)\n",
         "in-band consume hook"),
    ],
    R: [
        ("        recv_reqs = self._pull_raw_reqs()\n",
         "        recv_reqs = self._pull_raw_reqs()\n"
         "        # peerkv: rank 0 appends queued lend/release messages; broadcast carries them to all ranks.\n"
         "        _pk = __import__(\"sys\").modules.get(\"peerkv_engine\")\n"
         "        if _pk is not None:\n"
         "            recv_reqs = _pk.inject(recv_reqs)\n",
         "rank-0 inject hook"),
    ],
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(rel: str) -> str:
    return (ROOT / rel).read_bytes().decode("utf-8")


def main() -> None:
    if all(sha(read(rel)) == after for rel, (_b, after) in EXPECTED.items()):
        print(f"{TAG} OK: peerkv patch already applied and verified on {len(EXPECTED)} files", flush=True)
        return
    staged = {}
    for rel, edits in EDITS.items():
        orig = read(rel)
        before, after = EXPECTED[rel]
        if sha(orig) == after:
            continue
        if sha(orig) != before:
            sys.exit(f"{TAG} {rel}: unrecognized source sha256 {sha(orig)} (expected v7 {before})")
        s = orig
        for old, new, label in edits:
            n = s.count(old)
            if n != 1:
                sys.exit(f"{TAG} {rel} {label}: expected 1 anchor, found {n}")
            s = s.replace(old, new)
            print(f"{TAG} {label}: staged", flush=True)
        if sha(s) != after:
            sys.exit(f"{TAG} {rel}: patched sha256 {sha(s)} != expected {after}")
        staged[rel] = s
    for rel, s in staged.items():
        compile(s, rel, "exec")
        (ROOT / rel).write_bytes(s.encode("utf-8"))
        src = ROOT / rel
        for pyc in (src.parent / "__pycache__").glob(src.stem + ".*.pyc"):
            pyc.unlink()
        print(f"{TAG} {rel}: written", flush=True)
    if not ENGINE_SRC.exists() and os.environ.get("SGLANG_PEERKV", "0") == "1":
        sys.exit(f"{TAG} SGLANG_PEERKV=1 but {ENGINE_SRC} is missing")
    print(f"{TAG} OK: peerkv patch verified on {len(EXPECTED)} files", flush=True)


if __name__ == "__main__":
    main()
