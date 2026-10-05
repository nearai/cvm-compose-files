"""Build-time and CPU-only checks for the NEAR_SELF_PROFILE hook: importable, inert by default, and the
hook state machine plus summariser behave (synthetic traces, stub scheduler, no GPU)."""

import gzip
import json
import os
import subprocess
import sys
import time
import types

from sglang.srt.utils import near_self_profile as nsp

sched = types.SimpleNamespace(ps=types.SimpleNamespace(tp_rank=0))
os.environ.pop("NEAR_SELF_PROFILE", None)
assert nsp.maybe_create(sched) is None, "hook must be inert when NEAR_SELF_PROFILE is unset"
os.environ["NEAR_SELF_PROFILE"] = "0"
assert nsp.maybe_create(sched) is None
os.environ["NEAR_SELF_PROFILE"] = "1"
assert nsp.maybe_create(types.SimpleNamespace(ps=types.SimpleNamespace(tp_rank=1))) is None, "tp_rank != 0 inert"
assert nsp.maybe_create(sched) is not None

TRACE = {"traceEvents": [
    {"ph": "X", "cat": "python_function", "name": "sglang/srt/managers/scheduler.py(10): run_batch", "tid": 1, "ts": i * 1000, "dur": 900}
    for i in range(5)] + [
    {"ph": "X", "cat": "python_function", "name": "sglang/srt/x/y.py(7): resolve", "tid": 1, "ts": 1100, "dur": 500},
    {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "tid": 1, "ts": 1200, "dur": 100},
    {"ph": "X", "cat": "kernel", "name": "k", "tid": 7, "ts": 1000, "dur": 50},
    {"ph": "X", "cat": "kernel", "name": "k", "tid": 7, "ts": 1300, "dur": 50},
    {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH (Device -> Pinned)", "tid": 7, "ts": 1400, "dur": 5, "args": {"bytes": 64}}]}


def write(events, mode_cuda=True):
    os.makedirs(nsp.OUT_DIR, exist_ok=True)
    doc = TRACE if mode_cuda else {"traceEvents": [e for e in events if e["cat"] not in ("kernel", "gpu_memcpy", "cuda_runtime")]}
    with gzip.open(os.path.join(nsp.OUT_DIR, "near-TP-0.trace.json.gz"), "wt") as f:
        json.dump(doc, f)


# Summariser: cuda trace -> readable NEAR_PROFILE lines, trace dir deleted.
write(TRACE["traceEvents"])
out = subprocess.run([sys.executable, nsp.__file__, nsp.OUT_DIR, "cuda+cpu"], capture_output=True, text=True)
lines = out.stdout.splitlines()
assert out.returncode == 0 and lines and all(l.startswith("NEAR_PROFILE ") for l in lines), out
assert "srt/x/y.py:7 resolve" in out.stdout and "cudaStreamSynchronize=1/" in out.stdout and "DtoH=1/" in out.stdout, out.stdout
assert not os.path.exists(nsp.OUT_DIR), "trace dir must be deleted"
# Summariser: CUDA requested but no kernel events -> exit code NO_KERNELS, dir still deleted.
write(TRACE["traceEvents"], mode_cuda=False)
out = subprocess.run([sys.executable, nsp.__file__, nsp.OUT_DIR, "cuda+cpu"], capture_output=True, text=True)
assert out.returncode == nsp.NO_KERNELS and not os.path.exists(nsp.OUT_DIR), out
# Summariser errors print one line and never raise.
out = subprocess.run([sys.executable, nsp.__file__, nsp.OUT_DIR, "cpu-only"], capture_output=True, text=True)
assert out.returncode == 1 and out.stdout.startswith("NEAR_PROFILE error"), out


class PM:  # stub of SchedulerProfilerManager
    def __init__(self):
        self.profile_in_progress, self.started = False, []

    def _init_profile(self, d, start, steps, acts, *rest):
        self.acts = list(acts)

    def _start_profile(self):
        self.started.append(self.acts)
        self.profile_in_progress = True
        return types.SimpleNamespace(success=True, message="")


# Hook: waits for a decode batch, starts CUDA+CPU, summarises when stopped, retries CPU-only once on no kernels.
pm = PM()
s = types.SimpleNamespace(ps=types.SimpleNamespace(tp_rank=0), profiler_manager=pm)
os.environ["NEAR_SELF_PROFILE_AFTER_S"] = "0"
hook = nsp.maybe_create(s)
prefill = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: False))
decode = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: True))
hook.tick(prefill)
assert not pm.started, "must wait for a decode batch"
hook.tick(decode)
assert pm.started == [["CPU", "GPU"]] and pm.near_selfprof
write(TRACE["traceEvents"], mode_cuda=False)  # a CUDA run whose trace has no kernel events
pm.profile_in_progress = False
for _ in range(100):
    hook.tick(decode)
    if len(pm.started) == 2:
        break
    time.sleep(0.1)
assert pm.started == [["CPU", "GPU"], ["CPU"]], pm.started
write(TRACE["traceEvents"], mode_cuda=False)
pm.profile_in_progress = False
for _ in range(100):
    hook.tick(decode)
    if hook.state < 0:
        break
    time.sleep(0.1)
assert hook.state < 0 and len(pm.started) == 2, "hook must run once, then stay disabled"
print("near_self_profile checks passed")
