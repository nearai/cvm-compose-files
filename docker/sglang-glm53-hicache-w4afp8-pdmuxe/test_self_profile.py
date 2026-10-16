"""CPU checks for the NEAR_SELF_PROFILE self-profiling hook (sglang.srt.utils.near_self_profile).

1. Inert by default: importing the module does nothing, and maybe_create returns None unless
   NEAR_SELF_PROFILE=1 and tp_rank == 0.
2. Summariser: a synthetic CUDA trace gives NEAR_PROFILE lines (blocking calls, code paths, memcpy
   direction) and the trace dir is deleted; CUDA requested but no kernel events exits NO_KERNELS and
   still deletes the dir; errors print one line and never raise.
3. Hook state machine: waits for a decode batch, starts CUDA+CPU, retries once CPU-only when the trace
   has no kernel events, then runs once and stays disabled.
4. Hooks: the patched scheduler.py creates the hook in init_profiler and ticks it in run_batch before
   the existing profiler predicate; profiler_manager skips its all-rank barrier only for the hook.

Stdlib only; no GPU. The module under test is loaded from --module, so this runs on the installed
file in the image and on a source checkout alike.
"""
import argparse
import gzip
import importlib.util
import json
import os
import subprocess
import sys
import time
import types

ap = argparse.ArgumentParser()
ap.add_argument("--module", required=True)
ap.add_argument("--scheduler", required=False)
ap.add_argument("--profiler-manager", required=False)
args = ap.parse_args()

for key in [k for k in os.environ if k.startswith("NEAR_SELF_PROFILE")]:
    del os.environ[key]
try:
    import sglang.srt.environ  # noqa: F401  (maybe_create reads SGLANG_PROFILE_V2 from it)
except ImportError:  # source checkout without sglang: stand in for the one flag the hook reads
    for name in ("sglang", "sglang.srt", "sglang.srt.environ"):
        sys.modules[name] = types.ModuleType(name)
    flag = types.SimpleNamespace(get=lambda: False)
    sys.modules["sglang.srt.environ"].envs = types.SimpleNamespace(SGLANG_PROFILE_V2=flag)
spec = importlib.util.spec_from_file_location("near_self_profile_under_test", args.module)
nsp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nsp)
MODULE_PATH = os.path.abspath(args.module)
if os.path.exists(nsp.OUT_DIR):
    sys.exit(f"refusing to run: {nsp.OUT_DIR} already exists")


def stub(tp_rank=0, **extra):
    return types.SimpleNamespace(ps=types.SimpleNamespace(tp_rank=tp_rank), **extra)


# 1. Inert unless NEAR_SELF_PROFILE=1, and only on tp_rank 0.
assert nsp.maybe_create(stub()) is None, "hook must be inert when NEAR_SELF_PROFILE is unset"
for off in ("0", "", "false"):
    os.environ["NEAR_SELF_PROFILE"] = off
    assert nsp.maybe_create(stub()) is None, off
os.environ["NEAR_SELF_PROFILE"] = "1"
assert nsp.maybe_create(stub(tp_rank=1)) is None, "tp_rank != 0 must stay inert"
assert nsp.maybe_create(stub()) is not None
assert not os.path.exists(nsp.OUT_DIR), "creating the hook must not touch the disk"

TRACE = {
    "traceEvents": [
        {"ph": "X", "cat": "python_function", "name": "sglang/srt/managers/scheduler.py(10): run_batch", "tid": 1, "ts": i * 1000, "dur": 900}
        for i in range(5)
    ]
    + [
        {"ph": "X", "cat": "python_function", "name": "sglang/srt/x/y.py(7): resolve", "tid": 1, "ts": 1100, "dur": 500},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "tid": 1, "ts": 1200, "dur": 100},
        {"ph": "X", "cat": "kernel", "name": "k", "tid": 7, "ts": 1000, "dur": 50},
        {"ph": "X", "cat": "kernel", "name": "k", "tid": 7, "ts": 1300, "dur": 50},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH (Device -> Pinned)", "tid": 7, "ts": 1400, "dur": 5, "args": {"bytes": 64}},
    ]
}


def write(with_cuda=True):
    os.makedirs(nsp.OUT_DIR, exist_ok=True)
    events = TRACE["traceEvents"]
    if not with_cuda:
        events = [e for e in events if e["cat"] not in ("kernel", "gpu_memcpy", "cuda_runtime")]
    with gzip.open(os.path.join(nsp.OUT_DIR, "near-TP-0.trace.json.gz"), "wt") as f:
        json.dump({"traceEvents": events}, f)


def summarise(mode):
    return subprocess.run([sys.executable, MODULE_PATH, nsp.OUT_DIR, mode], capture_output=True, text=True)


# 2. Summariser.
write()
out = summarise("cuda+cpu")
lines = out.stdout.splitlines()
assert out.returncode == 0 and lines and all(l.startswith("NEAR_PROFILE ") for l in lines), out
assert "srt/x/y.py:7 resolve" in out.stdout and "cudaStreamSynchronize=1/" in out.stdout and "DtoH=1/" in out.stdout, out.stdout
assert not os.path.exists(nsp.OUT_DIR), "trace dir must be deleted"
write(with_cuda=False)
out = summarise("cuda+cpu")
assert out.returncode == nsp.NO_KERNELS and not os.path.exists(nsp.OUT_DIR), out
out = summarise("cpu-only")  # the directory is gone now: one error line, exit 1, no traceback
assert out.returncode == 1 and out.stdout.startswith("NEAR_PROFILE error") and "Traceback" not in out.stderr, out


# 3. Hook state machine against a stub profiler manager.
class PM:
    def __init__(self):
        self.profile_in_progress, self.started = False, []

    def _init_profile(self, d, start, steps, acts, *rest):
        self.acts = list(acts)
        return types.SimpleNamespace(success=True, message="")

    def _start_profile(self):
        self.started.append(self.acts)
        self.profile_in_progress = True
        return types.SimpleNamespace(success=True, message="")


pm = PM()
os.environ["NEAR_SELF_PROFILE_AFTER_S"] = "0"
hook = nsp.maybe_create(stub(profiler_manager=pm))
prefill = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: False))
decode = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: True))
hook.tick(prefill)
assert not pm.started, "must wait for a decode batch"
hook.tick(decode)
assert pm.started == [["CPU", "GPU"]] and pm.near_selfprof


def drive(done):
    pm.profile_in_progress = False
    for _ in range(100):
        hook.tick(decode)
        if done():
            return
        time.sleep(0.1)
    raise AssertionError(f"hook did not progress: started={pm.started} state={hook.state}")


write(with_cuda=False)  # a CUDA run whose trace has no kernel events
drive(lambda: len(pm.started) == 2)
assert pm.started == [["CPU", "GPU"], ["CPU"]], pm.started
write(with_cuda=False)
drive(lambda: hook.state < 0)
assert hook.state < 0 and len(pm.started) == 2, "hook must run once, then stay disabled"
hook.tick(decode)
assert len(pm.started) == 2, "a finished hook must never start another profile"
assert not os.path.exists(nsp.OUT_DIR)

assert pm.near_selfprof is False, "the barrier skip must end with the hook's profile"

# An API profile already in progress: the hook must not touch it, and must disable itself.
pm = PM()
pm.profile_in_progress = True
hook = nsp.maybe_create(stub(profiler_manager=pm))
hook.tick(decode)
assert pm.started == [] and hook.state < 0 and not getattr(pm, "near_selfprof", False), pm.started

# CPU-only requested up front: one start, no retry.
os.environ["NEAR_SELF_PROFILE_ACTIVITIES"] = "cpu"
pm = PM()
hook = nsp.maybe_create(stub(profiler_manager=pm))
hook.tick(decode)
assert pm.started == [["CPU"]], pm.started
write(with_cuda=False)
drive(lambda: hook.state < 0)
assert len(pm.started) == 1, pm.started

# 4. Hooks in the patched sources.
if args.scheduler:
    src = open(args.scheduler).read()
    assert "maybe_create as maybe_create_self_profile" in src
    init = src.index("self.near_self_profile = maybe_create_self_profile(self)")
    assert src.rindex("def init_", 0, init) > src.rindex("class Scheduler", 0, init)
    tick = src.index("self.near_self_profile.tick(batch)")
    predicate = src.index("self.profiler_manager._profile_batch_predicate(batch)", tick)
    assert 0 < predicate - tick < 200, (tick, predicate)
if args.profiler_manager:
    src = open(args.profiler_manager).read()
    guard = src.index('if not getattr(self, "near_selfprof", False)')
    barrier = src.index("torch.distributed.barrier(self.dp_tp_cpu_group)", guard)
    assert 0 < barrier - guard < 200 and src.count("near_selfprof") == 1
print("near_self_profile checks passed")
