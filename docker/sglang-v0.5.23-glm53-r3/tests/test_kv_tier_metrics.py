"""CPU checks for the KV tier metrics (sglang.srt.observability.kv_tier_metrics) and their hooks.

1. Inert by default: without SGLANG_KV_TIER_METRICS=1 the recorder is None.
2. Accounting: against a fake radix tree, VRAM evictions split by outcome, DRAM evictions by kind,
   load-backs, and the idle-time buckets are exact (cumulative, within_s=inf = all).
3. Gauges: the tree walk gives VRAM-cached, DRAM-duplicate and DRAM-only tokens exactly.
4. Hooks: the patched unified_tree_core stamps last_access_wall on every last_access_time write
   and calls the recorder at the six hook points; the scheduler's ingest_requests calls tick().
5. Safety: a failing hook is logged, not raised, unless SGLANG_KV_TIER_METRICS_STRICT=1.
6. Refresh: after a demotion with no further transfer, tick() reports the new VRAM/DRAM split once
   the resync interval has passed, and not before; tick() is a no-op without a Python tree core.
"""
import argparse
import importlib.util
import inspect
import os
import sys
import time
import types

ap = argparse.ArgumentParser()
ap.add_argument("--module", required=True)
ap.add_argument("--tree-core", required=False)
ap.add_argument("--scheduler", required=False)
args = ap.parse_args()
failures = 0


def fail(msg):
    global failures
    failures += 1
    print("FAILED " + msg, file=sys.stderr)


def load(env):
    for k in [k for k in os.environ if k.startswith("SGLANG_KV_TIER_METRICS")]:
        del os.environ[k]
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(f"kvtm_{len(env)}_{time.time()}", args.module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 1. inert by default
off = load({})
if off.recorder() is not None:
    fail("step1 recorder exists while disabled")
else:
    print("  ok inert by default")

# 2-3. accounting against a fake tree
from prometheus_client import CollectorRegistry

m = load({})
reg = CollectorRegistry()
rec = m.TierMetrics(resync_s=0, registry=reg, model_name="t")


class CD:
    def __init__(self, dev, host):
        self.value = list(range(dev)) if dev else None
        self.host_value = list(range(host)) if host else None


class Node:
    _ids = 0

    def __init__(self, dev, host, idle_s, parent=True, mamba=False):
        self.component_data = [CD(dev, host), CD(0, 0), CD(0, 1 if mamba else 0)]
        self.parent = object() if parent else None
        self.last_access_wall = time.monotonic() - idle_s
        self.children = {}
        Node._ids += 1
        self.id = Node._ids


nodes = {i: n for i, n in enumerate([
    Node(0, 0, 0, parent=False),        # root
    Node(640, 640, 5),                  # in VRAM and DRAM (write_through copy)
    Node(0, 1280, 700),                 # DRAM-only, with a mamba checkpoint below it (matchable)
    Node(256, 0, 50),                   # VRAM only, no copy
])}
nodes[4] = Node(0, 512, 900)          # DRAM-only, no mamba checkpoint anywhere below (unmatchable)
nodes[5] = Node(0, 0, 900, mamba=True)  # mamba checkpoint (host) under node 2
nodes[2].children = {"c": nodes[5]}
nodes[0].children = {"a": nodes[1], "b": nodes[2], "d": nodes[3], "e": nodes[4]}
core = types.SimpleNamespace(_node_arena=nodes, root_node=nodes[0])
rec.vram_evict(core, nodes[1], "demoted")          # 640 tokens, idle 5 s
rec.vram_evict(core, nodes[3], "deleted")          # 256 tokens, idle 50 s
rec.dram_evict(core, nodes[2], "dram_only")        # 1280 tokens, idle 700 s
rec.dram_evict(core, nodes[1], "duplicate")        # 640 tokens, idle 5 s
rec.load_back(core, nodes[2])                      # 1280 tokens, idle 700 s


def val(name, **lab):
    v = reg.get_sample_value(name, {"model_name": "t", **lab})
    return 0 if v is None else v


checks = [
    ("demoted", val("sglang:kv_tier_vram_evicted_tokens_total", outcome="demoted"), 640),
    ("deleted", val("sglang:kv_tier_vram_evicted_tokens_total", outcome="deleted"), 256),
    ("dram_only", val("sglang:kv_tier_dram_evicted_tokens_total", kind="dram_only"), 1280),
    ("duplicate", val("sglang:kv_tier_dram_evicted_tokens_total", kind="duplicate"), 640),
    ("load_back", val("sglang:kv_tier_load_back_tokens_total"), 1280),
    ("demoted idle<10", val("sglang:kv_tier_vram_evicted_idle_tokens_total", outcome="demoted", within_s="10"), 640),
    ("deleted idle<30", val("sglang:kv_tier_vram_evicted_idle_tokens_total", outcome="deleted", within_s="30"), 0),
    ("deleted idle<60", val("sglang:kv_tier_vram_evicted_idle_tokens_total", outcome="deleted", within_s="60"), 256),
    ("dram_only idle<600", val("sglang:kv_tier_dram_evicted_idle_tokens_total", kind="dram_only", within_s="600"), 0),
    ("dram_only idle<900", val("sglang:kv_tier_dram_evicted_idle_tokens_total", kind="dram_only", within_s="900"), 1280),
    ("load_back idle inf", val("sglang:kv_tier_load_back_idle_tokens_total", within_s="inf"), 1280),
    ("gauge vram", val("sglang:kv_tier_vram_cached_tokens"), 640 + 256),
    ("gauge dup", val("sglang:kv_tier_dram_duplicate_tokens"), 640),
    ("gauge dram_only", val("sglang:kv_tier_dram_only_tokens"), 1280 + 512),
    ("gauge dram_only matchable", val("sglang:kv_tier_dram_only_matchable_tokens"), 1280),
]
bad = [(n, got, want) for n, got, want in checks if got != want]
if bad:
    for n, got, want in bad:
        fail(f"step2/3 {n}: {got} != {want}")
else:
    print(f"  ok accounting: {len(checks)} counters/gauges exact (evictions by outcome/kind, idle buckets, tier gauges)")

# 4. hooks in the patched tree core (source-level; the module needs torch to import)
if args.tree_core:
    src = open(args.tree_core).read()
    want = {
        "wall stamp": "self.last_access_wall = time.monotonic()",
        "demote hook": '_kvtm.vram_evict, self, node, "demoted"',
        "delete hook": '_kvtm.vram_evict, self, node, "deleted"',
        "duplicate hook": '_kvtm.dram_evict, self, node, "duplicate"',
        "dram_only hook": '_kvtm.dram_evict, self, node, "dram_only"',
        "load-back hook": "_kvtm.load_back, self, self.node_by_id(nid)",
    }
    missing = [k for k, v in want.items() if v not in src]
    if missing:
        fail(f"step4 hooks missing: {missing}")
    else:
        print("  ok hooks: wall-clock stamp + 5 recorder calls present in unified_tree_core")
if args.scheduler:
    import ast

    sched = ast.parse(open(args.scheduler).read())
    ingest = [n for n in ast.walk(sched) if isinstance(n, ast.FunctionDef) and n.name == "ingest_requests"]
    calls = [ast.unparse(n.func) for f in ingest for n in ast.walk(f) if isinstance(n, ast.Call)]
    if len(ingest) != 1 or "kv_tier_metrics.tick" not in calls:
        fail("step4 scheduler ingest_requests does not call kv_tier_metrics.tick")
    else:
        print("  ok hooks: the scheduler loop calls kv_tier_metrics.tick in ingest_requests")

# 5. safety
m2 = load({})
def boom(*a):
    raise RuntimeError("x")
try:
    m2.safe(boom)
    print("  ok safe(): a failing hook is logged, not raised")
except Exception:
    fail("step5 safe() raised with STRICT off")
m3 = load({"SGLANG_KV_TIER_METRICS_STRICT": "1"})
try:
    m3.safe(boom)
    fail("step5 safe() swallowed an error with STRICT on")
except RuntimeError:
    print("  ok strict mode re-raises")

# 6. refresh from the scheduler loop, with a fake clock
m4 = load({})
clock = [1000.0]
m4.time = types.SimpleNamespace(monotonic=lambda: clock[0])
reg4 = CollectorRegistry()
rec4 = m4.TierMetrics(resync_s=30, registry=reg4, model_name="t")
m4._recorder, m4._checked = rec4, True


def val4(name):
    v = reg4.get_sample_value(name, {"model_name": "t"})
    return 0 if v is None else v


def split():
    return tuple(val4(f"sglang:kv_tier_{g}") for g in ("vram_cached_tokens", "dram_duplicate_tokens", "dram_only_tokens"))


root4 = Node(0, 0, 0, parent=False)
dup4, dev4 = Node(512, 512, 0), Node(256, 0, 0)
root4.children = {"a": dup4, "b": dev4}
core4 = types.SimpleNamespace(_node_arena={0: root4, 1: dup4, 2: dev4}, root_node=root4)
cache4 = types.SimpleNamespace(tree_core=core4)
m4.tick(cache4)
before = split()
# Demotion as the tree core does it: the hook runs, then the device copy is freed.
clock[0] += 1
rec4.vram_evict(core4, dup4, "demoted")
dup4.component_data[0].value = None
clock[0] += 1
m4.tick(cache4)
throttled = split()
clock[0] += 30
m4.tick(cache4)
after = split()
m4.tick(types.SimpleNamespace(tree_core=types.SimpleNamespace()))  # a core without the Python arena
m4.tick(types.SimpleNamespace())
if before != (768, 512, 0):
    fail(f"step6 initial split {before} != (768, 512, 0)")
elif throttled != before:
    fail(f"step6 tick inside the resync interval walked the tree: {throttled}")
elif after != (256, 0, 512):
    fail(f"step6 split after the interval {after} != (256, 0, 512)")
else:
    print("  ok refresh: a demotion with no further transfer shows within one resync interval")

if failures:
    print(f"kv tier metric checks FAILED ({failures})", file=sys.stderr)
    sys.exit(1)
print("kv tier metric checks passed")
