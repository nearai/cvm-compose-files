"""CPU checks for the ghost prefix cache (sglang.srt.observability.ghost_cache).

Run inside the image: python3 test_ghost_cache.py --module /sgl-workspace/sglang/python/sglang/srt/observability/ghost_cache.py
or locally against a copy of the module. Exits non-zero on any failure.

1. Exactness: with sampling off, the predicted hit tokens at every cache size equal a brute-force
   LRU simulation of a page cache over the same chained page hashes (inserting prompt + output).
2. Sampling: at 1/16 the prediction stays within 5 percentage points of the exact hit rate at the
   sizes that matter (where the exact rate is between 5% and 95%).
3. Compulsory misses: lookup - seen-at-any-distance equals the number of never-seen prompt pages.
4. Confidentiality: only 16-byte keyed digests are retained; no token id survives in the object
   graph; the same tokens hash differently under a different key.
5. Bounded memory: the tracked set never exceeds max_pages.
6. Inert by default: without SGLANG_GHOST_CACHE=1 the hook does nothing and starts no thread.
7. Shared key: replicas racing to create SGLANG_GHOST_CACHE_KEY_FILE all end up with one 32-byte key
   (mode 0600), so the same prefix has the same digest on every replica.
8. Wire format: digests round-trip through the aggregator messages, large requests split into
   datagrams under 64 KB, malformed messages are rejected.
9. Pooled accounting (ghost_aggregator.py): two replicas with conversation affinity and a failover
   halfway through. With sampling off, the pooled hit tokens at every size equal a brute-force LRU
   over the interleaved stream, and "seen before only on the other replica" equals a brute-force
   count; it is zero before the failover and positive after.
10. End to end (needs prometheus_client, i.e. inside the image): two engine recorders send over a
   real unix datagram socket to a running aggregator, whose /metrics match the direct computation.
"""

import argparse
import collections
import hashlib
import importlib.util
import os
import random
import sys
import threading

ap = argparse.ArgumentParser()
ap.add_argument("--module", required=True)
ap.add_argument("--aggregator", required=True)
args = ap.parse_args()


def load(env):
    for k in list(os.environ):
        if k.startswith("SGLANG_GHOST_CACHE"):
            del os.environ[k]
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(f"ghost_cache_{len(env)}_{random.random()}", args.module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gc_mod = load({})
PAGE = gc_mod.PAGE_TOKENS
failures = 0


def fail(msg):
    global failures
    failures += 1
    print("FAILED " + msg, file=sys.stderr)


def workload(seed, n_conv=300, n_req=6000, shared_prefix=2 * PAGE * 8):
    """Multi-turn conversations over a few shared system prompts; skewed conversation popularity."""
    rng = random.Random(seed)
    systems = [[rng.randrange(1, 150000) for _ in range(shared_prefix)] for _ in range(4)]
    convs = []
    for c in range(n_conv):
        convs.append(list(systems[c % 4]) + [rng.randrange(1, 150000) for _ in range(rng.randrange(50, 3000))])
    weights = [1.0 / (i + 1) ** 0.8 for i in range(n_conv)]
    for _ in range(n_req):
        c = rng.choices(range(n_conv), weights)[0]
        prompt = list(convs[c])
        output = [rng.randrange(1, 150000) for _ in range(rng.randrange(20, 600))]
        yield prompt, output
        convs[c] = prompt + output + [rng.randrange(1, 150000) for _ in range(rng.randrange(20, 2000))]
        if rng.random() < 0.03:  # conversation ends; a new one starts
            convs[c] = list(systems[rng.randrange(4)]) + [rng.randrange(1, 150000) for _ in range(rng.randrange(50, 3000))]


def brute_force(requests, key, capacities_pages):
    """Exact LRU page cache per capacity over the same digests: returns hit pages per capacity."""
    ghost = gc_mod.GhostCache(sample=1, key=key, distance_buckets=(1,))
    caches = {c: collections.OrderedDict() for c in capacities_pages}
    hits = {c: 0 for c in capacities_pages}
    lookups = 0
    for prompt, output in requests:
        digests = ghost._chain(prompt + output)
        n_prompt = len(prompt) // PAGE
        for c, lru in caches.items():
            for i, d in enumerate(digests):
                if d in lru:
                    lru.move_to_end(d)
                    if i < n_prompt:
                        hits[c] += 1
                else:
                    lru[d] = True
                    if len(lru) > c:
                        lru.popitem(last=False)
        lookups += n_prompt
    return hits, lookups


key = hashlib.sha256(b"test-key").digest()
requests = list(workload(1))
caps_pages = (64, 256, 1024, 4096, 16384, 65536)
buckets = tuple(c * PAGE for c in caps_pages)

# 1. exactness
exact_hits, exact_lookups = brute_force(requests, key, caps_pages)
g1 = gc_mod.GhostCache(sample=1, key=key, distance_buckets=buckets)
for p, o in requests:
    g1.observe(p, o, wall=0.0)
if g1.lookup_tokens != exact_lookups * PAGE:
    fail(f"step1 lookup tokens {g1.lookup_tokens} != {exact_lookups * PAGE}")
for c, b in zip(caps_pages, buckets):
    got, want = g1.hit_tokens_at(b), exact_hits[c] * PAGE
    if got != want:
        fail(f"step1 capacity {c} pages: ghost {got} != brute force {want}")
    else:
        print(f"  ok exact: {c:6d} pages -> hit rate {want / (exact_lookups * PAGE):.3f}")

# 2. sampling accuracy
g16 = gc_mod.GhostCache(sample=16, key=key, distance_buckets=buckets)
for p, o in requests:
    g16.observe(p, o, wall=0.0)
for c, b in zip(caps_pages, buckets):
    exact = exact_hits[c] / exact_lookups
    est = g16.hit_tokens_at(b) / g16.lookup_tokens
    if 0.05 < exact < 0.95 and abs(est - exact) > 0.05:
        fail(f"step2 capacity {c} pages: sampled {est:.3f} vs exact {exact:.3f}")
    print(f"  {'ok' if abs(est - exact) <= 0.05 or not 0.05 < exact < 0.95 else '!!'} sampled 1/16: {c:6d} pages -> {est:.3f} (exact {exact:.3f})")

# 3. compulsory misses
seen, first = set(), 0
chain = gc_mod.GhostCache(sample=1, key=key)._chain
for p, o in requests:
    ds = chain(p + o)
    for i, d in enumerate(ds):
        if i < len(p) // PAGE and d not in seen:
            first += 1
        seen.add(d)
compulsory = (g1.lookup_tokens - g1.reused_within[-1]) // PAGE
if compulsory != first:
    fail(f"step3 compulsory pages {compulsory} != never-seen pages {first}")
else:
    print(f"  ok compulsory misses: {first} pages ({first / exact_lookups:.1%} of lookups)")

# 4. confidentiality
token_values = set()
for p, o in requests[:200]:
    token_values.update(p)
    token_values.update(o)
bad = [k for k in g1.lru.last if not (isinstance(k, bytes) and len(k) == 16)]
if bad:
    fail(f"step4 non-digest keys retained: {bad[:3]}")
leaks = [v for v in list(g1.__dict__.values()) + list(g1.lru.__dict__.values()) if isinstance(v, (list, tuple)) and v and all(isinstance(x, int) for x in v) and len(set(v) & token_values) > len(v) // 2 and len(v) > 64]
if leaks:
    fail("step4 an int sequence resembling token ids is retained")
same = gc_mod.GhostCache(sample=1, key=key)._chain(requests[0][0])
other = gc_mod.GhostCache(sample=1, key=hashlib.sha256(b"other").digest())._chain(requests[0][0])
fresh = gc_mod.GhostCache(sample=1)._chain(requests[0][0])
if same != gc_mod.GhostCache(sample=1, key=key)._chain(requests[0][0]) or set(same) & set(other) or set(same) & set(fresh):
    fail("step4 digests are not keyed")
else:
    print("  ok confidentiality: only 16-byte keyed digests retained; a different key shares no digest")

# 5. bounded memory
small = gc_mod.GhostCache(sample=1, max_pages=1024, key=key, distance_buckets=buckets)
peak = 0
for p, o in requests:
    small.observe(p, o, wall=0.0)
    peak = max(peak, len(small.lru))
if peak > small.lru.slots or len(small.lru) > 1024 + small.lru.slots:
    fail(f"step5 tracked set grew to {peak}")
small.lru._compact()
if len(small.lru) > 1024:
    fail(f"step5 tracked set {len(small.lru)} > max 1024 after compaction")
else:
    print(f"  ok bounded memory: tracked set <= max_pages after compaction (peak {peak}, slots {small.lru.slots})")

# 6. inert by default
threads_before = {t.name for t in threading.enumerate()}


class Req:
    origin_input_ids = requests[0][0]
    output_ids = requests[0][1]
    cached_tokens = 0
    finished_reason = object()

    def finished(self):
        return True


off = load({})
off.observe_finished_req(Req())
if off._recorder is not None or "sglang-ghost-cache" in {t.name for t in threading.enumerate()} - threads_before:
    fail("step6 hook did work while disabled")
else:
    print("  ok inert by default: no recorder, no thread")

# 7. shared key
import multiprocessing
import stat
import tempfile

tmpdir = tempfile.mkdtemp()
key_path = os.path.join(tmpdir, "ghost.key")


def _load(q):
    q.put(gc_mod.load_shared_key(key_path))


ctx = multiprocessing.get_context("fork")
q = ctx.Queue()
procs = [ctx.Process(target=_load, args=(q,)) for _ in range(8)]
for pr in procs:
    pr.start()
keys = [q.get(timeout=30) for _ in procs]
for pr in procs:
    pr.join()
keys.append(gc_mod.load_shared_key(key_path))
mode = stat.S_IMODE(os.stat(key_path).st_mode)
leftovers = [f for f in os.listdir(tmpdir) if f != "ghost.key"]
if len(set(keys)) != 1 or len(keys[0]) != 32 or mode != 0o600 or leftovers:
    fail(f"step7 shared key: {len(set(keys))} distinct keys, mode {oct(mode)}, leftovers {leftovers}")
else:
    a = gc_mod.GhostCache(sample=1, key=keys[0])._chain(requests[0][0])
    b = gc_mod.GhostCache(sample=1, key=gc_mod.load_shared_key(key_path))._chain(requests[0][0])
    if a != b:
        fail("step7 same key file gave different digests")
    else:
        print("  ok shared key: 9 racing loaders got one 32-byte key (mode 0600); digests match across replicas")

# 8. wire format
recs = [(i % 3 != 0, hashlib.blake2b(str(i).encode(), digest_size=16).digest()) for i in range(8000)]
msgs = gc_mod.encode_messages("r1", 16, recs)
back = []
for m in msgs:
    rep, smp, rr = gc_mod.decode_records(m)
    assert rep == "r1" and smp == 16
    back += rr
rejected = 0
for bad in (b"junk" * 10, msgs[0][:-5]):
    try:
        gc_mod.decode_records(bad)
    except Exception:
        rejected += 1
if back != recs or max(len(m) for m in msgs) >= 65536 or rejected != 2:
    fail(f"step8 wire: roundtrip {back == recs}, max {max(len(m) for m in msgs)} bytes, rejected {rejected}/2")
else:
    print(f"  ok wire format: {len(recs)} records in {len(msgs)} datagrams (max {max(len(m) for m in msgs)} bytes); malformed rejected")

# 9. pooled accounting vs brute force, two replicas, affinity, failover at the halfway point
sys.modules.setdefault("sglang", type(sys)("sglang"))
sys.modules.setdefault("sglang.srt", type(sys)("sglang.srt"))
sys.modules.setdefault("sglang.srt.observability", type(sys)("sglang.srt.observability"))
sys.modules["sglang.srt.observability.ghost_cache"] = gc_mod
spec = importlib.util.spec_from_file_location("ghost_aggregator_under_test", args.aggregator)
agg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg_mod)

routed = []
half = len(requests) // 2
for idx, (p, o) in enumerate(requests):
    # conversation affinity: the first tokens after the shared 1024-token system prompt identify the
    # conversation, so conversations sharing a system prompt land on both replicas
    replica = "r1" if int.from_bytes(hashlib.sha256(repr(p[1024:1088]).encode()).digest()[:4], "little") & 1 else "r2"
    if idx >= half and replica == "r1":
        replica = "r2"  # r1 is down: everything goes to r2
    routed.append((replica, p, o))
chain = gc_mod.GhostCache(sample=1, key=key)._chain
pool = agg_mod.PoolAccounting(sample=1, max_pages=10 ** 7, distance_buckets=buckets)
bf = {c: collections.OrderedDict() for c in caps_pages}
bf_hits = {r: {c: 0 for c in caps_pages} for r in ("r1", "r2")}
seen_by = {}
bf_other = {"r1": 0, "r2": 0}
other_before = 0
for idx, (replica, p, o) in enumerate(routed):
    ds = chain(p + o)
    n_prompt = len(p) // PAGE
    pool.process(replica, [(i < n_prompt, d) for i, d in enumerate(ds)], wall=0.0)
    for c, lru in bf.items():
        for i, d in enumerate(ds):
            if d in lru:
                lru.move_to_end(d)
                if i < n_prompt:
                    bf_hits[replica][c] += 1
            else:
                lru[d] = True
                if len(lru) > c:
                    lru.popitem(last=False)
    for i, d in enumerate(ds):
        who = seen_by.setdefault(d, set())
        if i < n_prompt and who and replica not in who:
            bf_other[replica] += 1
            if idx < half:
                other_before += 1
        who.add(replica)
ok9 = True
for r in ("r1", "r2"):
    t = pool.totals[r]
    for j, c in enumerate(caps_pages):
        if t.within[j] != bf_hits[r][c] * PAGE:
            fail(f"step9 {r} pooled hits at {c} pages: {t.within[j]} != {bf_hits[r][c] * PAGE}")
            ok9 = False
    if t.other_only != bf_other[r] * PAGE:
        fail(f"step9 {r} other-replica-only: {t.other_only} != {bf_other[r] * PAGE}")
        ok9 = False
if bf_other["r2"] <= other_before or other_before == 0:
    fail(f"step9 expected cross-replica tokens both before ({other_before}) and after the failover ({bf_other['r2']})")
    ok9 = False
if ok9:
    print(
        f"  ok pooled accounting == brute force at {len(caps_pages)} sizes; other-replica-only tokens: "
        f"r1 {pool.totals['r1'].other_only}, r2 {pool.totals['r2'].other_only} "
        f"({other_before * PAGE} before the failover: shared system prompts; the rest after r1 failed over)"
    )

# 10. end to end over a real socket (image only)
try:
    import prometheus_client  # noqa: F401
    have_prom = True
except ImportError:
    have_prom = False
if not have_prom:
    print("  skip end-to-end: prometheus_client not installed here (runs inside the image)")
else:
    import socket as _socket
    import time as _time
    import urllib.request

    sock_path = os.path.join(tmpdir, "agg.sock")
    s_ = _socket.socket(); s_.bind(("127.0.0.1", 0)); port = s_.getsockname()[1]; s_.close()
    th = threading.Thread(target=agg_mod.serve, args=(sock_path, port, 4, 10 ** 6, "test-model"), daemon=True)
    th.start()
    for _ in range(100):
        if os.path.exists(sock_path):
            break
        _time.sleep(0.05)
    shared = gc_mod.load_shared_key(key_path)
    # one recorder per engine process in production; two here, so each gets its own metrics registry
    rec = {
        r: gc_mod._Recorder(sample=4, max_pages=10 ** 6, queue_size=10000, key=shared, socket_path=sock_path, replica=r, registry=prometheus_client.CollectorRegistry())
        for r in ("r1", "r2")
    }
    direct = agg_mod.PoolAccounting(sample=4, max_pages=10 ** 6)
    dchain = gc_mod.GhostCache(sample=4, key=shared)
    sub = routed[:1500]
    for replica, p, o in sub:
        rec[replica].submit(list(p), list(o), len(p), 0)
        # keep global order identical to the direct computation: wait for this request to be sent
        while not rec[replica].queue.empty():
            _time.sleep(0.001)
        _time.sleep(0.002)
        tracked = [(i < len(p) // PAGE, d) for i, d in enumerate(dchain._chain(p + o)) if dchain._tracked(d)]
        direct.process(replica, tracked)
    _time.sleep(1.0)
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()

    def metric(name, **lab):
        for line in body.splitlines():
            if line.startswith(name + "{") and all(f'{k}="{v}"' in line for k, v in lab.items()):
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    ok10 = True
    for r in ("r1", "r2"):
        t = direct.totals[r]
        got = (metric("sglang:ghost_pool_lookup_tokens_total", replica=r), metric("sglang:ghost_pool_reused_tokens_total", replica=r, within="inf"), metric("sglang:ghost_pool_other_replica_only_tokens_total", replica=r))
        want = (t.lookup, t.within[-1], t.other_only)
        if tuple(int(x) for x in got) != want:
            fail(f"step10 {r} /metrics {got} != direct {want}")
            ok10 = False
    bad = metric("sglang:ghost_pool_bad_messages_total")
    if ok10 and bad == 0:
        print(f"  ok end to end: two recorders -> unix socket -> aggregator /metrics match the direct computation ({int(metric('sglang:ghost_pool_messages_total'))} messages, 0 bad)")
    elif bad:
        fail(f"step10 {bad} bad messages")

if failures:
    print(f"ghost cache checks FAILED ({failures})", file=sys.stderr)
    sys.exit(1)
print("ghost cache checks passed")
