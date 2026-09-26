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

if failures:
    print(f"ghost cache checks FAILED ({failures})", file=sys.stderr)
    sys.exit(1)
print("ghost cache checks passed")
