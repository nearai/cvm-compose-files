# Keys the tree writes (node hashes at hash_page_size=64 over 256-token nodes, incl. after a split)
# equal the keys the controller queries on prefetch (get_hash_str over the fetched tokens at 64,
# chained from the anchor node's last hash). At page 256 (unpatched) they would not.
from types import SimpleNamespace as NS
from sglang.srt.mem_cache.radix_cache import RadixKey
import hashlib
import sglang.srt.mem_cache.utils as U
from sglang.srt.mem_cache.utils import compute_node_hash_values, get_hash_str, split_node_hash_value
# macOS: the native (C++, Linux-only) page hash is replaced by a pure-Python chained per-page sha256
# with the same contract (one hash per page, each seeded by the previous page's digest).
def _py_hash(token_ids, prior_digest, page_size):
    raw = list(getattr(token_ids, "token_ids", token_ids))[: len(token_ids)]
    out, prior = [], prior_digest
    for i in range(0, len(raw) - len(raw) % page_size, page_size):
        h = hashlib.sha256((prior or b"") + repr(raw[i : i + page_size]).encode()).digest()
        out.append(h.hex()); prior = h
    return out
U.get_native_hash = _py_hash
toks = list(range(1000, 1000 + 1024))
root = NS(parent=None, key=RadixKey([]), hash_value=[])
a = NS(parent=root, key=RadixKey(toks[:512]), hash_value=None)
a.hash_value = compute_node_hash_values(a, 64)
assert len(a.hash_value) == 8
b = NS(parent=a, key=RadixKey(toks[512:]), hash_value=None)
b.hash_value = compute_node_hash_values(b, 64)
# controller side: replica 2 has nothing on device, queries the full prompt from the root
q = get_hash_str(RadixKey(toks), None, page_size=64)
assert q == a.hash_value + b.hash_value
# controller side, anchored at node a (prefix already on device): chained from a's last hash
assert get_hash_str(RadixKey(toks[512:]), a.hash_value[-1], page_size=64) == b.hash_value
# a split at a 256-token tree page keeps the chain (split_pages = 256 // 64 = 4)
left, right = split_node_hash_value(a.hash_value, 256, 64)
assert left == q[:4] and right == q[4:8]
# unpatched: tree hashes at 256 cover 4x the tokens per key -> different keys than the 64-page query
assert compute_node_hash_values(a, 256)[0] not in q
print("7/7 tree write keys == controller prefetch keys at the 64-token transfer page")
