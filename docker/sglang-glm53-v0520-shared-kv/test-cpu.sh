#!/usr/bin/env bash
# CPU-only checks for the shared KV cache image (no GPU; run with the runc runtime).
set -euo pipefail
cd /sgl-workspace/sglang
export CUDA_VISIBLE_DEVICES=9

# 1. Every patched module imports.
python3 - <<'EOF'
import importlib
for mod in (
    "sglang.srt.mem_cache.unified_radix_cache",
    "sglang.srt.mem_cache.unified_cache.unified_tree_core",
    "sglang.srt.mem_cache.startup_ram_budget",
    "sglang.srt.mem_cache.hicache_storage",
):
    importlib.import_module(mod)
print("1/4 patched modules import")
EOF

# 2. The RAM budget reserves the shared store's remaining growth, and refuses unsafe configs.
python3 - <<'EOF'
import os, tempfile
from sglang.srt.mem_cache.startup_ram_budget import shared_store_growth_reserve

GiB = 1024**3
d = tempfile.mkdtemp()
with open(os.path.join(d, "page.bin"), "wb") as f:
    f.write(b"\0" * (3 * 1024**2))
env = {
    "SGLANG_HICACHE_SHARED_STORE_BUDGET": "4GiB",
    "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": d,
    "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": "2Gi",
}
os.environ.update(env)
budget, used, reserve = shared_store_growth_reserve("file")
assert (budget, used, reserve) == (4 * GiB, 3 * 1024**2, 4 * GiB - 3 * 1024**2), (budget, used, reserve)


def refused(**overrides):
    saved = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        shared_store_growth_reserve("file")
        return False
    except ValueError:
        return True
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


try:
    shared_store_growth_reserve("mooncake")
    raise AssertionError("non-file backend accepted")
except ValueError:
    pass
assert refused(SGLANG_HICACHE_SHARED_STORE_BUDGET=None), "missing store budget accepted"
assert refused(SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=None), "missing per-replica cap accepted"
assert refused(SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE="8Gi"), "per-replica cap above the store budget accepted"
assert refused(SGLANG_HICACHE_SHARED_STORE_BUDGET="50%"), "percentage store budget accepted"
print("2/4 shared-store RAM reserve")
EOF

# 3. Two TP4 replicas of the same model produce identical storage keys, so they share entries,
#    and a page written by one is read back byte-exact by the other.
python3 - <<'EOF'
import os, tempfile
import torch
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig

d = tempfile.mkdtemp()
os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = d
os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)


def replica():
    cfg = HiCacheStorageConfig(tp_rank=0, tp_size=4, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
                               is_mla_model=True, enable_storage_metrics=False, is_page_first_layout=True,
                               model_name="z-ai/glm-5.3-flash")
    return HiCacheFile(cfg)


r1, r2 = replica(), replica()
assert r1._get_suffixed_key("abc") == r2._get_suffixed_key("abc")
page = torch.randint(0, 255, (4096,), dtype=torch.uint8)
assert r1.set("page0", page)
out = r2.get("page0", torch.empty_like(page))
assert out is not None and torch.equal(out, page)
print("3/4 cross-replica file store round trip")
EOF

# 4. The storage hash chain runs at the 64-token transfer page even when the radix tree page is 256.
python3 - <<'EOF'
import inspect
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.unified_cache import unified_tree_core as utc

tree_src = inspect.getsource(utc)
assert "self.hash_page_size = params.page_size" in tree_src
assert "compute_node_hash_values(node, self.hash_page_size)" in tree_src
assert "compute_node_hash_values(new_node, self.hash_page_size)" in tree_src
cache_src = inspect.getsource(urc)
assert "self.tree_core.hash_page_size = self._transfer_page_size" in cache_src
assert 'reason="tree_page_align"' in cache_src
# The startup guard is lifted (cache host-memory mode); attaching a store at runtime stays refused.
assert "storage hashes and transfers require matching page sizes" not in cache_src
assert "Compressed DSA storage supports --hicache-host-memory-mode cache only." in cache_src
print("4/4 storage hash chain at the transfer page")
EOF
echo "test-cpu OK"
