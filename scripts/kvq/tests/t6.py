# Port additions: (a) mamba sidecar keys are per TP rank under MLA, KV keys stay shared;
# (b) a non-owner rank never adopts (so never evicts) a shared file it only read;
# (c) the compressed-DSA storage guard refuses the Rust tree core; the trim precedes alloc.
import inspect, os, tempfile
import torch
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig, PoolName
from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

d = tempfile.mkdtemp()
os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = d
os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
def rank(r, replica_tp=2):
    cfg = HiCacheStorageConfig(tp_rank=r, tp_size=replica_tp, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
                               is_mla_model=True, enable_storage_metrics=False, is_page_first_layout=True,
                               model_name="z-ai/glm-5.3-flash")
    return HiCacheFile(cfg)
a0, a1, b0, b1 = rank(0), rank(1), rank(0), rank(1)  # two TP2 replicas
assert a0._get_component_key("h") == a1._get_component_key("h") == b0._get_component_key("h"), "KV must be shared"
assert a0._get_component_key("h", PoolName.MAMBA) != a1._get_component_key("h", PoolName.MAMBA), "mamba must be per rank"
assert a0._get_component_key("h", PoolName.MAMBA) == b0._get_component_key("h", PoolName.MAMBA), "same rank shares across replicas"
assert a1._get_component_key("h", PoolName.INDEXER) == a0._get_component_key("h", PoolName.INDEXER), "indexer stays shared"
for be in (a0, a1):  # read/write path and existence path agree on the file name
    assert be._get_suffixed_key(be._log_key(PoolName.MAMBA, "h")) == be._get_component_key("h", PoolName.MAMBA)
s0, s1 = torch.full((64,), 1, dtype=torch.uint8), torch.full((64,), 2, dtype=torch.uint8)
assert a0.set(a0._log_key(PoolName.MAMBA, "h"), s0) and a1.set(a1._log_key(PoolName.MAMBA, "h"), s1)
assert torch.equal(b1.get(b1._log_key(PoolName.MAMBA, "h"), torch.empty(64, dtype=torch.uint8)), s1)
assert torch.equal(b0.get(b0._log_key(PoolName.MAMBA, "h"), torch.empty(64, dtype=torch.uint8)), s0)
print("6a mamba sidecar keys are per TP rank; KV and indexer keys shared")

d2 = tempfile.mkdtemp()
shared = os.path.join(d2, "kvpage_m.bin")
with open(shared, "wb") as f:
    f.write(b"\0" * 8192)
r1 = LRUFileEvictor(d2, "_m", tp_rank=1, is_mla_model=True, extra_config={"max_size": "16Ki"})
r1.touch("kvpage_m", shared)  # a read of a shared KV page
assert "kvpage_m" not in r1._lru, "non-owner adopted a shared file it only read"
for i in range(8):
    key = f"s{i}.mamba_tp1_2_m"
    assert r1.reserve(key, 4096, key=key)
    with open(os.path.join(d2, f"{key}.bin"), "wb") as f:
        f.write(b"\1" * 4096)
    r1.commit(key)
    r1.touch(key, os.path.join(d2, f"{key}.bin"))
assert os.path.exists(shared) and r1._total_bytes <= 4 * 1024, r1._total_bytes
r0 = LRUFileEvictor(d2, "_m", tp_rank=0, is_mla_model=True, extra_config={"max_size": "1Mi"})
assert "kvpage_m" in r0._lru, "owner must still adopt shared files"
print("6b non-owner never adopts or evicts shared files")

from sglang.srt.mem_cache import unified_radix_cache as urc
src = inspect.getsource(urc.UnifiedRadixCache)
assert 'self._tree_core_backend != "python"' in src
assert src.index('reason="tree_page_align"') < src.index("            alloc_len = hit_tokens\n")
assert "keep_pages = completed_tokens // self._transfer_page_size" in src
assert "// self.page_size]" not in src.split("def _try_alloc_storage_hit")[1].split("def _drain_and_alloc_storage_hit")[0]
print("6c guard, trim placement and transfer-page conversions present")
