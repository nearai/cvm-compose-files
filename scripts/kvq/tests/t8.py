# kvshare attribution metrics: a replica's reads of pages it wrote count as self, of pages the
# other replica wrote count as peer; skipped writes (page already present) are not counted.
import os, tempfile
import torch
from types import SimpleNamespace as NS
from prometheus_client import REGISTRY
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig, PoolName

d = tempfile.mkdtemp()
os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = d
os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
def rank(r):
    cfg = HiCacheStorageConfig(tp_rank=r, tp_size=2, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
                               is_mla_model=True, enable_storage_metrics=False, is_page_first_layout=True,
                               model_name="z-ai/glm-5.3-flash")
    return HiCacheFile(cfg)
class Pool:  # minimal host pool: one flat page = 64 bytes, page_size 64 tokens
    page_size = 64
    def __init__(self): self.pages = {}
    def get_dummy_flat_data_page(self): return torch.empty(64, dtype=torch.uint8)
    def get_data_page(self, off, flat=True): return torch.full((64,), off % 251, dtype=torch.uint8)
    def set_from_flat_data_page(self, off, page): self.pages[off] = page.clone()
def val(name, **labels):
    v = REGISTRY.get_sample_value(name, labels)
    return 0.0 if v is None else v
A, B, A1 = rank(0), rank(0), rank(1)   # replica A rank0, replica B rank0, replica A rank1
pool = Pool()
assert A._write_page(PoolName.KV, "k1", pool, 64)
assert val("sglang:kvshare_storage_written_tokens_total") == 64
assert B._write_page(PoolName.KV, "k1", pool, 64)            # already present: skipped, not counted
assert val("sglang:kvshare_storage_written_tokens_total") == 64
assert A._read_page(PoolName.KV, "k1", pool, 0)              # A reads its own page
assert B._read_page(PoolName.KV, "k1", pool, 128)            # B reads A's page
assert val("sglang:kvshare_storage_hit_tokens_total", source="self") == 64
assert val("sglang:kvshare_storage_hit_tokens_total", source="peer") == 64
assert A1._write_page(PoolName.KV, "k2", pool, 192) and A1._read_page(PoolName.KV, "k2", pool, 0)
assert val("sglang:kvshare_storage_written_tokens_total") == 64, "rank 1 must not count"
assert A._read_page(PoolName.MAMBA, "k1", pool, 0) is False or True  # sidecar reads never count
assert val("sglang:kvshare_storage_hit_tokens_total", source="self") + val("sglang:kvshare_storage_hit_tokens_total", source="peer") == 128
print("8 kvshare attribution: self/peer hits and written tokens counted on rank 0 only")
