# Threaded batch reads (SGLANG_KVSHARE_READ_THREADS) land every page in its own host slot exactly
# like the sequential path, reuse one scratch page per pool per thread, and report misses in order.
import os, tempfile
import torch
from types import SimpleNamespace
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig, PoolName

d = tempfile.mkdtemp()
os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = d
os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
cfg = HiCacheStorageConfig(tp_rank=0, tp_size=2, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
                           is_mla_model=True, enable_storage_metrics=False, is_page_first_layout=True,
                           model_name="z-ai/glm-5.3-flash")
PAGE, N = 64, 300
class Pool:
    page_size = PAGE
    def __init__(self): self.slots = torch.zeros(4 * N, 128, dtype=torch.uint8); self.allocs = 0
    def get_dummy_flat_data_page(self):
        self.allocs += 1; return torch.empty(128, dtype=torch.uint8)
    def get_data_page(self, off, flat=True): return torch.full((128,), (off // PAGE) % 251, dtype=torch.uint8)
    def set_from_flat_data_page(self, off, page): self.slots[off // PAGE].copy_(page)
src, w = Pool(), HiCacheFile(cfg)
keys = [f"h{i}" for i in range(N)]
for i, k in enumerate(keys):
    assert w._write_page(PoolName.KV, k, src, i * PAGE)
os.remove(os.path.join(d, w._get_component_key("h17") + ".bin"))  # one miss
def restore(threads):
    os.environ["SGLANG_KVSHARE_READ_THREADS"] = str(threads)
    r, dst = HiCacheFile(cfg), Pool()
    idx = torch.arange(N * PAGE) + 2 * N * PAGE  # land in a different region
    t = SimpleNamespace(name=PoolName.KV, keys=keys, host_indices=idx)
    r.registered_pools = {PoolName.KV: dst}
    return r._batch_io_v2([t], r._read_page)[PoolName.KV], dst
seq, a = restore(1)
par, b = restore(8)
assert seq == par and seq.count(False) == 1 and seq[17] is False
assert torch.equal(a.slots, b.slots)
assert all(int(b.slots[2 * N + i][0]) == i % 251 for i in range(N) if i != 17)
assert a.allocs == 1 and b.allocs <= 8, (a.allocs, b.allocs)
print("9 threaded batch reads match sequential; scratch page reused per thread")
