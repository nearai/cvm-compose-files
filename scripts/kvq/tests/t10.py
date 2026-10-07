# The controller's v1 KV restore delegates to HiCacheFile.kvshare_read_into_host: same contract as
# _generic_page_get (contiguous prefix count, stop at first miss), no per-page pinned allocation,
# self/peer attribution, and v1 batch_set writes counted once.
import os, tempfile, inspect
import torch
from types import SimpleNamespace
from prometheus_client import REGISTRY
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.managers import cache_controller as cc

assert "kvshare_read_into_host" in inspect.getsource(cc.HiCacheController._generic_page_get)
d = tempfile.mkdtemp()
os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = d
os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
cfg = HiCacheStorageConfig(tp_rank=0, tp_size=2, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
                           is_mla_model=True, enable_storage_metrics=False, is_page_first_layout=True,
                           model_name="z-ai/glm-5.3-flash")
PAGE, N = 64, 50
class Pool:
    def __init__(self): self.slots = torch.zeros(4 * N, 128, dtype=torch.uint8); self.allocs = 0
    def get_dummy_flat_data_page(self):
        self.allocs += 1; return torch.empty(128, dtype=torch.uint8)
    def set_from_flat_data_page(self, off, page): self.slots[int(off) // PAGE].copy_(page)
def val(name, **labels):
    v = REGISTRY.get_sample_value(name, labels); return 0.0 if v is None else v
A, B = HiCacheFile(cfg), HiCacheFile(cfg)
keys = [f"p{i}" for i in range(N)]
assert A.batch_set(keys, [torch.full((128,), i % 251, dtype=torch.uint8) for i in range(N)])  # v1 write
assert val("sglang:kvshare_storage_written_tokens_total") == N * PAGE
assert B.batch_set(keys[:5], [torch.zeros(128, dtype=torch.uint8)] * 5)  # already present: not counted
assert val("sglang:kvshare_storage_written_tokens_total") == N * PAGE
os.remove(os.path.join(d, A._get_component_key("p30") + ".bin"))
op = SimpleNamespace(is_terminated=lambda: False, request_id="r")
idx = torch.arange(N * PAGE)
for be, src in ((B, "peer"), (A, "self")):
    pool = Pool()
    got = be.kvshare_read_into_host(op, keys, idx, pool, PAGE)
    assert got == 30, got
    assert all(int(pool.slots[i][0]) == i % 251 for i in range(30))
    assert pool.allocs <= 9, pool.allocs  # 8 reader threads + the caller sizing the page
    assert val("sglang:kvshare_storage_hit_tokens_total", source=src) == 30 * PAGE, src
term = SimpleNamespace(is_terminated=lambda: True, request_id="r")
assert A.kvshare_read_into_host(term, keys, idx, Pool(), PAGE) == 0
print("10 controller v1 restore uses the fast reader; contiguous-prefix contract and attribution hold")
