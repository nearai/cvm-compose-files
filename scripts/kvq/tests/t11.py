# Direct reads into page_first_direct host slots: byte-identical to the scratch+copy path, no copy.
import os, tempfile
import torch
from types import SimpleNamespace
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig

d = tempfile.mkdtemp()
os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = d
os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
cfg = HiCacheStorageConfig(tp_rank=0, tp_size=2, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
                           is_mla_model=True, enable_storage_metrics=False, is_page_first_layout=True,
                           model_name="z-ai/glm-5.3-flash")
PAGE, N, L, DIM = 64, 40, 3, 8
class Pool:  # page_first_direct: kv_buffer[pages, layers, page, 1, dim]
    layout, page_size = "page_first_direct", PAGE
    def __init__(self): self.kv_buffer = torch.zeros(N + 4, L, PAGE, 1, DIM, dtype=torch.bfloat16); self.copies = 0
    def get_dummy_flat_data_page(self): return torch.zeros(L, PAGE, 1, DIM, dtype=torch.bfloat16).flatten()
    def get_data_page(self, index, flat=True): return self.kv_buffer[index // PAGE : index // PAGE + 1].flatten()
    def set_from_flat_data_page(self, index, page):
        self.copies += 1; self.kv_buffer[index // PAGE : index // PAGE + 1] = page.reshape(1, L, PAGE, 1, DIM)
src = Pool(); src.kv_buffer.normal_()
w = HiCacheFile(cfg)
keys = [f"d{i}" for i in range(N)]
assert w.batch_set(keys, [src.get_data_page(i * PAGE) for i in range(N)])
op = SimpleNamespace(is_terminated=lambda: False, request_id="r")
idx = torch.arange(N * PAGE)
for direct in ("1", "0"):
    os.environ["SGLANG_KVSHARE_DIRECT_READ"] = direct
    dst = Pool()
    assert w.kvshare_read_into_host(op, keys, idx, dst, PAGE) == N
    assert torch.equal(dst.kv_buffer[:N], src.kv_buffer[:N]), direct
    assert dst.copies == (0 if direct == "1" else N), (direct, dst.copies)
print("11 direct slot reads are byte-identical to scratch+copy and skip the copy")
