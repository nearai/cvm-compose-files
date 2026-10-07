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
print("3/5 cross-replica file store round trip")
