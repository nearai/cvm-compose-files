import os, tempfile
from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

d = tempfile.mkdtemp()
shared = os.path.join(d, "kvpage_m.bin")
with open(shared, "wb") as f:
    f.write(b"\0" * 4096)
os.environ["SGLANG_HICACHE_FILE_BACKEND_SIDECAR_MAX_FRACTION"] = "0.25"
cfg = {"max_size": "64Ki"}
owner = LRUFileEvictor(d, "_m", tp_rank=0, is_mla_model=True, extra_config=cfg)
rank1 = LRUFileEvictor(d, "_m", tp_rank=1, is_mla_model=True, extra_config=cfg)
assert owner.enabled and rank1.enabled
assert owner.max_size_bytes == 64 * 1024 and rank1.max_size_bytes == 16 * 1024
assert len(owner._lru) == 1, "owner adopts the existing shared file"
assert len(rank1._lru) == 0, "non-owner must not adopt shared files"
for i in range(8):  # 8 x 4 KiB of this rank's own sidecars against a 16 KiB cap
    key = f"s{i}.mamba_m"
    assert rank1.reserve(key, 4096, key=key), "non-owner refused its own sidecar write"
    with open(os.path.join(d, f"{key}.bin"), "wb") as f:
        f.write(b"\1" * 4096)
    rank1.commit(key)
assert rank1._total_bytes <= 16 * 1024, rank1._total_bytes
assert os.path.exists(shared), "non-owner evicted a shared file it does not own"
print("5/5 non-owner ranks bound their own sidecars")
