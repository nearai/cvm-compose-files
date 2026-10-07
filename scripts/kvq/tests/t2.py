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
print("2/5 shared-store RAM reserve")
