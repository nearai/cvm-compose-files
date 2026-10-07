import os, tempfile
import sglang.srt.mem_cache.startup_ram_budget as rb

# A replica starting while its peers write: a temp file is listed, then renamed away before it is
# stat'ed (gpu04 r4, 2026-10-07 22:47 UTC: FileNotFoundError aborted startup). It must be skipped.
d = tempfile.mkdtemp()
with open(os.path.join(d, "kept.bin"), "wb") as f:
    f.write(b"\0" * 4096)
gone = os.path.join(d, "page.bin.tmp.1.2.3")
with open(gone, "wb") as f:
    f.write(b"\0" * 8192)
os.environ.update({
    "SGLANG_HICACHE_SHARED_STORE_BUDGET": "4GiB",
    "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": d,
    "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": "2Gi",
})
real_scandir = os.scandir


class Vanishing:
    def __init__(self, path):
        self._it = real_scandir(path)

    def __enter__(self):
        entries = list(self._it)
        os.remove(gone)  # the peer's rename happens after the listing
        return iter(entries)

    def __exit__(self, *a):
        self._it.close()


rb.os.scandir = Vanishing
try:
    budget, used, reserve = rb.shared_store_growth_reserve("file")
finally:
    rb.os.scandir = real_scandir
assert used == 4096, used
assert reserve == 4 * 1024**3 - 4096, reserve
print("12 store scan skips a file a peer renames away mid-scan")
