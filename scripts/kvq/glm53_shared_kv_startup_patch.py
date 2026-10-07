# Start-up patch: in-CVM shared KV host tier (cvm-compose-files PR #304, recipe
# docker/sglang-glm53-v0520-shared-kv/shared-kv.diff) ported onto the production GLM-5.3 image v6
# (docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17,
# fork fc91d24 + sglang-glm53-hicache + admission-reserve v10 + sglang-glm53-hicache-w4afp8).
# Replicas in one CVM share an L3 HiCacheFile store on a tmpfs directory they all mount.
#  1) startup_ram_budget: SGLANG_HICACHE_RAM_BUDGET may run with the file backend; it reserves the
#     store's remaining growth (SGLANG_HICACHE_SHARED_STORE_BUDGET minus bytes already in the store).
#  2) lru_file_evictor: under a store cap, non-owner MLA ranks (TP rank > 0) write and bound their
#     own rank-sharded sidecars (mamba state) at SGLANG_HICACHE_FILE_BACKEND_SIDECAR_MAX_FRACTION
#     (default 0.25) of the cap instead of refusing every write. Port addition: a non-owner rank
#     never adopts a file it only read (shared KV / indexer pages), so it can never evict one.
#  3) unified_tree_core + unified_radix_cache: compressed DSA keeps 256-token tree pages but moves
#     64-token transfer pages; the storage hash chain runs at the transfer page, storage hits are
#     trimmed to the tree page, cache host-memory mode only. Port addition: refuse L3 storage on
#     compressed DSA unless the Python tree core is selected (the Rust core ignores hash_page_size).
#  4) hicache_storage (port addition): HiCacheFile keys MLA KV without a TP rank (replicated), and
#     mamba sidecars inherited that key, so every TP rank read and wrote one shared mamba file
#     although each rank holds a different head shard. Mamba sidecar keys now carry _tp<rank>_<size>.
# Inert unless --hicache-storage-backend file is set. Each file is checked against the exact v6
# sha256 before and the exact patched sha256 after; edits are exact-anchor replacements staged in
# memory and written only when every file verifies (fail closed: any mismatch exits non-zero).
import hashlib, os, pathlib, sys
ROOT = pathlib.Path(os.environ.get("KVQ_SHARED_KV_PATCH_ROOT", "/sgl-workspace/sglang/python/sglang/srt"))
TAG = "[shared-kv-patch]"
# rel -> (v6 sha256, patched sha256)
EXPECTED = {
    "mem_cache/startup_ram_budget.py": ("565ecfa113f46c5f1b7cfdf8e6799d3604eb1d86aae80505dcfbc7252ef79c5c",
        "d9f92e59e3ea5b75a573e97f7250b03eeba2f168754ceb2e87ade8a59fad281a"),
    "mem_cache/storage/file/lru_file_evictor.py": ("9e590e144c1169d463102e9aa147432cf5fa9108f1f8936de1184903c79a1330",
        "606894fcea4a5570180fc603d55c6dfbd8da6c1fc2c5358868057036e9ff1c05"),
    "mem_cache/unified_cache/unified_tree_core.py": ("0bbb0bb1a9581dc543dc4cc40cfe0b6f888e3b124c4739eac9635b86b176c168",
        "d9992a542941f6cf03af9459ef32e15d546f06a05ddd7ac2c58cd9317a285a50"),
    "mem_cache/unified_radix_cache.py": ("98f175f529614ebb33618a27c54248105eff3bf80c1365ffd3716bd346add278",
        "88263caaca4967d542a9f9e20af7230114d5c8ddcb5dfcd2d9c9cec1fee42c2d"),
    "managers/cache_controller.py": ("ffb53c980497d0a94f4ffea7c54efa86b3e97077a08d8b8a3282ccbf2531c779",
        "5a7c6a25d39de57be74ad094ff38e72de29c8b7a1629ba19892e77196dbbed60"),
    "mem_cache/hicache_storage.py": ("40d892d038557bbce41f3b35369c8a1bf2feeed56b907f709492f7e0f113d313",
        "e1c088ed5277dc630c0ad36c281aee3386e2c97aed5b8037d97787aa24761035"),
}
staged = {}
def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def read(rel):
    # Bytes, not read_text(): no locale decoding and no newline translation.
    return (ROOT / rel).read_bytes().decode("utf-8")
def load(rel):
    if rel not in staged:
        staged[rel] = read(rel)
    return staged[rel]
def sub(rel, old, new, label):
    s = load(rel)
    if new in s:
        print(f"{TAG} {label}: already applied", flush=True); return
    n = s.count(old)
    if n != 1:
        sys.exit(f"{TAG} {label}: expected 1 anchor, found {n}")
    staged[rel] = s.replace(old, new); print(f"{TAG} {label}: staged", flush=True)

# --- 1) startup_ram_budget.py (shared-kv.diff, verbatim) ---
R = "mem_cache/startup_ram_budget.py"
sub(R, '''ENV = "SGLANG_HICACHE_RAM_BUDGET"
RESERVE_BYTES''', '''ENV = "SGLANG_HICACHE_RAM_BUDGET"
STORE_ENV = "SGLANG_HICACHE_SHARED_STORE_BUDGET"
RESERVE_BYTES''', "ram budget STORE_ENV")
sub(R, '''    return re.sub(r"\\\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def cgroup_headrooms(''', '''    return re.sub(r"\\\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def shared_store_growth_reserve(backend: str) -> tuple[int, int, int]:
    """(store_budget, bytes_in_store, reserve) for an in-CVM shared `file` store.

    Replicas in one CVM share the store directory; each replica's LRU evictor must be capped
    (SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE) so that together they stay within the store budget.
    """
    if backend != "file":
        raise ValueError("RAM budget supports only the in-CVM `file` storage backend")
    raw = os.environ.get(STORE_ENV)
    store_dir = os.environ.get("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR")
    cap_raw = os.environ.get("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE")
    if not raw or not store_dir or not cap_raw:
        raise ValueError(
            f"{ENV} with a storage backend needs {STORE_ENV}, "
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR and SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"
        )
    unit, amount = parse_budget(raw)
    if unit == "%":
        raise ValueError(f"{STORE_ENV} must be GB or GiB, not a percentage")
    budget = int(amount * (1024**3 if unit == "GiB" else 10**9))
    from sglang.srt.mem_cache.storage.file.lru_file_evictor import _parse_size_to_bytes

    cap = _parse_size_to_bytes(cap_raw)
    if cap <= 0 or cap > budget:
        raise ValueError(
            "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE must be a size the file backend accepts "
            f"(e.g. 200Gi or 200G) and at most {STORE_ENV}"
        )
    used = 0
    if os.path.isdir(store_dir):
        with os.scandir(store_dir) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False):
                    used += entry.stat(follow_symlinks=False).st_size
    return budget, used, max(0, budget - used)


def cgroup_headrooms(''', "ram budget shared_store_growth_reserve")
sub(R, '''        if get_memory().hicache_storage_backend is not None:
            raise ValueError("RAM budget currently supports in-CVM host RAM only")
        record["available_bytes"] = available_ram_bytes()
''', '''        backend = get_memory().hicache_storage_backend
        store_reserve = 0
        if backend is not None:
            store_budget, store_used, store_reserve = shared_store_growth_reserve(backend)
            logger.info(
                "HiCache shared store reserve: budget_bytes=%d used_bytes=%d reserved_bytes=%d",
                store_budget,
                store_used,
                store_reserve,
            )
        record["available_bytes"] = max(0, available_ram_bytes() - store_reserve)
''', "ram budget reserves store growth")

# --- 2) lru_file_evictor.py (shared-kv.diff verbatim + non-owner touch fix) ---
E = "mem_cache/storage/file/lru_file_evictor.py"
sub(E, '''        self._eviction_enabled = self._eviction_configured and self._is_storage_owner
        if self._eviction_configured and not self._is_storage_owner:
            logger.info(
                f"HiCacheFile rank {self._tp_rank} (MLA): eviction handled by rank 0; "
                f"this rank skips LRU bookkeeping and will not create new files."
            )
''', '''        # Every rank bounds what it writes. Non-owner MLA ranks write only rank-sharded sidecars
        # (e.g. mamba state); they track and evict only their own files, never scan shared ones.
        self._eviction_enabled = self._eviction_configured
        if self._eviction_configured and not self._is_storage_owner:
            fraction = float(os.environ.get("SGLANG_HICACHE_FILE_BACKEND_SIDECAR_MAX_FRACTION", "0.25"))
            if not 0.0 < fraction <= 1.0:
                raise ValueError("SGLANG_HICACHE_FILE_BACKEND_SIDECAR_MAX_FRACTION must be in (0, 1]")
            if self.max_size_bytes > 0:
                self.max_size_bytes = int(self.max_size_bytes * fraction)
            logger.info(
                f"HiCacheFile rank {self._tp_rank} (MLA): shared files are evicted by rank 0; "
                f"this rank bounds only its own sidecar files, cap={self.max_size_bytes} B."
            )
''', "evictor non-owner bounds own sidecars")
sub(E, '''        self._scan_existing_files()
        with self._lock:
            if self.max_size_bytes > 0 and self._total_bytes > self.max_size_bytes:''',
'''        if self._is_storage_owner:
            self._scan_existing_files()
        with self._lock:
            if self.max_size_bytes > 0 and self._total_bytes > self.max_size_bytes:''', "evictor owner-only scan")
sub(E, '''            return True  # unbounded storage: nothing to enforce
        if not self._is_storage_owner:
            logger.warning(
                f"HiCacheFile rank {self._tp_rank} is not the MLA storage owner; "
                f"not caching new key {key} because file eviction is enabled."
            )
            return False
        if self.max_size_bytes > 0 and value_bytes > self.max_size_bytes:''',
'''            return True  # unbounded storage: nothing to enforce
        if self.max_size_bytes > 0 and value_bytes > self.max_size_bytes:''', "evictor non-owner may reserve")
sub(E, '''            if suffixed_key in self._lru:
                self._lru.move_to_end(suffixed_key, last=True)
                return
        # Untracked file: stat without holding the lock.''',
'''            if suffixed_key in self._lru:
                self._lru.move_to_end(suffixed_key, last=True)
                return
        if not self._is_storage_owner:
            # A non-owner reads shared KV/indexer pages it did not write; adopting them would let
            # it evict files rank 0 owns. It tracks only the sidecars it reserved itself.
            return
        # Untracked file: stat without holding the lock.''', "evictor non-owner never adopts")

# --- 3a) unified_tree_core.py (shared-kv.diff; hunk 1 re-anchored: the fork has no is_host_memory_buffer_only) ---
T = "mem_cache/unified_cache/unified_tree_core.py"
sub(T, '''        self.page_size = params.page_size
        self.is_eagle = params.is_eagle and ComponentType.MAMBA not in components
''', '''        self.page_size = params.page_size
        # Storage hash-chain granularity. Equals the tree page except for
        # compressed DSA, where the owning cache sets it to the transfer page.
        self.hash_page_size = params.page_size
        self.is_eagle = params.is_eagle and ComponentType.MAMBA not in components
''', "tree core hash_page_size")
sub(T, '''                node.hash_value = compute_node_hash_values(node, self.page_size)
                filled += 1''', '''                node.hash_value = compute_node_hash_values(node, self.hash_page_size)
                filled += 1''', "tree core backfill hashes")
sub(T, '''        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )''', '''        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.hash_page_size
        )''', "tree core split hashes")
sub(T, '''            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
''', '''            new_node.hash_value = compute_node_hash_values(new_node, self.hash_page_size)
''', "tree core insert hashes")
sub(T, '''            hash_value = hash_value[prefix_len // self.page_size :]''',
'''            hash_value = hash_value[prefix_len // self.hash_page_size :]''', "tree core insert_host hash walk")

# --- 3b) unified_radix_cache.py (shared-kv.diff; hunks 6-7 adapted to the fork) ---
U = "mem_cache/unified_radix_cache.py"
sub(U, '''            components=self.components,
        )
        # Components execute boundary actions through the tree core.''', '''            components=self.components,
        )
        if hasattr(self.tree_core, "hash_page_size"):
            self.tree_core.hash_page_size = self._transfer_page_size
        # Components execute boundary actions through the tree core.''', "cache sets transfer hash page")
sub(U, '''        if storage_backend is not None and self.page_size != params.page_size:
            raise ValueError(
                "Compressed DSA currently supports L2 HiCache only; "
                "storage hashes and transfers require matching page sizes."
            )
''', '''        if (
            storage_backend is not None
            and self.page_size != params.page_size
            and self.host_memory_mode != "cache"
        ):
            raise ValueError(
                "Compressed DSA storage supports --hicache-host-memory-mode cache only."
            )
        if (
            storage_backend is not None
            and self.page_size != params.page_size
            and self._tree_core_backend != "python"
        ):
            raise ValueError(
                "Compressed DSA storage needs SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python."
            )
''', "cache compressed-DSA storage guard")
sub(U, '''            hash_value[: completed_tokens // self.page_size],
        )''', '''            hash_value[: completed_tokens // self._transfer_page_size],
        )''', "cache prefetch insert hashes")
sub(U, '''        expected_tokens = len(hash_value) * self.page_size
''', '''        expected_tokens = len(hash_value) * self._transfer_page_size
''', "cache prefetch expected tokens")
sub(U, '''            keep_pages = completed_tokens // self.page_size
''', '''            keep_pages = completed_tokens // self._transfer_page_size
''', "cache prefetch keep_pages")
sub(U, '''            alloc_len = hit_tokens
            host_indices = cc.mem_pool_host.alloc(alloc_len)
''', '''            if not buffer_mode:
                # Storage hits end on a transfer page; the tree inserts whole tree pages.
                tree_aligned = hit_tokens - hit_tokens % self.page_size
                if tree_aligned != hit_tokens:
                    self._resolve_storage_prefetch_tokens(
                        req_id, hit_tokens - tree_aligned, reason="tree_page_align"
                    )
                    hit_tokens = tree_aligned
                    operation.storage_hit_count = hit_tokens
                    operation.hash_value = operation.hash_value[
                        : hit_tokens // self._transfer_page_size
                    ]
            alloc_len = hit_tokens
            host_indices = cc.mem_pool_host.alloc(alloc_len)
''', "cache trims storage hit to tree page")
sub(U, '''            operation.hash_value = operation.hash_value[: alloc_len // self.page_size]
''', '''            operation.hash_value = operation.hash_value[
                : alloc_len // self._transfer_page_size
            ]
''', "cache host-capacity hash trim")

# --- 4) hicache_storage.py: rank-sharded mamba sidecar keys under MLA (port addition) ---
H = "mem_cache/hicache_storage.py"
sub(H, '''            self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"
''', '''            self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"
        # MLA KV is rank-replicated and keyed without a rank, but Mamba/KDA state is a per-rank
        # head shard: key it by rank or every TP rank reads and writes the same file.
        self._sharded_sidecar_tag = (
            f"_tp{tp_rank}_{tp_size}" if is_mla_model and tp_size > 1 else ""
        )
''', "file backend sidecar rank tag")
sub(H, '''        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")
''', '''        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(
            f"{key}.{component_name}{self._sidecar_rank_tag(component_name)}"
        )

    def _sidecar_rank_tag(self, component_name) -> str:
        if component_name == PoolName.MAMBA:
            return getattr(self, "_sharded_sidecar_tag", "")
        return ""
''', "file backend component key rank tag")
sub(H, '''        return key if pool_name == PoolName.KV else f"{key}.{pool_name}"
''', '''        if pool_name == PoolName.KV:
            return key
        return f"{key}.{pool_name}{self._sidecar_rank_tag(pool_name)}"
''', "file backend page key rank tag")

# --- 5) hicache_storage.py: shared-tier attribution metrics for the prod A/B (port addition) ---
# sglang:kvshare_storage_hit_tokens_total{source}: Full-KV tokens loaded from the file tier, split by
# whether THIS process wrote the page (self) or another replica / a previous run did (peer).
# sglang:kvshare_storage_written_tokens_total: Full-KV tokens this process wrote (pages a peer had
# already written are skipped by set() and not counted). TP rank 0 only; SGLANG_KVSHARE_METRICS=0
# turns them off. The written-key set is bounded (SGLANG_KVSHARE_WRITTEN_KEYS, default 1,000,000).
sub(H, """        self._sharded_sidecar_tag = (
            f"_tp{tp_rank}_{tp_size}" if is_mla_model and tp_size > 1 else ""
        )
""", """        self._sharded_sidecar_tag = (
            f"_tp{tp_rank}_{tp_size}" if is_mla_model and tp_size > 1 else ""
        )
        self._kvshare_rank0 = tp_rank == 0
        self._kvshare_last_write = None
""", "file backend kvshare rank")
sub(H, """    def _sidecar_rank_tag(self, component_name) -> str:
        if component_name == PoolName.MAMBA:
            return getattr(self, "_sharded_sidecar_tag", "")
        return ""
""", """    def _sidecar_rank_tag(self, component_name) -> str:
        if component_name == PoolName.MAMBA:
            return getattr(self, "_sharded_sidecar_tag", "")
        return ""

    def _kvshare_metrics(self):
        m = getattr(self, "_kvshare_m", None)
        if m is not None:
            return m
        m = False
        self._kvshare_written = OrderedDict()
        self._kvshare_written_max = int(os.environ.get("SGLANG_KVSHARE_WRITTEN_KEYS", "1000000"))
        if getattr(self, "_kvshare_rank0", False) and os.environ.get("SGLANG_KVSHARE_METRICS", "1") == "1":
            try:
                m = _kvshare_counters()
            except Exception:
                logger.exception("kvshare metrics disabled")
                m = False
        self._kvshare_m = m
        return m

    def _kvshare_page_tokens(self) -> int:
        pool = getattr(self, "registered_pools", {}).get(PoolName.KV)
        size = getattr(pool, "page_size", None)
        return int(size or os.environ.get("SGLANG_KVSHARE_PAGE_TOKENS", "64"))

    def _kvshare_note_write(self, suffixed: str) -> None:
        m = self._kvshare_metrics()
        if m is False:
            return
        self._kvshare_last_write = suffixed
        self._kvshare_written[suffixed] = None
        if len(self._kvshare_written) > self._kvshare_written_max:
            self._kvshare_written.popitem(last=False)
        if "." not in suffixed[: len(suffixed) - len(self.config_suffix)]:
            # Full-KV page (sidecar keys carry a ".<pool>" component).
            m[1].inc(self._kvshare_page_tokens())

    def kvshare_read_into_host(self, operation, hash_values, host_indices, host_pool, page_size) -> int:
        # Restore Full-KV pages from the file tier into host_pool for the controller's v1 path.
        # Same contract as HiCacheController._generic_page_get (count of the contiguous prefix
        # of pages restored, stop at the first miss or on termination), but each worker thread
        # reuses one scratch page instead of allocating a pinned page per page, and pages are
        # read in parallel (SGLANG_KVSHARE_READ_THREADS, default 8). Counts self/peer hits.
        if operation.is_terminated():
            return 0
        t_start = time.perf_counter()
        offsets = [host_indices[i * page_size] for i in range(len(hash_values))]
        m = self._kvshare_metrics()

        stage = [0.0, 0.0]  # seconds in get() (open/read/touch) and in the host-slot copy, summed over threads

        def one(item):
            key, off = item
            t0 = time.perf_counter()
            data = self.get(key, self._scratch_page(host_pool))
            t1 = time.perf_counter()
            if data is None:
                return None
            host_pool.set_from_flat_data_page(off, data)
            stage[0] += t1 - t0
            stage[1] += time.perf_counter() - t1
            return "self" if self._get_suffixed_key(key) in self._kvshare_written else "peer"

        workers = int(os.environ.get("SGLANG_KVSHARE_READ_THREADS", "8"))
        items = list(zip(hash_values, offsets))
        if workers > 1 and len(items) > 1:
            pool = getattr(self, "_kvshare_read_pool", None)
            if pool is None:
                from concurrent.futures import ThreadPoolExecutor

                pool = self._kvshare_read_pool = ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="hicache-file-read"
                )
            sources = list(pool.map(one, items))
        else:
            sources = [one(item) for item in items]
        count = 0
        for i, source in enumerate(sources):
            if source is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            if m:
                m[0].labels(source=source).inc(page_size)
            count += 1
        if os.environ.get("SGLANG_KVSHARE_LOG_READS", "0") == "1":
            dt = time.perf_counter() - t_start
            page_bytes = self._scratch_page(host_pool).numel() * self._scratch_page(host_pool).element_size()
            logger.info(
                "kvshare read: req=%s pages=%d restored=%d MB=%.1f s=%.3f MBps=%.0f get_s=%.3f copy_s=%.3f threads=%d",
                operation.request_id, len(hash_values), count, count * page_bytes / 1e6, dt,
                count * page_bytes / 1e6 / max(dt, 1e-9), stage[0], stage[1], workers,
            )
        return count
""", "file backend kvshare helpers")
sub(H, """            os.replace(tmp_path, tensor_path)
            self._evictor.commit(suffixed)
""", """            os.replace(tmp_path, tensor_path)
            self._evictor.commit(suffixed)
            self._kvshare_note_write(suffixed)
""", "file backend kvshare note write")
sub(H, """        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        return True
""", """        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        m = self._kvshare_metrics() if pool_name == PoolName.KV else False
        if m:
            source = "self" if self._get_suffixed_key(storage_key) in self._kvshare_written else "peer"
            m[0].labels(source=source).inc(getattr(host_pool, "page_size", 1) or 1)
        return True
""", "file backend kvshare count hit")
sub(H, """        storage_key = self._log_key(pool_name, key)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)
""", """        storage_key = self._log_key(pool_name, key)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)
""", "file backend kvshare count write")
sub(H, """class HiCacheFile(HiCacheStorage):
""", """_KVSHARE_COUNTERS = None


def _kvshare_counters():
    # One registration per process (prometheus rejects duplicate names).
    global _KVSHARE_COUNTERS
    if _KVSHARE_COUNTERS is None:
        from prometheus_client import Counter

        _KVSHARE_COUNTERS = (
            Counter(
                "sglang:kvshare_storage_hit_tokens_total",
                "Full-KV tokens loaded from the shared L3 file tier, by writer: self = this "
                "process wrote the page, peer = another replica (or a previous run) did.",
                ["source"],
            ),
            Counter(
                "sglang:kvshare_storage_written_tokens_total",
                "Full-KV tokens this process wrote to the shared L3 file tier.",
            ),
        )
    return _KVSHARE_COUNTERS


class HiCacheFile(HiCacheStorage):
""", "file backend kvshare counters")

sub(H, """import os
import threading
""", """import os
import threading
from collections import OrderedDict
""", "file backend kvshare import")

# --- 6) hicache_storage.py: faster shared-tier restores (port addition, gpu13 CC 2026-10-07) ---
# A 218K-token cross-replica restore spent ~11 s in L3->L2 reads: one Python read per 64-token page
# per pool, sequential, each into a freshly allocated, zeroed, pinned (cudaHostAlloc) scratch page.
# Reuse one scratch page per pool per thread, and read the pages of a batch with a small thread pool
# (file reads release the GIL; each page lands in a distinct host slot; the evictor and metadata
# cache are already lock-protected). SGLANG_KVSHARE_READ_THREADS=1 restores the sequential path.
sub(H, """    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        \"\"\"Read one page from storage into host_pool at page_offset.\"\"\"
        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
""", """    def _scratch_page(self, host_pool):
        local = getattr(self, "_kvshare_tls", None)
        if local is None:
            local = self._kvshare_tls = threading.local()
        pages = getattr(local, "pages", None)
        if pages is None:
            pages = local.pages = {}
        page = pages.get(id(host_pool))
        if page is None:
            page = pages[id(host_pool)] = host_pool.get_dummy_flat_data_page()
        return page

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        \"\"\"Read one page from storage into host_pool at page_offset.\"\"\"
        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, self._scratch_page(host_pool))
""", "file backend reuse scratch page")
sub(H, """            results[transfer.name] = [
                op_fn(transfer.name, key, host_pool, host_indices[i * page_size].item())
                for i, key in enumerate(keys)
            ]
        return results
""", """            offsets = host_indices[::page_size].tolist()
            workers = int(os.environ.get("SGLANG_KVSHARE_READ_THREADS", "8"))
            if op_fn == self._read_page and workers > 1 and len(keys) > 1:
                pool = getattr(self, "_kvshare_read_pool", None)
                if pool is None:
                    from concurrent.futures import ThreadPoolExecutor

                    pool = self._kvshare_read_pool = ThreadPoolExecutor(
                        max_workers=workers, thread_name_prefix="hicache-file-read"
                    )
                name = transfer.name
                results[transfer.name] = list(
                    pool.map(
                        lambda item: op_fn(name, item[0], host_pool, item[1]),
                        zip(keys, offsets),
                    )
                )
            else:
                results[transfer.name] = [
                    op_fn(transfer.name, key, host_pool, offsets[i])
                    for i, key in enumerate(keys)
                ]
        return results
""", "file backend threaded batch reads")

# --- 7) cache_controller.py: route v1 KV restores through the backend's fast reader ---
# The controller's v1 path allocated a new pinned host page for every page of a prefetch, then
# read sequentially; HiCacheFile.kvshare_read_into_host (section 5) keeps the same contract.
C = "managers/cache_controller.py"
sub(C, """    def _generic_page_get(
        self, operation, hash_values, host_indices, extra_info=None
    ) -> int:
        dummy_page_dst = [
""", """    def _generic_page_get(
        self, operation, hash_values, host_indices, extra_info=None
    ) -> int:
        fast = getattr(self.storage_backend, "kvshare_read_into_host", None)
        if fast is not None:
            return fast(
                operation, hash_values, host_indices, self.storage_host_pool, self.page_size
            )
        dummy_page_dst = [
""", "controller v1 get uses the backend fast reader")

# --- verify every file, then write ---
for rel, (before, after) in EXPECTED.items():
    orig = read(rel)
    got = sha(staged.get(rel, orig))
    if sha(orig) == after:
        continue
    if sha(orig) != before:
        sys.exit(f"{TAG} {rel}: unrecognized source sha256 {sha(orig)} (expected v6 {before})")
    if got != after:
        sys.exit(f"{TAG} {rel}: patched sha256 {got} != expected {after}")
for rel in EXPECTED:
    s = staged.get(rel)
    if s is not None and sha(read(rel)) != sha(s):
        compile(s, rel, "exec")
        (ROOT / rel).write_bytes(s.encode("utf-8"))
        # Drop stale bytecode so an unchecked-hash .pyc can never shadow the patched source.
        src = ROOT / rel
        for pyc in (src.parent / "__pycache__").glob(src.stem + ".*.pyc"):
            pyc.unlink()
        print(f"{TAG} {rel}: written", flush=True)
if "KVQ_SHARED_KV_PATCH_ROOT" not in os.environ:
    # The patched tree must be the one this interpreter imports (editable install), or the patch
    # is silently inert. find_spec locates the package without executing it.
    import importlib.util
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        print(f"{TAG} WARNING: this python cannot locate sglang; run the engine with the same python", flush=True)
    elif os.path.realpath(list(spec.submodule_search_locations)[0]) != os.path.realpath(ROOT.parent):
        sys.exit(f"{TAG} sglang imports from {list(spec.submodule_search_locations)[0]}, not {ROOT.parent}")
print(f"{TAG} OK: shared KV host tier patch verified on {len(EXPECTED)} files", flush=True)
