# GLM-5.3 Flash engine with an in-CVM shared KV cache

SGLang v0.5.20 plus the NEAR AI GLM-5.3 long-tier port plus one new patch, so the two TP4
replicas in one CVM (gpu02's long-context tier) share a CPU-RAM KV store instead of each
re-prefilling what the other already computed. The store is a tmpfs directory both engine
containers mount: it is guest RAM, nothing leaves the CVM, and no network hop is involved.

| file | what |
|---|---|
| `v0520-port.diff` | v0.5.20 → branch `nearai-v0520` @ `4fb82813` (26 files): HCC-safe HiCache (CUDA-owned host memory, pooled transfers, startup RAM budget), W4AFP8 loader fix, chunked-prefill pool clamp, opt-in DSA indexer query split, four fork fixes |
| `shared-kv.diff` | new here (3 files): HiCache L3 storage for compressed-DSA models + a RAM budget that coexists with a shared store |

`apply-patches.py` checks every patch digest and the exact before/after digest of all 27 touched
files (`source-manifest.json`). The result is byte-identical to the image qualified on gpu32.

## Why a patch was needed

GLM-5.3's DSA indexer keeps one compressed row per 4 KV pages, so the radix tree works in
256-token pages while host pages, the cache controller and every storage backend move 64-token
pages. Upstream-of-us code rejects L3 storage for this layout. With that guard bypassed, write-through
files a quarter of the data under 256-token keys and prefetch looks up 64-token keys, so nothing is
ever reused. The patch:

- hashes the storage chain at the 64-token transfer page (`tree_core.hash_page_size`); the tree keeps
  256-token nodes and stores 4 chain hashes per node;
- converts every token↔hash-count on the storage path with the transfer page;
- trims storage hits to the 256-token tree page before insertion (mamba checkpoints are keyed by a
  node's last hash, and the backend's trailing-page check backs a hit off to a node end that has one);
- supports `--hicache-host-memory-mode cache` only (buffer mode is rejected);
- lets `SGLANG_HICACHE_RAM_BUDGET` run with the `file` backend by reserving the store's remaining
  growth: `available − (SGLANG_HICACHE_SHARED_STORE_BUDGET − bytes already in the store)`. tmpfs pages
  already count against MemAvailable, so the second replica to start does not double-count.

## Enabling sharing

Everything is opt-in; without a storage backend the engine behaves as the port does.

```
--enable-hierarchical-cache --hicache-write-policy write_through --hicache-io-backend direct
--hicache-mem-layout page_first_direct --hicache-storage-backend file
--hicache-storage-prefetch-policy wait_complete
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/shared-kv        # the same tmpfs volume in every replica
SGLANG_HICACHE_RAM_BUDGET=406GiB                          # per replica, as today
SGLANG_HICACHE_SHARED_STORE_BUDGET=400GiB                 # the whole store, reserved at startup
SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200Gi                # per replica, file-backend size syntax (Gi/G)
SGLANG_HICACHE_CUDA_HOST_MEMORY=1                         # required on HCC/PPCIe hosts
```

Each replica's rank-0 LRU evictor only counts its own writes and the files present when it started,
so the per-replica cap must be at most the store budget divided by the number of replicas. If one
replica evicts a page the other expected, that lookup fails and the request recomputes; nothing breaks.

**Sizing on gpu02.** The CVM had ~1.43 TB available at start-up (r2 saw 995 GB after r1 took its
406 GiB). Two 406 GiB private host tiers plus a 400 GiB store fits with ~100 GiB to spare.

**The private host tier must stay at least as large as the GPU KV pool.** KV reaches the store only
by being written through to host memory first; a host tier smaller than the GPU pool (a lab config
of 25 GB/rank, 1.15M tokens against a 3.5M-token GPU pool) fills up, stalls write-through and starves
the store.

## Qualification (gpu32, bare metal, CC off, 2026-09-25)

| | result |
|---|---|
| cross-replica restore, 131K / 262K | 1.9 s / 3.1 s vs 6.1 s / 12.3 s recompute (3.3× / 4.0×) |
| correctness (teacher-forced logprobs, restored vs computed) | mean abs Δ 0.13 / 0.19, top-1 98% / 97%; noise floor 0.20 / 0.20, 97% |
| writer under 15 min of long-tier load | store == host tier (35,476 pages), writer idle between batches |

**Known gaps:**

- The last prefill chunk (16K tokens) is recomputed on restore: the final mamba checkpoint does not
  reach the store yet.
- The HCC/PPCIe allocator path is not covered on gpu32 (CC off). A TEE-host soak is required before
  gpu02.
- `sglang:hicache_backup_tokens_total` overcounts 4× on this layout; use `backuped_tokens_total` and
  `hicache_host_used_tokens`.

## Test

```
docker run --rm --runtime runc --entrypoint /usr/share/nearai/glm53-v0520-shared-kv/test-cpu.sh <image>
```
