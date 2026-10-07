# GLM-5.3: peer prefix fetch between colocated replicas (GPU→GPU over NIXL cuda_ipc) — design and feasibility

Status: design study only. Nothing has been built, deployed or pushed. Written 2026-10-07.

Source studied: the prod SGLang tree "v6" (fork fc91d24 + recipes), reconstructed at
`scratchpad/kvq/port/v6/python/sglang/srt`. Every `file:line` below is relative to that `srt/` directory.
The shared-KV-patched copy (`port/try2`) differs only in five `mem_cache` files (`shared-kv-v6-port.diff`).
None of those files is on the peer-fetch path, except `unified_radix_cache.py`, whose storage paths this design does not touch.

## 0. Verdict up front

- **Feasible.** The hard parts already exist in v6:
  - P/D moves every pool GLM-5.3 needs over NIXL: target MLA KV, DSA indexer K, draft KV, draft DSA, and the Mamba state slot.
  - Rank consensus for asynchronous transfers exists in two places: the P/D poll all-reduce and the HiCache prefetch `rank_consensus`.
  - The radix tree's Mamba copy-on-write handles "a node owns a state slot".
- **What is new is the glue**, plus a NULL-mode scheduler that can act as both sender and receiver.
- **Recommended shape:**
  - The proxy routes by load and adds an `X-KV-Peer: <replica>` hint.
  - The receiver B asks the holder A in-band.
  - A matches and locks in its scheduler loop, then pushes the pages (NIXL WRITE, reusing `NixlKVSender`) into pages B has already preallocated.
  - B inserts the prefix into its own radix tree and schedules the request normally.
- **Effort:**
  - Prototype on gpu13: about **8 engineer-days**.
  - Prod quality, including the expandable-segments fix, failure handling, tests, the proxy hint and a prod A/B: about **5–6 more engineer-weeks**.
- **Be honest about the prize.**
  - Only 1.2–1.6% of prompt tokens are cached only on a sibling replica. The fetch is worth building only if load-based routing inside the host is worth something.
  - Measure that first; §5 gives the measurement.
  - If it is small, extend the existing shared L3 tier with a "push on move" hint (§5, alternative B) for about 1 week instead.

## 1. Discovery: how B learns that A has the prefix

| Option | Precision | Cost | Verdict |
|---|---|---|---|
| **Proxy hint header** (`X-KV-Peer: r1`): the in-host proxy knows which replica served this conversation last, and adds the header when load routing sends it elsewhere | High. A is almost always the last holder. A still verifies by matching, so a stale hint costs one RPC | Proxy change (inference-proxy repo) plus header plumbing. A precedent exists: `entrypoints/openai/serving_base.py:265-293` already reads `x-smg-routing-key` and `X-Data-Parallel-Rank` | **Recommend.** It targets exactly the case that matters: a turn moved for load |
| Ghost aggregator digests | Poor. Sampled 1/16 (`--sample 16`). It sees finished requests only (`observability/ghost_cache.py` header), knows nothing of evictions or Mamba checkpoints, and is an LRU of unbounded size | Free | Reject for discovery. Keep it as the yardstick |
| Host-local index of radix hash chains published by each replica | Good, if it tracks evictions. Needs every insert and evict published: the KV-events publisher exists (`managers/scheduler.py:2401`), and chain hashes exist when storage is on (`unified_tree_core` `compute_node_hash_values`). Compressed DSA hashes on the 64-token page, while the tree splits on 256 | A new sidecar plus a consistency protocol, and a query on every miss | Over-built for a 1.2–1.6% prize. Revisit only if the proxy hint proves too coarse |
| Try-fetch on every miss, to all siblings | Exact | Three RPCs per miss, and A's scheduler pays a match for each | Reject as the default. Acceptable as the fallback when the hint is absent and the miss is ≥32K tokens |

**Hint transport.**
- Add `kv_peer: Optional[str]` to `GenerateReqInput` / `TokenizedGenerateReqInput`. `bootstrap_host` and `bootstrap_room` at `managers/io_struct.py:279-281` are the model.
- Set it from the header in `serving_base.py`.
- A static env map turns the replica label into A's internal HTTP address and bootstrap port, e.g. `SGLANG_PEER_REPLICAS=r1=http://model-…-r1:8000@8998,…`.

## 2. Holder side (A)

### 2.1 Rank-consistent trigger

- **Every TP rank must take the same decision on the same scheduler iteration.**
- v6 gets this in two ways:
  - **In-band requests** are received on rank 0 and broadcast: `managers/scheduler_components/request_receiver.py:88`, with `broadcast_pyobj` and `attn_cp_tp_broadcast_pyobj` at lines 173-195.
  - **Out-of-band per-rank events** use a MIN all-reduce of poll states:
    - `disaggregation/utils.py:224 _all_reduce_polls`, `:231 poll_and_all_reduce`, `:246 poll_and_all_reduce_attn_cp_tp_group`;
    - the HiCache form is `@rank_consensus` on `check_prefetch_progress` (`mem_cache/unified_radix_cache.py:1959`), plus a MAX all-reduce for timeouts (`:1936 _can_terminate_prefetch`).
- **Use the same split as P/D prefill.**
  - **The export request is in-band.** B's rank 0 POSTs `/peer_export {room, input_ids, extra_key, cache_salt, from_len=P_B, max_len}` to A's HTTP server. A new io_struct carries it to the scheduler through the broadcast path.
  - **B's destination metadata is out-of-band.** B's rank i sends it to A's rank i over the existing ZMQ bootstrap channel.
  - **Readiness is a MIN all-reduce**, as in `PrefillBootstrapQueue.pop_bootstrapped` (`disaggregation/prefill.py:419-470`).
- **Never let A's per-rank bootstrap thread trigger a match.** Ranks would see the metadata on different iterations.
- **Gloo needs equal-length inputs.** gpu03 (2026-10-06) showed that a poll list which differs by one request between ranks aborts gloo. The #335 patch guards this (`scripts/glm53_pd_startup_patch.py` item 4). The new peer queues must poll only rooms queued on every rank, in rid order, from day one.

### 2.2 Match and lock

- **New `PeerExportQueue`** in the scheduler (a new file, `disaggregation/peer.py`). On add:
  1. `tree_cache.match_prefix(MatchPrefixParams(key=RadixKey(input_ids[:max_len], extra_key, cache_salt), cow_mamba=False))`.
     - Use device-only matching: A must not load back from its own L2 to serve a peer.
     - The validators make the match end at the deepest node that has both FULL KV and a device Mamba value: `unified_cache/components/mamba_component.py:142 create_match_validator`, `:155 finalize_match_result_in_tree_core`.
  2. Compute `L_A = len(device_indices)`, aligned down to the grid.
     - The grid is `mamba_checkpoint_grid(tree_page)`: `runtime_context.py:1920-1933`, the lcm of the Mamba chunk and the tree page, with tree page 256 (`unified_radix_cache.py:122-138`).
     - If `L_A - P_B < min_tokens` (suggest 4096), complete the protocol with `L=0` and stop.
  3. `lock = tree_cache.inc_lock_ref(result.last_device_node)` (`unified_radix_cache.py:826`).
     - This pins FULL and MAMBA together.
     - The Mamba slot is `tree_core.get_component_device_value(best_match_node, ComponentType.MAMBA)`, the same accessor COW uses at `mamba_component.py:193`.
  4. **Record a CUDA event on `forward_stream`.**
     - Under overlap scheduling, the node's pages, and its Mamba checkpoint copy, may still be in flight from the previous forward.
     - P/D early-send solves the same race at `prefill.py:1163-1170` (`_early_send_wait_event`). Reuse that mechanism.
  5. Create a `NixlKVSender(room)` (`disaggregation/nixl/conn.py:2914`).
- **Lock lifetime:** released on sender `Success` or `Failed` under consensus, or after a bootstrap timeout (`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`, `common/conn.py:253`). With WRITE (§3.2), A itself knows when the copy has finished, so a timeout never unlocks pages that are still being read.

### 2.3 What is reused from P/D prefill

- **KV-args construction:** `PrefillBootstrapQueue._init_kv_manager` (`prefill.py:212-332`). It already:
  - appends the draft KV pool with the same indices (`:250-261`);
  - fills state components through `setup_state_kv_args` (`disaggregation/utils.py:1329`). For GLM-5.3's `HybridLinearKVPool` with `use_dsa`, this yields **MAMBA, DSA, DSA_TAIL and draft DSA** (`utils.py:1445-1496`).
  - Factor it into a function both the export and fetch managers call.
- **Registration:** `NixlKVManager.register_buffer_to_engine` (`nixl/conn.py:1432-1488`).
  - It registers KV and state as VRAM.
  - Aux is registered as DRAM (`:1466`). Under CC this needs pinned `MetadataBuffers`. v6 does not pin them; the #335 patch shows how.
- **Send path.** `NixlKVSender.send(page_indices, state_indices)` calls `add_transfer_request` (`conn.py:2565`), then the `transfer_worker` WRITE (`:1090`, `:1505 _send_kvcache_generic`, `:2160 _send_mamba_state`, `:2106 _send_slot_state`).
  - **Source pages:** `kv_to_page_indices(device_indices[P_B:L_A], 64)`. Same as `send_kv_chunk` (`prefill.py:1177-1372`), but taken from the tree rather than `req_to_token`.
  - **State indices:** MAMBA = `[node_slot]`. DSA = the same 64-page list (as `_full_kv_pages_payload`, `prefill.py:1240-1244`). DSA_TAIL = `[]`, because `L_A` is a multiple of 256 (`utils.py:1159-1161` returns `[]` when `seq_len % index_kpool == 0`).
- **`decode_prefix_len`.** The protocol already carries it: `NixlKVReceiver.send_metadata(..., decode_prefix_len)` (`conn.py:3042`) and `req_to_decode_prefix_len` (`common/conn.py:247`). B sends `P_B`, and only the delta moves.
- **Partial send.** The transfer worker slices destination indices to the source length (`conn.py:1175-1191`). A can therefore send fewer pages than B preallocated.
- **Aux.** Put `L_A` in the aux `cached_tokens` slot, so B learns how much landed.

### 2.4 Two managers per process

- Today a scheduler has one manager, and its role is fixed by `disaggregation_mode`:
  - `CommonKVManager.__init__` branches at `common/conn.py:227` and `:254`;
  - NIXL starts 8 transfer threads only in PREFILL (`nixl/conn.py:434`);
  - the bootstrap server starts only in PREFILL (`managers/disagg_service.py:18-34`).
- **Prototype:** each NULL-mode replica builds one PREFILL-role manager (export) and one DECODE-role manager (fetch), and starts the bootstrap server.
- **Unknown:** two UCX agents in one process exporting the same VRAM through cuda_ipc. It is expected to work, since IPC handles are cached per allocation, but test it on day 1.

## 3. Receiver side (B)

### 3.1 Where it hooks

- **`Scheduler._add_request_to_queue`** (`managers/scheduler.py:3182`), NULL branch.
  - Today it calls `_prefetch_kvcache` (`:3104`, the HiCache L3 analogue) and then `waiting_queue.append`.
  - Add: if `req.kv_peer` is set and the local match leaves more than `min_tokens` uncovered, call `peer_fetch_queue.add(req)` instead.
  - Every input to this decision is rank-identical: the broadcast request and the replicated tree.
- **`get_next_batch_to_run`** (`:3508`) calls `peer_fetch_queue.step()` before `get_new_batch_prefill` (`:3691`). It is non-blocking, and finished requests are appended to `waiting_queue`.
- **Do not hold fetched requests inside the waiting queue the way HiCache does** with the `check_prefetch_progress` / `continue` at `:3866-3869`. A separate queue keeps the `PrefillAdder` loop and its `break` semantics untouched.

### 3.2 Steps (modelled on `DecodePreallocQueue` and `DecodeTransferQueue`)

1. **Local match.**
   - Match with `cow_mamba=False` and lock, as in `DecodePreallocQueue._match_prefix_and_lock` (`decode.py:660`).
   - `P_B` = the local device prefix. Set `max_len = floor((input_len-1)/grid)*grid`; the `-1` comes from `_compute_max_prefix_len`, `schedule_batch.py:1553`.
2. **Preallocate.**
   - Allocate pages for `[P_B, max_len)` and one Mamba slot from `req_to_token_pool.mamba_allocator`. Pattern: `_pre_alloc` (`decode.py:1760`) and `alloc_for_decode_prealloc` (`:1941`).
   - B would need these pages anyway to prefill the prompt, so this adds no peak memory.
   - It must be **counted in the admission reserve.** The admission-reserve OOM history and `num_tokens_pre_allocated` (`decode.py:1564`) are the precedents. Gate the preallocation on `_allocatable_token_budgets` (`decode.py:1625`).
3. **Create the receiver and send metadata.**
   - Create a `NixlKVReceiver(bootstrap_addr=A, room)` (`conn.py:3031`).
   - Call `send_metadata(dst_pages, aux_idx, state_indices=[[mamba_slot], dsa_pages, []], decode_prefix_len=P_B)`. The decode-side payload builders are at `decode.py:1382-1420`.
   - Rank 0 fires the in-band `/peer_export` POST to A on a thread pool. The pattern is decode's `_prefill_recompute_executor` (`common/conn.py:286`), which already drives a prefill's `/generate`.
4. **Poll.**
   - Each `step()` runs `poll_and_all_reduce` over the fetch queue: rid-ordered, and covering only rooms present on every rank. This is `DecodeTransferQueue.pop_transferred` (`decode.py:2280`), gated by `_poll_with_metadata_gate` (`:2248`).
   - **Who initiates:** A WRITEs, matching P/D.
   - **Why WRITE rather than a B-initiated READ:**
     - A sees completion locally (`check_xfer_state`, `conn.py:1374`), so it unlocks exactly when its pages are no longer read.
     - With READ, A must unlock on a TTL or an ack. A TTL shorter than a slow READ lets A evict and reuse pages mid-copy, and B then inserts corrupted KV silently.
     - The cost of WRITE is copy-engine work and 8 CPU transfer threads on A, the busier replica. In the lab, 218K tokens took ≤0.125 s.
5. **On Success**, read `L_A` from aux.
   - If `L_A <= P_B`, treat it as a miss.
   - Otherwise build `RadixKey(input_ids[:L_A], extra_key, cache_salt).page_aligned(256)`, and call `tree_cache.insert(InsertParams(key, value=cat(local_prefix_indices, new_pages[:L_A-P_B]), mamba_value=slot))` (`unified_radix_cache.py:579`, `mamba_component.py:218`).
   - **Free the duplicates.** If a concurrent insert on B already covers part of the range, free `value[P_B:result.prefix_len]` and free the slot when `result.mamba_exist`. This mirrors the free logic of `cache_finished_req` (`unified_radix_cache.py:886-955`).
   - Free the unused tail pages `[L_A, max_len)`.
   - `inc_lock_ref(result.last_device_node)` as a peer pin. Drop the step-1 lock.
   - Append the request to `waiting_queue`.
   - Normal admission then matches the node, COWs the Mamba state into the request's slot (`mamba_component.py:187-216`), and extends from `L_A`.
   - Release the peer pin once the request is admitted, or aborted.
6. **On Failed or timeout:**
   - free the pages and slot. This must be **deferred** if A may still be writing; see the hazards.
   - Append the request to `waiting_queue`. It falls back to a normal prefill and must never be aborted.

**Non-blocking.** Nothing waits on the network inside the loop. There are only two costs per `step()`: one gloo all-reduce when the queue is non-empty, and the HTTP POST on a thread.

## 4. Correctness hazards

| Hazard | Analysis | Mitigation |
|---|---|---|
| **Compressed DSA granularity** | The tree page is lcm(64, 64×index_kpool) = 256 (`unified_radix_cache.py:122-138`). One compressed index row lives in the first 64-page of each 256 group. The DSA compress tail is per-request (`memory_pool.py:4770-4911`) and is non-empty only when `seq_len % index_kpool != 0` (`utils.py:1144-1172`) | Export only 256-aligned `L_A`, so the tail is empty. Copy DSA state for every 64-page, which is a superset. Check that `_send_slot_state` with an empty index list is a no-op (`build_dsa_tail_transfer_blocks`); not verified |
| **Mamba checkpoint alignment** | States exist only at checkpoint nodes. `mamba_max_states_per_path` may drop intermediate ones (`mamba_component.py:252-308`). The match ends at the deepest node with a state, so `L_A` can be well short of the FULL hit (`mamba_branching_seqlen`) | Export exactly the best match: FULL up to the node that has the state. Never export FULL past the state, because B could not use it. Insert on B with `mamba_value` on the leaf |
| **Mamba int8 checkpoint pool** | With `--int8-mamba-ckpt-size`, tree states live in `MambaCheckpointPool` (`mamba_checkpoint_pool.py`, `memory_pool.py:1290`), which P/D does not register | Prod does not set it. The startup check refuses peer fetch if `mamba_ckpt_pool` is not None |
| **EAGLE draft KV** | The draft is DeepseekV3 NextN (MLA+DSA, no Mamba) (`models/glm5_next_nextn.py:24`). Draft KV shares the target's token indices, and P/D already appends it (`prefill.py:250-261`, `decode.py:557-568`, draft DSA at `utils.py:1490-1496`). `is_eagle` (bigram keys) is off whenever MAMBA is present (`unified_tree_core.py:412`), so the tree semantics for draft KV are the same as a local hit on A. B always extends ≥1 token, which regenerates the hidden states the first draft step needs | **Fetch the draft KV; do not recompute it.** It is the same descriptor list. Validate with the acceptance rate and greedy equality (§6) |
| **TP rank mapping** | MLA latent, indexer and draft KV are replicated across ranks. Mamba state is a per-rank head shard (see also the #304 key-collision bug the A/B patch fixed) | Rank i ↔ rank i only. P/D's equal-TP path already does this (`_prep_equal_tp_dlist`, `conn.py:673`). Refuse a peer whose `tp_size` or `kv_args` fingerprint differs |
| **Eviction on A** | A locked node cannot be evicted (`inc_lock_ref`). A node split keeps its pages. HiCache write-through copies but never moves pages. Unified-memory compaction (`flush_opportunistic`) *would* move them, but it is off in prod | Refuse peer export if `enable_unified_memory` is on. Unlock only on WRITE completion or Failed |
| **In-flight forward on A** | The node may be inserted before its forward finishes (overlap) | Event on `forward_stream` that the transfer worker waits on (reuse `_early_send_wait_event`) |
| **B aborts mid-fetch** (client disconnect) | Freeing the destination pages while A's WRITE is in flight corrupts whichever request reuses them | Use deferred release, as in `DecodeTransferQueue._defer_release` (`decode.py:2417-2470`, `deferred_kv_release_timeout` `:2063`): wait for A's drain ack before freeing |
| **A crashes mid-fetch** | B's poll goes `Failed`, via UCX error or `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` (`common/conn.py:280`). Partially written pages are garbage | Never insert unless the status is Success and aux `L_A` > `P_B`. Free the pages after the deferred-release timeout. Fall back to prefill |
| **B crashes mid-fetch** | A's WRITE fails, or completes into a dead mapping | Unlock on Failed or bootstrap timeout. Drop B's agent on reconnect (the agent name is a uuid per start, `conn.py:454`) |
| **Version skew** (rolling deploys; the A/B runs patched r3/r4 next to unpatched r1/r2) | KV from a different image, dtype or quant is silently wrong | Fingerprint in the handshake: model path, quant, `kv_cache_dtype_str`, page size, layer count, `index_kpool`, image digest. A mismatch means no fetch. Peer groups must be explicit (only r3↔r4 in an A/B) |
| **expandable_segments** | Prod uses `expandable_segments:True` (`prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml:175`). VMM memory cannot be exported with legacy cudaIpc, and UCX then **silently** falls back to host staging (0.3 GB/s in the lab; under CC it probably fails outright). Running with `False` cost decode mem-fraction in the lab | Preferred: keep `True`, and allocate only the KV, state and draft pools from a `torch.cuda.MemPool` backed by plain `cudaMalloc` (a `CUDAPluggableAllocator`). The hook already exists: `mem_cache/utils.py:81 maybe_init_custom_mem_pool`, honoured by MambaPool (`memory_pool.py:548`), MLA (`:4262`), the DSA tail (`:4797`) and the index cache (`index_key_cache.py:19`). Unknown: whether every GLM-5.3 buffer goes through it, and the CUDA-graph and fragmentation cost. Measure. At startup, assert `UCX_PROTO_INFO` shows `cuda_ipc` for a self-test transfer, and refuse peer mode otherwise |
| **CC host memory** | UCX `cuMemHostRegister` on pageable memory fails under PPCIe (error 801) | Pin the aux buffers (port of #335). KV and state are all VRAM, so the data path never touches host memory |
| **Accounting** | `cached_tokens` reported to the client and the ghost "actual cached" counter will include peer hits | Add a `source="peer"` breakdown (§6) so the A/B can attribute them |

## 5. Effort and alternatives

**A. Peer prefix fetch (this design)**

| Phase | Scope | Engineer-days |
|---|---|---|
| Prototype on gpu13 | Hint as a request JSON field. Two managers per process. Factored kv-args. `PeerExportQueue` and `PeerFetchQueue` on the happy path, with fallback on Failed. Pinned aux. `expandable_segments:False`. Gates G0–G3 | 8 (3 code, 1 CC bring-up, 2 correctness debugging, 2 lab and write-up) |
| Prod quality | Deferred release and abort paths. Both timeouts. Rank-divergence guards. Admission-reserve accounting. Version fingerprint. `cudaMalloc` MemPool for KV under `expandable_segments:True`, and requalifying its memory. Metrics. CPU unit tests of the queue state machines (style of `scripts/kvq/tests`). GPU fault-injection tests. Startup patch plus generator. Proxy `X-KV-Peer` and least-load routing within a sharing group (inference-proxy, about 4 d). CC qualification, then a one-host A/B | 25–30 |
| **Total** | | **about 6–7 weeks for one engineer** |

**B. Cheaper: "push on move" over the existing shared L3 tier**
- **How it works:**
  - The proxy sends the same hint.
  - B's prefetch already runs `wait_complete` against the shared `file` tier.
  - The missing piece is that A has usually not written the conversation out, because writes are selective and asynchronous.
  - Add a tiny `/peer_export` that makes A force write-through of the matched node (device → L2 → file). B's L3 prefetch then waits for it.
- **Cost:** about 4–6 days, all on code paths the current A/B already qualifies.
- **Pros:**
  - no NIXL in serving;
  - keeps `expandable_segments:True`;
  - no new rank protocol on A beyond one in-band request.
- **Cons:**
  - roughly 2 s or more per 200K tokens: D2H and H2D under CC go through bounce buffers;
  - holds RAM in the store;
  - B still waits on A's write.

**C. Route to the holder (status quo)**
- Costs 0 days. You give up load balance. `MAX_IMBALANCE` 8 already bounds the damage.

**D. "Both roles" P/D (remote prefill on the holder)**
- B acts as decode and A as prefill. A computes only the uncached suffix and ships the whole prefix.
- It reuses the P/D protocol end to end.
- But it needs one scheduler running NULL, PREFILL and DECODE queues together. The event loops are separate today (`prefill.py:605/644`, `decode.py:2486/2525`).
- It also puts compute on the busy replica.
- Estimate: 4–6 weeks at high risk. **Not recommended.**

**Decide with data before building A.**
- The fetch only pays if affinity is costing latency.
- Measure, per host, the time-weighted fraction during which the affinity replica has queued requests (`sglang:num_queue_reqs > 0`) while a sibling has spare capacity (`num_running_reqs` well below max). Weight it by the queue wait of those requests.
- Also count proxy affinity overrides by `MAX_IMBALANCE`, and how big their prefixes are: a moved 200K-token turn costs about 25–35 s to recompute versus about 0.1 s fetched.
- **If affinity-induced queueing is rare, do B or C.**

## 6. Test plan (gpu13 lab, under CC, `glm53kvq` project shape: two TP2 replicas on GPUs 4-5 and 6-7, each seeing 4-7, `pid: host`)

Before every launch, gate on the host coordination log and `nvidia-smi`.

| Gate | Pass criteria |
|---|---|
| **G0 bring-up** | Both managers register. A self-test transfer shows `cuda_ipc` in `UCX_PROTO_INFO`. No 801, Xid or NCCL errors |
| **G1 functional** | Turn 1 (218K tokens, hidden code) on r1, turn 2 on r2 with the hint. `peer_fetch_tokens` ≈ the prefix, rounded down to 256. Recall 3/3. TTFT < 1 s (cold is about 30 s). Fetch time ≤ 0.2 s |
| **G2 bit-level equivalence** | Greedy decoding: for 50 prompts, turn 2 on r1 (local hit) and turn 2 on r2 (peer fetch) produce identical tokens. Allow a small, quantified mismatch rate if the cold-vs-hit baseline also shows nondeterminism. EAGLE acceptance length matches within noise |
| **G3 edge matrix** | Prefix lengths 256k+{0,17,255}. `P_B` in {0, partial}. Concurrent identical fetches on B. A holding the prefix only on L2 (expect a miss, no crash). A's Mamba state shallower than its FULL hit |
| **G4 faults** | `docker kill` A mid-fetch: B falls back and its pool returns to baseline. Client abort on B mid-fetch: deferred release, no corruption (rerun G2 afterwards). Eviction pressure on A while exports are locked. Lock leaks: A's `protected_size` returns to baseline |
| **G5 quality** | GSM8K(150) ≥ 95%, with random routing and hints |
| **G6 load** | Base-tier mix at 3/5/7 req/s. Least-load routing plus hint, versus affinity. Compare TTFT p90, TPOT p90, unserved, and the holder's TPOT while exporting (copy-engine contention) |
| **G7 expandable** | Repeat G1 and G6 with `expandable_segments:True` plus the `cudaMalloc` KV MemPool. Still `cuda_ipc`; KV pool size within 2% of prod |
| **G8 soak** | 2 h, no Traceback, no gloo divergence |

**Metrics to add** (scheduler metrics collector, `model_name` label):
- `sglang:peer_fetch_requests_total{outcome="hit|miss|declined_short|declined_budget|fallback_failed|fallback_timeout|aborted"}`.
- `sglang:peer_fetch_tokens_total`: tokens inserted on B.
- `sglang:peer_export_tokens_total`: tokens sent by A.
- `sglang:peer_fetch_bytes_total`.
- `sglang:peer_fetch_latency_seconds` (histogram), from enqueue to insert. Split by `phase="handshake|transfer|insert"`.
- `sglang:peer_export_locked_tokens` (gauge on A) and `sglang:peer_fetch_inflight` (gauge on B).
- A `cached_tokens` breakdown with `source="peer"`, so ghost recovery (`ghost_actual_cached / ghost_pool_reused{within="inf"}`) can attribute gains.
- **Readout:** peer tokens as a share of `ghost_pool_other_replica_only_tokens_total`, plus the TTFT and TPOT deltas at equal load.

## 7. Unknowns, ranked

1. Two NIXL/UCX agents per process exporting the same VRAM through cuda_ipc, under CC.
2. Whether the `cudaMalloc` MemPool covers every KV, state and draft buffer, and what it does to CUDA-graph memory. Without it, prod must run with `expandable_segments:False`.
3. The copy-engine and NVLink contention cost to the holder's TPOT during large exports.
4. Whether the value of load-based routing justifies about 6 weeks (§5 measurement).
5. Small code-level checks: the empty DSA_TAIL no-op path, and `translate_mamba_indices` (identity unless unified memory is on, `memory_pool.py:1406`).
