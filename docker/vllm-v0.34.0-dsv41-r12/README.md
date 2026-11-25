# vLLM v0.34.0 for DeepSeek-V4.1-Flash-1017 with a LoRA adapter and for GLM-5.3-Flash

Stock `vllm/vllm-openai` v0.34.0, pinned by digest, with the DS4.1 LoRA and confidential-computing
patches, a FlashInfer workspace check fix, quantized GLM-5.3-Flash MTP draft loading, opt-in GLM-5.3-Flash KDA changes, bounded request handling, and pinned OpenSSL and
glibc versions. Upstream numbers are vllm-project/vllm pull requests or commits and name where a patch comes from.

This image is r10 (`docker/vllm-v0.34.0-dsv41-r10`) plus patches 28 to 30; patches 01 to 27 and the OS pins are
byte-identical to r10's. r10 is r9 (`docker/vllm-v0.34.0-dsv41-r9`) plus patches 26 and 27. r9 is r8 (`docker/vllm-v0.34.0-dsv41-r8`) plus patch 25. r8 is r7 (`docker/vllm-v0.34.0-dsv41-r7`) plus patch 24; r7 is r6 (`docker/vllm-v0.34.0-dsv41-r6`) plus patch 23. r6 is r5 (`docker/vllm-v0.34.0-dsv41-r5`) plus patch 22. r5 is r4 (`docker/vllm-v0.34.0-dsv41-r4`) with linux-libc-dev raised from 6.8.0-149.149
to 6.8.0-151.151. r4 is r3 (`docker/vllm-v0.34.0-dsv41-r3`) plus patches 20 and 21.
The series is carried forward from the v0.33.0 build (`docker/vllm-v0.33.0-dsv41-r3`). Two of
its patches are not needed on v0.34.0 and have no file here; the rest are renumbered 01-15 to keep
the series gap-free, and patches 16 to 22 are new. The "v0.33 no." column gives each patch's number in the v0.33.0 series.

| patch | v0.33 no. | what it does | status on v0.34.0 | origin |
|---|---|---|---|---|
| 01-dsv41-supports-lora-wo-b | 01 | the DS4.1 vision-language model accepts LoRA only when the adapter targets `wo_b` alone (no tower or connector LoRA) | reduced: v0.34.0 declares LoRA support itself, only the fail-closed check remains | NEAR |
| 02-moe-lora-only-when-targeted | 02 | builds MoE LoRA wrappers only when MoE modules are targeted | ported | NEAR |
| (none) | 03 | maps LoRA token indices onto decoder-replay rows | replaced by patch 17: v0.34.0 builds no decoder replay when LoRA is enabled | NEAR |
| 03-cc-disable-uva | 04 | `VLLM_NEAR_DISABLE_UVA=1` keeps the engine on explicit copies, so the GPU reads only memory it owns; checked before upstream's own detection | ported | NEAR |
| 04-engram-cpu-gather | 05 | gathers host Engram rows on the CPU under CC; inert with `cpu_offload:false` | ported (two hunks rewritten) | NEAR |
| 05-engram-fast-gather | 06 | one host sync per step and pinned async H2D for 04; inert with `cpu_offload:false` | ported | NEAR |
| 06-engram-peer-reads-kill-switch | 07 | `VLLM_NEAR_ENGRAM_PEER_READS=0` keeps the DP-collective Engram lookup instead of peer-HBM reads | ported | NEAR |
| 07-error-sanitizer-linear-time | 08 | the error-message path sanitizer runs in time linear in the message length | ported | #60233, tests from #52120 |
| 08-cached-tokens-from-engine-only | 09 | cached-token counts come from the engine; a client-sent count applies only with `VLLM_TRUST_REMOTE_PREFILL_CACHED_TOKENS=1` | ported | #64918 |
| 09-stop-and-bad-words-bounds | 10 | caps `stop_token_ids`, `stop` and `bad_words`, including `stop_token_ids` on the Responses API; also changes `deepseek_v32_encoding.py`: `render_message` takes a required `last_user_idx` and a new optional `last_assistant_idx`, `drop_thinking_messages` requires `last_user_idx`, and a tool message is rendered after the last assistant message seen by `encode_messages` instead of by scanning back | ported, extended | #54721 |
| 10-detok-bounded-holdback | 11 | flushes incremental detokenization after 16 tokens that never complete a UTF-8 character | ported | #63668 |
| 11-engine-add-rollback | 12 | rolls back request state when the engine add fails; requests with `top_k` below -1 or above 2**63-1, or `stream_interval` above 2**63-1, are rejected with a validation error; the Cohere `k` field has the same bound | ported, extended | #56212 |
| 12-mm-uuid-length-cap | 13 | caps a multimodal `uuid` at 128 characters, also for the renderer's direct `multi_modal_uuids` and for client-supplied multimodal feature hashes; a longer value is rejected | ported, extended | #57221 |
| 13-media-fetch-public-ips-only | 14 | engine-side media fetches resolve to public unicast IPs only (loopback, private, link-local, multicast, NAT64 and reserved addresses are refused), pin the resolved IP and re-check it after a redirect; both clients ignore proxy environment variables; the per-IP pool cache holds at most 64 pools and is closed with the session | ported | #57825, extended |
| 14-media-client-no-cookies | 15 | shared media HTTP clients keep no cookies: a `Set-Cookie` response header is not stored or sent back | ported | #56588 |
| 15-cc-explicit-copies | 16 | confidential-compute mode comes from `VLLM_CONFIDENTIAL_COMPUTE=1` alone (no hardware auto-detection); a failed D2H copy raises at the next event, stream or device synchronize or query; `VLLM_CONFIDENTIAL_COMPUTE_KEEP_UVA=1` keeps the runner's UVA buffers | reduced: the explicit-copy machinery is upstream in v0.34.0 | NEAR delta on top of #59981 |
| 16-engram-prestage-full-graph | (new) | Stage host-memory Engram rows before CUDA graph replay so decode can use FULL graphs; off unless `VLLM_NEAR_ENGRAM_PRESTAGE=1` (needs `VLLM_NEAR_ENGRAM_FAST_GATHER=1`); `VLLM_NEAR_ENGRAM_PRESTAGE_CHECK=N` compares the first N prestaged steps with the in-forward rows and logs counts only | new | NEAR |
| 17-lora-decoder-replay | (new) | with `VLLM_NEAR_LORA_DECODER_REPLAY=1`, decoder SWA bounded replay stays on when LoRA is enabled; the replay rows get their own LoRA token indices, and replay raises an error instead of running without them; replay under a CUDA graph is refused with LoRA; off by default, where v0.34.0's behavior (no replay with LoRA) is unchanged | new | NEAR |
| 18-dp-global-prefill-cadence | (new) | with `VLLM_NEAR_DP_GLOBAL_PREFILL_CADENCE=1` and a prefill schedule interval above 1, data-parallel engine ranks share one prefill release interval: one CPU all-reduce on each release step tells every rank whether any rank was capacity-bound, the per-rank capacity escape is cleared, the interval halves (floor `VLLM_NEAR_DP_GLOBAL_PREFILL_CADENCE_MIN`, default 2) when any rank was bound and doubles back after `VLLM_NEAR_DP_GLOBAL_PREFILL_CADENCE_RESTORE` (default 2) clear releases; the three variables must be equal on all ranks; off by default, where v0.34.0's behavior is unchanged | new | NEAR |
| 19-glm5next-mtp-quant-weight-mapper | (new) | the GLM-5.3-Flash MTP draft applies the quantization config's checkpoint weight and cache-scale mappers to its own layer's weights, so a W4AFP8 checkpoint's routed-expert tensors load under the packed names the W4A8 MoE method registers; checkpoints whose mappers rename nothing load as before | new | NEAR, glm5next counterpart of upstream #68632 |
| 20-glm5next-kda-fp8-projections | (new) | with `VLLM_NEAR_KDA_FP8=1`, the GLM-5.3-Flash KDA layer's `in_proj_qkvbfg_a` and `o_proj` linears are quantized online to fp8 (per-output-channel weight scale, dynamic per-token activation scale) while the weights load; other KDA parameters keep their dtype; off by default | new | NEAR |
| 21-glm5next-kda-contiguous-slices | (new) | with `VLLM_NEAR_KDA_SLICE=1`, the KDA layer in a mixed spec-decode step uses slices instead of `index_select` and `index_copy_` when the spec and non-spec tokens are two contiguous ranges, decided from host-side metadata without a device sync; any other layout takes the existing path; `VLLM_NEAR_KDA_SLICE_VERIFY=1` checks the plan against the index tensors (one sync per step, debugging only); off by default | new | NEAR |
| 22-nixl-agent-thread-sync | (new) | the NIXL KV connector creates its agent with `NIXL_THREAD_SYNC_STRICT`, so the handshake thread and the engine thread no longer call the agent concurrently without a lock | ported | #59877 (fixes #59870) |
| 23-nixl-no-progress-thread | (new) | the NIXL KV connector reads `enable_prog_thread` (default `true`) from `kv_connector_extra_config` and passes it to its NIXL agent config; with `false`, the UCX progress thread is off and the engine drives UCX progress from its own step calls. Under confidential computing with Protected PCIe, the progress thread can block in the CUDA driver while holding the UCX worker lock. The effective value is logged once at INFO; a non-boolean value stops the engine at startup; the build checks that NIXL accepts the option | new | NEAR |
| 24-flashinfer-cc-pinned-workspace-check | (new) | FlashInfer SM90 sparse MLA FP8 KV under confidential computing: the planner no longer requires the host scratch workspace to report `is_pinned()`, because pinned host tensors report false there although the allocation succeeds and the buffer is only copied to the device with an explicit copy; the CPU-device and contiguity checks stay. The patch is to the installed FlashInfer 0.7.3 (`flashinfer/jit/attention/modules.py`), not to vLLM; `apply-patches.py` refuses any other FlashInfer version or an unexpected original file | new | NEAR |
| 25-class-admission-429 | (new) | class-aware engine admission for prefill engines: an add request that would exceed a per-rank uncached-token backlog limit is refused before any KV or connector state is allocated, and the client gets HTTP 429 with `Retry-After`; see "Patch 25" below | new | NEAR |
| 26-mamba-mid-block-state-cache | (new) | on a hybrid model with a Mamba or KDA layer in `align` mode and a prefix match unit smaller than the block size, a block that a chunk leaves holding a state that is not its block-end state is no longer published under the full-block hash; the state is recorded per request and cached under the matching partial hash instead | ported | #67717 (merged upstream) |
| 27-partial-hit-cow-defer | (new) | a partial prefix-cache hit whose copy-on-write source block was keyed in the same scheduling step is deferred to the next step, because that block's KV is written by this step's forward pass, after the copy would run | ported | #67451 (open upstream, not merged) |
| 28-nixl-mamba-checkpoint-cut | (new) | NIXL P/D on a hybrid model with prefix caching and a Mamba or KDA group in `align` mode: the prefiller stops at the last Mamba checkpoint a local prefill would cache, and the decoder loads up to that point and recomputes the prompt tail, so the transferred state is one the decoder can prefix-cache; in pull mode the prefiller's reported count wins when the two sides disagree; the NIXL connector version becomes 1017; see "Patches 28 and 29" below | ported | #67064 (open upstream, not merged) |
| 29-requery-connector-after-hybrid-reconcile | (new) | when a decoder's full-attention prefix hit diverges from its Mamba hit and the first connector query finds nothing to load, the scheduler asks the connector again from the reconciled hit, so the decoder loads the missing state instead of recomputing from the older checkpoint | new | NEAR |
| 30-admission-long-reject-all | (new) | with `VLLM_ADMIT_LONG_REJECT_ALL=1`, patch 25's admission refuses every long request (uncached tokens above `VLLM_ADMIT_LONG_CUTOFF_TOKENS`) with `overloaded_long_count`, whatever the in-flight count; P/D requests stay exempt; off by default | new | NEAR |
| (none) | 17 | keeps the EP fault flag referenced until the copy stream has read it | upstream-obsolete: v0.34.0 holds the flag itself | upstream fbcadab2 |

Defaults this image sets, which differ from stock v0.34.0: 08 takes cached-token counts from the
engine; 09 caps stop lists and changes tool-message rendering for DeepSeek chat templates; 10
flushes detokenization after 16 incomplete tokens; 11 rejects out-of-range `top_k` and
`stream_interval`; 12 rejects a `uuid` over 128 characters; 13 resolves engine-side media fetches
to public unicast IPs only and ignores `HTTP_PROXY`/`HTTPS_PROXY`; 14 keeps no cookies; 15 turns
confidential-compute mode on only from `VLLM_CONFIDENTIAL_COMPUTE=1`.

`VLLM_CONFIDENTIAL_COMPUTE_KEEP_UVA=1` (patch 15) affects only the runner buffers.
`VLLM_NEAR_DISABLE_UVA=1` (patch 03) takes precedence over it.

Patch 16 changes nothing unless `VLLM_NEAR_ENGRAM_PRESTAGE=1` is set.
Patch 17 changes nothing unless `VLLM_NEAR_LORA_DECODER_REPLAY=1` is set.
Patch 18 changes nothing unless `VLLM_NEAR_DP_GLOBAL_PREFILL_CADENCE=1` is set.
Do not enable it with elastic expert parallelism or engine fault tolerance: a rank that restarts or joins starts from the base interval and its collectives would no longer match its peers.
Patch 19 changes only how the GLM-5.3-Flash MTP draft loads its weights.
Patch 23 changes nothing unless `enable_prog_thread` is set to `false` in `kv_connector_extra_config`.
Patch 24 relaxes only the pinned-memory check in FlashInfer's workspace validation; it changes nothing else.
Patch 25 changes nothing unless a `VLLM_ADMIT_*` limit is set.
Patches 26 and 27 matter only on hybrid models with prefix caching when the server runs with `--prefix-match-unit`; without it they change nothing.
Patch 28 is on by default for NIXL P/D on a hybrid model with prefix caching and a Mamba or KDA group in `align` mode, and it changes the NIXL connector version for every model; both sides of a P/D pair must run it, or the handshake fails with a compatibility hash mismatch.
Patch 29 runs only on a decoder whose full-attention and Mamba prefix hits diverge on a connector that allows it (NIXL).
With patch 28 every decoder recomputes a short prompt tail, so its steps mix prefill and decode tokens. Run decoders with `--compilation-config` `cudagraph_mode` `FULL_AND_PIECEWISE` and capture sizes up to 2048; with `FULL_DECODE_ONLY` those mixed steps run without CUDA graphs and decode slows.
Patch 30 changes nothing unless `VLLM_ADMIT_LONG_REJECT_ALL` is set.
Patches 20 and 21 change nothing unless `VLLM_NEAR_KDA_FP8=1` or `VLLM_NEAR_KDA_SLICE=1` is set.

Limits of patches 20 and 21. Set the variables before the server starts: `VLLM_NEAR_KDA_SLICE` is read
once at import and `VLLM_NEAR_KDA_FP8` when each KDA layer is built.

- `VLLM_NEAR_KDA_FP8=1` (patch 20) is not checked for shapes at startup; test each tensor-parallel size you deploy.
  The merged `in_proj_qkvbfg_a` projection has, per rank, an output width of
  `3 * (num_heads * head_dim / tp) + num_heads / tp + 2 * head_dim` (the two `head_dim` shards are replicated, not split).
  The fp8 kernels may need `num_heads / tp` to be a multiple of 16. If it is not, kernel selection can fail or fall
  back to a slower path when the weights load. The patch adds no check for this.
- The fp8 override replaces the layer's quantization config, whatever the checkpoint declares. It assumes the
  checkpoint stores `in_proj_qkvbfg_a` and `o_proj` in bf16 (the KDA layers already ignore the checkpoint's
  quantization config). A checkpoint that stores these projections already quantized is not supported and
  would fail at load, possibly with an unclear error.
- A separate online-quantization config is built for each layer, so the loader's per-layer quantization summary may not combine them.
  This affects only that summary.
- `VLLM_NEAR_KDA_SLICE=1` (patch 21) uses slices only when the spec-decode and non-spec tokens form one block each,
  with all spec sequences first or all non-spec sequences first. Any other layout takes the original
  `index_select` path for that step. One known case is a batch with zero-length padded sequences after a
  non-speculative step is followed by a speculative one. The fallback is silent and costs only speed; it
  does not mean the option is broken. `VLLM_NEAR_KDA_SLICE_VERIFY=1` checks the plan but does not report fallbacks.

## Patch 25: class-aware admission (HTTP 429 + Retry-After)

Off unless a limit below is set; with none set, behavior is unchanged. The engine core and the API server both
read these variables, so set them on the same container.

| variable | default | meaning |
|---|---|---|
| `VLLM_ADMIT_SHORT_MAX_TOKENS` | unset (off) | refuse a short request when the uncached tokens already queued for prefill plus its own exceed this, per engine core |
| `VLLM_ADMIT_LONG_MAX_INFLIGHT` | unset (off) | refuse a long request when this many long requests are already admitted and have not finished prefill, per engine core |
| `VLLM_ADMIT_LONG_YIELD_SHORT_TOKENS` | unset (off) | refuse a long request while queued short uncached tokens exceed this |
| `VLLM_ADMIT_LONG_CUTOFF_TOKENS` | 65536 | uncached prompt tokens at or below this make a request short |
| `VLLM_ADMIT_RETRY_AFTER_SHORT_S` | 1 | `Retry-After` seconds for a refused short request |
| `VLLM_ADMIT_RETRY_AFTER_LONG_S` | 10 | `Retry-After` seconds for a refused long request |
| `VLLM_ADMIT_LONG_REJECT_ALL` | unset (off) | patch 30: `1`, `true`, `yes` or `on` refuses every long request, whatever the in-flight count; meant for a decode engine run with `VLLM_ADMIT_ROLE=prefill`, so that a direct request that misses the decoder's prefix cache is refused while P/D requests stay exempt |
| `VLLM_ADMIT_ROLE` | unset | forces the role for any engine, including one with no KV connector; set it to `prefill` only on prefill engines (a `kv_producer` engine is enabled without it) |

- A lone short request is always admitted: a short request is refused only if the short backlog is above zero and
  the backlog plus its own uncached tokens exceed `VLLM_ADMIT_SHORT_MAX_TOKENS`. Size the limit with this in mind;
  one short request larger than the limit still gets through when nothing is queued.
- Naming: the module (`admission_p26`), its class, the `_p26_*` request attributes and the test variables use
  "p26"/"P26", a historic name; the patch file is number 25 in this series.
- Class is by uncached tokens: the prompt length minus the prefix-cache hit, read with the existing side-effect-free
  probe at the moment of the decision. A long conversation with a warm cache counts as short.
- A refused request gets an `overloaded_short`, `overloaded_long_count` or `overloaded_long_yield` error from the engine
  core before the request is added, so it holds no KV blocks or connector state. The API server answers 429 with
  `Retry-After`. The counter `vllm:admission_rejections{reason}` counts refusals per API server process.
- The backlog is recomputed from the scheduler's request table on each add, so finish, abort, disconnect and preemption
  cannot leave a stale count.
- The decode side never sheds: a `kv_consumer` engine, a request that carries remote-prefill parameters, session
  continuations, resumable and pooling requests are admitted. If the probe fails, the request is admitted.
- A streaming request refused by the engine core gets a 200 with an in-stream error, as in stock vLLM; use it with
  non-streaming prefill legs.

`tests/` holds 62 CPU-only tests (`test_admission.py`, `test_real_scheduler.py`). It is hashed in `SHA256SUMS` for
review and, deliberately, is not copied into the image. Run the tests against a patched vLLM tree with `PYTHONPATH`
pointing at it and three variables set: `P26_ADMISSION` (path to the patched tree's
`vllm/v1/core/sched/admission_p26.py`), `P26_UTILS` (vLLM's `tests/v1/core/utils.py`) and `P26_MODEL` (a local model
directory read only for its config): `python3 -m pytest -q -p no:cacheprovider tests/`.

Known limits of patch 25:

- The backlog is recomputed on every add by re-probing the prefix cache for each request that has not started. The cost
  of one add therefore grows with the queue depth (waiting requests times prompt blocks) and runs on the engine-core
  loop, only when a limit is set. It has not been measured under a deep queue.
- The serving layer's `Retry-After` header is derived from the error code and message mapping in the patch. Rewording
  those messages without updating the mapping drops the header; the 429 status is unaffected.

## Patches 26 and 27: prefix-cache fixes with `--prefix-match-unit`

Both patches touch only `vllm/v1/core/{block_pool,kv_cache_manager,single_type_kv_cache_manager}.py` and are verbatim
backports; they apply to v0.34.0 with line offsets only.

- Patch 26 (vllm-project/vllm#67717, merged upstream). With a hybrid model whose Mamba or KDA state is cached in `align`
  mode and a prefix match unit smaller than the block size, a request resumed from a mid-block state could publish a
  block under the full-block hash while that block held the earlier, mid-block state. A later request matching the full
  hash would then resume from the wrong state. The patch adds a `_refine_block_mask` hook to the base cache manager; the
  Mamba manager records which blocks hold a mid-block state and excludes them from the full-block hash.
- Patch 27 (vllm-project/vllm#67451, still open upstream, not merged). When a partial prefix hit copies a block
  (copy-on-write) whose source block was keyed in the same scheduling step, the source's KV is not yet written when the
  copy runs. The block pool now remembers the blocks keyed in the current step and the cache manager defers such a
  request to the next step. Because the pull request is unmerged, upstream may still change it; re-diff against its final
  form before carrying it into a later image.

Cost: patch 27 adds one scheduling step of latency to each affected partial hit. Patch 26 matters only for resumed
chunks that reach the block after the next one, which at a 4736-token block means prompts of roughly 9.5k tokens or more.

`tests/` also holds the backport tests: `upstream-67717-test.diff` and `upstream-67451-test.diff` are the upstream test
hunks for `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py` (apply them to vLLM v0.34.0's copy of that
file), and `test_pmu64_block4736.py` is a scheduler-level regression test at block size 4736 with a prefix match unit of
64; it imports helpers from the patched upstream test file. They are CPU-only, run from a vLLM v0.34.0 source tree with
patches 26 and 27 applied, and fail without the patches.

## Patches 28 and 29: NIXL P/D transfers stop at the Mamba checkpoint

- Patch 28 (vllm-project/vllm#67064, open upstream, not merged). Without it, the prefiller stops one token (or the
  speculative lookahead) short of the prompt, and the decoder receives a Mamba state at a position it cannot
  prefix-cache. With it, both sides compute the same cut from the prompt length: the last block-aligned Mamba
  checkpoint that a local prefill of the prompt would cache, after the hash lookahead and the EAGLE/MTP block drop.
  The decoder recomputes the tail past the cut, which is at most about one block. The connector binds to the cache
  manager only when prefix caching is on and a Mamba group is in `align` mode; otherwise the cut is as before.
  The prefiller then keeps one block less of its own prefix cache per P/D request, because the block ending at the
  cut is not hashed on the prefill side.
- The NIXL connector version is 1017 instead of upstream's 16, off the upstream sequence, so a peer without patch 28
  never matches. The build checks the value.
- Patch 29. NIXL lets a decoder report a full-attention hit that is longer than its Mamba hit. With patch 28 the load
  target is the cut, so a full-attention hit that already reaches the cut makes the first query return nothing; the
  scheduler then reconciles to the hit every group agrees on. Patch 29 queries the connector again from that point,
  so the decoder loads the blocks up to the cut, with the state at the cut. The re-query is skipped for requests that
  skip reading external KV, with bounded SWA replay, with an auxiliary output connector, and for a reconciled hit that
  is not block-aligned. A pending lookup falls back to recompute, as before.

`tests/test_nixl_checkpoint_cut.py` holds the CPU tests for patches 28 and 29: the upstream cases, cuts at block
size 2432, the re-query, the version change and a mixed-pair refusal through the real handshake path. They import the
patched vLLM and run inside the built image without a GPU:
`python3 -m pytest -q -p no:cacheprovider tests/test_nixl_checkpoint_cut.py`.

OS: openssl/libssl3t64 3.0.13-0ubuntu3.16, libc6/libc-bin/libc6-dev/libc-dev-bin 2.39-0ubuntu8.12
and linux-libc-dev 6.8.0-151.151, pinned by exact version; the `nearai.os.packages` label lists
all of them. The image needs no Python package upgrades.

To bump an apt pin, change the version in every place it appears: the `pkgs=` list and the
`apt-get install` arguments in the `Dockerfile`, and the `nearai.os.packages` and
`nearai.os.linux_libc_dev` labels. Then refresh `SHA256SUMS` and the `Dockerfile` and `SHA256SUMS`
pins in the publish workflow.

`apply-patches.py` refuses any vLLM other than 0.34.0, any FlashInfer other than 0.7.3 and any original
`flashinfer/jit/attention/modules.py` other than the expected file, dry-runs and applies the patches in order,
and checks every touched file against `PATCHED-FILES.sha256`. Publish with
`.github/workflows/publish-vllm-v0.34.0-dsv41-r12.yaml` from a merged commit on main.
