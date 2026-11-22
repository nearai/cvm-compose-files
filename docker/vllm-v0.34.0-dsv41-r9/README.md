# vLLM v0.34.0 for DeepSeek-V4.1-Flash-1017 with a LoRA adapter and for GLM-5.3-Flash

Stock `vllm/vllm-openai` v0.34.0, pinned by digest, with the DS4.1 LoRA and confidential-computing
patches, a FlashInfer workspace check fix, quantized GLM-5.3-Flash MTP draft loading, opt-in GLM-5.3-Flash KDA changes, bounded request handling, and pinned OpenSSL and
glibc versions. Upstream numbers are vllm-project/vllm pull requests or commits and name where a patch comes from.

This image is r8 (`docker/vllm-v0.34.0-dsv41-r8`) plus patch 25; patches 01 to 24 and the OS pins are
byte-identical to r8's. r8 is r7 (`docker/vllm-v0.34.0-dsv41-r7`) plus patch 24; r7 is r6 (`docker/vllm-v0.34.0-dsv41-r6`) plus patch 23. r6 is r5 (`docker/vllm-v0.34.0-dsv41-r5`) plus patch 22. r5 is r4 (`docker/vllm-v0.34.0-dsv41-r4`) with linux-libc-dev raised from 6.8.0-149.149
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
| `VLLM_ADMIT_ROLE` | unset | `prefill` enables admission on an engine whose KV role is `kv_both`; `kv_producer` is enabled without it |

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

`tests/` holds 46 CPU-only tests (`test_admission.py`, `test_real_scheduler.py`). Run them in the built image with
`PYTHONPATH` pointing at the patched tree, `P26_UTILS` set to vLLM's `tests/v1/core/utils.py` and `P26_MODEL` set to a
local model directory read only for its config: `python3 -m pytest -q -p no:cacheprovider tests/`.

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
`.github/workflows/publish-vllm-v0.34.0-dsv41-r9.yaml` from a merged commit on main.
