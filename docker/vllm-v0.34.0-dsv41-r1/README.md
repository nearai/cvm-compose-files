# vLLM v0.34.0 for DeepSeek-V4.1-Flash-1017 with a LoRA adapter

Stock `vllm/vllm-openai` v0.34.0, pinned by digest, with the DS4.1 LoRA and confidential-computing
patches, bounded request handling, and pinned OpenSSL and glibc versions. Upstream numbers are
vllm-project/vllm pull requests or commits and name where a patch comes from.

The series is carried forward from the v0.33.0 build (`docker/vllm-v0.33.0-dsv41-r3`). Two of
its patches are not needed on v0.34.0 and have no file here; the rest are renumbered 01-15 to keep
the series gap-free, and patches 16 and 17 are new. The "v0.33 no." column gives each patch's number in the v0.33.0 series.

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

OS: openssl/libssl3t64 3.0.13-0ubuntu3.16, libc6/libc-bin/libc6-dev/libc-dev-bin 2.39-0ubuntu8.12
and linux-libc-dev 6.8.0-149.149, pinned by exact version; the `nearai.os.packages` label lists
all of them. The image needs no Python package upgrades.

To bump an apt pin, change the version in every place it appears: the `pkgs=` list and the
`apt-get install` arguments in the `Dockerfile`, and the `nearai.os.packages` and
`nearai.os.linux_libc_dev` labels. Then refresh `SHA256SUMS` and the `Dockerfile` and `SHA256SUMS`
pins in the publish workflow.

`apply-patches.py` refuses any vLLM other than 0.34.0, dry-runs and applies the patches in order,
and checks every touched file against `PATCHED-FILES.sha256`. Publish with
`.github/workflows/publish-vllm-v0.34.0-dsv41-r1.yaml` from a merged commit on main.
