# vLLM v0.33.0 for DeepSeek-V4.1-Flash-1017 with a LoRA adapter

Stock `vllm/vllm-openai` v0.33.0, pinned by digest, with the DS4.1 LoRA and confidential-computing
patches, bounded request handling, and pinned OpenSSL and glibc versions. Upstream numbers are
vllm-project/vllm pull requests or commits and name where a patch comes from.

| patch | what it does | origin |
|---|---|---|
| 01-dsv41-supports-lora-wo-b | DS4.1 declares LoRA support for `wo_b` | NEAR |
| 02-moe-lora-only-when-targeted | builds MoE LoRA wrappers only when MoE modules are targeted | NEAR |
| 03-decoder-replay-lora-rows | maps LoRA token indices onto decoder-replay rows; padded rows get no adapter | NEAR, adapted to v0.33 row padding |
| 04-cc-disable-uva | `VLLM_NEAR_DISABLE_UVA=1` keeps the engine on explicit copies, so the GPU reads only memory it owns | NEAR |
| 05-engram-cpu-gather | gathers host Engram rows on the CPU under CC; inert with `cpu_offload:false` | NEAR |
| 06-engram-fast-gather | one host sync per step and pinned async H2D for 05; inert with `cpu_offload:false` | NEAR |
| 07-engram-peer-reads-kill-switch | `VLLM_NEAR_ENGRAM_PEER_READS=0` keeps the DP-collective Engram lookup instead of peer-HBM reads | NEAR |
| 08-error-sanitizer-linear-time | the error-message path sanitizer runs in time linear in the message length | #60233, tests from #52120 |
| 09-cached-tokens-from-engine-only | cached-token counts come from the engine; a client-sent count applies only with `VLLM_TRUST_REMOTE_PREFILL_CACHED_TOKENS=1` | #64918 |
| 10-stop-and-bad-words-bounds | caps `stop_token_ids`, `stop` and `bad_words` | #54721 |
| 11-detok-bounded-holdback | flushes incremental detokenization after 16 tokens that never complete a UTF-8 character | #63668 |
| 12-engine-add-rollback | rolls back request state when the engine add fails; bounds `top_k` and `stream_interval` | #56212 |
| 13-mm-uuid-length-cap | caps a multimodal content-part `uuid` at 128 characters | #57221 |
| 14-media-fetch-public-ips-only | engine-side media fetches resolve to public IPs only, pin the resolved IP and re-check it after a redirect; async resolution runs off the event loop | #57825, extended |
| 15-media-client-no-cookies | shared media HTTP clients keep no cookies | #56588 |
| 16-cc-explicit-copies | under `VLLM_CONFIDENTIAL_COMPUTE=1` only: H2D on a staging stream, D2H from a worker thread, pinned `NonUvaBuffer` copies; wraps `torch.Tensor.copy_`, `to` and `cuda` | NEAR port of #59981 |
| 17-cc-retain-ep-fault-flag | keeps the EP fault flag referenced until the copy stream has read it | upstream fbcadab2, ported onto 16 |

Patches 08-15 are byte-identical to 01-04, 08 and 10-12 of `docker/vllm-v0.33.0-secure-r1`.
Patches 04-07 differ from `docker/vllm-dsv41-v033` only in comment text.

Defaults this image sets, which differ from stock v0.33.0: 09 takes cached-token counts from the
engine; 10 caps stop lists; 14 resolves engine-side media fetches to public IPs only, and on the
async path honours `HTTP_PROXY`/`HTTPS_PROXY` only when the proxy itself is on a public IP (the
compose files set no proxy).

OS: openssl/libssl3t64 3.0.13-0ubuntu3.16, libc6/libc-bin/libc6-dev/libc-dev-bin 2.39-0ubuntu8.12
and linux-libc-dev 6.8.0-149.149, pinned by exact version. The image needs no Python package
upgrades.

`apply-patches.py` refuses any vLLM other than 0.33.0, dry-runs and applies the patches in order,
and checks every touched file against `PATCHED-FILES.sha256`. Publish with
`.github/workflows/publish-vllm-v0.33.0-dsv41-r3.yaml` from a merged commit on main.
