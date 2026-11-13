# vLLM v0.33.0 for DeepSeek-V4.1-Flash-1017 with a LoRA adapter

Stock `vllm/vllm-openai` v0.33.0 (pinned by digest) with the DS4.1 LoRA and confidential-computing patches, the upstream security fixes this deployment can reach, and pinned OpenSSL and glibc security updates. It supersedes the r2 dir proposed separately. Upstream numbers are vllm-project/vllm pull requests or commits; most are still open.

| patch | what it does | origin |
|---|---|---|
| 01-dsv41-supports-lora-wo-b | DS4.1 declares LoRA support for `wo_b` | NEAR |
| 02-moe-lora-only-when-targeted | builds MoE LoRA wrappers only when MoE modules are targeted | NEAR |
| 03-decoder-replay-lora-rows | maps LoRA token indices onto decoder-replay rows; padded rows get no adapter | NEAR, adapted to v0.33 row padding |
| 04-cc-disable-uva | `VLLM_NEAR_DISABLE_UVA=1` turns UVA off (TDX+PPCIe reads stale pinned host memory); UVA users fall back to explicit copies | NEAR |
| 05-engram-cpu-gather | gathers host Engram rows on the CPU under CC; inert with `cpu_offload:false` | NEAR |
| 06-engram-fast-gather | one host sync per step and pinned async H2D for 05; inert with `cpu_offload:false` | NEAR |
| 07-engram-peer-reads-kill-switch | `VLLM_NEAR_ENGRAM_PEER_READS=0` keeps the DP-collective Engram lookup instead of peer-HBM reads | NEAR |
| 08-error-sanitizer-redos | the error-message path sanitizer runs in linear time (GHSA-f2g9-pmwr-xwc7) | code of open #60233 |
| 09-remote-prefill-cache-accounting-trust | rejects and ignores client-sent remote-prefill cached-token counts unless `VLLM_TRUST_REMOTE_PREFILL_CACHED_TOKENS=1` | #64918 |
| 10-stop-and-bad-words-bounds | caps `stop_token_ids`, `stop` and `bad_words` | #54721 |
| 11-detok-bounded-holdback | flushes incremental detokenization after 16 tokens that never complete a UTF-8 character | #63668 |
| 12-engine-add-rollback | rolls back request state when the engine add fails; bounds `top_k` and `stream_interval` | #56212 |
| 13-mm-uuid-length-cap | caps a multimodal content-part `uuid` at 128 characters | #57221 |
| 14-media-fetch-ssrf-filter | engine-side media fetches refuse non-public IPs, pin the resolved IP and re-check redirects; async resolution runs off the event loop | #57825, extended |
| 15-media-client-no-cookies | shared media HTTP clients keep no cookies | #56588 |
| 16-cc-explicit-copies | under `VLLM_CONFIDENTIAL_COMPUTE=1` only: H2D on a staging stream, D2H from a worker thread, pinned `NonUvaBuffer` copies; wraps `torch.Tensor.copy_`, `to` and `cuda` | NEAR port of the CC copy paths, not merged #59981 |
| 17-cc-retain-ep-fault-flag | keeps the EP fault flag referenced until the copy stream has read it | upstream fbcadab2, ported onto 16 |

Patch numbers before the renumbering to `patches/NN-name.diff`: 01-09 unchanged, 12→10, 13→11, 17→12, 21→13, 22→14, 23→15, 25→16, 26→17 (18, the regex compile worker, was dropped). 08-15 are byte-identical to 01-04, 08 and 10-12 of `docker/vllm-v0.33.0-secure-r1`; 16 is byte-identical to the CC patch proposed for r2. 04-07 differ from `docker/vllm-dsv41-v033` only in comment text.

Default behaviour changes: 09 stops trusting client remote-prefill cached-token counts; 10 caps stop lists; 14 blocks media fetches to private IPs, and also refuses an `HTTP_PROXY`/`HTTPS_PROXY` on a non-public IP on the async path (the compose files set no proxy).

Like `docker/vllm-v0.33.0-secure-r1`, this image leaves out #52119 (regex compile in a killable worker; see that README). Also left out from secure-r1: this deployment serves no pooling, audio or video, sets no Qwen tool parser or KV connector, and runs the V2 model runner, whose thinking-budget code #51133 does not touch.

OS: openssl/libssl3t64 3.0.13-0ubuntu3.16, libc6/libc-bin/libc6-dev/libc-dev-bin 2.39-0ubuntu8.12 and linux-libc-dev 6.8.0-149.149, pinned. The image needs no Python package upgrades.

`apply-patches.py` refuses any vLLM other than 0.33.0, dry-runs and applies the patches in order, and checks every touched file against `PATCHED-FILES.sha256`. Publish with `.github/workflows/publish-vllm-v0.33.0-dsv41-r3.yaml` from a merged commit on main.
