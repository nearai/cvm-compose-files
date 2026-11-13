# SGLang v0.5.23 + NEAR AI security patches (stock-SGLang models)

This recipe is for the stock-SGLang models (Qwen3.6, Qwen3.8, FLUX.2). It builds upstream SGLang **v0.5.23** (tag 297e54fd) with the engine-agnostic security patches shared with `docker/sglang-v0.5.23-glm53-r2` and two sglang-diffusion patches for FLUX.2. It has none of the GLM-specific model or performance patches, which is why it is a separate image.

Upstream v0.5.23 already closes CVE-2026-15976 (#42760), CVE-2026-86793 (#39858, #40259) and CVE-2026-15969 (#41322, #40259).

## Patches

Applied in file-name order by `apply-patches.py`, which refuses any tree other than a clean 297e54fd and checks each patch against `SERIES`. Patches 01-14 are byte-identical to the GLM recipe's patches listed in `glm#`; `was` is the number in the first draft of this dir.

| # | glm# | was | patch | what it does | source | switch |
|---|---|---|---|---|---|---|
| 01 | 05 | 05 | rid-abort-exact-match | aborts match the exact rid; `<rid>_<n>` children only for an rid the worker does not own | own; CVE-2026-112266 | – |
| 02 | 14 | 14 | admin-auth-redaction | redacts credentials in server info, scheduler state and dumps | sglang#48518 (merged after v0.5.23); CVE-2026-15977 | – |
| 03 | 15 | 15 | customer-data-log-redaction | logs media URLs, base64, schemas and tool values as length and digest only | own | – |
| 04 | 26 | 26 | log-format-followup | log line format follow-up to 03 | own | – |
| 05 | 27 | 27 | sampling-param-bounds | rejects out-of-range `n`, `logit_bias`, `temperature` and `top_logprobs` with a 400; drops out-of-vocab `stop_token_ids` | sglang#44047 (merged) | `n` 1-128 on OpenAI routes, at most `SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ` on `/generate`; an explicit `max_completion_tokens=0` wins over `max_tokens` |
| 06 | 28 | 28 | top-logprobs-vocab-bound | rejects `top_logprobs` / `logprobs` above the vocab size with a 400 | sglang#49414 (merged) | – |
| 07 | 29 | 29 | logprob-start-len-bound | rejects `logprob_start_len < -1` on `/generate` with a 400 | sglang#49758 (merged) | – |
| 08 | 30 | 30 | spec-grammar-reject-prune | prunes a speculative draft branch whose token the grammar rejects, instead of raising in the scheduler | sglang#48117 (merged) | – |
| 09 | 31 | 31 | media-local-path-reject | refuses `file://` URLs and server paths in image, audio and video inputs with a 400, without probing the path | own | `SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1` allows them |
| 10 | 32 | 33 | session-params-validation | type-checks `session_params` (400 on a bad type) and lets a `replace` turn without `rid` clear the session safely | sglang#41535 (open) + own; CVE-2026-110807, CVE-2026-110809 | – |
| 11 | 33 | 35 | expert-distribution-guard | answers the `*_expert_distribution_record` routes with a 400 when the recorder is off | sglang#49358 (open); CVE-2026-112082 | – |
| 12 | 34 | 36 | send-weights-unknown-group | returns an error from the weight-send calls for an unknown group, an empty group name or a wrong port count | sglang#42093 (open) + own; CVE-2026-112083 | – |
| 13 | 35 | 37 | caller-mm-hashes-opt-in | ignores caller `mm_hashes` and the `hash` / `pad_value` of preprocessed items; the server computes the cache identity | own; CVE-2026-111392 | `SGLANG_TRUST_CALLER_MM_HASHES=1` keeps caller hashes |
| 14 | 36 | 39 | transcription-upload-cap | reads `/v1/audio/transcriptions` uploads in chunks and returns 413 above a cap, so memory use is bounded (the spooled temp file is not) | sglang#43915 (open); CVE-2026-109225 | `--audio-transcription-max-file-size-mb` (default 25, 0 off) |
| 15 | – | 50 | diffusion-media-url-ssrf-guard | fetches client image and video URLs only over http(s) to public addresses: every resolved address is checked, the connection is pinned to it, each redirect is checked, proxy env is ignored; the image and video loaders read files the server saved itself (the input save dir, `sglang_*` temp dirs) and other local paths only with the opt-in, decided on the path text before any filesystem probe | own; CVE-2026-15974 | `SGLANG_DIFFUSION_ALLOW_PRIVATE_MEDIA_URLS=1` drops the address check; `SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1` allows local paths |
| 16 | – | 51 | diffusion-operator-only-request-fields | answers `perf_dump_path`, `profile`, `profile_all_stages`, `upscaling_model_path` and `frame_interpolation_model_path` in diffusion requests with a 400 | own | `SGLANG_DIFFUSION_ALLOW_REQUEST_DEBUG_FIELDS=1` |

Not patched here:
- CVE-2026-15978: admin routes are open unless `--admin-api-key` is set; the files in `prod/` do not set it. sglang#49894 (keys in tokenizer workers and WebSocket routes) has no effect without a key;
- CVE-2026-93034 (`PickleWrapper` in the ZMQ decoder): host-only (loopback/IPC sockets, peer allowlist from #45250 in v0.5.23);
- CVE-2026-10775 (64-bit multimodal cache hash): the hash feeds pad values and the GPU hash kernels, so widening it is not a drop-in change;
- diffusers 0.37 → 0.38 (CVE-2026-44513, -44827, -45804): only reachable by loading an untrusted model repo;
- sglang#47546 (abort before dispatch) and sglang#47076 (source paths in 400 errors): no crash or data exposure;
- nltk 3.10.4, oauthlib 4.0.1, apt 2.8.4: the serving path does not load them.
- sglang-diffusion API and admin keys (sglang#47067): the diffusion admin routes have no key check, and the backport changes nothing without a key set.

The image also gets hash-pinned nltk 3.10.3 + defusedxml 0.7.1 (the base has nltk 3.9.1), the pinned `linux-libc-dev` fix, and removal of the two unused Nsight EFA sampler binaries.

## Tests
`test-cpu.sh` (CPU only) runs:
1. the credential-redaction checks (`tests/test_admin_auth_redaction.py`);
2. the customer-data log-redaction checks (`tests/test_customer_data_log_redaction.py`);
3. the upstream unit tests that patches 01-04 add or edit;
4. patches 05-14: `tests/test_security_backports.py` and the upstream unit tests those patches add or edit;
5. patches 15-16: `tests/test_diffusion_security.py` and the sglang-diffusion unit tests around them;
6. the upstream mem_cache / managers / entrypoints / constrained / multimodal / function_call suites; a failure not in `tests/stock-v0523-known-failures.txt` fails the script. It needs network access (about 50 tests fetch Hugging Face configs) and runs with `SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1`, because a few upstream tests read local files on purpose.

```
docker build --build-arg BUILD_SOURCE_REVISION=$(git rev-parse HEAD) -t sglang-v0.5.23-secure-r2 docker/sglang-v0.5.23-secure-r2
docker run --rm --runtime runc --entrypoint bash sglang-v0.5.23-secure-r2 /usr/share/nearai/sglang-v0523-secure/test-cpu.sh
```

Publish with `.github/workflows/publish-sglang-v0.5.23-secure-r2.yaml` from a merged commit on main.
