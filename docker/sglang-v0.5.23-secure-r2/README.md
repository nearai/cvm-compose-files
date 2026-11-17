# SGLang v0.5.23 with request validation and log redaction

Stock `lmsysorg/sglang` v0.5.23 (source 297e54fd), pinned by digest, with exact-rid aborts,
credential and request-content log redaction, request validation, and the sglang-diffusion media URL
and request-field policy. It carries no model-specific patches. Upstream numbers are sgl-project/sglang
pull requests and name where a patch comes from.

`apply-patches.py` refuses any tree other than a clean 297e54fd checkout, checks every patch against
`SERIES`, and applies the patches in file-name order.

| patch | what it does | origin |
|---|---|---|
| 01-rid-abort-exact-match | an abort matches the exact rid; `<rid>_<n>` children are reached only for an rid the worker does not own | NEAR |
| 02-credential-redaction | server info, scheduler state, dumps and `ServerArgs` output show credentials redacted | #48518 |
| 03-request-content-log-redaction | logs and error texts show media URLs, base64, schemas and tool values as scheme, length and digest only | NEAR |
| 04-log-redaction-more-sites | applies 03 to the grammar, tool-call parser, chat serving and diffusion download log lines | NEAR |
| 05-sampling-param-bounds | 400 on out-of-range `n`, `logit_bias`, `temperature` and `top_logprobs`; out-of-vocab `stop_token_ids` are dropped | #44047 |
| 06-top-logprobs-vocab-bound | 400 on `top_logprobs` / `logprobs` above the vocab size | #49414 |
| 07-logprob-start-len-bound | 400 on `logprob_start_len < -1` on `/generate` | #49758 |
| 08-spec-grammar-reject-prune | prunes a speculative draft branch whose token the grammar rejects | #48117 |
| 09-media-local-paths-opt-in | image, audio and video inputs accept http(s) and data: inputs; `file://` URLs and server paths get a 400, decided without probing the path, unless `SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1` | NEAR |
| 10-session-params-validation | 400 on a `session_params` field of the wrong type; a `replace` turn without `rid` clears the session | #41535, extended |
| 11-expert-distribution-needs-recorder | the `*_expert_distribution_record` routes answer 400 when no recorder is configured | #49358 |
| 12-send-weights-group-validation | the weight-send calls return an error for an unknown or empty group name or a wrong port count | #42093, extended |
| 13-caller-mm-hashes-opt-in | the server computes the multimodal cache identity; caller `mm_hashes` and the `hash` / `pad_value` of preprocessed items are used only with `SGLANG_TRUST_CALLER_MM_HASHES=1` | NEAR |
| 14-transcription-upload-cap | `/v1/audio/transcriptions` reads uploads in chunks and answers 413 above `--audio-transcription-max-file-size-mb` (default 25, 0 turns the cap off) | #43915 |
| 15-diffusion-media-public-urls-only | sglang-diffusion fetches client image and video URLs only over http(s) from public addresses: every resolved address is checked, the connection goes to the checked address, each redirect is checked and proxy settings are not read; the loaders read files the server saved itself (the input save dir, `sglang_*` temp dirs) and other local paths only with `SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1`; `SGLANG_DIFFUSION_ALLOW_PRIVATE_MEDIA_URLS=1` drops the address check | NEAR |
| 16-diffusion-operator-only-request-fields | sglang-diffusion answers 400 to `perf_dump_path`, `profile`, `profile_all_stages`, `upscaling_model_path` and `frame_interpolation_model_path` in requests unless `SGLANG_DIFFUSION_ALLOW_REQUEST_DEBUG_FIELDS=1` | NEAR |

Defaults this image sets, which differ from stock v0.5.23: 05-07 and 10-12 reject the out-of-range
values above; 09 and 15 refuse local media paths; 13 ignores caller cache hashes; 14 caps
transcription uploads at 25 MB; 15 fetches diffusion media URLs from public addresses only; 16
refuses the operator-only request fields.

Packages: nltk 3.10.3 and defusedxml 0.7.1 from a hash-pinned `requirements.txt`, and
`linux-libc-dev` 6.8.0-149.149 pinned by exact version. The image removes the two Nsight EFA
sampler binaries, the cargo registry cache and the radix-tree crate's build files; it builds no Rust
code.

## Tests
`test-cpu.sh` runs on CPU inside the built image:
1. credential redaction (`tests/test_credential_redaction.py`);
2. request-content log redaction (`tests/test_request_content_log_redaction.py`);
3. the upstream unit tests that 01, 02 and 04 add or edit;
4. request validation: `tests/test_request_validation.py` and the upstream unit tests the
   validation patches add or edit;
5. the sglang-diffusion media URL policy and operator-only request fields
   (`tests/test_diffusion_media_and_request_fields.py`) and the sglang-diffusion unit tests around them;
6. the upstream mem_cache, managers, entrypoints, constrained, multimodal and function_call suites;
   a failure not in `tests/stock-v0523-known-failures.txt` fails the script. This step needs network
   access (about 50 tests fetch Hugging Face configs) and runs with `SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1`,
   because some upstream tests read local files.

```
docker build --build-arg BUILD_SOURCE_REVISION=$(git rev-parse HEAD) -t sglang-v0.5.23-secure-r2 docker/sglang-v0.5.23-secure-r2
docker run --rm --runtime runc --entrypoint bash sglang-v0.5.23-secure-r2 /usr/share/nearai/sglang-v0.5.23-secure-r2/test-cpu.sh
```

Publish with `.github/workflows/publish-sglang-v0.5.23-secure-r2.yaml` from a merged commit on main.
