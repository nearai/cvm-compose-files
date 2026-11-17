# SGLang v0.5.23 for GLM-5.3 Flash

Stock `lmsysorg/sglang` v0.5.23 (source 297e54fd), pinned by digest, with the GLM-5.3 Flash patch
series, exact-rid aborts, credential and request-content log redaction, and request validation.
Upstream numbers are sgl-project/sglang pull requests and name where a patch comes from.

`apply-patches.py` refuses any tree other than a clean 297e54fd checkout, checks every patch against
`SERIES`, and applies the patches in file-name order; later patches edit lines earlier ones add.

| patch | what it does | flag / env | origin |
|---|---|---|---|
| 01-chunked-prefill-admission-reserve | reserves KV room for in-flight chunked prefills before admitting a new one | `SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE` (opt-in) | NEAR |
| 02-chunked-prefill-pool-clamp | clamps a prefill chunk to the free KV pool, also when no room is left | – | NEAR |
| 03-modules-to-not-convert | honours `modules_to_not_convert`, so `graphistry/GLM-5.3-Flash-W4AFP8` loads | – | NEAR |
| 04-near-self-profile | per-rank self-profiler | `NEAR_SELF_PROFILE` (opt-in) | NEAR |
| 05-rid-abort-exact-match | an abort matches the exact rid; `<rid>_<n>` children are reached only for an rid the worker does not own | – | NEAR |
| 06-dsa-indexer-qsplit | splits DSA indexer query rows inside the logits budget | `SGLANG_DSA_INDEXER_QSPLIT=1` (opt-in) | NEAR |
| 07-fp8kv-flashmla | FP8 NoPE DSA KV cache with a 656 B row on `flashmla_kv` | `--kv-cache-dtype fp8_e4m3` with `flashmla_kv` | NEAR |
| 08-dsa-q8-flashmla-sparse | `flashmla_sparse_q8` DSA prefill on the 656 B FP8 NoPE row | `--dsa-prefill-backend flashmla_sparse_q8` | NEAR |
| 09-tokenize-on-preprocessor-thread | tokenization runs on the single request-preprocessor thread, off the event loop | – | #30771 |
| 10-shm-off-loop | multimodal shared-memory transfers run off the event loop | – | NEAR |
| 11-event-loop-stall-dump | dumps thread stacks when the event loop stalls for 30 s | – | NEAR |
| 12-preprocess-pool | process pool for request preprocessing; a result that does not pickle uses the thread path | `SGLANG_PREPROCESS_WORKERS` (opt-in) | NEAR |
| 13-tool-schema-depth-cap | caps tool-schema depth and node count when a cap is set | `SGLANG_TOOL_SCHEMA_MAX_DEPTH`, `SGLANG_TOOL_SCHEMA_MAX_NODES` (default 0, off) | NEAR |
| 14-credential-redaction | server info, scheduler state, dumps and `ServerArgs` output show credentials redacted | – | #48518 |
| 15-request-content-log-redaction | logs and error texts show media URLs, base64, schemas and tool values as scheme, length and digest only | – | NEAR |
| 16-hicache-runtime | HiCache host-memory budget and pooled transfers to the MLA host pool | `SGLANG_HICACHE_RAM_BUDGET`, `SGLANG_HICACHE_*` | NEAR |
| 17-hcc-managed-memory | HiCache host buffers from `cudaMallocHost` | `SGLANG_HICACHE_CUDA_HOST_MEMORY=1` | NEAR |
| 18-tests | unit tests for 16 | – | NEAR |
| 19-hcc-tests | unit tests for 17 | – | NEAR |
| 20-ghost-prefix-cache | ghost prefix-cache hit metrics through a separate aggregator | `SGLANG_GHOST_CACHE` (opt-in) | NEAR |
| 21-kv-tier-metrics | KV cache tier metrics | `SGLANG_KV_TIER_METRICS` (opt-in) | NEAR |
| 22-thinking-token-budget | request `thinking_token_budget` maps to `custom_params.thinking_budget`: 8192 by default for `glm5_next`, `null` turns it off, an explicit native budget wins | request field | NEAR |
| 23-final-prefill-abort-exclusion | a request aborted during its final prefill chunk is excluded from the batch and returns its abort reason | – | #34153 |
| 24-glm5-perception-gate-tests | unit tests for the GLM-5 perception gate | – | NEAR |
| 25-glm53-epd-video-and-pd-draft-state | GLM video dispatch in the EPD encoder and hybrid-DSA draft state in P/D | – | NEAR |
| 26-log-redaction-more-sites | applies 15 to the grammar, tool-call parser, chat serving and diffusion download log lines | – | NEAR |
| 27-sampling-param-bounds | 400 on out-of-range `n`, `logit_bias`, `temperature` and `top_logprobs`; out-of-vocab `stop_token_ids` are dropped | – | #44047 |
| 28-top-logprobs-vocab-bound | 400 on `top_logprobs` / `logprobs` above the vocab size | – | #49414 |
| 29-logprob-start-len-bound | 400 on `logprob_start_len < -1` on `/generate` | – | #49758 |
| 30-spec-grammar-reject-prune | prunes a speculative draft branch whose token the grammar rejects | – | #48117 |
| 31-media-local-paths-opt-in | image, audio and video inputs accept http(s) and data: inputs; `file://` URLs and server paths get a 400, decided without probing the path | `SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1` allows them | NEAR |
| 32-session-params-validation | 400 on a `session_params` field of the wrong type; a `replace` turn without `rid` clears the session | – | #41535, extended |
| 33-expert-distribution-needs-recorder | the `*_expert_distribution_record` routes answer 400 when no recorder is configured | – | #49358 |
| 34-send-weights-group-validation | the weight-send calls return an error for an unknown or empty group name or a wrong port count | – | #42093, extended |
| 35-caller-mm-hashes-opt-in | the server computes the multimodal cache identity; caller `mm_hashes` and the `hash` / `pad_value` of preprocessed items are ignored | `SGLANG_TRUST_CALLER_MM_HASHES=1` keeps them | NEAR |
| 36-transcription-upload-cap | `/v1/audio/transcriptions` reads uploads in chunks and answers 413 above the cap | `--audio-transcription-max-file-size-mb` (default 25, 0 turns the cap off) | #43915 |
| 37-ghost-aggregator-short-message | the ghost-cache aggregator counts a datagram shorter than the message header as a bad message, drops it and keeps reading | `SGLANG_GHOST_CACHE` (opt-in) | NEAR |
| 38-token-id-vocab-bound | 400 on caller token ids outside `[0, vocab_size)`, ids beyond 64 bits included, on every route that takes token ids (`/generate`, `/encode`, `/v1/completions`, `/v1/embeddings`, `/v1/classify`, `/v1/score`) | – | NEAR |

Defaults this image sets, which differ from stock v0.5.23: 02 clamps prefill chunks; 09 moves
tokenization to the preprocessor thread; 11 dumps stacks after a 30 s loop stall; 22 applies an
8192 thinking budget to `glm5_next`; 27-29 and 32-34 reject the out-of-range values above; 31
refuses local media paths; 35 ignores caller cache hashes; 36 caps transcription uploads at 25 MB; 38 rejects token ids outside the vocabulary.

Launch notes for this image:
- `SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION` defaults to true in v0.5.23; set it to 0 to turn it off.
- Breakable prefill CUDA graphs are on by default for GLM-5.3 Flash and capture the KDA extend.
  With `--mem-fraction-static 0.86` they do not fit in memory during warmup; launch with
  `--cuda-graph-backend-prefill disabled`. `SGLANG_DISABLE_KDA_PREFILL_GRAPH_EXTEND=1` drops only
  the KDA capture.

Packages: nltk 3.10.3 and defusedxml 0.7.1 from a hash-pinned `requirements.txt`, and
`linux-libc-dev` 6.8.0-149.149 pinned by exact version. The image removes the two Nsight EFA
sampler binaries, the cargo registry cache and the radix-tree crate's build files; it builds no Rust
code.

## Tests
`test-cpu.sh` runs on CPU inside the built image: the recipe checks in `tests/` for 04, 06-08,
11-13, 20, 21 and 37 and for the redaction patches, the upstream unit tests the series adds or edits,
request validation (`tests/test_request_validation.py` and the upstream unit tests the validation
patches add or edit), and the upstream mem_cache, managers, entrypoints, constrained, multimodal and
function_call suites, where a failure not in `tests/stock-v0523-known-failures.txt` fails the script.
The last step needs network access and runs with `SGLANG_ALLOW_LOCAL_MEDIA_PATHS=1`, because some
upstream tests read local files.

```
docker build --build-arg BUILD_SOURCE_REVISION=$(git rev-parse HEAD) -t sglang-v0.5.23-glm53-r2 docker/sglang-v0.5.23-glm53-r2
docker run --rm --runtime runc --entrypoint bash sglang-v0.5.23-glm53-r2 /usr/share/nearai/sglang-v0.5.23-glm53-r2/test-cpu.sh
```

Publish with `.github/workflows/publish-sglang-v0.5.23-glm53-r2.yaml` from a merged commit on main.
