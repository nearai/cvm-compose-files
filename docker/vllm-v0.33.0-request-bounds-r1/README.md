# vLLM v0.33.0 with bounded request handling

Stock `vllm/vllm-openai` v0.33.0, pinned by digest, with bounds and validation on the request paths
the stock-vLLM services use, one Whisper language fallback, and pinned OpenSSL and glibc versions.
Upstream numbers are vllm-project/vllm pull requests and name where a patch comes from.

| patch | what it does | upstream |
|---|---|---|
| 01-error-sanitizer-linear-time | the error-message path sanitizer runs in time linear in the message length | #60233, tests from #52120 |
| 02-cached-tokens-from-engine-only | `kv_transfer_params.remote_prefill_cached_tokens` from a client is rejected (400) and left out of usage, unless `VLLM_TRUST_REMOTE_PREFILL_CACHED_TOKENS=1` | #64918 |
| 03-stop-and-bad-words-bounds | caps `stop_token_ids` (`VLLM_MAX_STOP_TOKEN_IDS`, 128), `stop` and `bad_words`; stop-token lookup uses a set | #54721 |
| 04-detok-bounded-holdback | flushes incremental detokenization after 16 tokens that never complete a UTF-8 character | #63668 |
| 05-pooling-prompt-count-bound | caps list inputs on /v1/embeddings, /pooling, /score, /rerank at `VLLM_MAX_COMPLETION_PROMPTS`, default raised to 2048 | #53473 |
| 06-vision-rope-cache-byte-bound | bounds the Qwen3-VL / Qwen2.5-VL rotary-position cache by bytes | #60607 |
| 07-media-reference-script-reject | accepts uploaded audio/video only when it is media, not an FFmpeg concat, HLS, SDP or XML script | #63701 |
| 08-engine-add-rollback | rolls back request state when the engine add fails; bounds `top_k` and `stream_interval` | #56212 |
| 09-audio-channel-cap | accepts audio up to `VLLM_MAX_AUDIO_CHANNELS` (8) channels, including on the torchcodec backend | #51894, extended |
| 10-mm-uuid-length-cap | caps a multimodal content-part `uuid` at 128 characters | #57221 |
| 11-media-fetch-public-ips-only | engine-side media fetches resolve to public IPs only (`VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS`, 0), pin the resolved IP and re-check it at every redirect hop; the async path resolves off the event loop | #57825, extended |
| 12-media-client-no-cookies | shared media HTTP clients keep no cookies | #56588 |
| 13-whisper-langdetect-fallback | a detected language vLLM does not list falls back to "en" instead of HTTP 500 | NEAR |

Defaults this image sets, which differ from stock v0.33.0:
- 02 takes cached-token counts from the engine.
- 05 caps list inputs at 2048 on the pooling routes; /v1/completions prompt lists get the same 2048
  default (upstream 1024).
- 11 resolves engine-side media fetches to public IPs only, and on the async path honours
  `HTTP_PROXY`/`HTTPS_PROXY` only when the proxy itself is on a public IP; the compose files set no
  proxy.
- 13 makes the inline Whisper build step in the compose files unnecessary.

Video decoding is patched by 07 only, so Qwen3-VL on this image runs with video disabled.

Regex compilation keeps the upstream v0.33.0 behaviour: each compile is bounded by
`VLLM_REGEX_COMPILATION_TIMEOUT_S`, on the thread that requested it.

OS: openssl/libssl3t64 3.0.13-0ubuntu3.16 and libc6/libc-bin/libc6-dev/libc-dev-bin
2.39-0ubuntu8.12, pinned by exact version; the engine's media fetches load them (TLS, DNS). pip,
opentelemetry, apt and gnupg are left as in the base: the serving process does not load them.

## Tests
`test-cpu.sh` (CPU only) checks the patched files against `PATCHED-FILES.sha256`, runs
`tests/test_request_bounds.py` (one check per patch; all but the listed-language Whisper case fail
on stock v0.33.0), and with `VLLM_TESTS` set runs the upstream tests the patches add or edit
(`tests/upstream-tests.diff`), failing on anything not in `tests/stock-v0330-known-failures.txt`.

```
docker build --build-arg BUILD_SOURCE_REVISION=$(git rev-parse HEAD) -t vllm-v0.33.0-request-bounds-r1 docker/vllm-v0.33.0-request-bounds-r1
docker run --rm --runtime runc --entrypoint bash vllm-v0.33.0-request-bounds-r1 /usr/share/nearai/vllm-v0.33.0-request-bounds-r1/test-cpu.sh
docker run --rm --runtime runc -v <vllm v0.33.0 checkout>:/src:ro -e VLLM_TESTS=/src/tests --entrypoint bash \
  vllm-v0.33.0-request-bounds-r1 /usr/share/nearai/vllm-v0.33.0-request-bounds-r1/test-cpu.sh
```

Publish with `.github/workflows/publish-vllm-v0.33.0-request-bounds-r1.yaml` from a merged commit on
main.
