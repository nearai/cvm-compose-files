# vLLM v0.33.0 + NEAR AI security backports

Stock `vllm/vllm-openai` v0.33.0 (pinned by digest) for the stock-vLLM services in `prod/small-models.yaml` and `prod/pack-small-models.yaml`: Qwen3-VL, the embedding and reranker models, and Whisper. It adds the upstream security fixes those services can reach, one Whisper fix, and pinned OpenSSL and glibc security updates. Upstream numbers are vllm-project/vllm pull requests; most are still open, so their final form may differ.

| patch | what it does | upstream |
|---|---|---|
| 01-error-sanitizer-redos | the error-message path sanitizer runs in linear time (GHSA-f2g9-pmwr-xwc7) | code of open #60233; tests from #52120 |
| 02-remote-prefill-cache-accounting-trust | rejects a client-sent `kv_transfer_params.remote_prefill_cached_tokens` (400) and ignores it in usage, unless `VLLM_TRUST_REMOTE_PREFILL_CACHED_TOKENS=1` | #64918 |
| 03-stop-and-bad-words-bounds | caps `stop_token_ids` (`VLLM_MAX_STOP_TOKEN_IDS`, 128), `stop` and `bad_words`; stop-token lookup uses a set | #54721 |
| 04-detok-bounded-holdback | flushes incremental detokenization after 16 tokens that never complete a UTF-8 character | #63668 |
| 05-pooling-prompt-count-bound | caps list inputs on /v1/embeddings, /pooling, /score, /rerank at `VLLM_MAX_COMPLETION_PROMPTS`, default raised to 2048 | #53473 |
| 06-vision-rope-cache-byte-bound | bounds the Qwen3-VL / Qwen2.5-VL rotary-position cache by bytes | #60607 |
| 07-media-reference-script-reject | rejects uploaded audio/video that is an FFmpeg concat, HLS, SDP or XML script | #63701 |
| 08-engine-add-rollback | rolls back request state when the engine add fails; bounds `top_k` and `stream_interval` | #56212 |
| 09-audio-channel-cap | rejects audio with more than `VLLM_MAX_AUDIO_CHANNELS` (8) channels, including on the torchcodec backend | #51894, extended |
| 10-mm-uuid-length-cap | caps a multimodal content-part `uuid` at 128 characters | #57221 |
| 11-media-fetch-ssrf-filter | engine-side media fetches refuse non-public IPs (`VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS`, 0), pin the resolved IP and re-check every redirect hop; the async path resolves off the event loop | #57825, extended |
| 12-media-client-no-cookies | shared media HTTP clients keep no cookies | #56588 |
| 13-whisper-langdetect-fallback | functional, not security: a detected language vLLM does not list falls back to "en" instead of HTTP 500 | NEAR |

Patch numbers before the renumbering to `patches/NN-name.diff`: 08→01, 09→02, 12→03, 13→04, 14→05, 15→06, 16→07, 17→08, 20→09, 21→10, 22→11, 23→12, 24→13 (18, the regex compile worker, was dropped). Patches 01-04, 08 and 10-12 are byte-identical to 08-15 of `docker/vllm-v0.33.0-dsv41-r3`.

Default behaviour changes:
- 02 stops trusting client remote-prefill cached-token counts.
- 05 caps list inputs at 2048 on the pooling routes; /v1/completions prompt lists get the same 2048 default (upstream 1024).
- 11 blocks media fetches to private IPs. It also refuses an `HTTP_PROXY`/`HTTPS_PROXY` on a non-public IP on the async path; the compose files set no proxy.
- 13 makes the inline Whisper build step in the compose files unnecessary.

Video decoding is patched only by 07; the later upstream video-decode fixes (#57525, #67314, #60612) are not included, so Qwen3-VL on this image must run with video disabled.

Not included: #52119 (regex compile in a killable worker). Its per-compile process cost legitimate regex structured outputs up to 10x and could delay unrelated requests, so it was dropped. Stock v0.33.0 still times out a regex compile after `VLLM_REGEX_COMPILATION_TIMEOUT_S`, but the timed-out compile keeps running in a thread until it finishes.

Not included, because no service on this image reaches the code: #56814 (`vllm_xargs` transfer params need a KV or EC connector), #51133 (thinking budget needs a reasoning parser), #58407 (no Qwen3 tool parser is set).

OS: openssl/libssl3t64 3.0.13-0ubuntu3.16 and libc6/libc-bin/libc6-dev/libc-dev-bin 2.39-0ubuntu8.12, pinned; the engine's media fetches load them (TLS, DNS). pip, opentelemetry, apt and gnupg are left as in the base: the serving process does not load them.

## Tests
`test-cpu.sh` (CPU only) checks the patched files against `PATCHED-FILES.sha256`, runs `tests/test_security_backports.py` (one check per patch; all but the listed-language Whisper case fail on stock v0.33.0), and with `VLLM_TESTS` set runs the upstream tests the patches add or edit (`tests/upstream-tests.diff`), failing on anything not in `tests/stock-v0330-known-failures.txt`.

```
docker build --build-arg BUILD_SOURCE_REVISION=$(git rev-parse HEAD) -t vllm-v0.33.0-secure-r1 docker/vllm-v0.33.0-secure-r1
docker run --rm --runtime runc --entrypoint bash vllm-v0.33.0-secure-r1 /usr/share/nearai/vllm-v0.33.0-secure-r1/test-cpu.sh
docker run --rm --runtime runc -v <vllm v0.33.0 checkout>:/src:ro -e VLLM_TESTS=/src/tests --entrypoint bash \
  vllm-v0.33.0-secure-r1 /usr/share/nearai/vllm-v0.33.0-secure-r1/test-cpu.sh
```

Publish with `.github/workflows/publish-vllm-v0.33.0-secure-r1.yaml` from a merged commit on main.
