# vLLM v0.33.0 with bounded request handling

Stock `vllm/vllm-openai` v0.33.0, pinned by digest, with bounds and validation on request handling,
one Whisper language fallback, and pinned OpenSSL, glibc and linux-libc-dev versions.
Upstream numbers are vllm-project/vllm pull requests and name where a patch comes from.

| patch | what it does | upstream |
|---|---|---|
| 01-error-sanitizer-linear-time | the error-message path sanitizer runs in time linear in the message length | #60233, tests from #52120 |
| 02-cached-tokens-from-engine-only | `kv_transfer_params.remote_prefill_cached_tokens` from a client is rejected (400) and left out of usage, unless `VLLM_TRUST_REMOTE_PREFILL_CACHED_TOKENS=1` | #64918 |
| 03-stop-and-bad-words-bounds | caps `stop_token_ids` (`VLLM_MAX_STOP_TOKEN_IDS`, 128), `stop` and `bad_words`; stop-token lookup uses a set; also changes `deepseek_v32_encoding.py`: `render_message` takes a required `last_user_idx` and a new optional `last_assistant_idx`, `drop_thinking_messages` requires `last_user_idx`, and a tool message is rendered after the last assistant message seen by `encode_messages` instead of by scanning back | #54721 |
| 04-detok-bounded-holdback | flushes incremental detokenization after 16 tokens that never complete a UTF-8 character, keeping a trailing incomplete character so that streamed text matches a one-shot decode; a token id the decode stream rejects is skipped with a warning that carries the error type only | #63668, extended |
| 05-pooling-prompt-count-bound | caps list inputs on /v1/embeddings, /pooling, /score, /rerank at `VLLM_MAX_COMPLETION_PROMPTS`, default raised to 2048 | #53473 |
| 06-vision-rope-cache-byte-bound | bounds the Qwen3-VL / Qwen2.5-VL rotary-position cache by bytes | #60607 |
| 07-media-reference-script-reject | accepts uploaded audio/video only when it is media, not an FFmpeg concat, HLS, SDP or XML script | #63701 |
| 08-engine-add-rollback | rolls back request state when the engine add fails; bounds `top_k` and `stream_interval` | #56212 |
| 09-audio-channel-cap | accepts audio up to `VLLM_MAX_AUDIO_CHANNELS` (8) channels, including on the torchcodec backend | #51894, extended |
| 10-mm-uuid-length-cap | caps a multimodal content-part `uuid` at 128 characters | #57221 |
| 11-media-fetch-public-ips-only | engine-side media fetches resolve to public unicast IPs only (`VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS`, 0; loopback, private, link-local, multicast, NAT64 and reserved addresses are refused), connect only to the validated addresses, in answer order, and re-check them after a redirect; one fetch timeout covers name lookup, every connect attempt and every redirect hop; the async path resolves off the event loop; both clients ignore proxy environment variables; the per-host pool cache holds at most 64 pools and is closed with the session | #57825, extended |
| 12-media-client-no-cookies | shared media HTTP clients keep no cookies: a `Set-Cookie` response header is not stored or sent back | #56588 |
| 13-whisper-langdetect-fallback | a detected language vLLM does not list falls back to "en" instead of HTTP 500 | NEAR |
| 14-media-retry-log-fields | media fetch retry warnings carry the attempt count and error type | NEAR |

Defaults this image sets, which differ from stock v0.33.0:
- 02 takes cached-token counts from the engine.
- 03 caps stop lists and changes tool-message rendering for DeepSeek chat templates.
- 05 caps list inputs at 2048 on the pooling routes; /v1/completions prompt lists get the same 2048
  default (upstream 1024).
- 11 resolves engine-side media fetches to public unicast IPs only and ignores
  `HTTP_PROXY`/`HTTPS_PROXY`.
- 12 keeps no cookies.
- 13 answers "en" for a detected language vLLM does not list.

Video decoding is changed by 07 only.

Regex compilation keeps the upstream v0.33.0 behaviour: each compile is bounded by
`VLLM_REGEX_COMPILATION_TIMEOUT_S`, on the thread that requested it.

OS: openssl/libssl3t64 3.0.13-0ubuntu3.16, libc6/libc-bin/libc6-dev/libc-dev-bin 2.39-0ubuntu8.12
and linux-libc-dev 6.8.0-149.149, pinned by exact version; the `nearai.os.packages` label lists all
of them. The engine's media fetches load OpenSSL and glibc (TLS, DNS). pip,
opentelemetry, apt and gnupg are left as in the base: the serving process does not load them.

## Tests
`test-cpu.sh` (CPU only, installed in the image) checks the patched files against
`PATCHED-FILES.sha256`, runs `tests/test_request_bounds.py` (checks per patch; all but the
listed-language Whisper case fail on stock v0.33.0), and with `VLLM_TESTS` set to a v0.33.0 `tests/`
tree runs the upstream tests the patches add or edit (`tests/upstream-tests.diff`), failing on
anything not in `tests/stock-v0330-known-failures.txt`.

The patch 04 checks compare streamed text with a one-shot decode for 2-, 3- and 4-byte characters
across the flush boundary and bound the held-back tokens on an endless run of lead bytes. The patch
11 checks cover refused destinations (literal, alternate numeric forms, IPv4-mapped IPv6, NAT64,
names that resolve to them, mixed answers, redirects), proxy environment variables, a lookup slower
than the timeout, and fallback to the next validated address when one is refused or does not
answer, within one timeout.
