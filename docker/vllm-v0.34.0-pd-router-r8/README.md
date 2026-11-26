# P/D sidecar router for vLLM NixlConnector (pull mode)

One router fronts one prefill/decode pair. A client request goes through two sequential legs:

1. **Prefill leg**: the body with `max_tokens=1`, `stream=false`, `kv_transfer_params={"do_remote_decode": true}`
   (and without `stream_options`, `n`, `logprobs`, structured-output fields) is sent to a prefill rank.
   The engine answers with the block list in `kv_transfer_params`.
2. **Decode leg**: the client's own body plus that `kv_transfer_params` is sent to a decode rank and streamed back.

Only the decode leg's response reaches the client, so usage is counted once. Nothing is sent to the client before
the decode leg has produced its first byte, so every leg-1 failure is a clean error response.

With `ROUTER_DDIRECT=1` a chat follow-up whose new tail is small goes straight to the decode rank and skips leg 1
(see "Decode-direct"). It is off by default; with it off the router runs only the two-leg path. Two further
opt-in paths, first-turn decode-direct and overflow to decode, are described under "Decode-direct".

`router.py` is a single file for the vLLM image (fastapi, aiohttp and uvicorn are in it; orjson is used when
present), so it also runs as `python3 router.py --prefill H:P ... --decode H:P ...` with the same CLI and
environment names as the r2 router. The image sets that as its entrypoint and runs as a non-root user.

## Options

| flag / env | default | meaning |
|---|---|---|
| `--prefill`, `--decode` | required | `host:port` of every rank behind this pair |
| `--host`, `--port` | `127.0.0.1`, `31150` | listen address |
| `--max-imbalance`, `PD_MAX_IMBALANCE` | `64` | stay on the sticky rank unless it has this many more requests in flight than the least loaded rank of the same role |
| `PD_KV_TIMEOUT_S` | `20` | decode request sent to first byte, streaming requests. Must be above `0` and below `PD_KV_LEASE_S - 5`, or the router refuses to start |
| `PD_KV_LEASE_S` | `30` | the prefill engine's `kv_lease_duration` in seconds; must be set to the same value (see "KV lease coupling") |
| `PD_PREFILL_TIMEOUT_S` | `600` | whole prefill leg; `0` disables |
| `PD_NONSTREAM_TIMEOUT_S` | `0` (off) | decode first byte for non-streaming requests, which answer only after the full generation; see "Timeout" for the accepted limit and the recommended setting |
| `PD_HEALTH_TIMEOUT_S` | `2` | per-probe timeout for `/health`; a timeout counts as down |
| `PD_METRICS_TIMEOUT_S` | `5` | per-rank timeout for the `/metrics` scrape |
| `PD_P_DROP_STRUCT` | `1` | drop structured-output fields from the prefill leg |
| `PD_STATS_LOG_S` | `60` | period of the one-line counters log, `0` = off |
| `ROUTER_DDIRECT` | `0` | `1` turns on decode-direct; `0` keeps the two-leg path only, with no classification or memory |
| `ROUTER_DDIRECT_BLOCK_TOKENS` | `2432` | the decode engine's Mamba checkpoint granularity in tokens; set it to the engines' `--prefix-match-unit` when that is used, else their block size |
| `ROUTER_DDIRECT_MAX_TAIL_TOKENS` | `1024` | largest estimated new tail (tokens after the last assistant message) sent direct |
| `ROUTER_DDIRECT_MAX_CHUNK_TOKENS` | `2048` | largest estimated chunk the decode engine has to prefill itself (the tail plus the part of the history past its last checkpoint) |
| `ROUTER_DDIRECT_CHARS_PER_TOKEN` | `3.5` | characters per token for the estimates; must be above `0` |
| `ROUTER_DDIRECT_MAX_D_INFLIGHT` | `48` | the request goes the two-leg way when the router already has this many requests in flight on the decode rank |
| `ROUTER_DDIRECT_MAX_AGE_S` | `30` | send direct only if this router served the conversation within this many seconds |
| `ROUTER_DDIRECT_MEM_CAP` | `100000` | most conversations remembered (least recently used dropped first); at least `1` |
| `ROUTER_DDIRECT_MEM_TTL_S` | `600` | a remembered conversation older than this is dropped on lookup |
| `ROUTER_DDIRECT_COUNT_PREV_REPLY` | `0` | `1` counts the previous assistant reply as part of the new tail in the estimates, because the decode engine recomputes it when it is not cached |
| `ROUTER_DDIRECT_CACHE_BLOCK_TOKENS` | `0` (off) | the engines' prefix-cache block size in tokens; when above `0`, the part of a remembered prefix past its last full block counts as uncached in the first-turn and overflow estimates |
| `ROUTER_DDIRECT_HISTORY_STATS` | `0` | `1` hashes each message of a request's history and keeps the hashes per conversation to fill `pd_router_history_change_total` and `pd_router_history_diverged_tokens_est`; with `0` nothing is computed or stored and those series stay at zero. Routing is the same either way |
| `ROUTER_DDIRECT_FIRST_TURN` | `0` | `1` sends a first turn direct when its stable prefix was recently served by a decode rank (see "First-turn decode-direct"); with `0` the prefix memory and its measurement series still run |
| `ROUTER_DDIRECT_FIRST_TURN_MIN_PREFIX_TOKENS` | `1024` | smallest remembered prefix (estimated tokens) for a first-turn direct request |
| `ROUTER_DDIRECT_FIRST_TURN_MAX_TOKENS` | the value of `ROUTER_DDIRECT_MAX_TAIL_TOKENS` | largest estimated part of a first turn past its remembered prefix |
| `ROUTER_DDIRECT_PREFIX_MEM_CAP` | `20000` | most stable prefixes remembered (least recently used dropped first); at least `1` |
| `ROUTER_DDIRECT_PREFIX_TTL_S` | `120` | a remembered prefix older than this is dropped on lookup |
| `ROUTER_DDIRECT_OVERFLOW` | `0` | `1` sends a request that is not otherwise sent direct to a decode rank while the prefill tier is busy (see "Overflow to decode") |
| `ROUTER_DDIRECT_OVERFLOW_P_INFLIGHT` | `8` | overflow applies when the prefill ranks together have at least this many requests in flight |
| `ROUTER_DDIRECT_OVERFLOW_MAX_TOKENS` | `2048` | largest estimated uncached part of a request that overflows |
| `ROUTER_STRICT_AFFINITY` | `0` | `1`: both legs always use the conversation's hash rank, with no max-imbalance escape; applies also with decode-direct off |

## Behaviour

* **Affinity**: both legs are sticky per conversation (SHA-1 of the messages before the first user message plus the first
  user message, or of the prompt for completions), within this pair's ranks only, with the max-imbalance escape. All
  turns of a conversation map to the same ranks. r2 hashed the first two messages, so a conversation without a system
  prompt changed key on its second turn; a conversation whose messages are `[system, user, ...]` keeps its r2 key, and
  other existing conversations remap once on upgrade. Conversations with identical openers share a key and so a rank;
  the max-imbalance escape bounds that skew, but `ROUTER_STRICT_AFFINITY=1` removes the bound, so identical openers pin
  to one prefill and one decode rank.
* **Cache namespace**: a request with a `cache_salt` hashes the salt with that prefix. Requests with different salts
  cannot share an engine prefix cache, so they never share a rank pick or a decode-direct memory entry. A request
  without `cache_salt` keeps the r3 key; salted conversations remap once on upgrade from r3.
* **Direct-attempt context metrics**: each decode-direct attempt is counted by outcome (`accepted` when the decoder
  answers 200, `rejected` when its admission refuses the request, `error` for any other failure) with histograms of the estimated tokens since the conversation was last served (previous reply
  plus new messages), the new-message estimate, the previous reply estimate and the seconds since it was served, and a
  counter of whether the template-rendered body fields outside the messages (`tools`, `tool_choice`,
  `chat_template_kwargs` and similar) changed since then. The router keeps only a hash of those fields per conversation.
* **Client disconnect** cancels the leg that is running (polled every 250 ms between legs and while waiting); closing the
  upstream connection makes the engine abort that request. A disconnect between the legs skips the decode leg.
  A prefill that already finished keeps its KV blocks until the connector's lease expires; the engine frees them itself.
  There is no router-side release call.
* **Timeout**: when the decode leg has no first byte after `PD_KV_TIMEOUT_S`, the decode request is cancelled and the client gets a 504.
  The wait covers queueing on the decode engine, the KV pull and the first token, and the router cannot tell them apart, so
  the counter is `pd_router_decode_first_byte_timeouts_total` (not a transfer-failure count). To tell overload from a stuck
  transfer, compare it with the decode ranks' own `vllm:num_requests_waiting`. For non-streaming requests the first byte is the
  whole answer, so `PD_NONSTREAM_TIMEOUT_S` is not bounded by the lease; keep the decode queue short enough that a request
  starts pulling within the lease.
  Accepted limit: with the default `PD_NONSTREAM_TIMEOUT_S=0` a non-streaming request has no first-byte bound, so a stalled KV
  pull can outlive the prefill lease and the request then waits for the engine instead of failing fast. Recommended: set
  `PD_NONSTREAM_TIMEOUT_S` above the longest generation you expect to serve, so a stuck request is cancelled with a 504 while
  normal ones are not cut short. The default is left off because no single value is safe for every deployment.
* **Mid-stream failure**: if the decode connection breaks after the response has started, the router aborts the client
  connection instead of ending the stream cleanly, so a truncated response is a transport error for the caller.
* **Request shaping**: on `/v1/completions`, `prompt` must be a string or one list of token ids; a batch of prompts (a list of strings or of lists) is rejected with 400. `n` must be absent or the integer 1; any other value (greater than 1, a float, a string, a boolean, null) is rejected with 400. `prompt_logprobs`, `logprobs`, `top_logprobs`, `stream_options`
  and `best_of` are not sent to the prefill leg. A client-supplied `kv_transfer_params` (also under `vllm_xargs`) is dropped on
  every path; the decode leg gets only what the prefill engine returned, so `remote_prefill_cached_tokens` can come only from it.
* **Usage** in the decode response (including `stream_options.include_usage`) is forwarded byte for byte; the router never
  adds, edits or synthesises usage, and the prefill leg's usage is never forwarded.
* **Errors** are `{"error": {"message", "type", "param", "code"}}`. The upstream status is kept (a 429 stays 429, as does
  `Retry-After`); an engine error body is forwarded as is, anything else is replaced by a generic body.
  Router-made errors: 400 bad body, 502 unreachable engine or no transfer parameters, 504 timeout, 499 client gone.
* **`GET /health`** probes `/health` on every prefill and decode rank on each call (no cache) and is 200 only if all answer 200;
  otherwise 503 with counts.
* **`GET /v1/metrics`, `GET /metrics`** scrape `/metrics` of every rank (each within `PD_METRICS_TIMEOUT_S`) and return pair totals: series are summed across ranks and
  roles with the `engine` label dropped (`vllm:num_requests_running` and `vllm:num_requests_waiting` are therefore pair totals,
  which is what a load probe wants); ratio gauges (`*_perc`, `*_ratio`) take the max, `*_info` the first value, `*_created` are
  dropped. `?per_rank=1` returns every series with `pd_role` and `pd_rank` labels instead. If any rank does not answer in time, the
  scrape is a 503 instead of a smaller sum, so a caller never reads a partial total as low load. Router series:
  `pd_router_decode_waiting_first_byte` (requests between the decode POST and its first byte),
  `pd_router_decode_first_byte_timeouts_total`, `pd_router_requests_total`, `pd_router_prefill_failures_total`,
  `pd_router_decode_failures_total`, `pd_router_client_disconnects_total`, `pd_router_backends_up`, `pd_router_backends_total`.
* **Router series** on `/metrics` (also when a scrape is per rank), counters unless noted, label values fixed:
  `pd_router_decode_requests_total{path}` (decode-leg requests, `path` = `direct`, `pd` or `pd-fb`),
  `pd_router_first_byte_seconds{path}` (histogram, router arrival to the first byte ready for the client, successful
  responses only), `pd_router_ddirect_eligible_total`, `pd_router_ddirect_attempts_total`,
  `pd_router_ddirect_rejected_fallback_total`, `pd_router_ddirect_fallback_reason_total{reason}`, `pd_router_ddirect_guard_fallback_total`, `pd_router_ddirect_errors_total`,
  `pd_router_ddirect_skipped_total{reason}` (`no_entry`, `stale`, `rank_changed`, `rejected_prev`, `context_changed`) and
  `pd_router_ddirect_ineligible_total{reason}` (`first_turn`, `tail_too_long`, `chunk_too_large`, `images`, `completions`).
  `pd_router_ddirect_attempt_context_total{outcome,context}` (`accepted`/`rejected`/`error`, `same`/`changed`), and the histograms
  `pd_router_ddirect_attempt_new_tokens_est`, `pd_router_ddirect_attempt_tail_tokens_est`,
  `pd_router_ddirect_attempt_prev_reply_tokens_est` and `pd_router_ddirect_attempt_gap_seconds`, each with `outcome`.
  `pd_router_assistant_reasoning_field_total{field}` counts chat requests with an assistant message by the reasoning
  field of the last one (`reasoning`, `reasoning_content`, `reasoning_details`, `inline_think` for a `<think>` tag in
  the content, `none`); it counts with decode-direct on or off.
  The decode-direct series stay at zero while it is off.
* `GET /v1/models` and `/version` pass through to the first decode rank; `GET /pd_stats` has leg timings and counters, and
  `GET /ddirect_stats` the decode-direct counters, settings and estimate histograms.
* **Logging**: durations, counts, status codes and backend indexes only. No request or response content, no URLs.

## Decode-direct

On with `ROUTER_DDIRECT=1`. It needs decode engines that keep a prefix-cached state at the cut a P/D transfer leaves,
so a later turn of the same conversation can resume on the decode rank. A request is sent direct when all of these hold:

* it is a `/v1/chat/completions` request with at least one assistant message and no image, video or audio part;
* the estimated new tail is at most `ROUTER_DDIRECT_MAX_TAIL_TOKENS` and the estimated chunk at most
  `ROUTER_DDIRECT_MAX_CHUNK_TOKENS`. Estimates count text characters (content, text parts, tool-call arguments and tool
  results) divided by `ROUTER_DDIRECT_CHARS_PER_TOKEN`; template tokens and reasoning text are not counted;
* this router served the conversation (same affinity hash) to completion within `ROUTER_DDIRECT_MAX_AGE_S`, on the
  decode rank it would pick now, and the decode engine did not refuse the last direct attempt since then;
* the router has fewer than `ROUTER_DDIRECT_MAX_D_INFLIGHT` requests in flight on that decode rank.

A direct request is the client's body as received, minus any `kv_transfer_params`, and goes through the same decode
path as leg 2 (timeouts, disconnects, error shapes). If the decode engine refuses it before the first byte (HTTP 429 or
503, or a streaming response whose first event is an error with that code), the router runs the normal two legs instead
(`pd-fb`). The same happens when a direct attempt fails before any byte went to the client: HTTP 500, 502 or 504 (or an
error event with that code), a connection error, no first byte or no first output event within the time budget, or a
stream that ends without any content, reasoning or tool-call delta (`pd_router_ddirect_fallback_reason_total{reason}`,
`reason` = `5xx`, `conn`, `timeout` or `empty`). For a streaming direct attempt the router holds back the leading
role-only or empty events until the first event with output arrives (at most 64 KiB or 5 s), then sends them together;
a usage chunk that reports generated tokens also counts as output. After the first byte went to the client nothing is
retried. The P/D decode leg is unchanged. Responses carry `x-pd-path: direct`, `pd` or `pd-fb` only with decode-direct on. The memory holds only the
hash, a rank, a time, a token estimate and a flag per conversation; it is per router process and empty after a restart,
so each conversation's next turn after a restart takes the two-leg path.

To run it, set the decode engines' admission so that a direct request that misses their prefix cache is refused (for
example the decode engine's long-request admission in the matching vLLM image), and keep `ROUTER_DDIRECT_BLOCK_TOKENS`
equal to the engines' checkpoint granularity.

### First-turn decode-direct

With `ROUTER_DDIRECT_FIRST_TURN=1` a chat request with no assistant message is also sent direct when the messages before
its first user message, its context fields and its `cache_salt` (the stable prefix) were served to completion by a decode
rank within `ROUTER_DDIRECT_PREFIX_TTL_S`, that rank has fewer than `ROUTER_DDIRECT_MAX_D_INFLIGHT` requests in flight,
the remembered prefix is at least `ROUTER_DDIRECT_FIRST_TURN_MIN_PREFIX_TOKENS` and the rest of the prompt is at most
`ROUTER_DDIRECT_FIRST_TURN_MAX_TOKENS`. With `ROUTER_DDIRECT_CACHE_BLOCK_TOKENS` set, the part of the prefix past its
last full cache block counts toward that limit. A refusal by the decode engine falls back to the two legs and forgets the
prefix. The memory holds a hash, a rank, a time and a token estimate per prefix. Results are counted by
`pd_router_ddirect_first_turn_total{result}` and `pd_router_ddirect_first_turn_outcome_total{outcome}`; with the flag off
the lookups are counted as `would_attempt` and nothing is sent direct.

### Overflow to decode

With `ROUTER_DDIRECT_OVERFLOW=1`, a request that is not otherwise sent direct goes straight to a decode rank when the
prefill ranks together have at least `ROUTER_DDIRECT_OVERFLOW_P_INFLIGHT` requests in flight and its estimated uncached
part (the whole prompt, or what is past a remembered prefix or conversation) is at most
`ROUTER_DDIRECT_OVERFLOW_MAX_TOKENS`. The decode rank is the remembered one, else the conversation's own rank, and the
in-flight guard of `ROUTER_DDIRECT_MAX_D_INFLIGHT` applies. A refusal falls back to the two legs. Counted by
`pd_router_ddirect_overflow_total{result}`.

### Context and history measurements

A direct attempt is skipped (`context_changed`) when the fields outside the messages that the chat template renders
(`tools`, `tool_choice`, `chat_template_kwargs` and similar) differ from the last served request of the conversation.
`pd_router_first_turn_prefix_tokens_est`, `pd_router_first_turn_block_residual_tokens_est`,
`pd_router_history_change_total{kind}`, `pd_router_history_diverged_tokens_est` and `pd_router_prefill_inflight` are
measurement series; they do not change routing. The two history series need `ROUTER_DDIRECT_HISTORY_STATS=1`.

The conversation memory and the prefix memory also drop expired entries on insert: each insert checks the 8 least
recently used entries against `ROUTER_DDIRECT_MEM_TTL_S` or `ROUTER_DDIRECT_PREFIX_TTL_S`, so a TTL bounds how long an
entry stays in memory even without a lookup. The size caps are unchanged.

## KV lease coupling

With NixlConnector the prefill engine keeps a finished request's blocks only for `kv_lease_duration` seconds (default 30), and the
decode side extends the lease only while a pull is running. A decode request that waits in a queue longer than the lease would
pull blocks the prefill engine has already freed. The key is read by both sides from the connector's extra config
(`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py`, `kv_lease_duration`, default 30, and
`.../nixl/base_worker.py`, same key and default), i.e. in the prefill engine's `--kv-transfer-config`, next to the existing connector settings:

```
"kv_connector_extra_config": {"kv_lease_duration": 150}
```

`PD_KV_LEASE_S` must equal the engines' `kv_lease_duration`; the router cannot read the engine's value, so it cannot verify this.
Set the same number as `PD_KV_LEASE_S` on the router, and keep `PD_KV_TIMEOUT_S` below `PD_KV_LEASE_S - 5`; the router checks this
at startup and exits with an error otherwise. The lease starts when the prefill leg finishes, and the 5 s margin covers the
hand-over between the legs. With the stock lease of 30 s the largest allowed timeout is just under 25 s (default 20 s); a
deployment whose decode queue can be deeper should raise the engine lease and the timeout together (for example lease 150 s,
timeout 120 s) rather than lower the timeout and fail queued but healthy requests.

## Tests

```
uvx --with fastapi --with aiohttp --with uvicorn --with orjson --with httpx pytest -q docker/vllm-v0.34.0-pd-router-r8/tests
```
