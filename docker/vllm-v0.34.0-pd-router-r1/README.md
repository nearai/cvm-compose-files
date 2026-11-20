# P/D sidecar router for vLLM NixlConnector (pull mode)

One router fronts one prefill/decode pair. A client request goes through two sequential legs:

1. **Prefill leg**: the body with `max_tokens=1`, `stream=false`, `kv_transfer_params={"do_remote_decode": true}`
   (and without `stream_options`, `n`, `logprobs`, structured-output fields) is sent to a prefill rank.
   The engine answers with the block list in `kv_transfer_params`.
2. **Decode leg**: the client's own body plus that `kv_transfer_params` is sent to a decode rank and streamed back.

Only the decode leg's response reaches the client, so usage is counted once. Nothing is sent to the client before
the decode leg has produced its first byte, so every leg-1 failure is a clean error response.

`router.py` is a single file for the vLLM image (fastapi, aiohttp and uvicorn are in it; orjson is used when
present), so it also runs as `python3 router.py --prefill H:P ... --decode H:P ...` with the same CLI and
environment names as the earlier prototype router. The image sets that as its entrypoint and runs as a non-root user.

## Options

| flag / env | default | meaning |
|---|---|---|
| `--prefill`, `--decode` | required | `host:port` of every rank behind this pair |
| `--host`, `--port` | `127.0.0.1`, `31150` | listen address |
| `--max-imbalance`, `PD_MAX_IMBALANCE` | `64` | stay on the sticky rank unless it has this many more requests in flight than the least loaded rank of the same role |
| `PD_KV_TIMEOUT_S` | `20` | decode request sent to first byte, streaming requests. Must be above `0` and below `PD_KV_LEASE_S - 5`, or the router refuses to start |
| `PD_KV_LEASE_S` | `30` | the prefill engine's `kv_lease_duration` in seconds; must be set to the same value (see "KV lease coupling") |
| `PD_PREFILL_TIMEOUT_S` | `600` | whole prefill leg; `0` disables |
| `PD_NONSTREAM_TIMEOUT_S` | `0` (off) | decode first byte for non-streaming requests, which answer only after the full generation |
| `PD_HEALTH_TIMEOUT_S` | `2` | per-probe timeout for `/health`; a timeout counts as down |
| `PD_METRICS_TIMEOUT_S` | `5` | per-rank timeout for the `/metrics` scrape |
| `PD_P_DROP_STRUCT` | `1` | drop structured-output fields from the prefill leg |
| `PD_STATS_LOG_S` | `60` | period of the one-line counters log, `0` = off |

## Behaviour

* **Affinity**: both legs are sticky per conversation (SHA-1 of the first two messages, or the prompt), within this pair's ranks only,
  with the max-imbalance escape. The hash is the earlier prototype router's, so conversations map to the same ranks.
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
* **Mid-stream failure**: if the decode connection breaks after the response has started, the router aborts the client
  connection instead of ending the stream cleanly, so a truncated response is a transport error for the caller.
* **Request shaping**: `n` greater than 1 is rejected with 400. `prompt_logprobs`, `logprobs`, `top_logprobs`, `stream_options`
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
* `GET /v1/models` and `/version` pass through to the first decode rank; `GET /pd_stats` has leg timings and counters.
* **Logging**: durations, counts, status codes and backend indexes only. No request or response content, no URLs.

## KV lease coupling

With NixlConnector the prefill engine keeps a finished request's blocks only for `kv_lease_duration` seconds (default 30), and the
decode side extends the lease only while a pull is running. A decode request that waits in a queue longer than the lease would
pull blocks the prefill engine has already freed. The key is read by both sides from the connector's extra config
(`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py`, `kv_lease_duration`, default 30, and
`.../nixl/base_worker.py`, same key and default), i.e. in the prefill engine's `--kv-transfer-config`, next to the existing connector settings:

```
"kv_connector_extra_config": {"kv_lease_duration": 150}
```

Set the same number as `PD_KV_LEASE_S` on the router, and keep `PD_KV_TIMEOUT_S` below `PD_KV_LEASE_S - 5`; the router checks this
at startup and exits with an error otherwise. The lease starts when the prefill leg finishes, and the 5 s margin covers the
hand-over between the legs. With the stock lease of 30 s the largest allowed timeout is just under 25 s (default 20 s); a
deployment whose decode queue can be deeper should raise the engine lease and the timeout together (for example lease 150 s,
timeout 120 s) rather than lower the timeout and fail queued but healthy requests.

## Tests

```
uvx --with fastapi --with aiohttp --with uvicorn --with orjson --with httpx pytest -q docker/vllm-v0.34.0-pd-router-r1/tests
```
