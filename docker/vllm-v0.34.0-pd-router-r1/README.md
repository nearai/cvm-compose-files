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
| `PD_KV_TIMEOUT_S` | `120` | decode request sent to first byte, streaming requests; `0` disables |
| `PD_PREFILL_TIMEOUT_S` | `0` (off) | whole prefill leg |
| `PD_NONSTREAM_TIMEOUT_S` | `0` (off) | decode first byte for non-streaming requests, which answer only after the full generation |
| `PD_HEALTH_TIMEOUT_S` | `2` | per-probe timeout for `/health` and the metrics scrape; a timeout counts as down |
| `PD_P_DROP_STRUCT` | `1` | drop structured-output fields from the prefill leg |
| `PD_STATS_LOG_S` | `60` | period of the one-line counters log, `0` = off |

## Behaviour

* **Affinity**: both legs are sticky per conversation (SHA-1 of the first two messages, or the prompt), within this pair's ranks only,
  with the max-imbalance escape. The hash is the earlier prototype router's, so conversations map to the same ranks.
* **Client disconnect** cancels the leg that is running (polled every 250 ms between legs and while waiting); closing the
  upstream connection makes the engine abort that request. A disconnect between the legs skips the decode leg.
  A prefill that already finished keeps its KV blocks until the connector's lease expires (default 30 s, `kv_lease_duration`
  in the prefill engine's `kv_connector_extra_config`); the engine frees them itself. There is no router-side release call.
* **Timeout**: when the decode leg has no first byte after `PD_KV_TIMEOUT_S`, the decode request is cancelled and the client gets a 504.
* **Errors** are `{"error": {"message", "type", "param", "code"}}`. The upstream status is kept (a 429 stays 429, as does
  `Retry-After`); an engine error body is forwarded as is, anything else is replaced by a generic body.
  Router-made errors: 400 bad body, 502 unreachable engine or no transfer parameters, 504 timeout, 499 client gone.
* **`GET /health`** probes `/health` on every prefill and decode rank on each call (no cache) and is 200 only if all answer 200;
  otherwise 503 with counts.
* **`GET /v1/metrics`, `GET /metrics`** scrape `/metrics` of every rank and return pair totals: series are summed across ranks and
  roles with the `engine` label dropped (`vllm:num_requests_running` and `vllm:num_requests_waiting` are therefore pair totals,
  which is what a load probe wants); ratio gauges (`*_perc`, `*_ratio`) take the max, `*_info` the first value, `*_created` are
  dropped. `?per_rank=1` returns every series with `pd_role` and `pd_rank` labels instead. Router series:
  `pd_router_decode_waiting_kv` (requests waiting on the decode leg's first byte, i.e. on the KV pull),
  `pd_router_kv_timeouts_total`, `pd_router_requests_total`, `pd_router_prefill_failures_total`,
  `pd_router_decode_failures_total`, `pd_router_client_disconnects_total`, `pd_router_backends_up`, `pd_router_backends_total`.
* `GET /v1/models` and `/version` pass through to the first decode rank; `GET /pd_stats` has leg timings and counters.
* **Logging**: durations, counts, status codes and backend indexes only. No request or response content, no URLs.
* A client-supplied `kv_transfer_params` is discarded.

## Tests

```
uvx --with fastapi --with aiohttp --with uvicorn --with orjson --with httpx pytest -q docker/vllm-v0.34.0-pd-router-r1/tests
```
