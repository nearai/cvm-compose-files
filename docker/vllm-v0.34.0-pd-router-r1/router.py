#!/usr/bin/env python3
"""P/D sidecar router for vLLM NixlConnector pull mode: one router per prefill/decode pair.

Request flow (two sequential legs, because in pull mode the decode leg cannot start before the prefill engine has
returned its block list):
  leg 1  POST to a prefill rank with max_tokens=1, stream=false, kv_transfer_params={"do_remote_decode": true}
  leg 2  POST the client's own body plus the kv_transfer_params from leg 1 to a decode rank, streamed back unchanged.

Behaviour contract:
  * Same CLI and env names as the prototype router (--prefill, --decode, --host, --port, --max-imbalance; PD_MAX_IMBALANCE,
    PD_P_DROP_STRUCT, PD_STATS_LOG_S) so it can replace it as a single file.
  * Both legs are sticky per conversation (hash of the first two messages or the prompt) unless the sticky rank has more
    than --max-imbalance more requests in flight than the least loaded rank of the same role. Ranks come from this pair only.
  * A client that disconnects cancels whichever leg is running. Closing the upstream connection makes the engine abort the
    request. A prefill that already finished holds its KV blocks until the connector's lease expires (see README).
  * PD_KV_TIMEOUT_S (default 20) bounds decode-leg start to first byte for streaming requests. On expiry the decode
    request is cancelled and the client gets a 504 in the engine's error shape. The prefill engine frees its blocks when
    its KV lease ends (kv_lease_duration in kv_connector_extra_config, default 30 s), so the router refuses to start
    unless PD_KV_TIMEOUT_S < PD_KV_LEASE_S - 5; PD_KV_LEASE_S must equal the engine's value (see README).
  * A body with n > 1 is rejected with 400. kv_transfer_params from the client is dropped on every path; the decode leg
    gets only what the prefill engine returned.
  * Errors keep the upstream status and are engine-shaped JSON. Nothing reaches the client before leg 2 has produced its
    first byte, so a leg 1 failure is always a clean error response.
  * Only the decode leg's response is forwarded, byte for byte, so usage is counted once and never altered. A decode
    stream that breaks mid-way aborts the client connection instead of ending cleanly.
  * GET /health probes every prefill and decode rank on every call (no cache, PD_HEALTH_TIMEOUT_S default 2, a timeout is
    unhealthy) and is 200 only when all of them answer /health with 200.
  * GET /v1/metrics and /metrics aggregate the ranks' Prometheus text (pair totals; ?per_rank=1 keeps every series).
    Every rank must answer within PD_METRICS_TIMEOUT_S (default 5); otherwise the scrape is a 503 rather than a smaller sum.
  * Logs carry counts, durations, status codes and backend indexes only. No request or response content, no URLs.
"""
import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import time
from collections import deque

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

try:
    import orjson

    def jloads(b):
        return orjson.loads(b)

    def jdumps(o) -> bytes:
        return orjson.dumps(o)
except Exception:  # orjson is optional
    def jloads(b):
        return json.loads(b)

    def jdumps(o) -> bytes:
        return json.dumps(o).encode()

log = logging.getLogger("pd-router")

POP_PREFILL = ["stream_options", "max_completion_tokens", "min_tokens", "logprobs", "top_logprobs", "prompt_logprobs",
               "n", "best_of", "kv_transfer_params"]
POP_STRUCT = ["response_format", "structured_outputs", "guided_json", "guided_regex", "guided_choice", "guided_grammar",
              "guided_decoding_backend", "guided_whitespace_pattern"]
FWD_HEADERS = ("authorization", "x-request-id")
GET_PASSTHROUGH = ("v1/models", "version")
ERR_CAP = 1 << 20
POLL_S = 0.25
LEASE_MARGIN_S = 5.0


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


class Gone(Exception):
    """The client hung up."""


class Timeout(Exception):
    """A leg exceeded its time budget."""


class StreamAborted(Exception):
    """The decode stream broke after the response had started; raised so the client connection is aborted."""


class GuardedStream(StreamingResponse):
    """StreamingResponse that always runs `cleanup`, also when the body generator never started (client gone early)."""

    def __init__(self, content, cleanup, **kw):
        super().__init__(content, **kw)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cleanup()


def engine_error(status: int, message: str, etype: str, headers: dict | None = None) -> Response:
    body = {"error": {"message": message, "type": etype, "param": None, "code": status}}
    return Response(jdumps(body), status_code=status, media_type="application/json", headers=headers or None)


ETYPE = {400: "BadRequestError", 401: "AuthenticationError", 403: "PermissionDeniedError", 404: "NotFoundError",
         413: "BadRequestError", 422: "BadRequestError", 429: "TooManyRequestsError", 503: "ServiceUnavailableError",
         504: "GatewayTimeoutError"}


def upstream_error(status: int, data: bytes, retry_after: str | None) -> Response:
    """Keep the upstream status; forward its body only if the engine produced an error object, else build one."""
    headers = {"retry-after": retry_after} if retry_after else None
    try:
        obj = jloads(data)
    except Exception:
        obj = None
    if isinstance(obj, dict) and ("error" in obj or obj.get("object") == "error"):
        return Response(data, status_code=status, media_type="application/json", headers=headers)
    return engine_error(status, f"upstream engine returned status {status}",
                        ETYPE.get(status, "InternalServerError"), headers)


# ---------------------------------------------------------------- Prometheus aggregation

SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*$")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
SUFFIXES = ("_bucket", "_sum", "_count", "_total", "_created")


def fmt(v: float) -> str:
    if v != v:
        return "NaN"
    if v in (float("inf"), float("-inf")):
        return "+Inf" if v > 0 else "-Inf"
    return repr(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)


def family_type(name: str, types: dict) -> str:
    if name in types:
        return types[name]
    for s in SUFFIXES:
        if name.endswith(s) and name[: -len(s)] in types:
            return types[name[: -len(s)]]
    return "untyped"


def aggregate_metrics(scrapes: list, per_rank: bool = False) -> str:
    """scrapes: [(role, rank, text)]. Sum series across ranks (max for ratio gauges, first for *_info), or with
    per_rank keep every series and add pd_role and pd_rank labels."""
    meta, types, order, acc = [], {}, [], {}
    seen_meta = set()
    for role, rank, text in scrapes:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.split(None, 3)
                if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                    if parts[1] == "TYPE" and len(parts) == 4:
                        types[parts[2]] = parts[3]
                    if (parts[1], parts[2]) not in seen_meta:
                        seen_meta.add((parts[1], parts[2]))
                        meta.append(line)
                continue
            m = SAMPLE.match(line)
            if not m:
                continue
            name, raw_labels, raw_val = m.groups()
            if name.endswith("_created"):
                continue
            try:
                val = float(raw_val)
            except ValueError:
                continue
            labels = LABEL.findall(raw_labels or "")
            if per_rank:
                labels = labels + [("pd_role", role), ("pd_rank", str(rank))]
                key = (name, tuple(labels))
                order.append(key)
                acc[key] = val
                continue
            labels = tuple((k, v) for k, v in labels if k != "engine")
            key = (name, labels)
            if key not in acc:
                order.append(key)
                acc[key] = val
                continue
            if name.endswith("_info") or family_type(name, types) == "info":
                continue
            if family_type(name, types) == "gauge" and name.endswith(("_perc", "_ratio")):
                acc[key] = max(acc[key], val)
            else:
                acc[key] += val
    out = list(meta)
    for key in order:
        name, labels = key
        lab = ",".join(f'{k}="{v}"' for k, v in labels)
        out.append(f"{name}{{{lab}}} {fmt(acc[key])}" if lab else f"{name} {fmt(acc[key])}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- router

def pct(xs, q):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))] * 1000, 1) if s else None


class Router:
    def __init__(self, prefill, decode, max_imbalance=64, kv_timeout=None, prefill_timeout=None,
                 nonstream_timeout=None, health_timeout=None, drop_struct=None, stats_log_s=None, kv_lease=None,
                 metrics_timeout=None):
        self.prefill, self.decode = list(prefill), list(decode)
        self.max_imbalance = max_imbalance
        self.kv_timeout = env_float("PD_KV_TIMEOUT_S", 20) if kv_timeout is None else kv_timeout
        self.kv_lease = env_float("PD_KV_LEASE_S", 30) if kv_lease is None else kv_lease
        if not 0 < self.kv_timeout < self.kv_lease - LEASE_MARGIN_S:
            raise ValueError(
                f"PD_KV_TIMEOUT_S={self.kv_timeout:g} must be > 0 and below PD_KV_LEASE_S - {LEASE_MARGIN_S:g} "
                f"(PD_KV_LEASE_S={self.kv_lease:g}): the prefill engine frees its KV blocks when kv_lease_duration "
                "ends, so a decode request must not wait longer than the lease. Raise kv_lease_duration on the "
                "prefill engine and PD_KV_LEASE_S together, or lower PD_KV_TIMEOUT_S.")
        self.prefill_timeout = env_float("PD_PREFILL_TIMEOUT_S", 600) if prefill_timeout is None else prefill_timeout
        self.metrics_timeout = env_float("PD_METRICS_TIMEOUT_S", 5) if metrics_timeout is None else metrics_timeout
        self.nonstream_timeout = (env_float("PD_NONSTREAM_TIMEOUT_S", 0) if nonstream_timeout is None
                                  else nonstream_timeout)
        self.health_timeout = env_float("PD_HEALTH_TIMEOUT_S", 2) if health_timeout is None else health_timeout
        self.drop_struct = os.environ.get("PD_P_DROP_STRUCT", "1") != "0" if drop_struct is None else drop_struct
        self.stats_log_s = env_float("PD_STATS_LOG_S", 60) if stats_log_s is None else stats_log_s
        self.inflight = {"prefill": [0] * len(self.prefill), "decode": [0] * len(self.decode)}
        self.t_prefill = deque(maxlen=4000)
        self.t_dfirst = deque(maxlen=4000)
        self.counts = {"requests": 0, "client_gone_in_prefill": 0, "client_gone_in_decode": 0, "prefill_fail": 0,
                       "decode_fail": 0, "decode_first_byte_timeout": 0, "bad_request": 0}
        self.decode_waiting = 0  # requests between decode POST and first byte (decode queueing plus the remote KV pull)
        self._session = None

    # -- plumbing
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # keepalive shorter than uvicorn's 5 s server keep-alive: never reuse a connection about to be closed
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=6 * 3600, sock_connect=10),
                connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=4.0))
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def summary(self) -> dict:
        return {**self.counts, "inflight": {k: list(v) for k, v in self.inflight.items()}, "n": len(self.t_prefill),
                "prefill_ms": {"p50": pct(self.t_prefill, .5), "p95": pct(self.t_prefill, .95)},
                "decode_first_byte_ms": {"p50": pct(self.t_dfirst, .5), "p95": pct(self.t_dfirst, .95)}}

    # -- routing
    @staticmethod
    def affinity(body: dict) -> int:
        # stdlib json on purpose: same key as the prototype router, so a conversation maps to the same ranks as before
        msgs = body.get("messages")
        key = json.dumps(msgs[:2] if isinstance(msgs, list) else body.get("prompt", ""), sort_keys=True)
        return int(hashlib.sha1(key.encode()).hexdigest(), 16)

    def pick(self, role: str, h: int) -> int:
        load = self.inflight[role]
        pref = h % len(load)
        least = min(range(len(load)), key=lambda i: load[i])
        return pref if load[pref] - load[least] <= self.max_imbalance else least

    # -- cancellation helper
    async def guarded(self, req: Request, task: asyncio.Task, timeout: float):
        """Wait for task; cancel it (closing its upstream connection) if the client hangs up or the time budget ends."""
        deadline = time.monotonic() + timeout if timeout > 0 else None
        try:
            while True:
                wait = POLL_S if deadline is None else max(0.0, min(POLL_S, deadline - time.monotonic()))
                done, _ = await asyncio.wait({task}, timeout=wait)
                if done:
                    return task.result()
                if await req.is_disconnected():
                    raise Gone()
                if deadline is not None and time.monotonic() >= deadline:
                    raise Timeout()
        except BaseException:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
            elif not task.cancelled() and task.exception() is None:
                # The task finished between the poll and the disconnect/deadline check: nobody will use its result.
                self._discard(task.result())
            raise

    @staticmethod
    def _discard(result):
        """Close an upstream response that was produced but will not be used."""
        for item in result if isinstance(result, tuple) else (result,):
            close = getattr(item, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

    # -- legs
    async def _prefill_leg(self, idx: int, path: str, data: bytes, headers: dict):
        async with self.session().post(f"http://{self.prefill[idx]}{path}", data=data, headers=headers) as r:
            raw = await r.content.read(ERR_CAP) if r.status != 200 else await r.read()
            return r.status, raw, r.headers.get("retry-after")

    async def _decode_start(self, idx: int, path: str, data: bytes, headers: dict):
        """POST leg 2 and wait for the first body byte. Returns (response, first_chunk)."""
        r = await self.session().post(f"http://{self.decode[idx]}{path}", data=data, headers=headers)
        try:
            if r.status != 200:
                return r, await r.content.read(ERR_CAP)
            return r, await r.content.readany()
        except BaseException:
            r.close()
            raise

    async def handle(self, req: Request, path: str) -> Response:
        c = self.counts
        c["requests"] += 1
        try:
            body = jloads(await req.body())
            if not isinstance(body, dict):
                raise ValueError
        except Exception:
            c["bad_request"] += 1
            return engine_error(400, "request body must be a JSON object", "BadRequestError")
        headers = {k: v for k, v in req.headers.items() if k.lower() in FWD_HEADERS}
        headers["content-type"] = "application/json"
        n = body.get("n")
        if isinstance(n, int) and not isinstance(n, bool) and n > 1:
            c["bad_request"] += 1
            return engine_error(400, "n > 1 is not supported", "BadRequestError")
        h = self.affinity(body)
        # Never accept transfer parameters from a client, on any path: leg 2 gets only what the prefill engine returned.
        body.pop("kv_transfer_params", None)
        if isinstance(body.get("vllm_xargs"), dict):
            body["vllm_xargs"].pop("kv_transfer_params", None)

        pre = dict(body)
        pre.update(stream=False, max_tokens=1, kv_transfer_params={"do_remote_decode": True})
        for k in POP_PREFILL + (POP_STRUCT if self.drop_struct else []):
            if k != "kv_transfer_params":
                pre.pop(k, None)

        # ---- leg 1
        p = self.pick("prefill", h)
        self.inflight["prefill"][p] += 1
        t0 = time.perf_counter()
        task = asyncio.ensure_future(self._prefill_leg(p, path, jdumps(pre), headers))
        try:
            status, raw, retry_after = await self.guarded(req, task, self.prefill_timeout)
        except Gone:
            c["client_gone_in_prefill"] += 1
            return Response(status_code=499)
        except Timeout:
            c["prefill_fail"] += 1
            log.warning("prefill leg timeout backend=%d", p)
            return engine_error(504, "prefill leg timed out", "GatewayTimeoutError")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            c["prefill_fail"] += 1
            log.warning("prefill leg failed backend=%d kind=%s", p, type(e).__name__)
            return engine_error(502, "prefill engine unreachable", "BadGatewayError")
        finally:
            self.inflight["prefill"][p] -= 1
        t1 = time.perf_counter()
        self.t_prefill.append(t1 - t0)
        if status != 200:
            c["prefill_fail"] += 1
            log.warning("prefill leg status=%d backend=%d", status, p)
            return upstream_error(status, raw, retry_after)
        try:
            kvp = jloads(raw).get("kv_transfer_params")
        except Exception:
            kvp = None
        if not isinstance(kvp, dict) or not kvp:
            c["prefill_fail"] += 1
            log.warning("prefill leg returned no transfer parameters backend=%d", p)
            return engine_error(502, "prefill engine returned no KV transfer parameters", "BadGatewayError")
        del raw

        # Client left between the legs: do not start a decode whose KV pull nobody wants.
        if await req.is_disconnected():
            c["client_gone_in_prefill"] += 1
            return Response(status_code=499)

        # ---- leg 2
        body["kv_transfer_params"] = kvp
        d = self.pick("decode", h)
        self.inflight["decode"][d] += 1
        released = False

        def release():
            nonlocal released
            if not released:
                released = True
                self.inflight["decode"][d] -= 1

        timeout = self.kv_timeout if body.get("stream") else self.nonstream_timeout
        task = asyncio.ensure_future(self._decode_start(d, path, jdumps(body), headers))
        t2 = time.perf_counter()
        self.decode_waiting += 1
        try:
            try:
                r, first = await self.guarded(req, task, timeout)
            finally:
                self.decode_waiting -= 1
        except Gone:
            c["client_gone_in_decode"] += 1
            release()
            return Response(status_code=499)
        except Timeout:
            c["decode_first_byte_timeout"] += 1
            c["decode_fail"] += 1
            log.warning("decode first byte timeout backend=%d after_ms=%d", d, int((time.perf_counter() - t2) * 1000))
            release()
            return engine_error(504, "KV transfer or decode did not produce a first byte in time",
                                "GatewayTimeoutError")
        except asyncio.CancelledError:
            release()
            raise
        except Exception as e:
            c["decode_fail"] += 1
            log.warning("decode leg failed backend=%d kind=%s", d, type(e).__name__)
            release()
            return engine_error(502, "decode engine unreachable", "BadGatewayError")
        self.t_dfirst.append(time.perf_counter() - t2)
        ctype = r.headers.get("content-type", "application/json")
        if r.status != 200:
            c["decode_fail"] += 1
            log.warning("decode leg status=%d backend=%d", r.status, d)
            retry_after = r.headers.get("retry-after")
            r.close()
            release()
            return upstream_error(r.status, first, retry_after)

        async def gen():
            try:
                if first:
                    yield first
                async for chunk in r.content.iter_any():
                    yield chunk
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                c["decode_fail"] += 1
                log.warning("decode stream broke backend=%d kind=%s", d, type(e).__name__)
                raise StreamAborted from None  # abort the client connection: a truncated stream must not look complete
            finally:
                cleanup()

        def cleanup():
            r.close()
            release()

        return GuardedStream(gen(), cleanup, status_code=200, media_type=ctype.split(";")[0],
                             headers={"x-pd-prefill-ms": str(round((t1 - t0) * 1000, 1))})

    # -- health / metrics
    async def _get(self, addr: str, path: str, timeout: float | None = None):
        try:
            async with self.session().get(
                    f"http://{addr}{path}",
                    timeout=aiohttp.ClientTimeout(total=self.health_timeout if timeout is None else timeout)) as r:
                return r.status, await r.text()
        except Exception:
            return None, ""

    async def health(self) -> Response:
        targets = [("prefill", i, a) for i, a in enumerate(self.prefill)] + \
                  [("decode", i, a) for i, a in enumerate(self.decode)]
        res = await asyncio.gather(*[self._get(a, "/health") for _, _, a in targets])
        up = {"prefill": 0, "decode": 0}
        for (role, _, _), (st, _) in zip(targets, res):
            up[role] += st == 200
        ok = up["prefill"] == len(self.prefill) and up["decode"] == len(self.decode)
        return JSONResponse({"status": "ok" if ok else "unhealthy",
                             "prefill": {"up": up["prefill"], "total": len(self.prefill)},
                             "decode": {"up": up["decode"], "total": len(self.decode)}},
                            status_code=200 if ok else 503)

    async def metrics(self, per_rank: bool) -> Response:
        targets = [("prefill", i, a) for i, a in enumerate(self.prefill)] + \
                  [("decode", i, a) for i, a in enumerate(self.decode)]
        res = await asyncio.gather(*[self._get(a, "/metrics", self.metrics_timeout) for _, _, a in targets])
        scrapes = [(role, i, text) for (role, i, _), (st, text) in zip(targets, res) if st == 200]
        if len(scrapes) < len(targets):
            # A sum over fewer ranks would read as low load; fail the scrape instead of under-counting.
            return engine_error(503, f"metrics incomplete: {len(scrapes)} of {len(targets)} backends answered",
                                "ServiceUnavailableError")
        text = aggregate_metrics(scrapes, per_rank)
        own = [("pd_router_decode_waiting_first_byte", "gauge", self.decode_waiting),
               ("pd_router_backends_up", "gauge", len(scrapes)), ("pd_router_backends_total", "gauge", len(targets)),
               ("pd_router_requests_total", "counter", self.counts["requests"]),
               ("pd_router_prefill_failures_total", "counter", self.counts["prefill_fail"]),
               ("pd_router_decode_failures_total", "counter", self.counts["decode_fail"]),
               ("pd_router_decode_first_byte_timeouts_total", "counter", self.counts["decode_first_byte_timeout"]),
               ("pd_router_client_disconnects_total", "counter",
                self.counts["client_gone_in_prefill"] + self.counts["client_gone_in_decode"])]
        for name, typ, v in own:
            text += f"# TYPE {name} {typ}\n{name} {v}\n"
        return Response(text, media_type="text/plain; version=0.0.4")

    async def passthrough(self, path: str) -> Response:
        st, text = await self._get(self.decode[0], "/" + path)
        if st is None:
            return engine_error(502, "decode engine unreachable", "BadGatewayError")
        return Response(text, status_code=st, media_type="application/json")


def create_app(prefill, decode, **kw) -> FastAPI:
    router = Router(prefill, decode, **kw)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        stats = None
        if router.stats_log_s > 0:
            async def loop():
                while True:
                    await asyncio.sleep(router.stats_log_s)
                    if router.counts["requests"]:
                        log.warning("pd_stats %s", router.summary())
            stats = asyncio.ensure_future(loop())
        yield
        if stats:
            stats.cancel()
        await router.close()

    app = FastAPI(lifespan=lifespan)
    app.state.router = router

    @app.exception_handler(Exception)
    async def boom(request: Request, exc: Exception):
        log.warning("unhandled kind=%s", type(exc).__name__)
        return engine_error(500, "internal router error", "InternalServerError")

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        return await router.handle(req, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def comp(req: Request):
        return await router.handle(req, "/v1/completions")

    @app.get("/health")
    async def health():
        return await router.health()

    @app.get("/v1/metrics")
    @app.get("/metrics")
    async def metrics(per_rank: int = 0):
        return await router.metrics(bool(per_rank))

    @app.get("/pd_stats")
    async def pd_stats():
        return JSONResponse(router.summary())

    @app.get("/{p:path}")
    async def passthrough(p: str):
        if p not in GET_PASSTHROUGH:
            return engine_error(404, "not found", "NotFoundError")
        return await router.passthrough(p)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", nargs="+", required=True)  # host:port ...
    ap.add_argument("--decode", nargs="+", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=31150)
    ap.add_argument("--max-imbalance", type=int, default=int(os.environ.get("PD_MAX_IMBALANCE", "64")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        app = create_app(args.prefill, args.decode, max_imbalance=args.max_imbalance)
    except ValueError as e:
        ap.error(str(e))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
