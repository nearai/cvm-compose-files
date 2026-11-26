"""Router tests: real uvicorn router in front of fake aiohttp engines, all in one event loop."""
import asyncio
import contextlib
import importlib
import json
import logging
import os
import socket
import sys

import aiohttp
import pytest
import uvicorn
from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
R = importlib.import_module("router")

SECRET = "ZX-SECRET-PROMPT-TEXT"
PREFILL_USAGE = {"prompt_tokens": 1111, "completion_tokens": 1, "total_tokens": 1112}
DECODE_USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Fake:
    """Fake engine. mode: ok | hang | error"""

    def __init__(self, role, name="m"):
        self.role, self.name = role, name
        self.bodies, self.cancelled, self.active = [], 0, 0
        self.mode = "ok"
        self.status, self.error_body, self.headers = 500, None, {}
        self.error_segments = 0  # >0: send the error body in this many separate writes
        self.health = 200
        self.health_delay = 0
        self.metrics = ""
        self.metrics_delay = 0.0
        self.break_after = None  # decode stream: abort the connection after this many chunks
        self.prefill_extra = {}
        self.stream_delay = 0.0
        self.port = free_port()

    async def chat(self, request):
        body = await request.json()
        self.bodies.append(body)
        self.active += 1
        try:
            if self.mode == "error":
                data = self.error_body if self.error_body is not None else json.dumps(
                    {"error": {"message": "engine says no", "type": "X", "code": self.status}})
                if self.error_segments:
                    raw = data.encode() if isinstance(data, str) else data
                    size = -(-len(raw) // self.error_segments)
                    resp = web.StreamResponse(status=self.status, headers={**self.headers, "content-type": "application/json"})
                    await resp.prepare(request)
                    for i in range(0, len(raw), size):
                        await resp.write(raw[i:i + size])
                        await asyncio.sleep(0.05)
                    return resp
                return web.Response(body=data, status=self.status, headers=self.headers,
                                    content_type="application/json")
            if self.mode == "hang":
                await asyncio.sleep(3600)
            if self.mode == "nokvp":
                return web.json_response({"choices": [], "usage": PREFILL_USAGE})
            if self.role == "prefill":
                return web.json_response({"choices": [], "usage": PREFILL_USAGE, "kv_transfer_params": {
                    "do_remote_prefill": True, "remote_block_ids": [[1, 2]], "remote_host": "h",
                    **self.prefill_extra}})
            if body.get("stream"):
                resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
                await resp.prepare(request)
                for i in range(3):
                    await resp.write(f'data: {{"i": {i}}}\n\n'.encode())
                    if self.break_after is not None and i + 1 >= self.break_after:
                        request.transport.abort()
                        return resp
                    await asyncio.sleep(self.stream_delay)
                await resp.write(f'data: {{"usage": {json.dumps(DECODE_USAGE)}}}\n\ndata: [DONE]\n\n'.encode())
                return resp
            return web.json_response({"choices": [], "usage": DECODE_USAGE})
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1

    async def health_h(self, request):
        if self.health_delay:
            await asyncio.sleep(self.health_delay)
        return web.Response(status=self.health)

    async def metrics_h(self, request):
        if self.metrics_delay:
            await asyncio.sleep(self.metrics_delay)
        return web.Response(text=self.metrics)

    async def models(self, request):
        return web.json_response({"data": [{"id": self.name}]})

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_post("/v1/completions", self.chat)
        app.router.add_get("/health", self.health_h)
        app.router.add_get("/metrics", self.metrics_h)
        app.router.add_get("/v1/models", self.models)
        self.runner = web.AppRunner(app, handler_cancellation=True)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        self.addr = f"127.0.0.1:{self.port}"

    async def stop(self):
        await self.runner.cleanup()


@contextlib.asynccontextmanager
async def stack(n_p=1, n_d=1, **kw):
    ps, ds = [Fake("prefill") for _ in range(n_p)], [Fake("decode") for _ in range(n_d)]
    for f in ps + ds:
        await f.start()
    kw.setdefault("stats_log_s", 0)
    app = R.create_app([f.addr for f in ps], [f.addr for f in ds], **kw)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    task = asyncio.ensure_future(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    sess = aiohttp.ClientSession()
    ctx = type("Ctx", (), {})()
    ctx.p, ctx.d, ctx.url, ctx.sess, ctx.router = ps, ds, f"http://127.0.0.1:{port}", sess, app.state.router
    try:
        yield ctx
    finally:
        await sess.close()
        server.should_exit = True
        await task
        for f in ps + ds:
            await f.stop()


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 60))


async def wait_for(cond, t=5.0):
    end = asyncio.get_event_loop().time() + t
    while asyncio.get_event_loop().time() < end:
        if cond():
            return True
        await asyncio.sleep(0.05)
    return cond()


def chat_body(text="hi", **kw):
    return {"model": "m", "messages": [{"role": "user", "content": text}], "stream": True,
            "stream_options": {"include_usage": True}, **kw}


async def post(c, body, **kw):
    async with c.sess.post(c.url + "/v1/chat/completions", json=body, **kw) as r:
        return r.status, await r.read(), dict(r.headers)


# ---- 1. client disconnect cancels the running leg

def test_disconnect_during_prefill_cancels_prefill_and_skips_decode():
    async def t():
        async with stack() as c:
            c.p[0].mode = "hang"
            task = asyncio.ensure_future(post(c, chat_body()))
            assert await wait_for(lambda: c.p[0].active == 1)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert await wait_for(lambda: c.p[0].cancelled == 1)
            assert await wait_for(lambda: c.router.inflight["prefill"] == [0])
            assert c.d[0].bodies == []
    run(t())


def test_disconnect_while_waiting_for_decode_cancels_decode():
    async def t():
        async with stack() as c:
            c.d[0].mode = "hang"
            task = asyncio.ensure_future(post(c, chat_body()))
            assert await wait_for(lambda: c.d[0].active == 1)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert await wait_for(lambda: c.d[0].cancelled == 1)
            assert await wait_for(lambda: c.router.inflight["decode"] == [0] and c.router.decode_waiting == 0)
    run(t())


def test_disconnect_mid_stream_cancels_decode():
    async def t():
        async with stack() as c:
            c.d[0].stream_delay = 5
            async with c.sess.post(c.url + "/v1/chat/completions", json=chat_body()) as r:
                assert r.status == 200
                await r.content.readany()
            # leaving the context closes the connection
            assert await wait_for(lambda: c.d[0].cancelled == 1)
            assert await wait_for(lambda: c.router.inflight["decode"] == [0])
    run(t())


# ---- 2. KV transfer / decode first byte timeout

def test_kv_timeout_returns_engine_shaped_504_and_cancels_decode():
    async def t():
        async with stack(kv_timeout=0.6) as c:
            c.d[0].mode = "hang"
            st, data, _ = await post(c, chat_body())
            assert st == 504
            err = json.loads(data)["error"]
            assert err["code"] == 504 and err["type"] == "GatewayTimeoutError" and err["message"]
            assert await wait_for(lambda: c.d[0].cancelled == 1)
            assert c.router.counts["decode_first_byte_timeout"] == 1
            assert c.router.decode_waiting == 0
    run(t())


def test_default_timeouts_fit_the_default_lease(monkeypatch):
    for k in ("PD_KV_TIMEOUT_S", "PD_KV_LEASE_S", "PD_PREFILL_TIMEOUT_S"):
        monkeypatch.delenv(k, raising=False)
    r = R.Router(["a:1"], ["b:2"])
    assert r.kv_timeout == 20 and r.kv_lease == 30 and r.prefill_timeout == 600
    monkeypatch.setenv("PD_KV_TIMEOUT_S", "7")
    assert R.Router(["a:1"], ["b:2"]).kv_timeout == 7


def test_router_refuses_a_kv_timeout_that_outlives_the_lease(monkeypatch):
    monkeypatch.delenv("PD_KV_TIMEOUT_S", raising=False)
    monkeypatch.delenv("PD_KV_LEASE_S", raising=False)
    for timeout, lease in ((25, 30), (30, 30), (120, 30), (0, 30), (120, 125)):
        with pytest.raises(ValueError, match="PD_KV_LEASE_S"):
            R.Router(["a:1"], ["b:2"], kv_timeout=timeout, kv_lease=lease)
    assert R.Router(["a:1"], ["b:2"], kv_timeout=24.9, kv_lease=30).kv_timeout == 24.9
    assert R.Router(["a:1"], ["b:2"], kv_timeout=120, kv_lease=150).kv_lease == 150
    monkeypatch.setenv("PD_KV_TIMEOUT_S", "120")  # the old default against the stock lease
    with pytest.raises(ValueError):
        R.Router(["a:1"], ["b:2"])
    monkeypatch.setenv("PD_KV_LEASE_S", "150")
    assert R.Router(["a:1"], ["b:2"]).kv_timeout == 120


def test_metrics_expose_kv_timeouts_and_waiting_requests():
    async def t():
        async with stack(kv_timeout=1.5) as c:
            c.d[0].mode = "hang"
            task = asyncio.ensure_future(post(c, chat_body()))
            assert await wait_for(lambda: c.d[0].active == 1)
            async with c.sess.get(c.url + "/metrics") as r:
                text = await r.text()
            assert "pd_router_decode_waiting_first_byte 1\n" in text
            st, _, _ = await task
            assert st == 504
            async with c.sess.get(c.url + "/v1/metrics") as r:
                text = await r.text()
            assert "pd_router_decode_waiting_first_byte 0\n" in text
            assert "pd_router_decode_first_byte_timeouts_total 1\n" in text
            assert "# TYPE pd_router_decode_first_byte_timeouts_total counter" in text
    run(t())


# ---- 3. health

def test_health_requires_all_ranks_and_is_never_cached():
    async def t():
        async with stack(n_p=2, n_d=2) as c:
            async def health():
                async with c.sess.get(c.url + "/health") as r:
                    return r.status, await r.json()
            assert (await health())[0] == 200
            c.d[1].health = 503  # one decode rank down: the pair is unhealthy at the very next call
            st, j = await health()
            assert st == 503 and j["decode"] == {"up": 1, "total": 2}
            c.d[1].health = 200
            assert (await health())[0] == 200
            c.p[1].health = 500
            assert (await health())[0] == 503
    run(t())


def test_health_timeout_counts_as_unhealthy():
    async def t():
        async with stack(health_timeout=0.3) as c:
            c.d[0].health_delay = 2
            async with c.sess.get(c.url + "/health") as r:
                assert r.status == 503
    run(t())


def test_health_default_probe_timeout_is_short(monkeypatch):
    monkeypatch.delenv("PD_HEALTH_TIMEOUT_S", raising=False)
    assert R.Router(["a:1"], ["b:2"]).health_timeout == 2


def test_health_unreachable_backend():
    async def t():
        async with stack() as c:
            await c.d[0].stop()
            async with c.sess.get(c.url + "/health") as r:
                assert r.status == 503
            await c.d[0].start()
    run(t())


# ---- 4. metrics aggregation

P_METRICS = """# HELP vllm:num_requests_running Running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 2.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="m"} 100.0
vllm:prompt_tokens_created{engine="0",model_name="m"} 1.7e9
# TYPE lat histogram
lat_bucket{le="1.0",engine="0"} 1.0
lat_bucket{le="+Inf",engine="0"} 2.0
lat_sum{engine="0"} 1.5
lat_count{engine="0"} 2.0
"""
D_METRICS = P_METRICS.replace("2.0\n# TYPE vllm:num_requests_waiting", "5.0\n# TYPE vllm:num_requests_waiting") \
    .replace("0.25", "0.75")


def test_metrics_sum_pair_totals():
    async def t():
        async with stack(n_p=1, n_d=2) as c:
            c.p[0].metrics, c.d[0].metrics, c.d[1].metrics = P_METRICS, D_METRICS, D_METRICS
            async with c.sess.get(c.url + "/v1/metrics") as r:
                assert r.status == 200
                text = await r.text()
            assert 'vllm:num_requests_running{model_name="m"} 12\n' in text   # 2 + 5 + 5
            assert 'vllm:num_requests_waiting{model_name="m"} 3\n' in text    # 1 + 1 + 1
            assert 'vllm:kv_cache_usage_perc{model_name="m"} 0.75\n' in text  # ratio gauges take the max
            assert 'vllm:prompt_tokens_total{model_name="m"} 300\n' in text
            assert "_created" not in text
            assert 'lat_bucket{le="1.0"} 3\n' in text and 'lat_bucket{le="+Inf"} 6\n' in text
            assert text.count("# TYPE vllm:num_requests_running") == 1
            assert "pd_router_backends_up 3\n" in text
            async with c.sess.get(c.url + "/metrics") as r:
                assert (await r.text()).count("vllm:num_requests_running{") == 1
    run(t())


def test_metrics_per_rank_keeps_series_with_labels():
    async def t():
        async with stack(n_p=1, n_d=2) as c:
            c.p[0].metrics, c.d[0].metrics, c.d[1].metrics = P_METRICS, D_METRICS, D_METRICS
            async with c.sess.get(c.url + "/metrics?per_rank=1") as r:
                text = await r.text()
            assert 'vllm:num_requests_running{engine="0",model_name="m",pd_role="prefill",pd_rank="0"} 2\n' in text
            assert 'pd_role="decode",pd_rank="1"' in text
    run(t())


def test_metrics_fail_when_any_rank_is_missing_or_slow():
    async def t():
        async with stack(n_p=1, n_d=2, metrics_timeout=0.3) as c:
            for f in c.p + c.d:
                f.metrics = P_METRICS
            async with c.sess.get(c.url + "/metrics") as r:
                assert r.status == 200 and "pd_router_backends_up 3\n" in await r.text()
            await c.d[1].stop()
            async with c.sess.get(c.url + "/metrics") as r:
                assert r.status == 503
                assert "2 of 3" in (await r.json())["error"]["message"]
            await c.d[1].start()
            c.d[1].metrics_delay = 1.0  # slower than the scrape timeout
            async with c.sess.get(c.url + "/metrics") as r:
                assert r.status == 503
            c.d[1].metrics_delay = 0.0
            c.d[1].health_delay = 1.0  # a slow /health probe does not limit the scrape
            async with c.sess.get(c.url + "/metrics") as r:
                assert r.status == 200
    run(t())


# ---- 5. errors

def test_prefill_429_stays_429_with_engine_body_and_no_bytes_leak():
    async def t():
        async with stack() as c:
            p = c.p[0]
            p.mode, p.status, p.headers = "error", 429, {"Retry-After": "3"}
            st, data, h = await post(c, chat_body())
            assert st == 429 and h.get("Retry-After") == "3"
            assert json.loads(data)["error"]["message"] == "engine says no"
            assert not c.d[0].bodies  # decode never contacted after a leg 1 failure
    run(t())


def test_prefill_non_json_error_is_wrapped_with_status_kept():
    async def t():
        async with stack() as c:
            p = c.p[0]
            p.mode, p.status, p.error_body = "error", 503, "<html>upstream text</html>"
            st, data, _ = await post(c, chat_body())
            err = json.loads(data)["error"]
            assert st == 503 and err["code"] == 503 and "upstream text" not in data.decode()
    run(t())


def test_queue_full_body_forwarded_verbatim():
    async def t():
        async with stack() as c:
            body = json.dumps({"object": "error", "message": "queue is full", "type": "ServiceUnavailable", "code": 503})
            c.p[0].mode, c.p[0].status, c.p[0].error_body = "error", 503, body
            st, data, _ = await post(c, chat_body())
            assert st == 503 and data.decode() == body
    run(t())


def test_multi_segment_error_body_is_read_whole():
    async def t():
        async with stack() as c:
            body = json.dumps({"object": "error", "message": "queue is full " + "x" * 3000, "type": "ServiceUnavailable",
                               "code": 503})
            c.p[0].mode, c.p[0].status, c.p[0].error_body, c.p[0].error_segments = "error", 503, body, 5
            st, data, _ = await post(c, chat_body())
            assert st == 503 and data.decode() == body
    run(t())


def test_multi_segment_decode_error_body_is_read_whole():
    async def t():
        async with stack() as c:
            body = json.dumps({"error": {"message": "bad " + "y" * 3000, "type": "X", "code": 400}})
            c.d[0].mode, c.d[0].status, c.d[0].error_body, c.d[0].error_segments = "error", 400, body, 5
            st, data, _ = await post(c, chat_body())
            assert st == 400 and data.decode() == body
    run(t())


def test_read_capped_stops_at_the_cap():
    class S:
        def __init__(self):
            self.parts = [b"ab", b"cd", b"ef"]

        async def read(self, n):
            return self.parts.pop(0)[:n] if self.parts else b""
    assert asyncio.run(R.read_capped(S(), 5)) == b"abcde"
    assert asyncio.run(R.read_capped(S(), 100)) == b"abcdef"


def test_serialization_failure_does_not_leak_inflight(monkeypatch):
    async def t():
        async with stack() as c:
            monkeypatch.setattr(R, "jdumps", lambda o: (_ for _ in ()).throw(ValueError("x")))
            await post(c, chat_body())
            assert c.router.inflight["prefill"] == [0] and c.router.inflight["decode"] == [0]
            assert not c.p[0].bodies
    run(t())


def test_serialization_failure_on_decode_leg_does_not_leak_inflight(monkeypatch):
    async def t():
        async with stack() as c:
            real, calls = R.jdumps, []

            def flaky(o):
                calls.append(1)
                if len(calls) == 2:
                    raise ValueError("x")
                return real(o)
            monkeypatch.setattr(R, "jdumps", flaky)
            await post(c, chat_body())
            assert c.router.inflight["prefill"] == [0] and c.router.inflight["decode"] == [0]
            assert not c.d[0].bodies
    run(t())


def test_batched_completions_prompt_is_rejected():
    async def t():
        async with stack() as c:
            async def comp(prompt):
                async with c.sess.post(c.url + "/v1/completions", json={"model": "m", "prompt": prompt}) as r:
                    return r.status, await r.read()
            for bad in (["a", "b"], ["a"], [[1, 2], [3]], [[1, 2]]):
                st, data = await comp(bad)
                assert st == 400 and json.loads(data)["error"]["type"] == "BadRequestError", bad
            assert not c.p[0].bodies and not c.d[0].bodies
            for ok in ("hello", [1, 2, 3]):
                st, _ = await comp(ok)
                assert st == 200, ok
    run(t())


def test_prefill_without_transfer_params_is_502():
    async def t():
        async with stack() as c:
            c.p[0].mode = "nokvp"
            st, data, _ = await post(c, chat_body())
            assert st == 502 and json.loads(data)["error"]["code"] == 502
            assert not c.d[0].bodies
    run(t())


def test_decode_429_stays_429():
    async def t():
        async with stack() as c:
            c.d[0].mode, c.d[0].status = "error", 429
            st, data, _ = await post(c, chat_body())
            assert st == 429 and json.loads(data)["error"]["code"] == 429
    run(t())


def test_unreachable_engines_are_502_and_bad_body_is_400():
    async def t():
        async with stack() as c:
            await c.d[0].stop()
            st, data, _ = await post(c, chat_body())
            assert st == 502 and "error" in json.loads(data)
            async with c.sess.post(c.url + "/v1/chat/completions", data=b"[1,2]") as r:
                assert r.status == 400 and "error" in await r.json()
            await c.d[0].start()
    run(t())


# ---- 6. usage only from decode, body shaping

def test_only_decode_usage_reaches_client_and_bodies_are_shaped():
    async def t():
        async with stack() as c:
            body = chat_body(logprobs=True, n=1, kv_transfer_params={"do_remote_decode": False, "evil": 1},
                             response_format={"type": "json_object"}, temperature=0.3)
            st, data, h = await post(c, body)
            text = data.decode()
            assert st == 200 and "x-pd-prefill-ms" in h
            assert '"total_tokens": 10' in text and "1111" not in text
            pre, dec = c.p[0].bodies[0], c.d[0].bodies[0]
            assert pre["max_tokens"] == 1 and pre["stream"] is False
            assert pre["kv_transfer_params"] == {"do_remote_decode": True}
            for k in ("stream_options", "logprobs", "n", "response_format"):
                assert k not in pre
            assert dec["stream_options"] == {"include_usage": True} and dec["temperature"] == 0.3
            assert dec["kv_transfer_params"]["remote_block_ids"] == [[1, 2]] and "evil" not in dec["kv_transfer_params"]
            assert dec["stream"] is True and "max_tokens" not in dec
    run(t())


def test_non_streaming_request_returns_decode_json():
    async def t():
        async with stack() as c:
            st, data, _ = await post(c, chat_body(stream=False))
            assert st == 200 and json.loads(data)["usage"] == DECODE_USAGE
    run(t())


def test_get_models_passthrough_and_unknown_get_404():
    async def t():
        async with stack() as c:
            async with c.sess.get(c.url + "/v1/models") as r:
                assert r.status == 200 and (await r.json())["data"][0]["id"] == "m"
            async with c.sess.get(c.url + "/server_info") as r:
                assert r.status == 404
    run(t())


# ---- 7. affinity

def test_sticky_per_conversation_on_both_legs_and_spread():
    async def t():
        async with stack(n_p=3, n_d=3) as c:
            for _ in range(4):
                assert (await post(c, chat_body("conv-a")))[0] == 200
            assert sorted(len(f.bodies) for f in c.p) == [0, 0, 4]
            assert sorted(len(f.bodies) for f in c.d) == [0, 0, 4]
            for i in range(30):
                await post(c, chat_body(f"conv-{i}"))
            assert all(len(f.bodies) > 0 for f in c.p) and all(len(f.bodies) > 0 for f in c.d)
    run(t())


def test_affinity_ignores_turns_after_the_first_user_message():
    a = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
                      {"role": "user", "content": "z"}]}
    b = {"messages": a["messages"][:2] + [{"role": "user", "content": "other"}]}
    assert R.Router.affinity(a) == R.Router.affinity(b)
    assert R.Router.affinity({"prompt": "p"}) != R.Router.affinity({"prompt": "q"})


def test_max_imbalance_moves_off_a_hot_sticky_rank():
    r = R.Router(["p0:1", "p1:1", "p2:1"], ["d0:1"], max_imbalance=2)
    h = 3 * 10 + 1  # sticky rank 1
    assert r.pick("prefill", h) == 1
    r.inflight["prefill"] = [0, 3, 1]
    assert r.pick("prefill", h) == 0  # 3 - 0 > 2: least loaded wins
    r.inflight["prefill"] = [1, 3, 1]
    assert r.pick("prefill", h) == 1  # 3 - 1 <= 2: stay sticky


def test_ranks_come_from_this_pair_only():
    async def t():
        async with stack(n_p=2, n_d=1) as c:
            for i in range(10):
                await post(c, chat_body(f"c{i}"))
            assert len(c.d[0].bodies) == 10 and sum(len(f.bodies) for f in c.p) == 10
    run(t())


# ---- 8. no content in logs

def test_logs_never_contain_request_content(caplog):
    async def t():
        async with stack(kv_timeout=0.4) as c:
            c.p[0].mode = "error"
            c.p[0].status = 500
            c.p[0].error_body = json.dumps({"error": {"message": f"echo {SECRET}"}})
            await post(c, chat_body(SECRET))
            c.p[0].mode = "ok"
            c.d[0].mode = "hang"
            await post(c, chat_body(SECRET))
            c.d[0].mode = "error"
            await post(c, chat_body(SECRET))
            await c.d[0].stop()
            await post(c, chat_body(SECRET))
            await c.d[0].start()
    with caplog.at_level(logging.DEBUG):
        run(t())
    # the fake engines' own access logs are not the router's; check the router's logger
    recs = [r for r in caplog.records if r.name == "pd-router"]
    assert len(recs) >= 4
    text = "\n".join(r.getMessage() for r in recs)
    assert SECRET not in text and "127.0.0.1" not in text and "echo" not in text


# ---- 7. review fixes

class FakeReq:
    def __init__(self, body, disconnected):
        self._body, self._disc = json.dumps(body).encode(), disconnected
        self.headers = {}

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return await self._disc()


def test_guarded_closes_a_result_that_completed_at_the_disconnect_check():
    async def t():
        router = R.Router(["a:1"], ["b:2"], kv_timeout=10)
        gate, closed = asyncio.Event(), []

        class Resp:
            def close(self):
                closed.append(1)

        async def work():
            await gate.wait()
            return Resp(), b"first"

        async def disconnected():  # the engine answers exactly while the client-gone check runs
            gate.set()
            for _ in range(5):
                await asyncio.sleep(0)
            return True

        task = asyncio.ensure_future(work())
        with pytest.raises(R.Gone):
            await router.guarded(FakeReq({}, disconnected), task, 0)
        assert closed == [1]

        # same race at the deadline
        gate.clear()
        closed.clear()
        task = asyncio.ensure_future(work())

        async def not_gone():
            gate.set()
            for _ in range(5):
                await asyncio.sleep(0)
            return False

        with pytest.raises(R.Timeout):
            await router.guarded(FakeReq({}, not_gone), task, 0.001)
        assert closed == [1]
    run(t())


def test_unstarted_stream_still_releases_connection_and_inflight():
    async def t():
        async with stack() as c:
            async def never_gone():
                return False
            resp = await c.router.handle(FakeReq(chat_body(), never_gone), "/v1/chat/completions")
            assert isinstance(resp, R.GuardedStream) and c.router.inflight["decode"] == [1]

            async def send(msg):
                raise OSError("client went away before the first write")

            async def receive():
                return {"type": "http.disconnect"}

            with contextlib.suppress(BaseException):
                await resp({"type": "http", "asgi": {"version": "3.0"}, "method": "POST", "headers": []},
                           receive, send)
            assert c.router.inflight["decode"] == [0]
            assert await wait_for(lambda: c.d[0].active == 0)
    run(t())


def test_mid_stream_upstream_break_aborts_the_client_stream():
    async def t():
        async with stack() as c:
            c.d[0].break_after = 1
            async with c.sess.post(c.url + "/v1/chat/completions", json=chat_body()) as r:
                assert r.status == 200
                with pytest.raises(aiohttp.ClientError):
                    await r.read()
            assert await wait_for(lambda: c.router.counts["decode_fail"] == 1)
            assert await wait_for(lambda: c.router.inflight["decode"] == [0])
    run(t())


def test_n_greater_than_one_is_rejected_and_n_one_passes():
    async def t():
        async with stack() as c:
            st, data, _ = await post(c, chat_body(n=2))
            err = json.loads(data)["error"]
            assert st == 400 and err["type"] == "BadRequestError" and err["code"] == 400
            assert c.p[0].bodies == [] and c.d[0].bodies == []
            st, _, _ = await post(c, chat_body(n=1))
            assert st == 200
            for bad in (2.0, 1.0, "2", "1", True, False, 0, -1, None, [1]):
                st, data, _ = await post(c, chat_body(n=bad))
                assert st == 400, bad
                assert json.loads(data)["error"]["type"] == "BadRequestError"
            assert len(c.p[0].bodies) == 1 and len(c.d[0].bodies) == 1
    run(t())


def test_prompt_logprobs_is_not_sent_to_prefill():
    async def t():
        async with stack() as c:
            await post(c, chat_body(prompt_logprobs=1, logprobs=True, top_logprobs=2))
            assert "prompt_logprobs" not in c.p[0].bodies[0]
    run(t())


EVIL = {"do_remote_prefill": True, "do_remote_decode": False, "remote_prefill_cached_tokens": 999999,
        "remote_block_ids": [[666]], "remote_host": "evil"}


def test_client_transfer_params_never_reach_decode_on_any_path():
    async def t():
        async with stack() as c:
            c.p[0].prefill_extra = {"remote_prefill_cached_tokens": 2432}
            variants = [chat_body(), chat_body(stream=False), chat_body(stream=False, stream_options=None),
                        chat_body(vllm_xargs={"kv_transfer_params": "x"}), chat_body(stream_options={"include_usage": False})]
            for v in variants:
                v["kv_transfer_params"] = dict(EVIL)
                st, _, _ = await post(c, v)
                assert st == 200
            for pre in c.p[0].bodies:
                assert pre["kv_transfer_params"] == {"do_remote_decode": True}
            assert len(c.d[0].bodies) == len(variants)
            for dec in c.d[0].bodies:
                assert dec["kv_transfer_params"] == {"do_remote_prefill": True, "remote_block_ids": [[1, 2]],
                                                     "remote_host": "h", "remote_prefill_cached_tokens": 2432}
                assert "evil" not in json.dumps(dec) and "999999" not in json.dumps(dec)
    run(t())


def test_remote_prefill_cached_tokens_comes_only_from_the_prefill_engine():
    async def t():
        async with stack() as c:  # the prefill engine reports none: the client's value must not appear
            body = chat_body(stream=False)
            body["kv_transfer_params"] = {"remote_prefill_cached_tokens": 5}
            await post(c, body)
            assert "remote_prefill_cached_tokens" not in c.d[0].bodies[0]["kv_transfer_params"]
    run(t())


def test_usage_is_forwarded_unchanged_stream_and_non_stream():
    async def t():
        async with stack() as c:
            st, data, _ = await post(c, chat_body())
            events = [e for e in data.decode().split("\n\n") if e]
            assert json.loads(events[-2][len("data: "):]) == {"usage": DECODE_USAGE} and events[-1] == "data: [DONE]"
            assert c.d[0].bodies[0]["stream_options"] == {"include_usage": True}
            st, data, _ = await post(c, chat_body(stream=False, stream_options={"include_usage": True}))
            assert json.loads(data) == {"choices": [], "usage": DECODE_USAGE}
            assert c.d[0].bodies[1]["stream_options"] == {"include_usage": True}
    run(t())


def test_prefill_failure_forwards_no_usage_and_never_starts_decode():
    async def t():
        async with stack() as c:
            c.p[0].mode = "error"
            c.p[0].status = 500
            st, data, _ = await post(c, chat_body())
            assert st == 500 and b"usage" not in data and c.d[0].bodies == []
    run(t())
