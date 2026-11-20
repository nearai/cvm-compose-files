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
        self.health = 200
        self.health_delay = 0
        self.metrics = ""
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
                return web.Response(body=data, status=self.status, headers=self.headers,
                                    content_type="application/json")
            if self.mode == "hang":
                await asyncio.sleep(3600)
            if self.mode == "nokvp":
                return web.json_response({"choices": [], "usage": PREFILL_USAGE})
            if self.role == "prefill":
                return web.json_response({"choices": [], "usage": PREFILL_USAGE, "kv_transfer_params": {
                    "do_remote_prefill": True, "remote_block_ids": [[1, 2]], "remote_host": "h"}})
            if body.get("stream"):
                resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
                await resp.prepare(request)
                for i in range(3):
                    await resp.write(f'data: {{"i": {i}}}\n\n'.encode())
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
            assert c.router.counts["kv_timeout"] == 1
            assert c.router.decode_waiting == 0
    run(t())


def test_default_kv_timeout_is_120s(monkeypatch):
    monkeypatch.delenv("PD_KV_TIMEOUT_S", raising=False)
    assert R.Router(["a:1"], ["b:2"]).kv_timeout == 120
    monkeypatch.setenv("PD_KV_TIMEOUT_S", "7")
    assert R.Router(["a:1"], ["b:2"]).kv_timeout == 7


def test_metrics_expose_kv_timeouts_and_waiting_requests():
    async def t():
        async with stack(kv_timeout=1.5) as c:
            c.d[0].mode = "hang"
            task = asyncio.ensure_future(post(c, chat_body()))
            assert await wait_for(lambda: c.d[0].active == 1)
            async with c.sess.get(c.url + "/metrics") as r:
                text = await r.text()
            assert "pd_router_decode_waiting_kv 1\n" in text
            st, _, _ = await task
            assert st == 504
            async with c.sess.get(c.url + "/v1/metrics") as r:
                text = await r.text()
            assert "pd_router_decode_waiting_kv 0\n" in text
            assert "pd_router_kv_timeouts_total 1\n" in text
            assert "# TYPE pd_router_kv_timeouts_total counter" in text
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


def test_metrics_skips_dead_backend_and_reports_it():
    async def t():
        async with stack(n_p=1, n_d=2) as c:
            c.p[0].metrics, c.d[0].metrics = P_METRICS, P_METRICS
            await c.d[1].stop()
            async with c.sess.get(c.url + "/metrics") as r:
                text = await r.text()
            assert "pd_router_backends_up 2\n" in text and "pd_router_backends_total 3\n" in text
            await c.d[1].start()
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
            body = chat_body(logprobs=True, n=2, kv_transfer_params={"do_remote_decode": False, "evil": 1},
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


def test_affinity_ignores_turns_after_the_first_two_messages():
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
