"""Direct-attempt fallback tests for router.py: synthetic strings only, fake engines, ASGI client.

A direct attempt that fails before the first client byte (server error, connection error, stall, empty stream) is
handed to the P/D path; once a byte went to the client nothing is retried."""
import asyncio
import importlib.util
import json
import os
import re
import sys

import aiohttp.web as web
import httpx
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pd_router_fallback", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "router.py"))
rd = importlib.util.module_from_spec(_SPEC)
sys.modules["pd_router_fallback"] = rd
_SPEC.loader.exec_module(rd)


def U(n=35):
    return {"role": "user", "content": "a" * n}


def AS(n=35):
    return {"role": "assistant", "content": "b" * n}


FOLLOW = {"model": "m", "stream": True, "max_tokens": 77, "messages": [U(), AS(), U(35)]}


def chunk(delta=None, finish=None, usage=None):
    obj = {"choices": [{"index": 0, "delta": delta if delta is not None else {}, "finish_reason": finish}]}
    if usage is not None:
        obj["usage"] = usage
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


ROLE = chunk({"role": "assistant", "content": ""})
DONE = b"data: [DONE]\n\n"


class Backends:
    """P answers leg 1; D answers P/D leg 2 (with kv_transfer_params) with "pdcontent" and a direct attempt
    (without) according to `mode`."""

    def __init__(self, mode):
        self.mode = mode
        self.calls = []
        self.servers = []

    async def direct(self, request, body):
        m = self.mode
        if m.startswith("http"):
            code = int(m[4:])
            return web.json_response({"error": {"message": "x", "type": "x", "param": None, "code": code}}, status=code)
        if m == "conn":
            request.transport.abort()
            await asyncio.sleep(0.05)
            return web.Response()
        if m == "silent":
            await asyncio.sleep(3)
            return web.Response()
        resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await resp.prepare(request)
        if m == "sse500":
            await resp.write(b'data: {"error": {"code": 500}}\n\n' + DONE)
        elif m == "empty":  # role-only chunk, then a finish chunk and usage without generated tokens
            await resp.write(ROLE)
            await resp.write(chunk(finish="stop"))
            await resp.write(chunk(usage={"prompt_tokens": 5, "completion_tokens": 0}) + DONE)
        elif m == "empty_done_only":
            await resp.write(DONE)
        elif m == "empty_body":
            pass
        elif m == "role_stall":  # role-only chunk, then nothing within the first-event budget
            await resp.write(ROLE)
            await asyncio.sleep(1)
            await resp.write(chunk({"content": "late"}) + DONE)
        elif m == "role_then_content":
            await resp.write(ROLE)
            await asyncio.sleep(0.05)
            await resp.write(chunk({"content": "directcontent"}) + chunk(finish="stop") + DONE)
        elif m == "reasoning_first":
            await resp.write(chunk({"reasoning": "think"}) + chunk({"content": "directcontent"}) + DONE)
        elif m == "tool_first":
            await resp.write(ROLE + chunk({"tool_calls": [{"index": 0, "function": {"name": "f"}}]}) + DONE)
        elif m == "usage_tokens_only":  # no delta text, but the usage says tokens were generated
            await resp.write(chunk(usage={"completion_tokens": 3}) + DONE)
        elif m == "sent_then_break":
            await resp.write(ROLE + chunk({"content": "partial"}))
            await asyncio.sleep(0.05)
            request.transport.abort()
        elif m == "ok":
            await resp.write(chunk({"role": "assistant", "content": "directcontent"}) + DONE)
        return resp

    async def start(self):
        def handler_for(role):
            async def h(request):
                body = await request.json()
                self.calls.append((role, body))
                if role == "P":
                    return web.json_response({"choices": [], "kv_transfer_params": {"remote_block_ids": [1]}})
                if "kv_transfer_params" not in body:
                    return await self.direct(request, body)
                resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
                await resp.prepare(request)
                await resp.write(chunk({"role": "assistant", "content": "pdcontent"}) + DONE)
                return resp
            return h

        async def metrics_handler(request):
            return web.Response(text="# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 1\n")

        addrs = []
        for role in ("P", "D"):
            app = web.Application()
            app.router.add_post("/v1/chat/completions", handler_for(role))
            app.router.add_get("/metrics", metrics_handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            self.servers.append(runner)
            addrs.append("127.0.0.1:%d" % site._server.sockets[0].getsockname()[1])
        return addrs

    async def stop(self):
        for r in self.servers:
            await r.cleanup()


def run_case(mode, requests, **kw):
    async def go():
        be = Backends(mode)
        p, d = await be.start()
        app = rd.create_app([p], [d], stats_log_s=0, ddirect=True, **kw)
        router = app.state.router
        router._mem_served(router.affinity(FOLLOW), router.pick("decode", router.affinity(FOLLOW)), 100.0,
                           rd.ctx_hash(FOLLOW))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as cl:
                await requests(cl, be, router)
                stats = (await cl.get("/ddirect_stats")).json()
                metrics = (await cl.get("/metrics")).text
            return be, router, stats, metrics
        finally:
            await router.close()
            await be.stop()
    return asyncio.run(go())


def roles(be):
    return [c[0] for c in be.calls]


def fell_back(mode, reason, **kw):
    async def reqs(cl, be, router):
        r = await cl.post("/v1/chat/completions", json=FOLLOW)
        assert r.status_code == 200
        assert b"pdcontent" in r.content and b"[DONE]" in r.content
        assert b"directcontent" not in r.content and b"error" not in r.content
        assert r.headers["x-pd-path"] == "pd-fb"
        assert router.inflight == {"prefill": [0], "decode": [0]}
    be, router, stats, metrics = run_case(mode, reqs, **kw)
    assert roles(be) == ["D", "P", "D"]
    assert "kv_transfer_params" not in be.calls[0][1] and "kv_transfer_params" in be.calls[2][1]
    assert stats["sent_direct"] == 1 and stats["direct_rejected_fallback"] == 1
    assert stats["fallback_reason"] == {k: int(k == reason) for k in rd.DD_FB_REASONS}
    assert f'pd_router_ddirect_fallback_reason_total{{reason="{reason}"}} 1' in metrics
    return stats, router


@pytest.mark.parametrize("code", [500, 502, 504])
def test_direct_http_5xx_falls_back(code):
    fell_back(f"http{code}", "5xx")


def test_direct_sse_500_first_event_falls_back():
    fell_back("sse500", "5xx")


def test_direct_connection_error_falls_back():
    fell_back("conn", "conn")


def test_direct_first_byte_timeout_falls_back():
    fell_back("silent", "timeout", kv_timeout=0.3)


def test_direct_stall_after_role_event_falls_back(monkeypatch):
    monkeypatch.setattr(rd, "FIRST_EVENT_S", 0.1)
    fell_back("role_stall", "timeout")


def test_direct_unreachable_decode_falls_back_to_pd_error():
    async def go():
        app = rd.create_app(["127.0.0.1:1"], ["127.0.0.1:1"], stats_log_s=0, ddirect=True)
        rt = app.state.router
        rt._mem_served(rt.affinity(FOLLOW), 0, 100.0, rd.ctx_hash(FOLLOW))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as cl:
                return await cl.post("/v1/chat/completions", json=FOLLOW), rt
        finally:
            await rt.close()
    r, rt = asyncio.run(go())
    assert r.status_code == 502  # the P/D path was tried too and its prefill leg failed
    assert rt.dd["fallback_reason"]["conn"] == 1 and rt.counts["prefill_fail"] == 1
    assert rt.inflight == {"prefill": [0], "decode": [0]}


@pytest.mark.parametrize("mode", ["empty", "empty_done_only", "empty_body"])
def test_direct_empty_stream_falls_back(mode):
    fell_back(mode, "empty")


@pytest.mark.parametrize("mode,expect", [
    ("role_then_content", b"directcontent"),  # role-only first event, content next: served direct
    ("reasoning_first", b"think"),
    ("tool_first", b"tool_calls"),
    ("usage_tokens_only", b"completion_tokens"),
    ("ok", b"directcontent"),
])
def test_direct_with_output_is_served_direct(mode, expect):
    async def reqs(cl, be, router):
        r = await cl.post("/v1/chat/completions", json=FOLLOW)
        assert r.status_code == 200 and r.headers["x-pd-path"] == "direct"
        assert expect in r.content and b"pdcontent" not in r.content
        assert r.content.startswith(b"data: ") and r.content.endswith(DONE)
        assert router.inflight["decode"] == [0]
    be, _, stats, _ = run_case(mode, reqs)
    assert roles(be) == ["D"]
    assert stats["direct_rejected_fallback"] == 0 and stats["direct_errors"] == 0
    assert sum(stats["fallback_reason"].values()) == 0


def test_role_event_is_held_only_until_first_output():
    """The held-back role event reaches the client together with the first output chunk, in order."""
    async def reqs(cl, be, router):
        r = await cl.post("/v1/chat/completions", json=FOLLOW)
        assert r.content == ROLE + chunk({"content": "directcontent"}) + chunk(finish="stop") + DONE
    run_case("role_then_content", reqs)


def test_error_after_bytes_sent_is_not_retried():
    async def reqs(cl, be, router):
        got = b""
        try:
            async with cl.stream("POST", "/v1/chat/completions", json=FOLLOW) as r:
                async for part in r.aiter_bytes():
                    got += part
        except Exception:
            pass  # the router aborts the client connection: a truncated stream must not look complete
        assert DONE not in got and b"pdcontent" not in got  # never completed, never answered by P/D
        assert router.counts["decode_fail"] == 1
        await asyncio.sleep(0.1)
        assert router.inflight["decode"] == [0]
    be, router, stats, _ = run_case("sent_then_break", reqs)
    assert roles(be) == ["D"]
    assert stats["direct_rejected_fallback"] == 0 and stats["direct_errors"] == 1
    assert sum(stats["fallback_reason"].values()) == 0


def test_pd_leg_is_never_retried():
    """The P/D decode leg keeps its behaviour: a 5xx or an empty stream goes to the client as is."""
    async def go():
        be = Backends("ok")
        p, d = await be.start()
        be.mode = "ok"
        app = rd.create_app([p], [d], stats_log_s=0, ddirect=True)
        rt = app.state.router

        async def start(*a, **k):
            raise ConnectionError("x")
        rt._decode_start = start
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as cl:
                r = await cl.post("/v1/chat/completions", json={"model": "m", "stream": True, "messages": [U(), U()]})
            return r, rt, be
        finally:
            await rt.close()
            await be.stop()
    r, rt, be = asyncio.run(go())
    assert r.status_code == 502 and rt.dd["fallback_reason"]["conn"] == 0


# ---- judge unit
def test_sse_judge_unit():
    j = rd._sse_judge
    assert j(b"") == ("more", None) and j(b"", eof=True) == ("empty", None)
    assert j(ROLE) == ("more", None) and j(ROLE, eof=True) == ("empty", None)
    assert j(ROLE + chunk({"content": "x"})) == ("ok", None)
    assert j(ROLE + chunk({"content": ""}) + DONE) == ("empty", None)
    assert j(b": ping\n\n" + ROLE) == ("more", None)
    assert j(b'data: {"error": {"code": 502}}\n\n') == ("ok", 502)
    assert j(b'data: {"x": 1}\n\ndata: [DONE]\n\n') == ("ok", None)  # unknown shape passes through
    assert j(b'{"choices": []}') == ("ok", None)
    assert j(b"data: {\"choices\": [{\"text\": \"t\"}]}\n\n") == ("ok", None)
    assert j(chunk(finish="stop", usage={"completion_tokens": 2})) == ("ok", None)
    assert j(chunk({"content": "x"})[:-2]) == ("more", None)  # event not complete yet
    assert j(chunk({"content": "x"})[:-2], eof=True) == ("ok", None)


# ---- residual histogram: once per request
def test_block_residual_observed_once_across_first_turn_and_overflow():
    body = {"model": "m", "stream": True, "messages": [{"role": "system", "content": "s" * 350}, U(35), U(35)]}

    async def reqs(cl, be, router):
        pfx = rd.prefix_info(body, rd.ctx_key(body), router.dd_cpt)
        router._prefix_served(pfx[0], 0, pfx[1])
        r = await cl.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        res = router.ft_residual_hist
        # the first-turn check saw the prefix (measurement mode) and the overflow check saw it again
        assert router.ft_result["would_attempt"] == 1 and router.ov_result["attempted"] == 1
        assert sum(res["buckets"]) == 1
        text = (await cl.get("/metrics")).text
        assert "pd_router_first_turn_block_residual_tokens_est_count 1\n" in text
    run_case("ok", reqs, ddirect_first_turn=False, ddirect_first_turn_min_prefix_tokens=10,
             ddirect_cache_block_tokens=64, ddirect_overflow=True, ddirect_overflow_p_inflight=0)


def test_block_residual_unit():
    router = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_cache_block_tokens=64)
    seen = [False]
    assert router._block_residual(100.0, 20.0, seen) == 56.0
    assert router._block_residual(100.0, 20.0, seen) == 56.0  # same request: value returned, not observed again
    assert sum(router.ft_residual_hist["buckets"]) == 1 and router.ft_residual_hist["sum"] == 56.0
    assert router._block_residual(100.0, 20.0) == 56.0  # no flag: observed
    assert sum(router.ft_residual_hist["buckets"]) == 2
    off = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_cache_block_tokens=0)
    assert off._block_residual(100.0, 20.0, [False]) is None and sum(off.ft_residual_hist["buckets"]) == 0


def test_fallback_reason_labels_are_fixed_and_content_free():
    _, _, stats, metrics = run_case("empty", lambda cl, be, router: _post(cl))
    assert set(stats["fallback_reason"]) == {"5xx", "conn", "timeout", "empty"}
    for ln in metrics.splitlines():
        if ln.startswith("pd_router_ddirect_fallback_reason_total"):
            assert re.fullmatch(r'pd_router_ddirect_fallback_reason_total\{reason="(5xx|conn|timeout|empty)"\} \d+', ln)


async def _post(cl):
    await cl.post("/v1/chat/completions", json=FOLLOW)
