"""Decode-direct tests for router.py: synthetic strings only, fake engines, ASGI client."""
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
    "pd_router_ddirect", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "router.py"))
rd = importlib.util.module_from_spec(_SPEC)
sys.modules["pd_router_ddirect"] = rd
_SPEC.loader.exec_module(rd)


def U(n_chars=35):
    return {"role": "user", "content": "a" * n_chars}


def AS(n_chars=35):
    return {"role": "assistant", "content": "b" * n_chars}


# ---- estimator
def test_tail_after_last_assistant_only():
    msgs = [U(1000), AS(1000), U(70), AS(5000), U(35), U(35)]
    assert rd.tail_chars(msgs) == 70
    assert rd.estimate_tail_tokens(msgs, 3.5) == 20


def test_tail_text_parts_only():
    msgs = [AS(), {"role": "user", "content": [{"type": "text", "text": "a" * 7}, {"type": "text", "text": "a" * 7}]}]
    assert rd.tail_chars(msgs) == 14


def test_tail_tool_results_counted_and_last_assistant_args_not():
    msgs = [AS(), {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "x" * 20}}]},
            {"role": "tool", "content": "y" * 30}, {"role": "tool", "content": [{"type": "text", "text": "z" * 5}]}]
    assert rd.tail_chars(msgs) == 35  # the tool-call message is the last assistant message: not in the tail


def test_tail_tool_call_args_counted_when_in_tail():
    msgs = [AS(), {"role": "user", "content": "q", "tool_calls": [{"function": {"arguments": "x" * 20}},
                                                                   {"function": {"arguments": {"k": "v"}}}]}]
    assert rd.tail_chars(msgs) == 1 + 20 + len(json.dumps({"k": "v"}))


def test_tail_ignores_garbage():
    assert rd.tail_chars([AS(), "str", None, {"role": "user"}, {"role": "user", "content": 5}]) == 0


# ---- eligibility
def cls(path="/v1/chat/completions", **body):
    return rd.ddirect_classify(path, body, 3.5, 1024, 2432, 2048)


def test_first_turn():
    assert cls(messages=[U(), U()]) == ("first_turn", None, None)


def test_completions_api():
    assert cls("/v1/completions", prompt="a" * 10)[0] == "completions"
    assert cls("/v1/completions", messages=[U(), AS(), U()])[0] == "completions"
    assert cls(prompt="x")[0] == "completions"


def test_images_any_message():
    img = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}, {"type": "text", "text": "a"}]}
    assert cls(messages=[img, AS(), U()])[0] == "images"
    assert cls(messages=[U(), AS(), img])[0] == "images"


def test_boundary_and_too_long():
    assert cls(messages=[U(), AS(), U(int(1024 * 3.5))])[:2] == (None, 1024.0)
    reason, est, _ = cls(messages=[U(), AS(), U(int(1024 * 3.5) + 4)])
    assert reason == "tail_too_long" and est > 1024


def test_long_history_short_tail_eligible():
    assert cls(messages=[U(500000), AS(500000), U(10)])[0] is None


def test_chunk_estimate_formula():
    # N = 10000 tokens, tail = 100: held = 9900, floor(9900/2432)=4 -> chunk = 10000 - 9728 = 272
    msgs = [U(int(9900 * 3.5)), AS(0), U(int(100 * 3.5))]
    tail, chunk = rd.estimate_chunk_tokens(msgs, 3.5, 2432)
    assert tail == 100 and abs(chunk - 272) < 1e-6
    # short conversation: nothing cached yet beyond zero blocks -> chunk = N
    tail, chunk = rd.estimate_chunk_tokens([U(350), AS(0), U(35)], 3.5, 2432)
    assert abs(chunk - 110) < 1e-6


def test_chunk_too_large_even_with_small_tail():
    # held 2400 tokens (< one block) + tail 100 -> chunk 2500 > 2048 although tail <= 1024
    reason, tail, chunk = cls(messages=[U(int(2400 * 3.5)), AS(0), U(int(100 * 3.5))])
    assert reason == "chunk_too_large" and tail == 100 and chunk > 2048


def test_chunk_small_with_long_history_eligible():
    # held 2432*3 = 7296 exactly on a checkpoint, tail 500 -> chunk 500
    reason, tail, chunk = cls(messages=[U(int(7296 * 3.5)), AS(0), U(int(500 * 3.5))])
    assert reason is None and abs(chunk - 500) < 1e-6


def test_env_defaults(monkeypatch):
    for k in ("ROUTER_DDIRECT", "ROUTER_DDIRECT_MAX_TAIL_TOKENS", "ROUTER_DDIRECT_CHARS_PER_TOKEN",
              "ROUTER_DDIRECT_MAX_D_INFLIGHT"):
        monkeypatch.delenv(k, raising=False)
    r = rd.Router(["p:1"], ["d:1"])
    assert (r.ddirect, r.dd_max_tail, r.dd_cpt, r.dd_max_inflight, r.dd_block, r.dd_max_chunk) == (False, 1024, 3.5, 48, 2432, 2048)
    monkeypatch.setenv("ROUTER_DDIRECT", "1")
    monkeypatch.setenv("ROUTER_DDIRECT_MAX_D_INFLIGHT", "7")
    r = rd.Router(["p:1"], ["d:1"])
    assert r.ddirect is True and r.dd_max_inflight == 7


# ---- fake backends
class Backends:
    def __init__(self, d_reject=None):
        self.calls = []  # (role, path, body)
        self.servers = []
        # d_reject: None | "http429" | "sse429" | "sse400" | "http503" | "http400": D's answer to a request WITHOUT
        # kv_transfer_params (a direct attempt); P/D leg-2 requests (with kv_transfer_params) are always served.
        self.d_reject = d_reject

    async def start(self):
        async def handler_for(role):
            async def h(request):
                body = await request.json()
                self.calls.append((role, request.path, body))
                if role == "P":
                    return web.json_response({"choices": [], "kv_transfer_params": {"remote_block_ids": [1]}})
                rj = self.d_reject if "kv_transfer_params" not in body else None
                if rj and rj.startswith("http"):
                    code = int(rj[4:])
                    return web.json_response({"error": {"message": "rejected (overloaded_long_count)", "type": "x",
                                                        "param": None, "code": code}}, status=code,
                                             headers={"Retry-After": "10"})
                if rj and rj.startswith("sse"):
                    kind, _, variant = rj.partition(":")
                    code = int(kind[3:])
                    resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
                    await resp.prepare(request)
                    ev = json.dumps({"error": {"message": "m", "type": "t", "param": None, "code": code}}).encode()
                    nl = b"\r\n\r\n" if variant == "crlf" else b"\n\n"
                    if variant == "4b":
                        await resp.write(b"data")
                        await asyncio.sleep(0.02)
                        await resp.write(b": " + ev + nl)
                    elif variant == "nospace":
                        await resp.write(b"data:" + ev + nl)
                    elif variant == "ping":
                        await resp.write(b": ping\n\n")
                        await asyncio.sleep(0.02)
                        await resp.write(b"data: " + ev + nl)
                    else:
                        await resp.write(b"data: " + ev[:10])  # split first event across writes
                        await asyncio.sleep(0.02)
                        await resp.write(ev[10:] + nl)
                    await resp.write(b"data: [DONE]" + nl)
                    return resp
                if rj == "stall":
                    resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
                    await resp.prepare(request)
                    await resp.write(b'data: {"x"')
                    await asyncio.sleep(0.4)
                    await resp.write(b': 1}\n\ndata: [DONE]\n\n')
                    return resp
                if body.get("stream"):
                    resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
                    await resp.prepare(request)
                    await resp.write(b"data: {\"x\": 1}\n\n")
                    await resp.write(b"data: [DONE]\n\n")
                    return resp
                return web.json_response({"choices": [{"message": {"content": "ok"}}]})
            return h

        async def metrics_handler(request):
            return web.Response(text="# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 1\n")

        addrs = []
        for role in ("P", "D"):
            app = web.Application()
            h = await handler_for(role)
            app.router.add_post("/v1/chat/completions", h)
            app.router.add_post("/v1/completions", h)
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


def run_case(ddirect, requests, d_reject=None, seed=True, **kw):
    async def go():
        be = Backends(d_reject)
        p, d = await be.start()
        app = rd.create_app([p], [d], stats_log_s=0, ddirect=ddirect, **kw)
        router = app.state.router
        if seed:  # pretend FOLLOW's conversation was just served by its D rank, so direct is attempted
            router._mem_served(router.affinity(FOLLOW), router.pick("decode", router.affinity(FOLLOW)), 100.0,
                            rd.ctx_hash(FOLLOW))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as cl:
                await requests(cl, be, router)
                stats = (await cl.get("/ddirect_stats")).json()
            return be, router, stats
        finally:
            await router.close()
            await be.stop()
    return asyncio.run(go())


FOLLOW = {"model": "m", "stream": True, "max_tokens": 77, "messages": [U(), AS(), U(35)]}
FIRST = {"model": "m", "stream": True, "messages": [U(), U()]}


def post(cl, body, path="/v1/chat/completions"):
    return cl.post(path, json=body)


def test_on_followup_hits_only_decode_with_unchanged_body():
    async def reqs(cl, be, router):
        r = await post(cl, FOLLOW)
        assert r.status_code == 200 and b"[DONE]" in r.content
        assert "x-pd-prefill-ms" not in r.headers and r.headers["x-pd-path"] == "direct"
        assert router.inflight["decode"] == [0] and router.inflight["prefill"] == [0]
    be, router, stats = run_case(True, reqs)
    assert [c[0] for c in be.calls] == ["D"]
    assert be.calls[0][2] == FOLLOW  # no kv_transfer_params, max_tokens untouched
    assert stats["eligible"] == 1 and stats["sent_direct"] == 1 and stats["guard_fallback"] == 0
    assert stats["tail_est"]["le_64"] == 1 and stats["chunk_est"]["le_64"] == 1


def test_client_kv_transfer_params_still_dropped_on_direct():
    async def reqs(cl, be, router):
        await post(cl, {**FOLLOW, "kv_transfer_params": {"do_remote_prefill": True}})
    be, _, _ = run_case(True, reqs)
    assert [c[0] for c in be.calls] == ["D"] and "kv_transfer_params" not in be.calls[0][2]


def test_off_is_pd_order():
    async def reqs(cl, be, router):
        r = await post(cl, FOLLOW)
        assert r.status_code == 200 and "x-pd-prefill-ms" in r.headers and "x-pd-path" not in r.headers
    be, router, stats = run_case(False, reqs)
    assert [c[0] for c in be.calls] == ["P", "D"]
    p, d = be.calls[0][2], be.calls[1][2]
    assert p["max_tokens"] == 1 and p["kv_transfer_params"] == {"do_remote_decode": True}
    assert d["kv_transfer_params"] == {"remote_block_ids": [1]} and d["max_tokens"] == 77
    assert stats["enabled"] is False and stats["eligible"] == 0 and sum(stats["ineligible_reason"].values()) == 0


def test_on_ineligible_cases_use_pd():
    img = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}

    async def reqs(cl, be, router):
        await post(cl, FIRST)
        await post(cl, {"stream": True, "messages": [U(), AS(), img]})
        await post(cl, {"stream": True, "messages": [U(), AS(), U(5000)]})
        await post(cl, {"stream": True, "prompt": "a" * 10}, "/v1/completions")
    be, _, stats = run_case(True, reqs)
    assert [c[0] for c in be.calls] == ["P", "D"] * 4
    assert stats["ineligible_reason"] == {"first_turn": 1, "tail_too_long": 1, "chunk_too_large": 0, "images": 1, "completions": 1}
    assert stats["eligible"] == 0 and stats["sent_direct"] == 0


def test_busy_guard_falls_back_to_pd():
    async def reqs(cl, be, router):
        router.inflight["decode"][0] = 5
        assert (await post(cl, FOLLOW)).status_code == 200
        router.inflight["decode"][0] = 4  # below the guard of 5: direct
        assert (await post(cl, FOLLOW)).status_code == 200
    be, router, stats = run_case(True, reqs, ddirect_max_d_inflight=5)
    assert [c[0] for c in be.calls] == ["P", "D", "D"]
    assert stats["eligible"] == 2 and stats["guard_fallback"] == 1 and stats["sent_direct"] == 1


def test_direct_error_counted_and_inflight_released():
    async def go():
        app = rd.create_app(["127.0.0.1:1"], ["127.0.0.1:1"], stats_log_s=0, ddirect=True)
        rt = app.state.router
        rt._mem_served(rt.affinity(FOLLOW), 0, 100.0, rd.ctx_hash(FOLLOW))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as cl:
                r = await cl.post("/v1/chat/completions", json=FOLLOW)
                return r, (await cl.get("/ddirect_stats")).json(), app.state.router.inflight
        finally:
            await app.state.router.close()
    r, s, inflight = asyncio.run(go())
    assert r.status_code == 502 and s["direct_errors"] == 1 and s["sent_direct"] == 1
    assert inflight == {"prefill": [0], "decode": [0]}


def test_nonstream_direct():
    async def reqs(cl, be, router):
        r = await post(cl, {**FOLLOW, "stream": False})
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
    be, _, _ = run_case(True, reqs)
    assert [c[0] for c in be.calls] == ["D"]


def test_stats_have_no_content():
    async def reqs(cl, be, router):
        await post(cl, {"stream": True, "messages": [{"role": "user", "content": "SECRETMARK"}, AS(), U()]})
    _, _, stats = run_case(True, reqs)
    assert "SECRETMARK" not in json.dumps(stats)


# ---- D pre-stream rejection -> P/D fallback
def _fallback_case(kind, stream=True):
    async def reqs(cl, be, router):
        r = await post(cl, {**FOLLOW, "stream": stream})
        assert r.status_code == 200
        if stream:
            assert b"[DONE]" in r.content and b"error" not in r.content
        else:
            assert r.json()["choices"][0]["message"]["content"] == "ok"
        assert r.headers["x-pd-path"] == "pd-fb" and "x-pd-prefill-ms" in r.headers
        assert router.inflight["decode"] == [0] and router.inflight["prefill"] == [0]
    be, router, stats = run_case(True, reqs, d_reject=kind)
    assert [c[0] for c in be.calls] == ["D", "P", "D"]
    assert "kv_transfer_params" not in be.calls[0][2]
    assert be.calls[1][2]["max_tokens"] == 1 and be.calls[1][2]["kv_transfer_params"] == {"do_remote_decode": True}
    assert be.calls[2][2]["kv_transfer_params"] == {"remote_block_ids": [1]} and be.calls[2][2]["max_tokens"] == 77
    assert stats["sent_direct"] == 1 and stats["direct_rejected_fallback"] == 1 and stats["direct_errors"] == 0
    assert router.counts["decode_fail"] == 0


def test_d_http_429_falls_back_to_pd():
    _fallback_case("http429")


def test_d_http_429_nonstream_falls_back_to_pd():
    _fallback_case("http429", stream=False)


def test_d_sse_429_first_event_falls_back_to_pd():
    _fallback_case("sse429")


def test_d_http_503_falls_back_to_pd():
    _fallback_case("http503")


def test_d_200_goes_d_only_and_no_fallback_count():
    async def reqs(cl, be, router):
        r = await post(cl, FOLLOW)
        assert r.status_code == 200 and r.headers["x-pd-path"] == "direct"
    be, _, stats = run_case(True, reqs)
    assert [c[0] for c in be.calls] == ["D"] and stats["direct_rejected_fallback"] == 0


def test_d_http_400_not_retried():
    async def reqs(cl, be, router):
        r = await post(cl, FOLLOW)
        assert r.status_code == 400
        assert router.inflight["decode"] == [0]
    be, _, stats = run_case(True, reqs, d_reject="http400")
    assert [c[0] for c in be.calls] == ["D"]
    assert stats["direct_rejected_fallback"] == 0 and stats["direct_errors"] == 1


def test_d_sse_400_forwarded_not_retried():
    async def reqs(cl, be, router):
        r = await post(cl, FOLLOW)
        assert r.status_code == 200 and b'"code": 400' in r.content and b"[DONE]" in r.content
        assert r.headers["x-pd-path"] == "direct"
    be, _, stats = run_case(True, reqs, d_reject="sse400")
    assert [c[0] for c in be.calls] == ["D"] and stats["direct_rejected_fallback"] == 0


def test_pd_leg2_never_falls_back():
    async def reqs(cl, be, router):
        r = await post(cl, FIRST)  # first turn: P/D; leg 2 carries kv_transfer_params so the fake D serves it
        assert r.status_code == 200 and r.headers["x-pd-path"] == "pd"
    be, _, stats = run_case(True, reqs, d_reject="http429")
    assert [c[0] for c in be.calls] == ["P", "D"] and stats["direct_rejected_fallback"] == 0


def test_sse_error_code_parser():
    assert rd._sse_error_code(b'data: {"error": {"code": 429}}\n\ndata: [DONE]\n\n') == 429
    assert rd._sse_error_code(b'data: {"x": 1}\n\n') is None
    assert rd._sse_error_code(b'data: not json\n\n') is None


# ---- per-conversation memory
def _paths(be):
    return [c[0] for c in be.calls]


def test_mem_no_entry_goes_pd_then_next_turn_direct():
    async def reqs(cl, be, router):
        r1 = await post(cl, FOLLOW)
        assert r1.headers["x-pd-path"] == "pd"  # nothing remembered: straight to P/D, no wasted D attempt
        r2 = await post(cl, FOLLOW)
        assert r2.headers["x-pd-path"] == "direct"  # the first completed response created the entry
    be, router, stats = run_case(True, reqs, seed=False)
    assert _paths(be) == ["P", "D", "D"]
    assert stats["skip_reason"] == {"no_entry": 1, "stale": 0, "rank_changed": 0, "rejected_prev": 0, "context_changed": 0}
    assert stats["sent_direct"] == 1 and stats["mem_size"] == 1


def test_mem_stale_entry_skips():
    async def reqs(cl, be, router):
        h = router.affinity(FOLLOW)
        router.dd_mem[h][1] -= 31  # older than the 30 s default
        r = await post(cl, FOLLOW)
        assert r.headers["x-pd-path"] == "pd"
    be, router, stats = run_case(True, reqs)
    assert _paths(be) == ["P", "D"] and stats["skip_reason"]["stale"] == 1 and stats["sent_direct"] == 0


def test_mem_max_age_knob():
    async def reqs(cl, be, router):
        router.dd_mem[router.affinity(FOLLOW)][1] -= 31
        assert (await post(cl, FOLLOW)).headers["x-pd-path"] == "direct"
    run_case(True, reqs, ddirect_max_age_s=60)


def test_mem_rank_changed_skips():
    async def go():
        be = Backends()
        p, d = await be.start()
        app = rd.create_app([p], [d, d], stats_log_s=0, ddirect=True)
        rt = app.state.router
        h = rt.affinity(FOLLOW)
        rt._mem_served(h, 1 - rt.pick("decode", h), 10.0)  # served by the other D rank
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as cl:
                r = await post(cl, FOLLOW)
                return r, (await cl.get("/ddirect_stats")).json(), rt.dd_mem[h][0] == rt.pick("decode", h)
        finally:
            await rt.close()
            await be.stop()
    r, stats, updated = asyncio.run(go())
    assert r.headers["x-pd-path"] == "pd" and stats["skip_reason"]["rank_changed"] == 1
    assert updated  # completed P/D response updated the memory to the rank that served it


def test_mem_rejected_prev_skips_until_served():
    async def reqs(cl, be, router):
        h = router.affinity(FOLLOW)
        r = await post(cl, FOLLOW)  # D rejects, falls back; the fallback's completion clears the flag
        assert r.headers["x-pd-path"] == "pd-fb" and router.dd_mem[h][3] is False
        router._mem_rejected(h)  # a concurrent rejection nothing has served since
        r = await post(cl, FOLLOW)
        assert r.headers["x-pd-path"] == "pd"
    be, router, stats = run_case(True, reqs, d_reject="http429")
    assert stats["skip_reason"]["rejected_prev"] == 1 and stats["direct_rejected_fallback"] == 1


def test_mem_updated_by_direct_and_stores_no_content():
    async def reqs(cl, be, router):
        h = router.affinity(FOLLOW)
        router.dd_mem[h][1] -= 20
        await post(cl, FOLLOW)
        e = router.dd_mem[h]
        assert router._now() - e[1] < 5 and e[2] > 0
        assert all(type(x) in (int, float, bool, bytes, type(None)) for x in e)
    be, router, stats = run_case(True, reqs)
    assert all(type(k) is int for k in router.dd_mem)
    assert "aaaa" not in json.dumps(stats)


def test_mem_lru_cap_and_ttl():
    r = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_mem_cap=2, ddirect_mem_ttl_s=100)
    r._mem_served(1, 0, 1.0)
    r._mem_served(2, 0, 1.0)
    r._mem_get(1)  # touch: 2 is now least recently used
    r._mem_served(3, 0, 1.0)
    assert list(r.dd_mem) == [1, 3] and r.dd["mem_evicted"] == 1
    r.dd_mem[1][1] -= 101
    assert r._mem_get(1) is None and 1 not in r.dd_mem


def test_mem_off_when_ddirect_off():
    async def reqs(cl, be, router):
        await post(cl, FOLLOW)
        assert not router.dd_mem
    run_case(False, reqs, seed=False)


def test_mem_env_knobs(monkeypatch):
    monkeypatch.setenv("ROUTER_DDIRECT_MAX_AGE_S", "5")
    monkeypatch.setenv("ROUTER_DDIRECT_MEM_CAP", "9")
    monkeypatch.setenv("ROUTER_DDIRECT_MEM_TTL_S", "77")
    r = rd.Router(["p:1"], ["d:1"])
    assert (r.dd_max_age, r.dd_mem_cap, r.dd_mem_ttl) == (5, 9, 77)


# ---- strict affinity
def test_strict_affinity_ignores_imbalance():
    r = rd.Router(["p:1", "p:2"], ["d:1", "d:2"], max_imbalance=0, strict_affinity=False)
    h = 4  # 4 % 2 == 0
    r.inflight["prefill"] = [5, 0]
    r.inflight["decode"] = [5, 0]
    assert (r.pick("prefill", h), r.pick("decode", h)) == (1, 1)  # escape fires
    r.strict_affinity = True
    assert (r.pick("prefill", h), r.pick("decode", h)) == (0, 0)
    assert r.pick("prefill", 7) == 1 == r.pick("decode", 7)


def test_strict_affinity_env(monkeypatch):
    monkeypatch.delenv("ROUTER_STRICT_AFFINITY", raising=False)
    assert rd.Router(["p:1"], ["d:1"]).strict_affinity is False
    monkeypatch.setenv("ROUTER_STRICT_AFFINITY", "1")
    assert rd.Router(["p:1"], ["d:1"]).strict_affinity is True


# ---- first-event detection
@pytest.mark.parametrize("kind", ["sse429:4b", "sse429:nospace", "sse429:ping", "sse429:crlf", "sse503:4b"])
def test_sse_error_first_event_variants_fall_back(kind):
    _fallback_case(kind)


def test_sse_scan_unit():
    assert rd._sse_scan(b"data") == (False, None)
    assert rd._sse_scan(b": ping\n\n") == (False, None)
    assert rd._sse_scan(b": ping\r\n\r\ndata:{\"error\":{\"code\":429}}\r\n\r\n") == (True, 429)
    assert rd._sse_scan(b'data: {"x": 1}\n\n') == (True, None)
    assert rd._sse_scan(b'{"choices": []}') == (True, None)
    assert rd._sse_scan(b'data: {"error": {"code": 503}}', eof=True) == (True, 503)
    assert rd._sse_error_code(b'data:{"error":{"code":429}}\n\n') == 429


def test_first_event_stall_passes_through_not_502(monkeypatch):
    monkeypatch.setattr(rd, "FIRST_EVENT_S", 0.1)

    async def reqs(cl, be, router):
        r = await post(cl, FOLLOW)
        assert r.status_code == 200 and r.headers["x-pd-path"] == "direct"
        assert r.content == b'data: {"x": 1}\n\ndata: [DONE]\n\n'  # buffered bytes then the rest, exactly
        assert router.inflight["decode"] == [0]
    be, router, stats = run_case(True, reqs, d_reject="stall")
    assert [c[0] for c in be.calls] == ["D"]
    assert stats["direct_rejected_fallback"] == 0 and stats["direct_errors"] == 0 and router.counts["decode_fail"] == 0


def test_client_disconnect_during_first_event_read():
    class R:
        status = 200
        headers = {"content-type": "text/event-stream"}
        closed = False

        class content:
            @staticmethod
            async def readany():
                await asyncio.sleep(30)

        def close(self):
            R.closed = True

    class Req:
        async def is_disconnected(self):
            return True

    async def go():
        router = rd.Router(["p:1"], ["d:1"], ddirect=True)
        r = R()

        async def start(*a, **k):
            return r, b"da"
        router._decode_start = start
        try:
            out = await asyncio.wait_for(
                router._decode_leg(Req(), "/v1/chat/completions", {"stream": True}, {}, 0, {}, direct=True), 3)
        finally:
            await router.close()
        return router, out
    router, out = asyncio.run(go())
    assert out.status_code == 499 and R.closed
    assert router.inflight["decode"] == [0] and router.counts["client_gone_in_decode"] == 1
    assert router.counts["decode_fail"] == 0


# ---- Prometheus series on /metrics
SAMPLE_RE = re.compile(r"^([a-z_:]+)(\{[^}]*\})? (\S+)$")


def scrape(text):
    out = {}
    for line in text.splitlines():
        m = SAMPLE_RE.match(line)
        if m:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def run_with_metrics(ddirect, bodies, **kw):
    got = {}

    async def reqs(cl, be, router):
        got["headers"] = []
        for b in bodies:
            r = await post(cl, b)
            assert r.status_code == 200
            got["headers"].append(dict(r.headers))
        m = await cl.get("/metrics")
        assert m.status_code == 200
        got["text"] = m.text
    run_case(ddirect, reqs, **kw)
    return got["headers"], got["text"], scrape(got["text"])


def test_metrics_direct_and_pd_by_path():
    _, text, s = run_with_metrics(True, [FOLLOW, FIRST])
    assert s["pd_router_ddirect_eligible_total"] == 1 and s["pd_router_ddirect_attempts_total"] == 1
    assert s['pd_router_ddirect_ineligible_total{reason="first_turn"}'] == 1
    assert s['pd_router_decode_requests_total{path="direct"}'] == 1
    assert s['pd_router_decode_requests_total{path="pd"}'] == 1
    assert s['pd_router_decode_requests_total{path="pd-fb"}'] == 0
    for path, n in (("direct", 1), ("pd", 1), ("pd-fb", 0)):
        assert s[f'pd_router_first_byte_seconds_count{{path="{path}"}}'] == n
        assert s[f'pd_router_first_byte_seconds_bucket{{path="{path}",le="+Inf"}}'] == n
        buckets = [s[f'pd_router_first_byte_seconds_bucket{{path="{path}",le="{rd.fmt(b)}"}}'] for b in rd.FB_BUCKETS]
        assert buckets == sorted(buckets) and buckets[-1] <= n
    assert "# TYPE pd_router_first_byte_seconds histogram" in text
    assert s["vllm:num_requests_running"] == 2  # engine series are still summed across both ranks


def test_metrics_rejected_fallback():
    _, _, s = run_with_metrics(True, [FOLLOW], d_reject="http429")
    assert s["pd_router_ddirect_attempts_total"] == 1 and s["pd_router_ddirect_rejected_fallback_total"] == 1
    assert s['pd_router_decode_requests_total{path="direct"}'] == 1
    assert s['pd_router_decode_requests_total{path="pd-fb"}'] == 1
    assert s['pd_router_first_byte_seconds_count{path="direct"}'] == 0
    assert s['pd_router_first_byte_seconds_count{path="pd-fb"}'] == 1


def test_metrics_skip_reason():
    _, _, s = run_with_metrics(True, [FOLLOW], seed=False)
    assert s['pd_router_ddirect_skipped_total{reason="no_entry"}'] == 1
    assert s["pd_router_ddirect_attempts_total"] == 0
    assert s['pd_router_decode_requests_total{path="pd"}'] == 1


def test_metrics_off_no_header_and_zero_ddirect_series():
    headers, _, s = run_with_metrics(False, [FOLLOW])
    assert "x-pd-path" not in headers[0] and "x-pd-prefill-ms" in headers[0]
    assert s["pd_router_ddirect_attempts_total"] == 0 and s["pd_router_ddirect_eligible_total"] == 0
    assert s['pd_router_decode_requests_total{path="pd"}'] == 1
    assert s['pd_router_first_byte_seconds_count{path="pd"}'] == 1


def test_metrics_label_values_are_fixed():
    _, text, _ = run_with_metrics(True, [FOLLOW, FIRST])
    allowed = (set(rd.DD_SKIP_REASONS) | set(rd.DD_REASONS) | set(rd.PATHS) | {rd.fmt(b) for b in rd.FB_BUCKETS}
               | {"+Inf"} | set(rd.DD_OUTCOMES) | {"same", "changed"} | {rd.fmt(b) for b in rd.DD_BUCKETS}
               | {rd.fmt(b) for b in rd.DD_GAP_BUCKETS} | set(rd.REASONING_FIELDS)
               | set(rd.FT_RESULTS) | set(rd.OV_RESULTS) | set(rd.HIST_KINDS))
    own = [ln for ln in text.splitlines() if ln.startswith("pd_router_")]
    assert own
    for ln in own:
        for value in re.findall(r'="([^"]*)"', ln):
            assert value in allowed


# -- affinity key: the stable prefix of a conversation (messages before the first user message + first user message)
def _aff(msgs):
    return rd.Router.affinity({"messages": msgs})


def _conv(system):
    u1, a1, u2, a2, u3 = ({"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                          {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
                          {"role": "user", "content": "q3"})
    return system + [u1], system + [u1, a1, u2], system + [u1, a1, u2, a2, u3]


@pytest.mark.parametrize("system", [
    [], [{"role": "system", "content": "s"}],
    [{"role": "system", "content": "s1"}, {"role": "developer", "content": "s2"}, {"role": "system", "content": "s3"}],
    [{"role": "system", "content": [{"type": "text", "text": "s"}]}]])
def test_affinity_same_across_turns(system):
    t1, t2, t3 = _conv(system)
    assert _aff(t1) == _aff(t2) == _aff(t3)


def test_affinity_differs_by_first_user_or_system():
    a = _aff([{"role": "user", "content": "x"}])
    assert a != _aff([{"role": "user", "content": "y"}])
    s = {"role": "system", "content": "s"}
    assert _aff([s, {"role": "user", "content": "x"}]) != _aff([{"role": "system", "content": "t"}, {"role": "user", "content": "x"}])
    assert _aff([s, {"role": "user", "content": "x"}]) != a


def test_affinity_content_parts_and_tool_messages():
    u1 = {"role": "user", "content": [{"type": "text", "text": "q1"}, {"type": "image_url", "image_url": {"url": "u"}}]}
    turn1 = [u1]
    turn2 = [u1, {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c", "content": "r"},
             {"role": "user", "content": "q2"}]
    assert _aff(turn1) == _aff(turn2)
    # tool message before the first user message is part of the prefix
    assert _aff([{"role": "tool", "content": "r"}, u1]) != _aff(turn1)


def test_affinity_edge_cases():
    assert isinstance(_aff([]), int)
    assert _aff([]) == _aff([])
    # first message is the assistant: prefix runs through the first user message
    a = {"role": "assistant", "content": "hi"}
    u = {"role": "user", "content": "q"}
    assert _aff([a, u]) == _aff([a, u, {"role": "assistant", "content": "x"}, {"role": "user", "content": "q2"}])
    # no user message at all: first two messages (previous behaviour)
    assert _aff([a, {"role": "assistant", "content": "b"}, {"role": "assistant", "content": "c"}]) == \
        _aff([a, {"role": "assistant", "content": "b"}])
    # non-dict entries do not raise
    assert isinstance(_aff(["x", u]), int)


def test_affinity_prompt_unchanged():
    import hashlib
    key = json.dumps("hello", sort_keys=True)
    assert rd.Router.affinity({"prompt": "hello"}) == int(hashlib.sha1(key.encode()).hexdigest(), 16)


def test_memory_hit_on_turn_two_without_system_prompt():
    r = rd.Router(["p:1"], ["d:1"])
    t1, t2, _ = _conv([])
    h1 = r.affinity({"messages": t1})
    r._mem_served(h1, r.pick("decode", h1), 100.0)
    assert r._mem_skip_reason(r.affinity({"messages": t2})) is None


def test_affinity_system_user_matches_r2_key():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"}]
    r2_key = json.dumps(msgs[:2], sort_keys=True)
    import hashlib
    assert _aff(msgs) == int(hashlib.sha1(r2_key.encode()).hexdigest(), 16)
    assert _aff(msgs + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}]) == _aff(msgs)


def test_three_turns_without_system_prompt_go_direct_on_turn_two():
    t1 = [U(40)]
    t2 = t1 + [AS(), U(35)]
    t3 = t2 + [AS(), U(35)]

    async def reqs(cl, be, router):
        paths = []
        for msgs in (t1, t2, t3):
            r = await post(cl, {"model": "m", "stream": True, "messages": msgs})
            assert r.status_code == 200
            paths.append(r.headers["x-pd-path"])
        assert paths == ["pd", "direct", "direct"]
    be, router, stats = run_case(True, reqs, seed=False)
    assert _paths(be) == ["P", "D", "D", "D"]
    assert stats["sent_direct"] == 2 and stats["mem_size"] == 1


# -- cache_salt in the affinity key: callers that cannot share an engine cache never share a memory entry
def test_affinity_salted_same_across_turns_and_split_by_salt():
    t1, t2, t3 = _conv([{"role": "system", "content": "s"}])
    salted = [rd.Router.affinity({"messages": m, "cache_salt": "a"}) for m in (t1, t2, t3)]
    assert salted[0] == salted[1] == salted[2]
    assert rd.Router.affinity({"messages": t1, "cache_salt": "b"}) != salted[0]
    assert rd.Router.affinity({"messages": t1}) != salted[0]
    assert rd.Router.affinity({"prompt": "hello", "cache_salt": "a"}) != rd.Router.affinity({"prompt": "hello"})


def test_affinity_unsalted_key_is_r3_key():
    import hashlib
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"}]
    assert rd.Router.affinity({"messages": msgs}) == int(hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest(), 16)
    assert rd.Router.affinity({"messages": msgs, "cache_salt": None}) == rd.Router.affinity({"messages": msgs})


def test_affinity_salt_values_do_not_collide():
    m = [{"role": "user", "content": "q"}]
    keys = {rd.Router.affinity({"messages": m, "cache_salt": s}) for s in ("1", 1, "", ["1"], {"a": 1})}
    assert len(keys) == 5


def test_other_salt_does_not_reuse_direct_memory():
    t1 = [U(40)]
    t2 = t1 + [AS(), U(35)]

    async def reqs(cl, be, router):
        paths = []
        for salt, msgs in (("a", t1), ("a", t2), ("b", t2), ("b", t2)):
            r = await post(cl, {"model": "m", "stream": True, "messages": msgs, "cache_salt": salt})
            assert r.status_code == 200
            paths.append(r.headers["x-pd-path"])
        # salt b has no memory entry of its own: its first follow-up goes P/D, then it has one
        assert paths == ["pd", "direct", "pd", "direct"]
    be, router, stats = run_case(True, reqs, seed=False)
    assert stats["skip_reason"]["no_entry"] == 1 and stats["mem_size"] == 2


# -- direct-attempt context metrics (r5)
def test_ctx_hash_ignores_messages_and_tracks_tools():
    a = {"messages": [U(10)], "tools": [{"type": "function", "function": {"name": "f"}}]}
    b = dict(a, messages=[U(10), AS(), U(5)])
    c = dict(a, tools=[{"type": "function", "function": {"name": "g"}}])
    assert rd.ctx_hash(a) == rd.ctx_hash(b) != rd.ctx_hash(c)
    assert rd.ctx_hash({"messages": []}) == rd.ctx_hash({"messages": [U(3)], "tools": None})


def test_attempt_context_accepted_and_tools_changed():
    t1 = [U(40)]
    t2 = t1 + [AS(), U(35)]
    tools_a = [{"type": "function", "function": {"name": "a"}}]
    tools_b = [{"type": "function", "function": {"name": "b"}}]

    async def reqs(cl, be, router):
        for msgs, tools in ((t1, tools_a), (t2, tools_a), (t2 + [AS(), U(30)], tools_b)):
            r = await post(cl, {"model": "m", "stream": True, "messages": msgs, "tools": tools})
            assert r.status_code == 200
    be, router, stats = run_case(True, reqs, seed=False)
    ctx = router.dd_ctx["accepted"]
    # A request whose context fields changed is not sent direct, so it is skipped and never attempted.
    assert ctx["ctx_same"] == 1 and ctx["ctx_changed"] == 0
    assert stats["skip_reason"]["context_changed"] == 1
    assert sum(ctx["gap"]) == 1 and sum(ctx["new"]) == 1 and sum(ctx["reply"]) == 1
    text = router.own_metrics()
    assert 'pd_router_ddirect_attempt_context_total{outcome="accepted",context="same"} 1' in text
    assert 'pd_router_ddirect_attempt_gap_seconds_count{outcome="accepted"} 1' in text
    assert 'pd_router_ddirect_attempt_new_tokens_est_bucket{outcome="rejected",le="+Inf"} 0' in text


def test_attempt_context_rejected_outcome():
    async def reqs(cl, be, router):
        r = await post(cl, FOLLOW)
        assert r.status_code == 200
    be, router, stats = run_case(True, reqs, d_reject="http429")
    assert stats["direct_rejected_fallback"] == 1
    assert sum(router.dd_ctx["rejected"]["new"]) == 1 and sum(router.dd_ctx["accepted"]["new"]) == 0


# -- r6: error outcome and reasoning-field counter
def test_attempt_http400_is_error_not_accepted():
    async def reqs(cl, be, router):
        r = await post(cl, FOLLOW)
        assert r.status_code == 400
    be, router, stats = run_case(True, reqs, d_reject="http400")
    assert sum(router.dd_ctx["error"]["new"]) == 1
    assert sum(router.dd_ctx["accepted"]["new"]) == 0 and sum(router.dd_ctx["rejected"]["new"]) == 0
    st = router.dd_ctx["error"]
    assert st["ctx_same"] + st["ctx_changed"] == 1
    assert 'pd_router_ddirect_attempt_gap_seconds_count{outcome="error"} 1' in router.own_metrics()


def test_attempt_success_is_accepted():
    async def reqs(cl, be, router):
        r = await post(cl, FOLLOW)
        assert r.status_code == 200
    be, router, stats = run_case(True, reqs)
    assert sum(router.dd_ctx["accepted"]["new"]) == 1 and sum(router.dd_ctx["error"]["new"]) == 0


def test_reasoning_field_classification():
    A = lambda **kw: dict({"role": "assistant", "content": "x"}, **kw)
    u = {"role": "user", "content": "q"}
    assert rd.reasoning_field([u]) is None
    assert rd.reasoning_field([u, A(), u]) == "none"
    assert rd.reasoning_field([u, A(reasoning="r"), u]) == "reasoning"
    assert rd.reasoning_field([u, A(reasoning_content="r"), u]) == "reasoning_content"
    assert rd.reasoning_field([u, A(reasoning_details=[{"type": "reasoning.text"}]), u]) == "reasoning_details"
    assert rd.reasoning_field([u, A(content="<think>r</think>x"), u]) == "inline_think"
    assert rd.reasoning_field([u, A(reasoning="r", reasoning_content="r"), u]) == "reasoning"
    # only the last assistant message counts
    assert rd.reasoning_field([u, A(reasoning="r"), u, A(), u]) == "none"


def test_reasoning_field_counter_counts_every_chat_request_with_history():
    body = {"model": "m", "stream": True,
            "messages": [U(), dict(AS(), reasoning_content="r"), U(35)]}

    async def reqs(cl, be, router):
        for b in (body, FIRST):
            r = await post(cl, b)
            assert r.status_code == 200
    be, router, stats = run_case(False, reqs, seed=False)
    text = router.own_metrics()
    assert 'pd_router_assistant_reasoning_field_total{field="reasoning_content"} 1' in text
    assert sum(router.reasoning_field.values()) == 1
