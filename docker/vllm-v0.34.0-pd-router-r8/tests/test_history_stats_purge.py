"""History hash flag and expiry purge tests for router.py: synthetic strings only, no network."""
import asyncio
import importlib.util
import os
import sys

import httpx

_SPEC = importlib.util.spec_from_file_location(
    "pd_router_history", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "router.py"))
rd = importlib.util.module_from_spec(_SPEC)
sys.modules["pd_router_history"] = rd
_SPEC.loader.exec_module(rd)


def U(n=35):
    return {"role": "user", "content": "a" * n}


def AS(n=35):
    return {"role": "assistant", "content": "b" * n}


BODY = {"model": "m", "stream": False, "messages": [U(), AS(), U()]}


def observe(router, body=BODY):
    """Run the part of request handling that touches the history state, without engines."""
    h = router.affinity(body)
    if router.ddirect and router.hist_stats and isinstance(body.get("messages"), list):
        router._hist_observe(h, body["messages"])
        hist = rd.history_pack(body["messages"])
    else:
        hist = None
    router._mem_served(h, 0, 10.0, 0, hist)
    return h


def test_history_stats_default_off(monkeypatch):
    monkeypatch.delenv("ROUTER_DDIRECT_HISTORY_STATS", raising=False)
    r = rd.Router(["p:1"], ["d:1"], ddirect=True)
    assert r.hist_stats is False


def test_history_stats_env(monkeypatch):
    monkeypatch.setenv("ROUTER_DDIRECT_HISTORY_STATS", "1")
    assert rd.Router(["p:1"], ["d:1"], ddirect=True).hist_stats is True


def test_flag_off_stores_nothing_and_counters_stay_zero():
    r = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_history_stats=False)
    h = observe(r)
    observe(r)
    assert r.dd_mem[h][5] is None
    assert all(v == 0 for v in r.hist_kind.values())
    assert all(sum(st["buckets"]) == 0 for st in r.hist_diverged.values())


def test_flag_on_stores_hashes_and_counts():
    r = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_history_stats=True)
    h = observe(r)
    assert isinstance(r.dd_mem[h][5], bytes) and len(r.dd_mem[h][5]) == 8 * len(BODY["messages"])
    observe(r)
    assert r.hist_kind["append_only"] == 1 and r.hist_kind["unknown"] == 1


def test_routing_unchanged_by_flag():
    """Same requests, same engines' view: the direct/pd decisions do not depend on the flag."""
    async def run(flag):
        import aiohttp.web as web
        calls = []

        async def chat(req):
            body = await req.json()
            calls.append("D" if "kv_transfer_params" not in body or not body["kv_transfer_params"] else "x")
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                      "kv_transfer_params": {"x": 1}})
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        addr = "127.0.0.1:%d" % site._server.sockets[0].getsockname()[1]
        rapp = rd.create_app([addr], [addr], stats_log_s=0, ddirect=True, ddirect_history_stats=flag)
        router = rapp.state.router
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rapp), base_url="http://t") as cl:
                paths = []
                for msgs in ([U(), U()], [U(), AS(), U()], [U(), AS(), U(), AS(), U()]):
                    r = await cl.post("/v1/chat/completions", json={"model": "m", "stream": False, "messages": msgs})
                    paths.append(r.headers.get("x-pd-path"))
                return paths, dict(router.dd["skip_reason"]), router.dd["sent_direct"]
        finally:
            await router.close()
            await runner.cleanup()
    assert asyncio.run(run(False)) == asyncio.run(run(True))


def test_expired_entries_purged_on_insert_without_lookups():
    r = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_mem_ttl_s=100)
    now = [1000.0]
    r._now = lambda: now[0]
    for k in range(5):
        r._mem_served(k, 0, 1.0)
    now[0] += 101
    r._mem_served(99, 0, 1.0)
    assert list(r.dd_mem) == [99]
    assert r.dd["mem_evicted"] == 5


def test_live_entries_survive_purge_and_cap_unchanged():
    r = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_mem_ttl_s=100, ddirect_mem_cap=3)
    now = [0.0]
    r._now = lambda: now[0]
    for k in range(5):
        r._mem_served(k, 0, 1.0)
        now[0] += 10
    assert list(r.dd_mem) == [2, 3, 4]


def test_purge_is_bounded_per_insert():
    r = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_mem_ttl_s=100)
    now = [0.0]
    r._now = lambda: now[0]
    n = rd.PURGE_PER_INSERT * 3
    for k in range(n):
        r._mem_served(k, 0, 1.0)
    now[0] += 101
    r._mem_served(1000, 0, 1.0)
    # at most PURGE_PER_INSERT expired entries go per insert; the rest wait for later inserts
    assert len(r.dd_mem) == n + 1 - rd.PURGE_PER_INSERT
    r._mem_served(1001, 0, 1.0)
    assert len(r.dd_mem) == n + 2 - 2 * rd.PURGE_PER_INSERT


def test_prefix_memory_expired_entries_purged_on_insert():
    r = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_prefix_ttl_s=120, ddirect_prefix_mem_cap=50)
    now = [0.0]
    r._now = lambda: now[0]
    for k in range(4):
        r._prefix_served(k, 0, 5.0)
    now[0] += 121
    r._prefix_served(77, 0, 5.0)
    assert list(r.dd_pfx) == [77]


def test_prefix_memory_cap_unchanged():
    r = rd.Router(["p:1"], ["d:1"], ddirect=True, ddirect_prefix_ttl_s=120, ddirect_prefix_mem_cap=2)
    for k in range(4):
        r._prefix_served(k, 0, 5.0)
    assert list(r.dd_pfx) == [2, 3]
