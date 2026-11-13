"""CPU-only tests for preprocess_pool (no GPU, no model). Loads the module by path.

Usage: python3 test_preprocess_pool.py [path/to/preprocess_pool.py]
"""
import asyncio, importlib.util, os, signal, sys, time, types

os.environ["SGLANG_PREPROCESS_TEST_HOOK"] = "1"
os.environ.pop("SGLANG_PREPROCESS_WORKERS", None)  # the default must be "off"
os.environ["SGLANG_PREPROCESS_TIMEOUT_S"] = "3"
path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python/sglang/srt/managers/preprocess_pool.py"
spec = importlib.util.spec_from_file_location("preprocess_pool", path)
pp = importlib.util.module_from_spec(spec); sys.modules["preprocess_pool"] = pp; spec.loader.exec_module(pp)


class Req:
    def __init__(self, text, rid=None):
        self.messages = [types.SimpleNamespace(content=text)]
        self.rid = rid; self.tools = None


class Serving:
    def _convert_to_internal_request(self, request, raw):
        txt = request.messages[0].content
        if "__BAD__" in txt:
            raise ValueError("bad request body")
        return {"ids": list(range(len(txt))), "pid": os.getpid(), "hdr": raw.headers.get("x-test") if raw else None}, request


class Raw:
    class headers:
        @staticmethod
        def items():
            return [("X-Test", "v1")]


async def main():
    s = Serving()
    # Default (unset), 0, a negative count and garbage all leave the pool off: nothing is forked and
    # no attribute is set, so serving_base takes the unchanged thread path.
    os.environ["SGLANG_PREPROCESS_WORKERS"] = "3"
    os.environ["SGLANG_PREPROCESS_TIMEOUT_S"] = "nan"
    probe = await pp.install(Serving()); assert probe is not None and probe.timeout_s == 60.0, probe
    probe.shutdown(); os.environ["SGLANG_PREPROCESS_TIMEOUT_S"] = "3"
    for value in (None, "0", "-2", "", "four"):
        if value is None:
            os.environ.pop("SGLANG_PREPROCESS_WORKERS", None)
        else:
            os.environ["SGLANG_PREPROCESS_WORKERS"] = value
        assert await pp.install(s) is None, value
        assert not hasattr(s, "_preprocess_pool"), value
    # With the pool off, maybe_hooked() is only a wrapper while the test hook is on.
    os.environ["SGLANG_PREPROCESS_WORKERS"] = "3"
    # A forked child must not signal the parent through the inherited asyncio wakeup socket.
    loop = asyncio.get_running_loop()
    got = []
    loop.add_signal_handler(signal.SIGTERM, lambda: got.append("TERM"))
    pool = await pp.install(s)
    assert pool is not None and s._preprocess_pool is pool
    pids = set()
    t0 = time.time(); r, _ = await pool.run(Req("hello"), Raw()); assert r["hdr"] == "v1", r; pids.add(r["pid"])
    assert time.time() - t0 < 1.0, "fast request too slow"
    try:
        await pool.run(Req("__BAD__"), Raw()); raise SystemExit("no error")
    except ValueError as e:
        assert "bad request" in str(e)
    slow = asyncio.ensure_future(pool.run(Req("x __PP_SLOW_30__"), Raw(), label="slow"))
    await asyncio.sleep(0.2)
    t0 = time.time()
    fast = await asyncio.gather(*[pool.run(Req(f"fast{i}"), Raw()) for i in range(20)])
    fast_s = time.time() - t0
    assert fast_s < 2.0, f"fast requests blocked behind slow one: {fast_s:.2f}s"
    print(f"20 fast requests finished in {fast_s:.2f}s while one slow request spun")
    try:
        await slow; raise SystemExit("slow did not time out")
    except pp.PreprocessTimeout as e:
        assert 2.5 < e.elapsed_s < 6, e.elapsed_s
        print(f"slow request failed alone after {e.elapsed_s:.1f}s")
    assert pool.stats["timeouts"] == 1 and pool.stats["replaced"] == 1, pool.stats
    res = await asyncio.gather(*[pool.run(Req(f"__PP_SLOW_30__ {i}"), Raw()) for i in range(3)], return_exceptions=True)
    assert all(isinstance(x, pp.PreprocessTimeout) for x in res), res
    r, _ = await pool.run(Req("after"), Raw()); pids.add(r["pid"])
    try:
        await pool.run(Req("__PP_CRASH__"), Raw()); raise SystemExit("crash not reported")
    except pp.PreprocessWorkerDied:
        pass  # per-request error: a request that kills its worker is not retried on the thread path
    r, _ = await pool.run(Req("after crash"), Raw())
    t = asyncio.ensure_future(pool.run(Req("__PP_SLOW_1__"), Raw())); await asyncio.sleep(0.2); t.cancel()
    await asyncio.sleep(1.5)
    assert pool._idle.qsize() == 3, pool._idle.qsize()
    # SIGTERM to an idle worker kills that worker only; the parent's signal handler must not fire.
    victim = pool._idle._queue[0]
    os.kill(victim.pid, signal.SIGTERM)
    await asyncio.sleep(0.3)
    assert got == [], f"parent saw a SIGTERM delivered to a worker: {got}"
    for _ in range(4):  # whichever request lands on the dead worker is reported, then it is replaced
        try:
            await pool.run(Req("after sigterm"), Raw())
        except pp.PreprocessWorkerDied:
            pass
    r, _ = await pool.run(Req("recovered"), Raw())
    assert pool._idle.qsize() == 3, pool._idle.qsize()
    assert got == [], got
    big = "y" * 5_000_000
    t0 = time.time(); r, _ = await pool.run(Req(big), Raw()); assert len(r["ids"]) == len(big)
    print(f"big payload (5M ids) round trip {time.time()-t0:.2f}s")
    # A broken pool wakes requests queued for a worker instead of hanging them.
    while not pool._idle.empty():
        pool._idle.get_nowait()
    waiter = asyncio.ensure_future(pool.run(Req("queued"), Raw()))
    await asyncio.sleep(0.1)
    assert not waiter.done()
    pool.broken = True
    for _ in range(pool.n + 1):
        pool._idle.put_nowait(None)
    try:
        await asyncio.wait_for(waiter, 2)
        raise SystemExit("queued request should fail over")
    except pp.PreprocessPoolError:
        pass
    hooked = pp.maybe_hooked(s._convert_to_internal_request)
    t0 = time.time(); hooked(Req("__PP_SLOW_1__"), None); assert time.time() - t0 >= 0.9
    print("stats", pool.stats)
    pool.shutdown()
    # The zygote exits when its control socket closes (it is our child, so reap it to see that).
    exited = False
    for _ in range(40):
        done, _status = os.waitpid(pool._zygote_pid, os.WNOHANG)
        if done:
            exited = True
            break
        await asyncio.sleep(0.1)
    print("zygote exited after shutdown:", exited)
    assert exited, "zygote must exit when the control socket closes"

asyncio.run(main())
print("ALL PASS")
