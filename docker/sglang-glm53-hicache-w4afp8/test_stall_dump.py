#!/usr/bin/env python3
"""Functional test of event_loop_stall_dump (CPU only; stdlib plus the engine image's uvloop,
uvicorn and fastapi). Every scenario runs in its own child process, because faulthandler's
watchdog and the detector are both per process; the parent captures the child's stderr and
asserts on it.

  uvloop-sleep   uvloop (the production loop); a coroutine blocks the loop with time.sleep(7)
  asyncio-sleep  the same on the stdlib asyncio loop
  uvloop-spin    a coroutine spins in pure Python for 7 s (CPU-bound; the GIL still switches)
  uvloop-gil     a coroutine sits 7 s in a C call that keeps the GIL (libc sleep through
                 ctypes.PyDLL), which starves every Python thread: only faulthandler's C
                 watchdog can dump, and the Python dump comes late
  uvicorn        uvicorn + FastAPI on uvloop like `sglang serve`; install() is called with no
                 arguments from the FastAPI lifespan (as the http_server.py hook does) and reads
                 the env vars; an async POST /v1/chat/completions blocks the loop with
                 time.sleep(7) while the parent polls GET /health every 200 ms
  asyncio-fork   fork() + a child exiting through interpreter shutdown, a fork-context
                 ProcessPoolExecutor (the multimodal CPU pool pattern), then a GIL-held block
                 with no await in between: the child must not hang on the inherited
                 faulthandler timer, and the at-fork re-arm must still produce the C dump
  disabled       SGLANG_EVENT_LOOP_STALL_DUMP_SECS=0: no watcher thread, no dump
  bad-env        SGLANG_EVENT_LOOP_STALL_DUMP_SECS=abc: one warning, nothing raised, no dump

Usage: python3 test_stall_dump.py --module PATH/event_loop_stall_dump.py [--out FILE]
"""

import argparse
import asyncio
import importlib.util
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

BLOCK_S, AFTER_S = 7, 3
PREFIX = "[event-loop-stall]"
T0 = time.monotonic()


def mark(msg):
    os.write(2, f"[test +{time.monotonic() - T0:6.2f}s] {msg}\n".encode())


def load(path):
    spec = importlib.util.spec_from_file_location("event_loop_stall_dump", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- children
async def block_releasing_gil():
    time.sleep(BLOCK_S)  # the loop thread blocks, the GIL is released


async def block_spinning():
    end = time.monotonic() + BLOCK_S
    while time.monotonic() < end:  # pure Python, CPU-bound
        pass


async def block_holding_gil():
    import ctypes

    ctypes.PyDLL(None).sleep(BLOCK_S)  # libc sleep() without releasing the GIL


async def fork_then_hold_gil():
    """fork() with a child that exits through full interpreter shutdown, then the multimodal CPU
    pool pattern (fork-context ProcessPoolExecutor), then a GIL-held block. There is no await in
    between, so only the at-fork re-arm (not the heartbeat) can have armed the C watchdog."""
    import concurrent.futures
    import multiprocessing as mp

    sys.stdout.flush()
    t = time.monotonic()
    pid = os.fork()
    if pid == 0:
        sys.exit(0)  # unwinds asyncio.run in the child, then Py_Finalize
    while time.monotonic() - t < 5 and not os.waitpid(pid, os.WNOHANG)[0]:
        time.sleep(0.01)
    else:
        if time.monotonic() - t >= 5:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    mark(f"FORK child {'HUNG, killed' if time.monotonic() - t >= 5 else 'exited'} after "
         f"{time.monotonic() - t:.2f}s")
    with concurrent.futures.ProcessPoolExecutor(2, mp_context=mp.get_context("fork")) as ex:
        mark(f"POOL ok {list(ex.map(abs, [-1, -2, -3]))}")
    await block_holding_gil()


def child_block(mod, loop_kind, blocker, env_install=False):
    async def main():
        loop = asyncio.get_running_loop()
        mod.install(loop) if env_install else mod.install(loop, 2, 2, PREFIX)
        mark(f"loop={type(loop).__module__}.{type(loop).__name__} threads="
             f"{sorted(t.name for t in threading.enumerate())}")
        await asyncio.sleep(1.5)
        mark("BLOCK START")
        await blocker()
        mark("BLOCK END")
        await asyncio.sleep(AFTER_S)
        mark("CHILD EXIT")

    if loop_kind == "uvloop":
        import uvloop

        uvloop.run(main())
    else:
        asyncio.run(main())


def child_uvicorn(mod, port):
    from contextlib import asynccontextmanager

    import uvicorn
    from fastapi import FastAPI

    @asynccontextmanager
    async def lifespan(app):
        mod.install(asyncio.get_running_loop())  # same call as the http_server.py hook
        yield
        mark("LIFESPAN SHUTDOWN")

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_completions():
        mark("BLOCK START (request handler)")
        time.sleep(BLOCK_S)  # synchronous work on the loop, like the chat preprocessing
        mark("BLOCK END")
        return {"ok": True}

    uvicorn.run(app, host="127.0.0.1", port=port, loop="uvloop", log_level="warning")


# ----------------------------------------------------------------------------- parsing
def parse(err):
    """Return (python dumps, faulthandler blocks, loop ident) from a child's stderr."""
    lines = err.splitlines()
    m = re.search(r"loop thread (0x[0-9a-f]{16})", err)
    ident = m.group(1) if m else None
    dumps, cur = [], None
    for i, line in enumerate(lines):
        h = re.match(rf"{re.escape(PREFIX)} pid=\d+ event loop stalled for ([\d.]+)s .*dump (\d+), "
                     r"(\d+) threads, loop thread used ([-\d.na]+)s CPU", line)
        if h:
            cur = {"age": float(h.group(1)), "n": int(h.group(2)), "threads": int(h.group(3)),
                   "cpu": float(h.group(4)), "line": i, "sections": {}, "header": line}
            sec = None
        elif cur is not None and line.startswith(f"{PREFIX} --- thread"):
            sec = line
            cur["sections"][sec] = []
        elif cur is not None and re.match(rf"{re.escape(PREFIX)} pid=\d+ end of dump", line):
            dumps.append(cur)
            cur = None
        elif cur is not None and sec is not None:
            cur["sections"][sec].append(line[len(PREFIX) + 1:])
    blocks = []
    for i, line in enumerate(lines):
        if line.startswith("Timeout ("):
            secs, sec = {}, None
            for nxt in lines[i + 1:]:
                if nxt.startswith(("Thread 0x", "Current thread 0x")):
                    sec = nxt
                    secs[sec] = []
                elif nxt.startswith("  File ") and sec is not None:
                    secs[sec].append(nxt)
                elif nxt.strip() == "":
                    continue
                else:
                    break
            blocks.append({"header": line, "line": i, "sections": secs})
    return dumps, blocks, ident, lines


def loop_section(dump):
    for head, body in dump["sections"].items():
        if "(event loop)" in head:
            return head, "\n".join(body)
    return None, ""


def line_of(lines, pattern):
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            return i
    return None


# ----------------------------------------------------------------------------- checks
def check_block(name, err, blocker, needle=None, expect_cpu_bound=False, gil_held=False):
    dumps, blocks, ident, lines = parse(err)
    res = []
    ok = lambda cond, what: res.append((bool(cond), what))
    ok(re.search(r"event-loop stall dump armed: stall=2s repeat=2s", err), "armed INFO line logged")
    hit = [d for d in dumps if blocker in loop_section(d)[1]]
    if not gil_held:
        with_src = [d for d in hit if needle in loop_section(d)[1]]
        ok(len(with_src) >= 2, f">=2 dumps whose event-loop thread stack shows {blocker} at "
                               f"'{needle}' (got {len(with_src)} of {len(dumps)} dumps)")
        ok(dumps and 2.0 <= dumps[0]["age"] < 4.5, f"first dump after 2-4.5 s of stall (got "
                                                  f"{dumps[0]['age'] if dumps else None})")
        ok(not blocks, f"no faulthandler C-watchdog block (the Python watcher kept up; got {len(blocks)})")
        if expect_cpu_bound:
            ok(dumps and all(d["cpu"] >= 0.5 * d["age"] for d in dumps),
               "loop-thread CPU >= 50% of stall age in every dump (CPU-bound): "
               + ", ".join(f"{d['cpu']:.1f}/{d['age']:.1f}s" for d in dumps))
        else:
            ok(dumps and all(d["cpu"] < 0.5 for d in dumps),
               "loop-thread CPU < 0.5 s in every dump (blocked, not spinning): "
               + ", ".join(f"{d['cpu']:.1f}/{d['age']:.1f}s" for d in dumps))
    else:
        loop_secs = [body for b in blocks for head, body in b["sections"].items()
                     if ident and ident in head]
        ok(any(f" in {blocker}" in "\n".join(s) for s in loop_secs),
           f"faulthandler C-watchdog block shows the loop thread {ident} in {blocker} "
           f"({len(blocks)} block(s): {[b['header'] for b in blocks]})")
        ok(dumps and dumps[0]["age"] >= BLOCK_S - 1.5,
           f"Python watcher was starved while the GIL was held: its first dump came at "
           f"{dumps[0]['age'] if dumps else None}s of stall, not ~2-3 s")
    rec = line_of(lines, rf"{re.escape(PREFIX)} pid=\d+ event loop recovered after ([\d.]+)s")
    ok(rec is not None, "'event loop recovered after N s' line present"
       + (f": {lines[rec].split('] ', 1)[1]}" if rec is not None else ""))
    if rec is not None:
        after = [d for d in dumps if d["line"] > rec] + [b for b in blocks if b["line"] > rec]
        ok(not after, "no dump after the recovery line")
    return res, dumps, blocks


def check_uvicorn(err, health):
    res, dumps, blocks = check_block("uvicorn", err, "chat_completions", needle="time.sleep(")
    res = [r for r in res if "armed INFO" not in r[1]]
    ok = lambda cond, what: res.append((bool(cond), what))
    ok(re.search(r"event-loop stall dump armed: stall=2s repeat=2s", err),
       "armed INFO line logged from the FastAPI lifespan, values read from the env vars")
    frames = "\n".join(loop_section(d)[1] for d in dumps)
    ok(re.search(r"(starlette|fastapi)/", frames) and "uvicorn/" in frames,
       "event-loop stack runs through uvicorn and starlette/fastapi frames")
    during = [lat for lat in health if lat is not None]
    worst = max(during) if during else None
    ok(worst is not None and worst >= BLOCK_S - 2.5,
       f"GET /health stalled while the loop was blocked (max latency {worst:.2f}s)" if worst
       else "GET /health latencies recorded")
    tail = health[-5:]
    ok(tail and all(lat is not None and lat < 0.5 for lat in tail),
       f"GET /health answers again after the stall (last 5: {[round(x, 3) for x in tail if x]})")
    return res, dumps, blocks


# ----------------------------------------------------------------------------- parent
def run_child(args, extra_env=None):
    """Start a child with stderr in a temp file (no pipe back-pressure on the timing)."""
    env = dict(os.environ, **(extra_env or {}))
    errf = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), *args], env=env,
                         stdout=subprocess.DEVNULL, stderr=errf, text=True)
    return p, errf


def finish(p, errf, timeout=120):
    p.wait(timeout=timeout)
    errf.seek(0)
    return errf.read()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method, url, timeout):
    t = time.monotonic()
    try:
        req = urllib.request.Request(url, method=method, data=b"{}" if method == "POST" else None)
        urllib.request.urlopen(req, timeout=timeout).read()
        return time.monotonic() - t
    except Exception:
        return None


def drive_uvicorn(module):
    port = free_port()
    env = {"SGLANG_EVENT_LOOP_STALL_DUMP_SECS": "2", "SGLANG_EVENT_LOOP_STALL_DUMP_REPEAT_SECS": "2"}
    p, errf = run_child(["child-uvicorn", module, str(port)], env)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while http("GET", base + "/health", 1) is None and time.monotonic() < deadline:
        if p.poll() is not None:
            break
        time.sleep(0.2)
    time.sleep(1.5)
    health = []
    blocker = threading.Thread(target=http, args=("POST", base + "/v1/chat/completions", 30))
    blocker.start()
    t_end = time.monotonic() + BLOCK_S + AFTER_S + 1
    while time.monotonic() < t_end:
        t = time.monotonic()
        health.append(http("GET", base + "/health", 15))
        time.sleep(max(0.0, 0.2 - (time.monotonic() - t)))
    blocker.join()
    p.send_signal(signal.SIGTERM)
    err = finish(p, errf, timeout=60)
    summary = (f"[parent] GET /health polled every 200 ms, {len(health)} probes; latencies (s): "
               + " ".join(f"{lat:.2f}" if lat is not None else "timeout" for lat in health)
               + f"\n[parent] child exit code after SIGTERM: {p.returncode}\n")
    return err, health, summary, p.returncode


def check_fork(err):
    res = check_block("", err, "block_holding_gil", gil_held=True)[0]
    m = re.search(r"FORK child (exited|HUNG, killed) after ([\d.]+)s", err)
    res.append((m is not None and m.group(1) == "exited" and float(m.group(2)) < 5,
                "fork()-ed child exiting via full interpreter shutdown did not hang"
                + (f" ({m.group(0)})" if m else "")))
    res.append(("POOL ok [1, 2, 3]" in err, "fork-context ProcessPoolExecutor ran and shut down"))
    return res


def check_quiet(err, rc, bad_env):
    dumps, blocks, _, _ = parse(err)
    res = [(rc == 0, f"child exited 0 (got {rc})"),
           (not dumps and not blocks, "no dump at all although the loop was blocked 7 s"),
           ("event-loop-stall-watch" not in err, "no watcher thread was started"),
           ("stall dump armed:" not in err, "no 'armed' line")]
    if bad_env:
        res.append(("event-loop stall dump not armed" in err and "ValueError" in err,
                    "a single 'not armed' warning (ValueError), nothing raised into the caller"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    m = a.module
    children = {  # name: (child args, extra env, check)
        "uvloop-sleep": (["child", m, "uvloop", "block_releasing_gil"], None,
                         lambda e: check_block("", e, "block_releasing_gil", needle="time.sleep(")[0]),
        "asyncio-sleep": (["child", m, "asyncio", "block_releasing_gil"], None,
                          lambda e: check_block("", e, "block_releasing_gil", needle="time.sleep(")[0]),
        "uvloop-spin": (["child", m, "uvloop", "block_spinning"], None,
                        lambda e: check_block("", e, "block_spinning", needle="in block_spinning",
                                              expect_cpu_bound=True)[0]),
        "uvloop-gil": (["child", m, "uvloop", "block_holding_gil"], None,
                       lambda e: check_block("", e, "block_holding_gil", gil_held=True)[0]),
        "asyncio-fork": (["child", m, "asyncio", "fork_then_hold_gil"], None, check_fork),
        "disabled": (["child-env", m, "uvloop", "block_releasing_gil"],
                     {"SGLANG_EVENT_LOOP_STALL_DUMP_SECS": "0"}, None),
        "bad-env": (["child-env", m, "uvloop", "block_releasing_gil"],
                    {"SGLANG_EVENT_LOOP_STALL_DUMP_SECS": "abc"}, None),
    }
    started = {name: run_child(args, env) for name, (args, env, _) in children.items()}
    uv_err, health, uv_summary, uv_rc = drive_uvicorn(m)
    report = []
    for name, (p, errf) in started.items():
        err = finish(p, errf)
        check = children[name][2]
        if check is None:
            res = check_quiet(err, p.returncode, bad_env=name == "bad-env")
        else:
            res = [(p.returncode == 0, f"child exited 0 (got {p.returncode})")] + check(err)
        report.append((name, err, res, ""))
    # uvicorn re-raises a captured SIGTERM after its graceful shutdown, so -15 is the normal code
    res = [(uv_rc in (0, -signal.SIGTERM) and "LIFESPAN SHUTDOWN" in uv_err,
            f"uvicorn child shut down gracefully on SIGTERM (lifespan exit ran, exit code {uv_rc})")]
    report.append(("uvicorn", uv_err, res + check_uvicorn(uv_err, health)[0], uv_summary))

    failures = 0
    text = [f"python {sys.version.split()[0]} | {sys.executable} | module {m}\n"]
    for name, err, res, extra in report:
        text.append(f"\n{'=' * 30} scenario {name} {'=' * 30}\n--- child stderr ---\n{err}")
        text.append(extra)
        text.append("--- checks ---\n")
        for good, what in res:
            failures += not good
            text.append(f"{'PASS' if good else 'FAIL'}  {what}\n")
    total = sum(len(r) for _, _, r, _ in report)
    text.append(f"\n{'=' * 70}\nSUMMARY: {total - failures} passed, {failures} failed\n")
    for name, _, res, _ in report:
        text.append(f"  {name:14s} {'ok' if all(g for g, _ in res) else 'FAILED'}\n")
    out = "".join(text)
    sys.stdout.write(out)
    if a.out:
        with open(a.out, "w") as f:
            f.write(out)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("child", "child-env"):
        logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                            format="[child %(asctime)s %(levelname)s] %(message)s")
        mod = load(sys.argv[2])
        child_block(mod, sys.argv[3], globals()[sys.argv[4]], env_install=sys.argv[1] == "child-env")
    elif len(sys.argv) > 1 and sys.argv[1] == "child-uvicorn":
        logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                            format="[child %(asctime)s %(levelname)s] %(message)s")
        child_uvicorn(load(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
