#!/usr/bin/env python3
"""One-shot in-CVM kernel profile of a DS4.1 DEP8 replica; prints only an aggregate table to stdout.

Embedded by scripts/prepare_dsv41_lora_dep8.py into the DEP8 Profile pack (service dsv41-profiler) and run
there with the engine image's python3, on the host network. Waits for all eight ranks, starts and stops the
torch profiler on each, summarizes the per-rank traces (decode-only GPU time by category), prints one JSON
line per rank plus a final done line. The traces are deleted on success and on failure (a failure also tries
/stop_profile on every rank that was sent /start_profile and not yet stopped). Stdlib only. Trace content never leaves
/traces; stdout (Loki) gets the aggregates. On any error: one {"profile_error": ...} line, exit 1, no retry.
Env: PROFILE_DELAY_S (900), PROFILE_DUR_S (3), PROFILE_WAIT_S (7200, max wait for rank health), PROFILE_SETTLE_S (120, wait for stable trace files),
PROFILE_TRACE_DIR (/traces).
"""

import collections
import concurrent.futures
import glob
import gzip
import json
import os
import re
import sys
import time
import urllib.request

RANKS = 8
RANK_PORT0 = 9100
# Category regexes and the +-60 ms prefill margin come from the gpu31 lab's ktrace2.py.
CATS = [
    ("comm (nccl)", r"nccl"),
    ("moe", r"humming|moe|expert|marlin|permute|topk_ids"),
    ("attention+indexer", r"flash|mla|attn|attention|indexer|mqa_logits|topk|compress|kv_cache|rope"),
    ("gemm-dense", r"gemm|cutlass|cublas|sm90_fp8|wgmma|matmul|mxfp8"),
    ("mhc/norm", r"mhc|sinkhorn|rms|norm"),
    ("sampling/spec", r"sampl|reject|softmax|argmax|gumbel|logit"),
    ("engram", r"engram"),
    ("copy/memset", r"memcpy|memset|copy"),
]
MARGIN_US = 60_000
KERNEL_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def load_rank_events(trace_dir, ranks):
    """{rank: [events with absolute microsecond start in e['abs']]} from the flat dp<N>_*.pt.trace.json[.gz] files."""
    files = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "dp*.pt.trace.json*"))):
        match = re.match(r"dp(\d+)", os.path.basename(path))
        if not match:
            continue
        rank = int(match.group(1))
        if rank in files:
            raise RuntimeError(f"more than one trace file for rank {rank}")
        files[rank] = path
    if not files:
        raise RuntimeError(f"no dp*.pt.trace.json* files in {trace_dir}")
    missing = [r for r in range(ranks) if r not in files]
    if missing:
        raise RuntimeError(f"missing trace files for ranks {missing}")
    out = {}
    for rank, path in sorted(files.items()):
        with (gzip.open if path.endswith(".gz") else open)(path, "rt") as handle:
            data = json.load(handle)
        if "baseTimeNanoseconds" not in data:
            raise RuntimeError(f"rank {rank}: trace has no baseTimeNanoseconds")
        base = int(data["baseTimeNanoseconds"]) // 1000
        events = [e for e in data.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in KERNEL_CATS]
        for e in events:
            e["abs"] = base + e["ts"]
        out[rank] = events
    return out


def prefill_intervals(events_by_rank):
    starts = sorted(e["abs"] for ev in events_by_rank.values() for e in ev if "sm90::prefill" in e["name"])
    intervals = []
    for t in starts:
        if intervals and t - MARGIN_US <= intervals[-1][1]:
            intervals[-1][1] = t + MARGIN_US
        else:
            intervals.append([t - MARGIN_US, t + MARGIN_US])
    return intervals


def in_intervals(intervals, t):
    lo, hi = 0, len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid][1] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(intervals) and intervals[lo][0] <= t <= intervals[lo][1]


def summarize(trace_dir, ranks=RANKS):
    events_by_rank = load_rank_events(trace_dir, ranks)
    intervals = prefill_intervals(events_by_rank)
    rows = []
    for rank, events in sorted(events_by_rank.items()):
        if not events:
            raise RuntimeError(f"rank {rank}: no GPU events in trace")
        t0 = min(e["abs"] for e in events)
        t1 = max(e["abs"] + e["dur"] for e in events)
        excluded = sum(min(b, t1) - max(a, t0) for a, b in intervals if b > t0 and a < t1)
        decode_window = (t1 - t0) - max(excluded, 0)
        by_cat = collections.Counter()
        by_name = collections.Counter()
        copies = {k: [0, 0.0] for k in ("h2d", "d2h", "d2d")}
        allgatherv = []
        for e in events:
            if in_intervals(intervals, e["abs"]):
                continue
            name = e["name"].lower()
            category = "other"
            for label, rx in CATS:
                if re.search(rx, name):
                    category = label
                    break
            by_cat[category] += e["dur"]
            by_name[e["name"][:64]] += e["dur"]
            if e.get("cat") == "gpu_memcpy":
                for key, tag in (("h2d", "htod"), ("d2h", "dtoh"), ("d2d", "dtod")):
                    if tag in name:
                        copies[key][0] += 1
                        copies[key][1] += e["dur"]
            if "allgatherv" in name:
                allgatherv.append(e["dur"])
        allgatherv.sort()
        n = len(allgatherv)
        ag = {"p10": None, "p50": None, "p90": None, "n": n}
        if n:
            ag.update(p10=allgatherv[n // 10], p50=allgatherv[n // 2], p90=allgatherv[9 * n // 10])
        memcpy = {}
        for key, (count, dur) in copies.items():
            memcpy[f"{key}_n"] = count
            memcpy[f"{key}_ms"] = round(dur / 1e3, 3)
        rows.append({
            "rank": rank,
            "window_ms": round((t1 - t0) / 1e3, 1),
            "decode_window_ms": round(decode_window / 1e3, 1),
            "busy_ms": round(sum(by_cat.values()) / 1e3, 1),
            "categories": {label: round(by_cat.get(label, 0) / 1e3, 1) for label, _ in CATS + [("other", "")]},
            "memcpy": memcpy,
            "top_kernels": [[name, round(dur / 1e3, 1)] for name, dur in by_name.most_common(8)],
            "allgatherv_us": ag,
            "prefill_intervals": len(intervals),
        })
    return rows


def remove_traces(trace_dir):
    for path in glob.glob(os.path.join(trace_dir, "*")):
        if os.path.isfile(path):
            os.remove(path)


def rank_url(rank, path):
    return f"http://127.0.0.1:{RANK_PORT0 + rank}{path}"


def wait_healthy(deadline):
    pending = set(range(RANKS))
    while pending:
        for rank in sorted(pending):
            try:
                with urllib.request.urlopen(rank_url(rank, "/health"), timeout=5) as resp:
                    if resp.status == 200:
                        pending.discard(rank)
            except OSError:
                pass
        if pending:
            if time.monotonic() > deadline:
                raise RuntimeError(f"ranks not healthy before the wait limit: {sorted(pending)}")
            time.sleep(10)


def post_ranks(path, timeout, ranks):
    """POST path to the given ranks in parallel; {rank: None on HTTP 200, else the exception}."""
    def post(rank):
        try:
            request = urllib.request.Request(rank_url(rank, path), data=b"", method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"rank {rank} {path}: HTTP {resp.status}")
            return None
        except Exception as exc:
            return exc

    ranks = sorted(ranks)
    with concurrent.futures.ThreadPoolExecutor(RANKS) as pool:
        return dict(zip(ranks, pool.map(post, ranks)))


def first_error(results):
    for rank, exc in sorted(results.items()):
        if exc is not None:
            return RuntimeError(f"rank {rank}: {type(exc).__name__}: {exc}")
    return None


def trace_sizes(trace_dir):
    sizes = {}
    for path in glob.glob(os.path.join(trace_dir, "dp*.pt.trace.json*")):
        match = re.match(r"dp(\d+)", os.path.basename(path))
        if match:
            sizes[int(match.group(1))] = os.path.getsize(path)
    return sizes


def wait_traces_settled(trace_dir, settle_s, ranks=RANKS, interval=2.0):
    """Block until every rank has a non-empty trace whose size is unchanged across two checks `interval` apart."""
    deadline = time.monotonic() + settle_s
    previous = None
    while True:
        sizes = trace_sizes(trace_dir)
        complete = all(sizes.get(r, 0) > 0 for r in range(ranks))
        if complete and sizes == previous:
            return
        previous = sizes if complete else None
        if time.monotonic() + interval > deadline:
            raise RuntimeError(f"trace files not complete and stable within {settle_s:g} s")
        time.sleep(interval)


def main():
    trace_dir = os.environ.get("PROFILE_TRACE_DIR", "/traces")
    attempted = set()  # ranks sent /start_profile (a failed reply may still have started the profiler)
    stopped = set()
    code = 0
    try:
        delay = float(os.environ.get("PROFILE_DELAY_S", "900"))
        duration = float(os.environ.get("PROFILE_DUR_S", "3"))
        wait = float(os.environ.get("PROFILE_WAIT_S", "7200"))
        settle = float(os.environ.get("PROFILE_SETTLE_S", "120"))
        wait_healthy(time.monotonic() + wait)
        time.sleep(delay)
        remove_traces(trace_dir)
        attempted.update(range(RANKS))
        error = first_error(post_ranks("/start_profile", 600, range(RANKS)))
        if error:
            raise error
        time.sleep(duration)
        results = post_ranks("/stop_profile", 600, range(RANKS))
        stopped.update(r for r, exc in results.items() if exc is None)
        error = first_error(results)
        if error:
            raise error
        wait_traces_settled(trace_dir, settle)
        rows = summarize(trace_dir)
        for row in rows:
            print(json.dumps(row, separators=(",", ":")), flush=True)
        print(json.dumps({"profile_done": True, "ranks": len(rows)}), flush=True)
    except Exception as exc:  # one error line, no retry, nothing else on stdout
        print(json.dumps({"profile_error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
        code = 1
    finally:  # best effort, silent: stop profilers still running, then delete the traces
        try:
            post_ranks("/stop_profile", 10, attempted - stopped)
        except Exception:
            pass
        try:
            remove_traces(trace_dir)
        except Exception:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
