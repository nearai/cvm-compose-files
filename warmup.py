#!/usr/bin/env python3
"""
Warmup and auto-tuning for GLM-5/5.1 on SGLang.

Called by build.sh inside the container on the GPU host.
Produces /sgl-workspace/tuned_config.json which entrypoint.sh reads at serve time.

Three phases:

  1. deepgemm  — JIT-compile DeepGemm kernels with real autotuning (~30 min).
                 Cached in /root/.cache/deep_gemm/ (baked into the image).

  2. memory   — Binary search for the max safe mem_fraction_static on this GPU.

  3. bench    — Sweep max_running_requests x num_continuous_decode_steps,
                pick the combo with the highest generation throughput.

Usage (called by build.sh, not directly):
  python3 /sgl-workspace/warmup.py --model-path /models/zai-org/GLM-5.1-FP8 --tp 8
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# ── Constants ────────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 30000
HEALTH_URL = f"http://{HOST}:{PORT}/v1/models"
COMPLETIONS_URL = f"http://{HOST}:{PORT}/v1/chat/completions"
SERVER_INFO_URL = f"http://{HOST}:{PORT}/get_server_info"

COMMON_ARGS = [
    "--reasoning-parser", "glm45",
    "--tool-call-parser", "glm47",
    "--context-length", "200000",
    "--enable-cache-report",
    "--enable-metrics",
    "--log-requests-level", "0",
    '--model-loader-extra-config', '{"enable_multithread_load": "true", "num_threads": 64}',
    "--host", HOST,
    "--port", str(PORT),
]

MEM_FRACTIONS = [0.92, 0.90, 0.88, 0.86, 0.85]
MAX_RUNNING_CANDIDATES = [35, 45, 55, 65, 75]
DECODE_STEPS_CANDIDATES = [3, 5, 8, 10]

BENCH_PROMPT = (
    "Explain the theory of general relativity in detail, covering spacetime "
    "curvature, geodesics, the Einstein field equations, and experimental "
    "confirmations including gravitational lensing and LIGO."
)
BENCH_MAX_TOKENS = 256
WARMUP_N = 3
BENCH_N = 10

OUTPUT_PATH = "/sgl-workspace/tuned_config.json"


def log(msg: str):
    print(f"[warmup] {msg}", flush=True)


# ── Server lifecycle ─────────────────────────────────────────────────────────

def start_server(model_path: str, tp: int, extra_args: list,
                 env_override: dict | None = None) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tp", str(tp),
        *COMMON_ARGS,
        *extra_args,
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    if env_override:
        env.update(env_override)
    log(f"Starting server: mem_frac={next((extra_args[i+1] for i in range(len(extra_args)) if extra_args[i]=='--mem-fraction-static'), '?')}")
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    return proc


def wait_ready(timeout: int = 600) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(HEALTH_URL, timeout=5)
            if r.status_code == 200:
                log("Server ready.")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    log("Timeout waiting for server.")
    return False


def stop_server(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    log("Stopping server...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def server_alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


# ── Benchmarking ─────────────────────────────────────────────────────────────

def get_model_name() -> str:
    try:
        r = requests.get(HEALTH_URL, timeout=10)
        data = r.json()
        return data["data"][0]["id"]
    except Exception:
        return "default"


def send_request(model: str, prompt: str = BENCH_PROMPT,
                 max_tokens: int = BENCH_MAX_TOKENS) -> dict | None:
    try:
        r = requests.post(COMPLETIONS_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }, timeout=180)
        if r.status_code != 200:
            return None
        return r.json().get("usage", {})
    except Exception:
        return None


def bench_throughput(model: str) -> dict:
    for _ in range(WARMUP_N):
        send_request(model)

    total_gen = 0
    t0 = time.monotonic()
    ok = 0
    fail = 0
    for _ in range(BENCH_N):
        usage = send_request(model)
        if usage:
            total_gen += usage.get("completion_tokens", 0)
            ok += 1
        else:
            fail += 1
    elapsed = time.monotonic() - t0
    tps = total_gen / elapsed if elapsed > 0 else 0
    return {"gen_tok_s": round(tps, 1), "ok": ok, "fail": fail,
            "tokens": total_gen, "elapsed_s": round(elapsed, 2)}


def get_server_throughput() -> float | None:
    try:
        r = requests.get(SERVER_INFO_URL, timeout=10)
        return r.json().get("internal_states", [{}])[0].get("last_gen_throughput")
    except Exception:
        return None


# ── Phase 1: DeepGemm ───────────────────────────────────────────────────────

def phase_deepgemm(model_path: str, tp: int, mem_frac: float):
    cache = Path("/root/.cache/deep_gemm")
    cubins = len(list(cache.glob("**/*.cubin"))) if cache.exists() else 0
    if cubins > 50:
        log(f"DeepGemm cache warm ({cubins} cubins), skipping.")
        return

    log("Phase 1: DeepGemm autotuning (SGLANG_JIT_DEEPGEMM_FAST_WARMUP=0)...")
    proc = start_server(model_path, tp, [
        "--mem-fraction-static", str(mem_frac),
        "--max-running-requests", "55",
        "--num-continuous-decode-steps", "5",
    ], env_override={"SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "0"})

    try:
        if not wait_ready(timeout=1800):
            log("WARN: Server didn't start for DeepGemm warmup.")
            return
        model = get_model_name()
        for i, (p, mt) in enumerate([
            ("Hi", 16),
            ("Explain quantum computing in detail.", 256),
            (BENCH_PROMPT, 512),
            (BENCH_PROMPT * 3, 512),
        ]):
            log(f"  Warmup {i+1}/4...")
            send_request(model, prompt=p, max_tokens=mt)
        cubins = len(list(cache.glob("**/*.cubin"))) if cache.exists() else 0
        log(f"  Autotuning done, {cubins} kernels cached.")
    finally:
        stop_server(proc)
        time.sleep(5)


# ── Phase 2: Memory ─────────────────────────────────────────────────────────

def phase_memory(model_path: str, tp: int) -> float:
    log("Phase 2: Probing max mem_fraction_static...")
    for frac in MEM_FRACTIONS:
        log(f"  Trying {frac}...")
        proc = start_server(model_path, tp, [
            "--mem-fraction-static", str(frac),
            "--max-running-requests", "55",
            "--num-continuous-decode-steps", "5",
        ], env_override={"SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1"})
        try:
            if not wait_ready(timeout=600) or not server_alive(proc):
                log(f"  {frac} failed to start.")
                continue
            model = get_model_name()
            usage = send_request(model)
            if usage and server_alive(proc):
                log(f"  {frac} OK — generated {usage.get('completion_tokens', 0)} tokens")
                return frac
            log(f"  {frac} crashed under load.")
        finally:
            stop_server(proc)
            time.sleep(5)
    log("  All fractions failed, falling back to 0.85.")
    return 0.85


# ── Phase 3: Bench ──────────────────────────────────────────────────────────

def phase_bench(model_path: str, tp: int, mem_frac: float) -> dict:
    log("Phase 3: Parameter sweep...")
    best = {"gen_tok_s": 0, "mrr": 55, "ncd": 5}
    results = []

    for mrr in MAX_RUNNING_CANDIDATES:
        for ncd in DECODE_STEPS_CANDIDATES:
            tag = f"mrr={mrr} ncd={ncd}"
            proc = start_server(model_path, tp, [
                "--mem-fraction-static", str(mem_frac),
                "--max-running-requests", str(mrr),
                "--num-continuous-decode-steps", str(ncd),
            ], env_override={"SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1"})
            try:
                if not wait_ready(timeout=600):
                    log(f"  {tag}: failed to start, skip.")
                    continue
                model = get_model_name()
                stats = bench_throughput(model)
                sglang_thr = get_server_throughput()
                entry = {"mrr": mrr, "ncd": ncd, **stats,
                         "sglang_throughput": sglang_thr}
                results.append(entry)
                log(f"  {tag}: {stats['gen_tok_s']} tok/s "
                    f"({stats['ok']}/{stats['ok']+stats['fail']} ok)")
                if stats["gen_tok_s"] > best["gen_tok_s"] and stats["fail"] == 0:
                    best = {"gen_tok_s": stats["gen_tok_s"], "mrr": mrr, "ncd": ncd}
            finally:
                stop_server(proc)
                time.sleep(3)

    log(f"  Best: mrr={best['mrr']} ncd={best['ncd']} → {best['gen_tok_s']} tok/s")
    return {"best": best, "all": results}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--phases", nargs="+", default=["deepgemm", "memory", "bench"],
                   choices=["deepgemm", "memory", "bench"])
    p.add_argument("--mem-fraction-static", type=float, default=None)
    p.add_argument("--output", default=OUTPUT_PATH)
    args = p.parse_args()

    log(f"Model: {args.model_path}  TP: {args.tp}  Phases: {args.phases}")

    config = {
        "model_path": args.model_path,
        "tp": args.tp,
        "mem_fraction_static": args.mem_fraction_static or 0.85,
        "max_running_requests": 55,
        "num_continuous_decode_steps": 5,
    }

    if "deepgemm" in args.phases:
        phase_deepgemm(args.model_path, args.tp, config["mem_fraction_static"])

    if "memory" in args.phases and args.mem_fraction_static is None:
        config["mem_fraction_static"] = phase_memory(args.model_path, args.tp)
    elif args.mem_fraction_static:
        config["mem_fraction_static"] = args.mem_fraction_static

    if "bench" in args.phases:
        res = phase_bench(args.model_path, args.tp, config["mem_fraction_static"])
        b = res["best"]
        config["max_running_requests"] = b["mrr"]
        config["num_continuous_decode_steps"] = b["ncd"]
        config["bench_results"] = res["all"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(config, indent=2))
    log(f"Config written to {args.output}")
    log(json.dumps({k: v for k, v in config.items() if k != "bench_results"}, indent=2))


if __name__ == "__main__":
    main()
