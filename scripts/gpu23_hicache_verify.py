#!/usr/bin/env python3
"""Bounded, synthetic HiCache verification from the engine's Docker network.

Only structural results, timing, token counts, metric deltas and SHA-256
digests are emitted. Request and response content is never printed.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
from urllib.parse import urlsplit


ENDPOINT = os.environ.get(
    "EVAL_ENDPOINT", "http://model-sg-glm53-w4afp8-tp8-hicache:8000"
).rstrip("/")
PHASES = tuple(
    part.strip()
    for part in os.environ.get("EVAL_PHASES", "smoke,full,restore").split(",")
    if part.strip()
)
TOKEN = os.environ.get("PROXY_TOKEN", "")
TIMEOUT = int(os.environ.get("EVAL_TIMEOUT_SECONDS", "2400"))
MODEL = "z-ai/glm-5.3"
METRICS = (
    "sglang:hicache_host_used_tokens",
    "sglang:hicache_host_total_tokens",
    "sglang:hicache_backup_tokens_total",
    "sglang:hicache_backup_bytes_total",
    "sglang:load_back_tokens_total",
    "sglang:load_back_bytes_total",
    "sglang:hicache_dropped_tokens_total",
    "sglang:evicted_tokens_total",
)


def emit(event: str, **fields) -> None:
    print(
        json.dumps(
            {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "event": event, **fields},
            sort_keys=True,
        ),
        flush=True,
    )


def connection(timeout: int = TIMEOUT):
    parsed = urlsplit(ENDPOINT)
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return cls(parsed.hostname, parsed.port, timeout=timeout), parsed


def request(method: str, path: str, payload=None, *, auth=True, timeout=TIMEOUT):
    conn, parsed = connection(timeout)
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {TOKEN}"
    body = None if payload is None else json.dumps(payload, separators=(",", ":"))
    started = time.monotonic()
    conn.request(method, (parsed.path.rstrip("/") + path) or "/", body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    elapsed = time.monotonic() - started
    conn.close()
    try:
        parsed_body = json.loads(raw) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed_body = {}
    return response.status, parsed_body, elapsed


def wait_ready() -> None:
    deadline = time.monotonic() + TIMEOUT
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        try:
            status, _, _ = request("GET", "/health", auth=False, timeout=10)
            if status == 200:
                emit("ready", attempts=attempts)
                return
        except OSError:
            pass
        time.sleep(5)
    raise RuntimeError("engine did not become ready before timeout")


def metric_snapshot():
    conn, parsed = connection(20)
    conn.request("GET", (parsed.path.rstrip("/") + "/metrics") or "/")
    response = conn.getresponse()
    raw = response.read().decode("utf-8", errors="replace")
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"metrics HTTP {response.status}")
    totals = {name: 0.0 for name in METRICS}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in totals:
            continue
        try:
            totals[name] += float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
    return totals


def delta(after, before):
    return {key: round(after.get(key, 0) - before.get(key, 0), 6) for key in METRICS}


def content_digest(body) -> tuple[str, int, str | None]:
    choices = body.get("choices") or []
    text = "".join(
        ((choice.get("message") or {}).get("reasoning_content") or "")
        + ((choice.get("message") or {}).get("content") or "")
        + (choice.get("text") or "")
        for choice in choices
    )
    finish = choices[0].get("finish_reason") if choices else None
    return hashlib.sha256(text.encode()).hexdigest(), len(text), finish


def post_result(phase: str, path: str, payload):
    status, body, elapsed = request("POST", path, payload)
    digest, chars, finish = content_digest(body) if status == 200 else (None, 0, None)
    usage = body.get("usage") or {}
    emit(
        "request_result",
        phase=phase,
        status=status,
        elapsed_seconds=round(elapsed, 6),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        output_sha256=digest,
        output_characters=chars,
        finish_reason=finish,
        error_type=(body.get("error") or {}).get("type") if isinstance(body.get("error"), dict) else None,
    )
    return status, body, digest


def smoke() -> None:
    post_result(
        "short_chat",
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "Reply exactly OK."}],
         "temperature": 0, "max_tokens": 16},
    )
    post_result(
        "tool_call",
        "/v1/chat/completions",
        {"model": MODEL,
         "messages": [{"role": "user", "content": "Call record_value with integer 17."}],
         "tools": [{"type": "function", "function": {"name": "record_value",
                    "description": "Record one integer", "parameters": {"type": "object",
                    "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}],
         "tool_choice": {"type": "function", "function": {"name": "record_value"}},
         "temperature": 0, "max_tokens": 128},
    )


def token_prompt(token: int, count: int):
    # OpenAI completions accepts one tokenized prompt as list[int]. Prefixes
    # differ at token zero so churn requests cannot share a radix prefix.
    return [token] + [1000] * (count - 1)


def full_context() -> None:
    status, body, _ = post_result(
        "full_context_1m",
        "/v1/completions",
        {"model": MODEL, "prompt": token_prompt(1100, 1_000_000),
         "temperature": 0, "max_tokens": 1},
    )
    prompt_tokens = (body.get("usage") or {}).get("prompt_tokens")
    emit("full_context_gate", passed=status == 200 and prompt_tokens == 1_000_000,
         observed_prompt_tokens=prompt_tokens)


def flush_cache() -> None:
    status, _, elapsed = request("POST", "/flush_cache", {})
    emit("flush_cache", status=status, elapsed_seconds=round(elapsed, 6))
    if status != 200:
        raise RuntimeError(f"flush_cache HTTP {status}")


def restore_gate() -> None:
    flush_cache()
    before = metric_snapshot()
    target = {"model": MODEL, "prompt": token_prompt(1200, 230_000),
              "temperature": 0, "max_tokens": 1}
    prime_status, _, prime_digest = post_result("cache_prime", "/v1/completions", target)
    post_result("cache_gpu_hit", "/v1/completions", target)
    after_gpu_hit = metric_snapshot()
    for index in range(5):
        post_result(
            f"cache_churn_{index}",
            "/v1/completions",
            {"model": MODEL, "prompt": token_prompt(1300 + index, 220_000),
             "temperature": 0, "max_tokens": 1},
        )
    after_churn = metric_snapshot()
    restore_status, _, restore_digest = post_result("cache_host_restore", "/v1/completions", target)
    after_restore = metric_snapshot()
    restore_delta = delta(after_restore, after_churn)
    passed = (
        prime_status == 200
        and restore_status == 200
        and prime_digest == restore_digest
        and restore_delta["sglang:load_back_tokens_total"] > 0
        and restore_delta["sglang:load_back_bytes_total"] > 0
    )
    emit(
        "restore_gate",
        passed=passed,
        exact_output_match=prime_digest == restore_digest,
        gpu_hit_metrics_delta=delta(after_gpu_hit, before),
        churn_metrics_delta=delta(after_churn, after_gpu_hit),
        restore_metrics_delta=restore_delta,
        final_metrics_delta=delta(after_restore, before),
    )


def main() -> None:
    if not TOKEN:
        raise SystemExit("PROXY_TOKEN is required")
    wait_ready()
    emit("verification_start", endpoint_host=urlsplit(ENDPOINT).hostname, phases=PHASES)
    if "smoke" in PHASES:
        smoke()
    if "full" in PHASES:
        full_context()
    if "restore" in PHASES:
        restore_gate()
    emit("verification_end")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("verification_error", error_type=type(exc).__name__, message=str(exc))
        raise
