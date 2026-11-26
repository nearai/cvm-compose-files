#!/usr/bin/env python3
"""P/D sidecar router for vLLM NixlConnector pull mode: one router per prefill/decode pair.

Request flow (two sequential legs, because in pull mode the decode leg cannot start before the prefill engine has
returned its block list):
  leg 1  POST to a prefill rank with max_tokens=1, stream=false, kv_transfer_params={"do_remote_decode": true}
  leg 2  POST the client's own body plus the kv_transfer_params from leg 1 to a decode rank, streamed back unchanged.

Behaviour contract:
  * Same CLI and env names as the r2 router (--prefill, --decode, --host, --port, --max-imbalance; PD_MAX_IMBALANCE,
    PD_P_DROP_STRUCT, PD_STATS_LOG_S) so it can replace it as a single file.
  * Both legs are sticky per conversation (hash of the messages up to and including the first user message, or of the prompt, plus cache_salt when present) unless the sticky rank has more
    than --max-imbalance more requests in flight than the least loaded rank of the same role. Ranks come from this pair only.
  * A client that disconnects cancels whichever leg is running. Closing the upstream connection makes the engine abort the
    request. A prefill that already finished holds its KV blocks until the connector's lease expires (see README).
  * PD_KV_TIMEOUT_S (default 20) bounds decode-leg start to first byte for streaming requests. On expiry the decode
    request is cancelled and the client gets a 504 in the engine's error shape. The prefill engine frees its blocks when
    its KV lease ends (kv_lease_duration in kv_connector_extra_config, default 30 s), so the router refuses to start
    unless PD_KV_TIMEOUT_S < PD_KV_LEASE_S - 5; PD_KV_LEASE_S must equal the engine's value (see README).
  * A body whose n is present and not the integer 1 is rejected with 400. kv_transfer_params from the client is dropped on every path; the decode leg
    gets only what the prefill engine returned.
  * Errors keep the upstream status and are engine-shaped JSON. Nothing reaches the client before leg 2 has produced its
    first byte, so a leg 1 failure is always a clean error response.
  * Only the decode leg's response is forwarded, byte for byte, so usage is counted once and never altered. A decode
    stream that breaks mid-way aborts the client connection instead of ending cleanly.
  * GET /health probes every prefill and decode rank on every call (no cache, PD_HEALTH_TIMEOUT_S default 2, a timeout is
    unhealthy) and is 200 only when all of them answer /health with 200.
  * GET /v1/metrics and /metrics aggregate the ranks' Prometheus text (pair totals; ?per_rank=1 keeps every series).
    Every rank must answer within PD_METRICS_TIMEOUT_S (default 5); otherwise the scrape is a 503 rather than a smaller sum.
  * ROUTER_DDIRECT=1 (default off) sends a chat follow-up with a small new tail straight to the decode rank that served
    the conversation last, skipping leg 1. A direct attempt that the decode engine refuses before the first byte is
    retried on the P/D path. Responses then carry x-pd-path (direct, pd or pd-fb). See README.
  * /metrics also carries the router's own series: request and failure counters, decode-direct counters and a
    first-byte histogram by path. Fixed label values, counts and durations only.
  * Logs carry counts, durations, status codes and backend indexes only. No request or response content, no URLs.
"""
import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import struct
import time
from collections import OrderedDict, deque

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

try:
    import orjson

    def jloads(b):
        return orjson.loads(b)

    def jdumps(o) -> bytes:
        return orjson.dumps(o)
except Exception:  # orjson is optional
    def jloads(b):
        return json.loads(b)

    def jdumps(o) -> bytes:
        return json.dumps(o).encode()

log = logging.getLogger("pd-router")

POP_PREFILL = ["stream_options", "max_completion_tokens", "min_tokens", "logprobs", "top_logprobs", "prompt_logprobs",
               "n", "best_of", "kv_transfer_params"]
POP_STRUCT = ["response_format", "structured_outputs", "guided_json", "guided_regex", "guided_choice", "guided_grammar",
              "guided_decoding_backend", "guided_whitespace_pattern"]
FWD_HEADERS = ("authorization", "x-request-id")
GET_PASSTHROUGH = ("v1/models", "version")
ERR_CAP = 1 << 20
POLL_S = 0.25
LEASE_MARGIN_S = 5.0
# A direct attempt refused before the first client byte (HTTP 429/503, or an SSE stream whose first event is an error
# with that code, as vLLM reports an admission rejection on a streaming request) is retried on the P/D path.
DD_FALLBACK_STATUS = (429, 503)
# Reasons the per-conversation memory gives for not attempting a direct request (see Router._mem_skip_reason).
DD_SKIP_REASONS = ("no_entry", "stale", "rank_changed", "rejected_prev", "context_changed")
DD_REJECTED = object()  # sentinel returned by _decode_leg(direct=True) when D rejected the request pre-stream


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


async def read_capped(content, cap: int) -> bytes:
    """Read a response body until EOF or `cap` bytes; a single StreamReader.read() returns only what is buffered."""
    buf = bytearray()
    while len(buf) < cap:
        chunk = await content.read(cap - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


class Gone(Exception):
    """The client hung up."""


class Timeout(Exception):
    """A leg exceeded its time budget."""


class StreamAborted(Exception):
    """The decode stream broke after the response had started; raised so the client connection is aborted."""


class GuardedStream(StreamingResponse):
    """StreamingResponse that always runs `cleanup`, also when the body generator never started (client gone early)."""

    def __init__(self, content, cleanup, **kw):
        super().__init__(content, **kw)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cleanup()


def engine_error(status: int, message: str, etype: str, headers: dict | None = None) -> Response:
    body = {"error": {"message": message, "type": etype, "param": None, "code": status}}
    return Response(jdumps(body), status_code=status, media_type="application/json", headers=headers or None)


ETYPE = {400: "BadRequestError", 401: "AuthenticationError", 403: "PermissionDeniedError", 404: "NotFoundError",
         413: "BadRequestError", 422: "BadRequestError", 429: "TooManyRequestsError", 503: "ServiceUnavailableError",
         504: "GatewayTimeoutError"}


def upstream_error(status: int, data: bytes, retry_after: str | None) -> Response:
    """Keep the upstream status; forward its body only if the engine produced an error object, else build one."""
    headers = {"retry-after": retry_after} if retry_after else None
    try:
        obj = jloads(data)
    except Exception:
        obj = None
    if isinstance(obj, dict) and ("error" in obj or obj.get("object") == "error"):
        return Response(data, status_code=status, media_type="application/json", headers=headers)
    return engine_error(status, f"upstream engine returned status {status}",
                        ETYPE.get(status, "InternalServerError"), headers)


# ---------------------------------------------------------------- Prometheus aggregation

SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*$")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
SUFFIXES = ("_bucket", "_sum", "_count", "_total", "_created")


def fmt(v: float) -> str:
    if v != v:
        return "NaN"
    if v in (float("inf"), float("-inf")):
        return "+Inf" if v > 0 else "-Inf"
    return repr(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)


def family_type(name: str, types: dict) -> str:
    if name in types:
        return types[name]
    for s in SUFFIXES:
        if name.endswith(s) and name[: -len(s)] in types:
            return types[name[: -len(s)]]
    return "untyped"


def aggregate_metrics(scrapes: list, per_rank: bool = False) -> str:
    """scrapes: [(role, rank, text)]. Sum series across ranks (max for ratio gauges, first for *_info), or with
    per_rank keep every series and add pd_role and pd_rank labels."""
    meta, types, order, acc = [], {}, [], {}
    seen_meta = set()
    for role, rank, text in scrapes:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.split(None, 3)
                if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                    if parts[1] == "TYPE" and len(parts) == 4:
                        types[parts[2]] = parts[3]
                    if (parts[1], parts[2]) not in seen_meta:
                        seen_meta.add((parts[1], parts[2]))
                        meta.append(line)
                continue
            m = SAMPLE.match(line)
            if not m:
                continue
            name, raw_labels, raw_val = m.groups()
            if name.endswith("_created"):
                continue
            try:
                val = float(raw_val)
            except ValueError:
                continue
            labels = LABEL.findall(raw_labels or "")
            if per_rank:
                labels = labels + [("pd_role", role), ("pd_rank", str(rank))]
                key = (name, tuple(labels))
                order.append(key)
                acc[key] = val
                continue
            labels = tuple((k, v) for k, v in labels if k != "engine")
            key = (name, labels)
            if key not in acc:
                order.append(key)
                acc[key] = val
                continue
            if name.endswith("_info") or family_type(name, types) == "info":
                continue
            if family_type(name, types) == "gauge" and name.endswith(("_perc", "_ratio")):
                acc[key] = max(acc[key], val)
            else:
                acc[key] += val
    out = list(meta)
    for key in order:
        name, labels = key
        lab = ",".join(f'{k}="{v}"' for k, v in labels)
        out.append(f"{name}{{{lab}}} {fmt(acc[key])}" if lab else f"{name} {fmt(acc[key])}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- router

def pct(xs, q):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))] * 1000, 1) if s else None


# ---- Decode-direct: follow-up turns with a small new tail skip the prefill leg and go straight to decode.
DD_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
PATHS = ("direct", "pd", "pd-fb")
# Direct-attempt context by outcome (accepted: D answered 200; rejected: D admission refused it; error: any other
# failure): token estimates, gap since last served, and whether the template-rendered fields outside the messages changed.
DD_OUTCOMES = ("accepted", "rejected", "error")
# Which reasoning field the last assistant message of a request carries (first match in this order; "none" if no field).
REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_details", "inline_think", "none")
DD_GAP_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
# Body fields that the chat template can render ahead of the messages; a change moves D's prefix hit to token 0.
CTX_FIELDS = ("tools", "functions", "tool_choice", "chat_template_kwargs", "documents", "chat_template")
FB_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)
DD_REASONS = ("first_turn", "tail_too_long", "chunk_too_large", "images", "completions")
# How a follow-up's message history relates to the previous request of the same conversation (see history_kind).
HIST_KINDS = ("append_only", "reasoning_dropped", "edited", "truncated", "unknown")
HIST_MAX_MSGS = 256  # per-conversation memory keeps hashes of at most this many messages; more counts as unknown
_REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning_details")
_MEDIA_PART_MARKERS = ("image", "video", "audio")


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def _text_chars(content) -> int:
    """Characters of text in a message content: a string, or the text parts of a list. Non-text parts count zero."""
    if isinstance(content, str):
        return len(content)
    n = 0
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                n += len(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                n += len(part["text"])
    return n


def _has_media(msgs) -> bool:
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for part in c:
                t = part.get("type") if isinstance(part, dict) else None
                if isinstance(t, str) and any(k in t for k in _MEDIA_PART_MARKERS):
                    return True
    return False


def msg_chars(m) -> int:
    """Text chars of one message: content, plus tool-call arguments."""
    if not isinstance(m, dict):
        return 0
    n = _text_chars(m.get("content"))
    tcs = m.get("tool_calls")
    if isinstance(tcs, list):
        for tc in tcs:
            fn = tc.get("function") if isinstance(tc, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                n += len(args)
            elif args is not None:
                n += len(json.dumps(args))
    return n


def ctx_hash(body: dict) -> int:
    """Hash of the template-rendered body fields outside the messages (kept in memory only, never logged)."""
    key = json.dumps({k: body.get(k) for k in CTX_FIELDS if body.get(k) is not None}, sort_keys=True, default=str)
    return int(hashlib.sha1(key.encode()).hexdigest()[:15], 16)


def _h32(obj) -> int:
    return int(hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)


def msg_hashes(m) -> tuple:
    """(full, base) 32-bit hashes of one message. base covers role, content, tool calls and ids; full adds the
    reasoning fields (an empty one counts as absent). Memory only, never logged or exported."""
    if not isinstance(m, dict):
        return _h32(None), _h32(None)
    base = _h32([m.get(k) for k in ("role", "content", "tool_calls", "tool_call_id", "name")])
    rs = [m.get(k) for k in _REASONING_KEYS if m.get(k)]
    return (_h32([base, rs]) if rs else base), base


def history_pack(msgs) -> bytes | None:
    """Compact per-message hashes (full, base) of a request's messages, or None past HIST_MAX_MSGS (kept as unknown)."""
    if not isinstance(msgs, list) or len(msgs) > HIST_MAX_MSGS:
        return None
    return b"".join(struct.pack("<II", *msg_hashes(m)) for m in msgs)


def history_kind(prev: bytes | None, msgs: list, chars_per_token: float) -> tuple:
    """Compare the stored hashes of the previous request with the new messages: (kind in HIST_KINDS, estimated tokens
    from the first differing message to the end). The previous request carried reasoning where full != base."""
    if prev is None or len(msgs) > HIST_MAX_MSGS:
        return "unknown", 0.0
    old = [struct.unpack_from("<II", prev, o) for o in range(0, len(prev), 8)]
    new = [msg_hashes(m) for m in msgs]
    i = 0
    while i < min(len(old), len(new)) and old[i][0] == new[i][0]:
        i += 1
    est = sum(msg_chars(m) for m in msgs[i:]) / chars_per_token
    if len(new) < len(old):
        return "truncated", est
    if i == len(old):
        return "append_only", est
    m = msgs[i]
    if (isinstance(m, dict) and m.get("role") == "assistant" and old[i][0] != old[i][1]
            and new[i][0] == new[i][1] and old[i][1] == new[i][1]):
        return "reasoning_dropped", est
    return "edited", est


def reasoning_field(msgs: list):
    """The reasoning field of the last assistant message (a REASONING_FIELDS value), or None without one."""
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant":
            for k in ("reasoning", "reasoning_content", "reasoning_details"):
                if m.get(k):
                    return k
            c = m.get("content")
            return "inline_think" if isinstance(c, str) and "<think>" in c else "none"
    return None


def last_assistant_chars(msgs: list) -> int:
    """Text chars of the last assistant message (the reply the client sends back)."""
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return msg_chars(m)
    return 0


def tail_chars(msgs: list) -> int:
    """Text chars in all messages after the last assistant message: content, tool-call arguments, tool results."""
    last = -1
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant":
            last = i
    return sum(msg_chars(m) for m in msgs[last + 1:])


def estimate_tail_tokens(msgs: list, chars_per_token: float, count_prev_reply: bool = False) -> float:
    return (tail_chars(msgs) + (last_assistant_chars(msgs) if count_prev_reply else 0)) / chars_per_token


def estimate_chunk_tokens(msgs: list, chars_per_token: float, block_tokens: float,
                          count_prev_reply: bool = False) -> tuple:
    """(tail_est, chunk_est). D's hit stops at the last block checkpoint of what it already holds (everything before
    the tail), so chunk = N - floor((N - tail) / block) * block, N = estimated tokens of the whole request.
    With count_prev_reply the previous assistant reply counts as tail too: D has to recompute it when it is not cached."""
    tail = estimate_tail_tokens(msgs, chars_per_token, count_prev_reply)
    n = sum(msg_chars(m) for m in msgs) / chars_per_token
    return tail, n - math.floor((n - tail) / block_tokens) * block_tokens


def ddirect_classify(path: str, body: dict, chars_per_token: float, max_tail_tokens: float,
                     block_tokens: float = 2432, max_chunk_tokens: float = 2048,
                     count_prev_reply: bool = False):
    """Return (reason, tail_est, chunk_est). reason is None when eligible, else one of DD_REASONS.
    The estimates are None when not computed (reasons completions, first_turn, images)."""
    msgs = body.get("messages")
    if path != "/v1/chat/completions" or not isinstance(msgs, list):
        return "completions", None, None
    if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in msgs):
        return "first_turn", None, None
    if _has_media(msgs):
        return "images", None, None
    tail, chunk = estimate_chunk_tokens(msgs, chars_per_token, block_tokens, count_prev_reply)
    if tail > max_tail_tokens:
        return "tail_too_long", tail, chunk
    if chunk > max_chunk_tokens:
        return "chunk_too_large", tail, chunk
    return None, tail, chunk


_SSE_TERM = re.compile(rb"\r\n\r\n|\n\n")
FIRST_EVENT_CAP = 64 * 1024  # bytes buffered while looking for the first SSE event of a direct attempt
FIRST_EVENT_S = 5.0  # time budget for that read


def _sse_scan(buf: bytes, eof: bool = False):
    """Look at the start of an SSE stream. Returns (done, code): done=False means more bytes are needed before the first
    real event can be judged; code is the error code of the first non-comment event if it is an error object, else None.
    Leading comment-only events (": ping"), "data:" with or without a space, and \n\n or \r\n\r\n separators are handled."""
    s = buf.lstrip(b"\r\n")
    while True:
        if not s:
            return (eof, None)
        m = _SSE_TERM.search(s)
        if m is None:
            if not eof:
                head = s[:5]
                if s.startswith(b":") or s.startswith(b"data:") or b"data:".startswith(head):
                    return (False, None)
                return (True, None)  # not an SSE field we look at: pass through
            ev, s = s, b""
        else:
            ev, s = s[:m.start()], s[m.end():]
        lines = [ln for ln in ev.splitlines() if ln and not ln.startswith(b":")]
        if not lines:
            s = s.lstrip(b"\r\n")
            continue  # comment-only event
        data = [ln[5:][1:] if ln[5:6] == b" " else ln[5:] for ln in lines if ln.startswith(b"data:")]
        if not data:
            return (True, None)
        try:
            obj = jloads(b"\n".join(data).strip())
            err = obj.get("error") if isinstance(obj, dict) else None
            code = err.get("code") if isinstance(err, dict) else None
            return (True, int(code) if code is not None else None)
        except Exception:
            return (True, None)


def _sse_error_code(buf: bytes):
    """Error code of the first SSE event if it is an engine error object, else None."""
    return _sse_scan(buf, eof=True)[1]


class Router:
    def __init__(self, prefill, decode, max_imbalance=64, kv_timeout=None, prefill_timeout=None,
                 nonstream_timeout=None, health_timeout=None, drop_struct=None, stats_log_s=None, kv_lease=None,
                 metrics_timeout=None, ddirect=None, ddirect_max_tail_tokens=None, ddirect_chars_per_token=None,
                 ddirect_max_d_inflight=None, ddirect_block_tokens=None, ddirect_max_chunk_tokens=None, ddirect_max_age_s=None, ddirect_mem_cap=None,
                 ddirect_mem_ttl_s=None, strict_affinity=None, ddirect_count_prev_reply=None):
        self.prefill, self.decode = list(prefill), list(decode)
        self.max_imbalance = max_imbalance
        self.kv_timeout = env_float("PD_KV_TIMEOUT_S", 20) if kv_timeout is None else kv_timeout
        self.kv_lease = env_float("PD_KV_LEASE_S", 30) if kv_lease is None else kv_lease
        if not 0 < self.kv_timeout < self.kv_lease - LEASE_MARGIN_S:
            raise ValueError(
                f"PD_KV_TIMEOUT_S={self.kv_timeout:g} must be > 0 and below PD_KV_LEASE_S - {LEASE_MARGIN_S:g} "
                f"(PD_KV_LEASE_S={self.kv_lease:g}): the prefill engine frees its KV blocks when kv_lease_duration "
                "ends, so a decode request must not wait longer than the lease. Raise kv_lease_duration on the "
                "prefill engine and PD_KV_LEASE_S together, or lower PD_KV_TIMEOUT_S.")
        self.prefill_timeout = env_float("PD_PREFILL_TIMEOUT_S", 600) if prefill_timeout is None else prefill_timeout
        self.metrics_timeout = env_float("PD_METRICS_TIMEOUT_S", 5) if metrics_timeout is None else metrics_timeout
        self.nonstream_timeout = (env_float("PD_NONSTREAM_TIMEOUT_S", 0) if nonstream_timeout is None
                                  else nonstream_timeout)
        self.health_timeout = env_float("PD_HEALTH_TIMEOUT_S", 2) if health_timeout is None else health_timeout
        self.drop_struct = os.environ.get("PD_P_DROP_STRUCT", "1") != "0" if drop_struct is None else drop_struct
        self.stats_log_s = env_float("PD_STATS_LOG_S", 60) if stats_log_s is None else stats_log_s
        self.inflight = {"prefill": [0] * len(self.prefill), "decode": [0] * len(self.decode)}
        self.t_prefill = deque(maxlen=4000)
        self.t_dfirst = deque(maxlen=4000)
        self.counts = {"requests": 0, "client_gone_in_prefill": 0, "client_gone_in_decode": 0, "prefill_fail": 0,
                       "decode_fail": 0, "decode_first_byte_timeout": 0, "bad_request": 0}
        self.ddirect = env_bool("ROUTER_DDIRECT") if ddirect is None else ddirect
        self.dd_max_tail = (env_float("ROUTER_DDIRECT_MAX_TAIL_TOKENS", 1024)
                            if ddirect_max_tail_tokens is None else ddirect_max_tail_tokens)
        self.dd_cpt = (env_float("ROUTER_DDIRECT_CHARS_PER_TOKEN", 3.5)
                       if ddirect_chars_per_token is None else ddirect_chars_per_token)
        if self.dd_cpt <= 0:
            raise ValueError("ROUTER_DDIRECT_CHARS_PER_TOKEN must be > 0")
        self.dd_max_inflight = (int(env_float("ROUTER_DDIRECT_MAX_D_INFLIGHT", 48))
                                if ddirect_max_d_inflight is None else ddirect_max_d_inflight)
        self.dd_block = (env_float("ROUTER_DDIRECT_BLOCK_TOKENS", 2432)
                         if ddirect_block_tokens is None else ddirect_block_tokens)
        self.dd_max_chunk = (env_float("ROUTER_DDIRECT_MAX_CHUNK_TOKENS", 2048)
                             if ddirect_max_chunk_tokens is None else ddirect_max_chunk_tokens)
        if self.dd_block <= 0:
            raise ValueError("ROUTER_DDIRECT_BLOCK_TOKENS must be > 0")
        self.dd_max_age = (env_float("ROUTER_DDIRECT_MAX_AGE_S", 30) if ddirect_max_age_s is None else ddirect_max_age_s)
        self.dd_mem_cap = (int(env_float("ROUTER_DDIRECT_MEM_CAP", 100000)) if ddirect_mem_cap is None
                           else ddirect_mem_cap)
        self.dd_mem_ttl = (env_float("ROUTER_DDIRECT_MEM_TTL_S", 600) if ddirect_mem_ttl_s is None
                           else ddirect_mem_ttl_s)
        if self.dd_mem_cap < 1:
            raise ValueError("ROUTER_DDIRECT_MEM_CAP must be >= 1")
        self.dd_count_reply = (env_bool("ROUTER_DDIRECT_COUNT_PREV_REPLY") if ddirect_count_prev_reply is None
                               else ddirect_count_prev_reply)
        self.strict_affinity = env_bool("ROUTER_STRICT_AFFINITY") if strict_affinity is None else strict_affinity
        # h -> [D rank, served time, prompt tokens (estimate), rejected since served, hash of the context fields, packed message hashes]: numbers
        # and hashes only, never request content. LRU order, bounded by dd_mem_cap and TTL.
        self.dd_mem = OrderedDict()
        def hist():
            return {**{f"le_{b}": 0 for b in DD_BUCKETS}, "gt_%d" % DD_BUCKETS[-1]: 0}

        self.dd = {"eligible": 0, "sent_direct": 0, "guard_fallback": 0, "direct_errors": 0, "direct_rejected_fallback": 0,
                   "skip_reason": {k: 0 for k in DD_SKIP_REASONS}, "mem_evicted": 0,
                   "ineligible_reason": {k: 0 for k in DD_REASONS},
                   "tail_est": hist(), "chunk_est": hist()}
        # Per direct attempt, by outcome: estimate histograms (new tokens since served, tail, previous reply), gap since
        # served, and whether the context fields changed. Counts only.
        self.dd_ctx = {o: {"new": [0] * (len(DD_BUCKETS) + 1), "tail": [0] * (len(DD_BUCKETS) + 1),
                           "reply": [0] * (len(DD_BUCKETS) + 1), "gap": [0] * (len(DD_GAP_BUCKETS) + 1),
                           "ctx_changed": 0, "ctx_same": 0, "sum": {"new": 0.0, "tail": 0.0, "reply": 0.0, "gap": 0.0}}
                       for o in DD_OUTCOMES}
        self.reasoning_field = {k: 0 for k in REASONING_FIELDS}
        # Follow-up history vs the previous request of the conversation, by kind: a counter and an estimate histogram.
        self.hist_kind = {k: 0 for k in HIST_KINDS}
        self.hist_diverged = {k: {"buckets": [0] * (len(DD_BUCKETS) + 1), "sum": 0.0} for k in HIST_KINDS}
        self.decode_waiting = 0  # requests between decode POST and first byte (decode queueing plus the remote KV pull)
        # Decode-leg requests by path, and router arrival to first client byte by path (cumulative bucket counts).
        self.path_requests = {k: 0 for k in PATHS}
        self.fb_hist = {k: {"buckets": [0] * len(FB_BUCKETS), "sum": 0.0, "count": 0} for k in PATHS}
        self._session = None

    # -- plumbing
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # keepalive shorter than uvicorn's 5 s server keep-alive: never reuse a connection about to be closed
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=6 * 3600, sock_connect=10),
                connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=4.0))
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def summary(self) -> dict:
        return {**self.counts, "inflight": {k: list(v) for k, v in self.inflight.items()}, "n": len(self.t_prefill),
                "prefill_ms": {"p50": pct(self.t_prefill, .5), "p95": pct(self.t_prefill, .95)},
                "decode_first_byte_ms": {"p50": pct(self.t_dfirst, .5), "p95": pct(self.t_dfirst, .95)}}

    def ddirect_summary(self) -> dict:
        return {"enabled": self.ddirect, "max_tail_tokens": self.dd_max_tail, "chars_per_token": self.dd_cpt,
                "max_d_inflight": self.dd_max_inflight, "block_tokens": self.dd_block,
                "max_chunk_tokens": self.dd_max_chunk, "max_age_s": self.dd_max_age,
                "mem_cap": self.dd_mem_cap, "mem_ttl_s": self.dd_mem_ttl, "mem_size": len(self.dd_mem),
                "strict_affinity": self.strict_affinity, "count_prev_reply": self.dd_count_reply, **json.loads(json.dumps(self.dd)),
                "attempt_context": json.loads(json.dumps(self.dd_ctx)),
                "assistant_reasoning_field": dict(self.reasoning_field), "history_change": dict(self.hist_kind)}

    def _fb_observe(self, path: str, seconds: float):
        h = self.fb_hist[path]
        for i, b in enumerate(FB_BUCKETS):
            if seconds <= b:
                h["buckets"][i] += 1
        h["sum"] += seconds
        h["count"] += 1

    def _path_header(self, label: str) -> dict:
        """x-pd-path is sent only with decode-direct on."""
        return {"x-pd-path": label} if self.ddirect else {}

    def own_metrics(self) -> str:
        """The router's own Prometheus series. Fixed label values, counts and durations only."""
        dd = self.dd
        out = []

        def series(name, typ, samples):
            out.append(f"# TYPE {name} {typ}")
            out.extend(f"{name}{lab} {fmt(v)}" for lab, v in samples)

        for name, typ, v in (
                ("pd_router_ddirect_eligible_total", "counter", dd["eligible"]),
                ("pd_router_ddirect_attempts_total", "counter", dd["sent_direct"]),
                ("pd_router_ddirect_rejected_fallback_total", "counter", dd["direct_rejected_fallback"]),
                ("pd_router_ddirect_guard_fallback_total", "counter", dd["guard_fallback"]),
                ("pd_router_ddirect_errors_total", "counter", dd["direct_errors"])):
            series(name, typ, [("", v)])
        series("pd_router_ddirect_skipped_total", "counter",
               [(f'{{reason="{k}"}}', dd["skip_reason"][k]) for k in DD_SKIP_REASONS])
        series("pd_router_ddirect_ineligible_total", "counter",
               [(f'{{reason="{k}"}}', dd["ineligible_reason"][k]) for k in DD_REASONS])
        series("pd_router_decode_requests_total", "counter",
               [(f'{{path="{k}"}}', self.path_requests[k]) for k in PATHS])
        samples = []
        for k in PATHS:
            h = self.fb_hist[k]
            samples += [(f'_bucket{{path="{k}",le="{fmt(b)}"}}', c) for b, c in zip(FB_BUCKETS, h["buckets"])]
            samples += [(f'_bucket{{path="{k}",le="+Inf"}}', h["count"]), (f'_sum{{path="{k}"}}', h["sum"]),
                        (f'_count{{path="{k}"}}', h["count"])]
        series("pd_router_assistant_reasoning_field_total", "counter",
               [(f'{{field="{k}"}}', self.reasoning_field[k]) for k in REASONING_FIELDS])
        series("pd_router_ddirect_attempt_context_total", "counter",
               [(f'{{outcome="{o}",context="{c}"}}', self.dd_ctx[o]["ctx_" + c]) for o in DD_OUTCOMES
                for c in ("same", "changed")])
        for key, name, bounds in (("new", "pd_router_ddirect_attempt_new_tokens_est", DD_BUCKETS),
                                  ("tail", "pd_router_ddirect_attempt_tail_tokens_est", DD_BUCKETS),
                                  ("reply", "pd_router_ddirect_attempt_prev_reply_tokens_est", DD_BUCKETS),
                                  ("gap", "pd_router_ddirect_attempt_gap_seconds", DD_GAP_BUCKETS)):
            hs = []
            for o in DD_OUTCOMES:
                cum = 0
                for b, c in zip(bounds, self.dd_ctx[o][key]):
                    cum += c
                    hs.append((f'_bucket{{outcome="{o}",le="{fmt(b)}"}}', cum))
                n = sum(self.dd_ctx[o][key])
                hs += [(f'_bucket{{outcome="{o}",le="+Inf"}}', n), (f'_sum{{outcome="{o}"}}', self.dd_ctx[o]["sum"][key]),
                       (f'_count{{outcome="{o}"}}', n)]
            out.append(f"# TYPE {name} histogram")
            out.extend(f"{name}{lab} {fmt(v)}" for lab, v in hs)
        series("pd_router_history_change_total", "counter",
               [(f'{{kind="{k}"}}', self.hist_kind[k]) for k in HIST_KINDS])
        name = "pd_router_history_diverged_tokens_est"
        out.append(f"# TYPE {name} histogram")
        for k in HIST_KINDS:
            h, cum = self.hist_diverged[k], 0
            for b, c in zip(DD_BUCKETS, h["buckets"]):
                cum += c
                out.append(f'{name}_bucket{{kind="{k}",le="{fmt(b)}"}} {fmt(cum)}')
            n = sum(h["buckets"])
            out.append(f'{name}_bucket{{kind="{k}",le="+Inf"}} {fmt(n)}')
            out.append(f'{name}_sum{{kind="{k}"}} {fmt(h["sum"])}')
            out.append(f'{name}_count{{kind="{k}"}} {fmt(n)}')
        out.append("# TYPE pd_router_first_byte_seconds histogram")
        out.extend(f"pd_router_first_byte_seconds{lab} {fmt(v)}" for lab, v in samples)
        return "\n".join(out) + "\n"

    @staticmethod
    def _bucket(bounds: tuple, v: float) -> int:
        for i, b in enumerate(bounds):
            if v <= b:
                return i
        return len(bounds)

    def _dd_attempt_ctx(self, h: int, body: dict, tokens: float, tail_est: float, ctx: int) -> dict:
        """Context of one direct attempt, from the memory entry that allowed it: numbers and a boolean only."""
        e = self.dd_mem.get(h)
        msgs = body.get("messages") if isinstance(body.get("messages"), list) else []
        return {"new": max(0.0, tokens - e[2]) if e else tokens, "tail": tail_est or 0.0,
                "reply": last_assistant_chars(msgs) / self.dd_cpt,
                "gap": (self._now() - e[1]) if e else 0.0,
                "ctx_changed": bool(e) and len(e) > 4 and e[4] != ctx}

    def _dd_record(self, a: dict, outcome: str):
        st = self.dd_ctx[outcome]
        for k in ("new", "tail", "reply"):
            st[k][self._bucket(DD_BUCKETS, a[k])] += 1
            st["sum"][k] += a[k]
        st["gap"][self._bucket(DD_GAP_BUCKETS, a["gap"])] += 1
        st["sum"]["gap"] += a["gap"]
        st["ctx_changed" if a["ctx_changed"] else "ctx_same"] += 1

    def _dd_observe(self, which: str, est: float):
        for b in DD_BUCKETS:
            if est <= b:
                self.dd[which][f"le_{b}"] += 1
                return
        self.dd[which][f"gt_{DD_BUCKETS[-1]}"] += 1

    # -- routing
    @staticmethod
    def _stable_prefix(msgs: list) -> list:
        # Messages up to and including the first user message: they do not change as the conversation grows.
        # With no user message at all the first two messages are used.
        for i, m in enumerate(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return msgs[:i + 1]
        return msgs[:2]

    @staticmethod
    def affinity(body: dict) -> int:
        # stdlib json on purpose: same hashing as the r2 router. The key is the conversation's stable prefix (the messages
        # before the first user message plus the first user message), so every turn of a conversation maps to the same ranks.
        msgs = body.get("messages")
        prefix = Router._stable_prefix(msgs) if isinstance(msgs, list) else body.get("prompt", "")
        salt = body.get("cache_salt")
        # A request with a cache_salt is keyed by salt and prefix, so callers that cannot share an engine cache never
        # share a rank pick or a decode-direct memory entry. Without a salt the key is the r3 key.
        key = json.dumps(prefix if salt is None else {"cache_salt": salt, "prefix": prefix}, sort_keys=True)
        return int(hashlib.sha1(key.encode()).hexdigest(), 16)

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _mem_get(self, h: int):
        e = self.dd_mem.get(h)
        if e is not None:
            if self._now() - e[1] > self.dd_mem_ttl:
                del self.dd_mem[h]
                self.dd["mem_evicted"] += 1
                return None
            self.dd_mem.move_to_end(h)
        return e

    def _hist_observe(self, h: int, msgs: list):
        """Record how this request's history relates to the entry left by the previous request (counts only)."""
        e = self.dd_mem.get(h)
        live = e is not None and self._now() - e[1] <= self.dd_mem_ttl
        kind, est = history_kind(e[5] if live and len(e) > 5 else None, msgs, self.dd_cpt)
        self.hist_kind[kind] += 1
        st = self.hist_diverged[kind]
        st["buckets"][self._bucket(DD_BUCKETS, est)] += 1
        st["sum"] += est

    def _mem_served(self, h: int, d: int, tokens: float, ctx: int = 0, hist: bytes | None = None):
        self.dd_mem[h] = [d, self._now(), tokens, False, ctx, hist]
        self.dd_mem.move_to_end(h)
        while len(self.dd_mem) > self.dd_mem_cap:
            self.dd_mem.popitem(last=False)
            self.dd["mem_evicted"] += 1

    def _mem_rejected(self, h: int):
        e = self.dd_mem.get(h)
        if e is not None:
            e[3] = True

    def _mem_skip_reason(self, h: int, ctx: int | None = None):
        """None when a direct attempt is worth making, else one of DD_SKIP_REASONS."""
        e = self._mem_get(h)
        if e is None:
            return "no_entry"
        if self._now() - e[1] > self.dd_max_age:
            return "stale"
        if e[0] != self.pick("decode", h):
            return "rank_changed"
        if e[3]:
            return "rejected_prev"
        if ctx is not None and len(e) > 4 and e[4] != ctx:
            return "context_changed"  # the whole prompt is uncached on D, so a direct attempt can only be refused
        return None

    @staticmethod
    def _est_tokens(path: str, body: dict, cpt: float) -> float:
        msgs = body.get("messages")
        if isinstance(msgs, list):
            return sum(msg_chars(m) for m in msgs) / cpt
        pr = body.get("prompt")
        return (len(pr) / cpt) if isinstance(pr, (str, list)) else 0.0

    def pick(self, role: str, h: int) -> int:
        load = self.inflight[role]
        pref = h % len(load)
        if self.strict_affinity:
            return pref
        least = min(range(len(load)), key=lambda i: load[i])
        return pref if load[pref] - load[least] <= self.max_imbalance else least

    # -- cancellation helper
    async def guarded(self, req: Request, task: asyncio.Task, timeout: float):
        """Wait for task; cancel it (closing its upstream connection) if the client hangs up or the time budget ends."""
        deadline = time.monotonic() + timeout if timeout > 0 else None
        try:
            while True:
                wait = POLL_S if deadline is None else max(0.0, min(POLL_S, deadline - time.monotonic()))
                done, _ = await asyncio.wait({task}, timeout=wait)
                if done:
                    return task.result()
                if await req.is_disconnected():
                    raise Gone()
                if deadline is not None and time.monotonic() >= deadline:
                    raise Timeout()
        except BaseException:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
            elif not task.cancelled() and task.exception() is None:
                # The task finished between the poll and the disconnect/deadline check: nobody will use its result.
                self._discard(task.result())
            raise

    @staticmethod
    def _discard(result):
        """Close an upstream response that was produced but will not be used."""
        for item in result if isinstance(result, tuple) else (result,):
            close = getattr(item, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

    # -- legs
    async def _prefill_leg(self, idx: int, path: str, data: bytes, headers: dict):
        async with self.session().post(f"http://{self.prefill[idx]}{path}", data=data, headers=headers) as r:
            raw = await read_capped(r.content, ERR_CAP) if r.status != 200 else await r.read()
            return r.status, raw, r.headers.get("retry-after")

    async def _decode_start(self, idx: int, path: str, data: bytes, headers: dict):
        """POST leg 2 and wait for the first body byte. Returns (response, first_chunk)."""
        r = await self.session().post(f"http://{self.decode[idx]}{path}", data=data, headers=headers)
        try:
            if r.status != 200:
                return r, await read_capped(r.content, ERR_CAP)
            return r, await r.content.readany()
        except BaseException:
            r.close()
            raise

    async def handle(self, req: Request, path: str) -> Response:
        t_in = time.perf_counter()
        c = self.counts
        c["requests"] += 1
        try:
            body = jloads(await req.body())
            if not isinstance(body, dict):
                raise ValueError
        except Exception:
            c["bad_request"] += 1
            return engine_error(400, "request body must be a JSON object", "BadRequestError")
        headers = {k: v for k, v in req.headers.items() if k.lower() in FWD_HEADERS}
        headers["content-type"] = "application/json"
        if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
            c["bad_request"] += 1
            return engine_error(400, "only n = 1 is supported", "BadRequestError")
        prompt = body.get("prompt")
        if path == "/v1/completions" and isinstance(prompt, list) and not all(
                type(t) is int for t in prompt):
            c["bad_request"] += 1
            return engine_error(400, "batched prompts are not supported; send one prompt per request",
                                "BadRequestError")
        h = self.affinity(body)
        if isinstance(body.get("messages"), list):
            rf = reasoning_field(body["messages"])
            if rf is not None:
                self.reasoning_field[rf] += 1
        # Never accept transfer parameters from a client, on any path: leg 2 gets only what the prefill engine returned.
        body.pop("kv_transfer_params", None)
        if isinstance(body.get("vllm_xargs"), dict):
            body["vllm_xargs"].pop("kv_transfer_params", None)

        pd_label = "pd"
        tokens = self._est_tokens(path, body, self.dd_cpt) if self.ddirect else 0.0
        ctx = ctx_hash(body) if self.ddirect else 0
        hist = None
        if self.ddirect and isinstance(body.get("messages"), list):
            self._hist_observe(h, body["messages"])
            hist = history_pack(body["messages"])
        if self.ddirect:
            reason, tail_est, chunk_est = ddirect_classify(path, body, self.dd_cpt, self.dd_max_tail, self.dd_block,
                                                           self.dd_max_chunk, self.dd_count_reply)
            if tail_est is not None:
                self._dd_observe("tail_est", tail_est)
                self._dd_observe("chunk_est", chunk_est)
            if reason is not None:
                self.dd["ineligible_reason"][reason] += 1
            else:
                self.dd["eligible"] += 1
                skip = self._mem_skip_reason(h, ctx)
                if skip is not None:
                    self.dd["skip_reason"][skip] += 1
                elif self.inflight["decode"][self.pick("decode", h)] >= self.dd_max_inflight:
                    self.dd["guard_fallback"] += 1
                else:
                    self.dd["sent_direct"] += 1
                    # client body as received (transfer params already dropped): no max_tokens override, no leg 1
                    attempt = self._dd_attempt_ctx(h, body, tokens, tail_est, ctx)
                    resp = await self._decode_leg(req, path, dict(body), headers, h, self._path_header("direct"),
                                                  direct=True, tokens=tokens, t_in=t_in, path_label="direct", ctx=ctx, hist=hist)
                    self._dd_record(attempt, "rejected" if resp is DD_REJECTED else
                                    "accepted" if getattr(resp, "status_code", 0) == 200 else "error")
                    if resp is not DD_REJECTED:
                        return resp
                    # D refused it before streaming anything (admission 429): run the normal P/D path instead.
                    self.dd["direct_rejected_fallback"] += 1
                    self._mem_rejected(h)
                    pd_label = "pd-fb"

        pre = dict(body)
        pre.update(stream=False, max_tokens=1, kv_transfer_params={"do_remote_decode": True})
        for k in POP_PREFILL + (POP_STRUCT if self.drop_struct else []):
            if k != "kv_transfer_params":
                pre.pop(k, None)

        # ---- leg 1
        pre_data = jdumps(pre)
        p = self.pick("prefill", h)
        self.inflight["prefill"][p] += 1
        t0 = time.perf_counter()
        task = asyncio.ensure_future(self._prefill_leg(p, path, pre_data, headers))
        try:
            status, raw, retry_after = await self.guarded(req, task, self.prefill_timeout)
        except Gone:
            c["client_gone_in_prefill"] += 1
            return Response(status_code=499)
        except Timeout:
            c["prefill_fail"] += 1
            log.warning("prefill leg timeout backend=%d", p)
            return engine_error(504, "prefill leg timed out", "GatewayTimeoutError")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            c["prefill_fail"] += 1
            log.warning("prefill leg failed backend=%d kind=%s", p, type(e).__name__)
            return engine_error(502, "prefill engine unreachable", "BadGatewayError")
        finally:
            self.inflight["prefill"][p] -= 1
        t1 = time.perf_counter()
        self.t_prefill.append(t1 - t0)
        if status != 200:
            c["prefill_fail"] += 1
            log.warning("prefill leg status=%d backend=%d", status, p)
            return upstream_error(status, raw, retry_after)
        try:
            kvp = jloads(raw).get("kv_transfer_params")
        except Exception:
            kvp = None
        if not isinstance(kvp, dict) or not kvp:
            c["prefill_fail"] += 1
            log.warning("prefill leg returned no transfer parameters backend=%d", p)
            return engine_error(502, "prefill engine returned no KV transfer parameters", "BadGatewayError")
        del raw

        # Client left between the legs: do not start a decode whose KV pull nobody wants.
        if await req.is_disconnected():
            c["client_gone_in_prefill"] += 1
            return Response(status_code=499)

        # ---- leg 2
        body["kv_transfer_params"] = kvp
        resp_headers = {**self._path_header(pd_label), "x-pd-prefill-ms": str(round((t1 - t0) * 1000, 1))}
        return await self._decode_leg(req, path, body, headers, h, resp_headers, tokens=tokens, t_in=t_in,
                                      path_label=pd_label, ctx=ctx, hist=hist)

    async def _decode_leg(self, req: Request, path: str, body: dict, headers: dict, h: int, resp_headers: dict,
                          direct: bool = False, tokens: float = 0.0, t_in: float | None = None,
                          path_label: str = "pd", ctx: int = 0, hist: bytes | None = None):
        """Send `body` to the decode rank for hash h and stream the reply back. Used by leg 2 and by decode-direct."""
        c = self.counts
        self.path_requests[path_label] += 1
        dec_data = jdumps(body)
        d = self.pick("decode", h)
        self.inflight["decode"][d] += 1
        released = False

        def release():
            nonlocal released
            if not released:
                released = True
                self.inflight["decode"][d] -= 1

        timeout = self.kv_timeout if body.get("stream") else self.nonstream_timeout
        task = asyncio.ensure_future(self._decode_start(d, path, dec_data, headers))
        t2 = time.perf_counter()
        self.decode_waiting += 1
        try:
            try:
                r, first = await self.guarded(req, task, timeout)
            finally:
                self.decode_waiting -= 1
        except Gone:
            c["client_gone_in_decode"] += 1
            release()
            return Response(status_code=499)
        except Timeout:
            c["decode_first_byte_timeout"] += 1
            c["decode_fail"] += 1
            self.dd["direct_errors"] += direct
            log.warning("decode first byte timeout backend=%d after_ms=%d", d, int((time.perf_counter() - t2) * 1000))
            release()
            return engine_error(504, "KV transfer or decode did not produce a first byte in time",
                                "GatewayTimeoutError")
        except asyncio.CancelledError:
            release()
            raise
        except Exception as e:
            c["decode_fail"] += 1
            self.dd["direct_errors"] += direct
            log.warning("decode leg failed backend=%d kind=%s", d, type(e).__name__)
            release()
            return engine_error(502, "decode engine unreachable", "BadGatewayError")
        if not direct:
            self.t_dfirst.append(time.perf_counter() - t2)
        ctype = r.headers.get("content-type", "application/json")
        if direct:
            rejected = r.status in DD_FALLBACK_STATUS
            if r.status == 200 and first:
                done, code = _sse_scan(first)
                if not done:
                    buf = bytearray(first)
                    eof = True
                    try:
                        await self.guarded(req, asyncio.ensure_future(self._fill_first_event(r, buf)), FIRST_EVENT_S)
                    except Timeout:
                        eof = False  # stalled: pass on what was read, as is
                    except Gone:
                        c["client_gone_in_decode"] += 1
                        r.close()
                        release()
                        return Response(status_code=499)
                    except asyncio.CancelledError:
                        r.close()
                        release()
                        raise
                    except Exception as e:
                        c["decode_fail"] += 1
                        self.dd["direct_errors"] += 1
                        log.warning("direct first event read failed backend=%d kind=%s", d, type(e).__name__)
                        r.close()
                        release()
                        return engine_error(502, "decode engine stream broke", "BadGatewayError")
                    first = bytes(buf)
                    code = _sse_scan(first, eof=eof)[1]
                rejected = rejected or code in DD_FALLBACK_STATUS
            if rejected:
                r.close()
                release()
                return DD_REJECTED
        if r.status != 200:
            c["decode_fail"] += 1
            self.dd["direct_errors"] += direct
            log.warning("decode leg status=%d backend=%d", r.status, d)
            retry_after = r.headers.get("retry-after")
            r.close()
            release()
            return upstream_error(r.status, first, retry_after)

        if t_in is not None:
            self._fb_observe(path_label, time.perf_counter() - t_in)

        async def gen():
            try:
                if first:
                    yield first
                async for chunk in r.content.iter_any():
                    yield chunk
                if self.ddirect:
                    self._mem_served(h, d, tokens, ctx, hist)  # completed response: remember the rank that holds the cache
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                c["decode_fail"] += 1
                self.dd["direct_errors"] += direct
                log.warning("decode stream broke backend=%d kind=%s", d, type(e).__name__)
                raise StreamAborted from None  # abort the client connection: a truncated stream must not look complete
            finally:
                cleanup()

        def cleanup():
            r.close()
            release()

        return GuardedStream(gen(), cleanup, status_code=200, media_type=ctype.split(";")[0],
                             headers=resp_headers)

    @staticmethod
    async def _fill_first_event(r, buf: bytearray) -> None:
        """Read on into buf until the first real SSE event is complete, EOF, or FIRST_EVENT_CAP bytes. buf keeps what
        was read even if this task is cancelled (stall timeout), so the caller can pass it through unchanged."""
        while len(buf) < FIRST_EVENT_CAP and not _sse_scan(bytes(buf))[0]:
            more = await r.content.readany()
            if not more:
                break
            buf += more

    # -- health / metrics
    async def _get(self, addr: str, path: str, timeout: float | None = None):
        try:
            async with self.session().get(
                    f"http://{addr}{path}",
                    timeout=aiohttp.ClientTimeout(total=self.health_timeout if timeout is None else timeout)) as r:
                return r.status, await r.text()
        except Exception:
            return None, ""

    async def health(self) -> Response:
        targets = [("prefill", i, a) for i, a in enumerate(self.prefill)] + \
                  [("decode", i, a) for i, a in enumerate(self.decode)]
        res = await asyncio.gather(*[self._get(a, "/health") for _, _, a in targets])
        up = {"prefill": 0, "decode": 0}
        for (role, _, _), (st, _) in zip(targets, res):
            up[role] += st == 200
        ok = up["prefill"] == len(self.prefill) and up["decode"] == len(self.decode)
        return JSONResponse({"status": "ok" if ok else "unhealthy",
                             "prefill": {"up": up["prefill"], "total": len(self.prefill)},
                             "decode": {"up": up["decode"], "total": len(self.decode)}},
                            status_code=200 if ok else 503)

    async def metrics(self, per_rank: bool) -> Response:
        targets = [("prefill", i, a) for i, a in enumerate(self.prefill)] + \
                  [("decode", i, a) for i, a in enumerate(self.decode)]
        res = await asyncio.gather(*[self._get(a, "/metrics", self.metrics_timeout) for _, _, a in targets])
        scrapes = [(role, i, text) for (role, i, _), (st, text) in zip(targets, res) if st == 200]
        if len(scrapes) < len(targets):
            # A sum over fewer ranks would read as low load; fail the scrape instead of under-counting.
            return engine_error(503, f"metrics incomplete: {len(scrapes)} of {len(targets)} backends answered",
                                "ServiceUnavailableError")
        text = aggregate_metrics(scrapes, per_rank)
        own = [("pd_router_decode_waiting_first_byte", "gauge", self.decode_waiting),
               ("pd_router_backends_up", "gauge", len(scrapes)), ("pd_router_backends_total", "gauge", len(targets)),
               ("pd_router_requests_total", "counter", self.counts["requests"]),
               ("pd_router_prefill_failures_total", "counter", self.counts["prefill_fail"]),
               ("pd_router_decode_failures_total", "counter", self.counts["decode_fail"]),
               ("pd_router_decode_first_byte_timeouts_total", "counter", self.counts["decode_first_byte_timeout"]),
               ("pd_router_client_disconnects_total", "counter",
                self.counts["client_gone_in_prefill"] + self.counts["client_gone_in_decode"])]
        for name, typ, v in own:
            text += f"# TYPE {name} {typ}\n{name} {v}\n"
        text += self.own_metrics()
        return Response(text, media_type="text/plain; version=0.0.4")

    async def passthrough(self, path: str) -> Response:
        st, text = await self._get(self.decode[0], "/" + path)
        if st is None:
            return engine_error(502, "decode engine unreachable", "BadGatewayError")
        return Response(text, status_code=st, media_type="application/json")


def create_app(prefill, decode, **kw) -> FastAPI:
    router = Router(prefill, decode, **kw)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        stats = None
        if router.stats_log_s > 0:
            async def loop():
                while True:
                    await asyncio.sleep(router.stats_log_s)
                    if router.counts["requests"]:
                        log.warning("pd_stats %s", router.summary())
            stats = asyncio.ensure_future(loop())
        yield
        if stats:
            stats.cancel()
        await router.close()

    app = FastAPI(lifespan=lifespan)
    app.state.router = router

    @app.exception_handler(Exception)
    async def boom(request: Request, exc: Exception):
        log.warning("unhandled kind=%s", type(exc).__name__)
        return engine_error(500, "internal router error", "InternalServerError")

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        return await router.handle(req, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def comp(req: Request):
        return await router.handle(req, "/v1/completions")

    @app.get("/health")
    async def health():
        return await router.health()

    @app.get("/v1/metrics")
    @app.get("/metrics")
    async def metrics(per_rank: int = 0):
        return await router.metrics(bool(per_rank))

    @app.get("/pd_stats")
    async def pd_stats():
        return JSONResponse(router.summary())

    @app.get("/ddirect_stats")
    async def ddirect_stats():
        return JSONResponse(router.ddirect_summary())

    @app.get("/{p:path}")
    async def passthrough(p: str):
        if p not in GET_PASSTHROUGH:
            return engine_error(404, "not found", "NotFoundError")
        return await router.passthrough(p)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", nargs="+", required=True)  # host:port ...
    ap.add_argument("--decode", nargs="+", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=31150)
    ap.add_argument("--max-imbalance", type=int, default=int(os.environ.get("PD_MAX_IMBALANCE", "64")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        app = create_app(args.prefill, args.decode, max_imbalance=args.max_imbalance)
    except ValueError as e:
        ap.error(str(e))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
