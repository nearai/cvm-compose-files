"""Real-Scheduler / EngineCore-hook / HTTP-handler tests for patch 25 (CPU only).

Needs vllm with patch 25 applied; run in the image.
Env: P26_UTILS (vLLM tests/v1/core/utils.py, default /t/core_utils.py),
P26_MODEL (local model dir for ModelConfig, default /models/qwen36).
The scheduler under test is the STOCK Scheduler: patch 25 does not edit it.
"""

import asyncio
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

for _m in ("tests", "tests.v1", "tests.v1.kv_connector", "tests.v1.kv_connector.unit"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
_ku = types.ModuleType("tests.v1.kv_connector.unit.utils")


@dataclass
class MockKVConfig:
    matched_tokens: int = 0
    is_async: bool = False
    num_defers_before_matching: int = 0


_ku.MockKVConfig = MockKVConfig
sys.modules["tests.v1.kv_connector.unit.utils"] = _ku


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


U = _load("p26_core_utils", os.environ.get("P26_UTILS", "/t/core_utils.py"))
MODEL = os.environ.get("P26_MODEL", "/models/qwen36")

import pytest

from vllm.platforms import current_platform

if not current_platform.device_type:
    current_platform.device_type = "cuda"

from vllm.exceptions import EngineOverloadedError
from vllm.outputs import RequestError
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.admission_p26 import (
    REASON_LONG,
    REASON_LONG_YIELD,
    REASON_SHORT,
    AdmissionP26,
)
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

B = 8192
CUTOFF = 8000  # scaled-down stand-in for 65536


def mk(**kw):
    args = dict(
        model=MODEL,
        max_num_batched_tokens=B,
        long_prefill_token_threshold=4864,
        max_model_len=200000,
        num_blocks=100000,
        max_num_seqs=64,
        enable_prefix_caching=True,
    )
    args.update(kw)
    return U.create_scheduler(**args)


def req(rid, n, kvp=None, same_prompt=False):
    sp = None
    if kvp is not None:
        sp = SamplingParams(max_tokens=1, extra_args={"kv_transfer_params": kvp})
    return U.create_requests(
        1, num_tokens=n, max_tokens=1, req_ids=[rid], sampling_params=sp
    )[0]


def submit(s, ctl, r):
    """What EngineCoreProc does on ADD: verdict first, add_request if admitted."""
    v = ctl.verdict(s, r)
    if v is None:
        s.add_request(r)
    return v


def step(s):
    out = s.schedule()
    ids = list(out.num_scheduled_tokens)
    toks = [
        [0] if s.requests[i].num_computed_tokens >= s.requests[i].num_prompt_tokens else []
        for i in ids
    ]
    s.update_from_output(out, ModelRunnerOutput(req_ids=ids, sampled_token_ids=toks))
    return dict(out.num_scheduled_tokens)


def drain(s, n=100):
    for _ in range(n):
        if not s.requests:
            return
        step(s)
    raise AssertionError("did not drain")


def ctl(short=40960, long=2, yield_short=0):
    return AdmissionP26(short, long, CUTOFF, long_yield_short_tokens=yield_short)


def test_cached_prompt_is_classed_short_by_uncached_tokens():
    s = mk()
    c = ctl(long=1)
    assert submit(s, c, req("warm", 20000)) is None
    drain(s)  # prompt now in the prefix cache (identical prompts: same token ids)
    r = req("again", 20000)
    assert submit(s, c, r) is None
    assert r.p26_long is False and r.p26_uncached < CUTOFF  # 20000 raw > cutoff
    # the cached one did not take the single long slot (long=1 here)
    assert c.backlog(s)[1] == 0


def test_probe_is_side_effect_free_and_reject_leaves_no_state():
    s = mk(enable_prefix_caching=False)
    c = ctl(short=10000, long=1)
    assert submit(s, c, req("a", 7000)) is None  # short (<= CUTOFF 8000)
    free0 = s.kv_cache_manager.block_pool.get_num_free_blocks()
    nreq0, nwait0 = len(s.requests), len(s.waiting)
    r = req("b", 7000)
    assert c.verdict(s, r) == REASON_SHORT
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == free0
    assert (len(s.requests), len(s.waiting)) == (nreq0, nwait0)
    assert "b" not in s.requests
    assert c.rejected[REASON_SHORT] == 1


def test_long_yields_to_short_backlog_but_shorts_still_admitted():
    s = mk(enable_prefix_caching=False)
    c = ctl(short=40000, long=0, yield_short=10000)
    assert submit(s, c, req("S1", 6000)) is None
    assert submit(s, c, req("S2", 6000)) is None  # short backlog 12000 > 10000
    assert submit(s, c, req("L1", 20000)) == REASON_LONG_YIELD
    assert submit(s, c, req("S3", 6000)) is None  # shorts only hit their own limit
    assert c.rejected[REASON_LONG_YIELD] == 1 and c.rejected[REASON_LONG] == 0
    drain(s)
    assert submit(s, c, req("L1", 20000)) is None  # backlog gone


def test_long_tokens_do_not_count_toward_yield():
    s = mk(enable_prefix_caching=False)
    c = ctl(short=0, long=0, yield_short=10000)
    assert submit(s, c, req("L1", 20000)) is None
    assert submit(s, c, req("L2", 20000)) is None  # no short backlog: longs unlimited


def test_lone_short_larger_than_budget_admitted():
    s = mk(enable_prefix_caching=False)
    assert submit(s, ctl(short=1000), req("a", 6000)) is None


def test_long_gate_counts_waiting_and_running_then_releases_on_finish():
    s = mk(enable_prefix_caching=False)
    c = ctl(long=2)
    assert submit(s, c, req("L1", 20000)) is None
    step(s)  # L1 running, mid-prefill
    assert submit(s, c, req("L2", 20000)) is None  # L2 still waiting
    assert submit(s, c, req("L3", 20000)) == REASON_LONG
    s.finish_requests(["L1"], RequestStatus.FINISHED_ABORTED)  # client disconnect
    assert "L1" not in s.requests
    assert submit(s, c, req("L3", 20000)) is None


def test_short_budget_follows_progress_and_finish_error():
    s = mk(enable_prefix_caching=False)
    c = ctl(short=10000, long=0)
    assert submit(s, c, req("a", 7000)) is None
    assert submit(s, c, req("b", 3000)) is None
    assert submit(s, c, req("c", 500)) == REASON_SHORT
    s.finish_requests(["a"], RequestStatus.FINISHED_ERROR)  # KV-transfer failure
    assert submit(s, c, req("c", 500)) is None
    drain(s)
    assert c.backlog(s) == (0, 0)


def test_finished_request_pending_lease_not_counted():
    s = mk(enable_prefix_caching=False)
    c = ctl(short=10000, long=0)
    r = req("a", 9000)
    assert submit(s, c, r) is None
    r.status = RequestStatus.FINISHED_STOPPED  # as a P request awaiting NIXL lease
    assert c.backlog(s) == (0, 0)


def test_decode_side_request_never_shed():
    s = mk(enable_prefix_caching=False)
    c = ctl(short=1, long=1)
    kvp = {"do_remote_prefill": True, "remote_engine_id": "x"}
    for i in range(3):
        r = req(f"d{i}", 30000, kvp=kvp)
        assert c.verdict(s, r) is None
        s.add_request(r)
    assert c.backlog(s) == (0, 0)


def test_off_by_default(monkeypatch):
    for n in ("SHORT_MAX_TOKENS", "LONG_MAX_INFLIGHT"):
        monkeypatch.delenv(f"VLLM_ADMIT_{n}", raising=False)
    s = mk(enable_prefix_caching=False)
    c = AdmissionP26.from_env(s)
    assert not c.enabled
    for i in range(5):
        assert submit(s, c, req(f"x{i}", 30000)) is None


def _with_role(s, kv_role):
    """Real scheduler, with the role a P/D connector config would carry.

    MockKVConnector (vLLM's test tree) is not shipped in the image, so the role
    is injected on the scheduler's vllm_config view instead of a real connector.
    """
    from vllm.config import KVTransferConfig

    s.vllm_config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(kv_connector="NixlConnector", kv_role=kv_role)
    )
    return s


@pytest.mark.parametrize(
    "role,forced,enabled",
    [
        ("kv_consumer", None, False),
        ("kv_both", None, False),
        ("kv_producer", None, True),
        ("kv_both", "prefill", True),
        ("kv_producer", "off", False),
    ],
)
def test_role_gate(monkeypatch, role, forced, enabled):
    monkeypatch.setenv("VLLM_ADMIT_SHORT_MAX_TOKENS", "1000")
    monkeypatch.setenv("VLLM_ADMIT_LONG_YIELD_SHORT_TOKENS", "500")
    monkeypatch.delenv("VLLM_ADMIT_ROLE", raising=False)
    if forced:
        monkeypatch.setenv("VLLM_ADMIT_ROLE", forced)
    s = _with_role(mk(enable_prefix_caching=False), role)
    assert AdmissionP26.from_env(s).enabled is enabled


# ---- EngineCore hook (real EngineCoreProc methods, stub self) ----

def _core_cls():
    import vllm.v1.engine.core as core

    for name in ("EngineCoreProc", "EngineCore"):
        cls = getattr(core, name, None)
        if cls is not None and hasattr(cls, "_reject_add_overloaded"):
            return cls
    raise AssertionError("_reject_add_overloaded not found; patch not applied")


def _fake_core(s, calls):
    cls = _core_cls()
    sent = []
    fake = SimpleNamespace(scheduler=s, sent=sent)
    fake._send_finish_outputs_to_client = lambda ids, ci, fr, err: sent.append(
        (ids, ci, fr, err)
    )
    fake._reject_add_in_shutdown = lambda r: False
    fake._reject_add_while_paused = lambda r: False
    fake._reject_add_overloaded = lambda r: cls._reject_add_overloaded(fake, r)
    fake.add_request = lambda r, w=0: (calls.append(r.request_id), s.add_request(r))
    return cls, fake


def test_core_add_dispatch_rejects_before_add_request(monkeypatch):
    monkeypatch.setenv("VLLM_ADMIT_SHORT_MAX_TOKENS", "10000")
    monkeypatch.setenv("VLLM_ADMIT_LONG_MAX_INFLIGHT", "1")
    monkeypatch.setenv("VLLM_ADMIT_LONG_CUTOFF_TOKENS", str(CUTOFF))
    monkeypatch.setenv("VLLM_ADMIT_ROLE", "prefill")  # no kv_producer in this stub
    from vllm.v1.engine import EngineCoreRequestType

    s = mk(enable_prefix_caching=False)
    calls = []
    cls, fake = _fake_core(s, calls)
    ADD = EngineCoreRequestType.ADD
    cls._handle_client_request(fake, ADD, (req("L1", 20000), 0))
    cls._handle_client_request(fake, ADD, (req("L2", 20000), 0))
    assert calls == ["L1"] and "L2" not in s.requests
    ids, ci, fr, err = fake.sent[0]
    assert ids == ["L2"] and fr == FinishReason.ERROR
    assert err.code == REASON_LONG and err.retryable
    # do_remote_prefill (decode side) is never shed
    cls._handle_client_request(
        fake, ADD, (req("D", 20000, kvp={"do_remote_prefill": True}), 0)
    )
    assert calls == ["L1", "D"]
    # unset knobs -> inert
    monkeypatch.delenv("VLLM_ADMIT_SHORT_MAX_TOKENS")
    monkeypatch.delenv("VLLM_ADMIT_LONG_MAX_INFLIGHT")
    s2 = mk(enable_prefix_caching=False)
    calls2 = []
    cls, fake2 = _fake_core(s2, calls2)
    for i in range(4):
        cls._handle_client_request(fake2, ADD, (req(f"z{i}", 20000), 0))
    assert len(calls2) == 4 and not fake2.sent


# ---- frontend: RequestError -> exception -> HTTP 429 + Retry-After ----

def _http(code):
    from vllm.entrypoints.serve.exception_handling.handlers.vllm_error import (
        vllm_error_handler,
    )

    exc = RequestError(code=code, message="x", retryable=True).to_exception()
    fake_req = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(args=SimpleNamespace(log_error_stack=False))),
        state=SimpleNamespace(),
    )
    return exc, asyncio.run(vllm_error_handler(fake_req, exc))


@pytest.mark.parametrize(
    "code,ra", [(REASON_SHORT, "1"), (REASON_LONG, "10"), (REASON_LONG_YIELD, "10")]
)
def test_http_429_retry_after_and_phrase(code, ra, monkeypatch):
    monkeypatch.delenv("VLLM_ADMIT_RETRY_AFTER_SHORT_S", raising=False)
    monkeypatch.delenv("VLLM_ADMIT_RETRY_AFTER_LONG_S", raising=False)
    exc, resp = _http(code)
    assert isinstance(exc, EngineOverloadedError) and exc.reason == code
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == ra
    body = json.loads(resp.body)
    assert "engine has reached its prefill token backlog limit" in body["error"]["message"]
    assert body["error"]["code"] == 429


def test_retry_after_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_ADMIT_RETRY_AFTER_SHORT_S", "3")
    monkeypatch.setenv("VLLM_ADMIT_RETRY_AFTER_LONG_S", "bad")
    assert _http(REASON_SHORT)[1].headers["retry-after"] == "3"
    assert _http(REASON_LONG)[1].headers["retry-after"] == "10"


def test_serving_layer_error_response_keeps_retry_after(monkeypatch):
    """Non-streaming path: an ErrorResponse built by serving still gets the header."""
    from vllm.entrypoints.serve.exception_handling.error_response import (
        create_error_response,
    )
    from vllm.entrypoints.serve.exception_handling.handlers.exception import (
        error_response_headers,
    )

    monkeypatch.delenv("VLLM_ADMIT_RETRY_AFTER_LONG_S", raising=False)
    err = create_error_response(EngineOverloadedError(REASON_LONG))
    assert err.error.code == 429
    assert error_response_headers(err) == {"Retry-After": "10"}
    assert error_response_headers(create_error_response("bad")) is None


def test_other_request_errors_unchanged():
    exc = RequestError(code="engine_paused", message="p", retryable=True).to_exception()
    assert not isinstance(exc, EngineOverloadedError)


def test_metric_reasons_registered():
    from vllm.v1.metrics.loggers import ADMISSION_REASONS

    assert {REASON_SHORT, REASON_LONG, REASON_LONG_YIELD} <= set(ADMISSION_REASONS)
