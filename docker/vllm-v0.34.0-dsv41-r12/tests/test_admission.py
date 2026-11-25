"""Pure-Python unit tests for patch 25 admission (no GPU, no vllm needed).

Env: P26_ADMISSION (path to admission_p26.py, default: next to this file under
vllm/v1/core/sched/). Run: python -m pytest -q test_admission.py
"""

import importlib.util
import logging
import os
import sys
import types
from types import SimpleNamespace

import pytest

if "vllm.logger" not in sys.modules:
    try:
        import vllm.logger
    except Exception:
        _vl = types.ModuleType("vllm.logger")
        _vl.init_logger = lambda name: logging.getLogger(name)
        sys.modules.setdefault("vllm", types.ModuleType("vllm"))
        sys.modules["vllm.logger"] = _vl

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.environ.get(
    "P26_ADMISSION", os.path.join(_HERE, "vllm/v1/core/sched/admission_p26.py")
)
_spec = importlib.util.spec_from_file_location("admission_p26", _PATH)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


class FakeReq:
    def __init__(self, rid, prompt, computed=0, hit=0, kvp=None, finished=False):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_preemptions = 0
        self.kv_transfer_params = kvp
        self.abort_immediately = False
        self.resumable = False
        self.sampling_params = object()
        self._hit = hit
        self._finished = finished

    def is_finished(self):
        return self._finished


class FakeSched:
    def __init__(self):
        self.requests = {}
        self.kv_cache_manager = SimpleNamespace(
            get_num_cached_tokens=lambda r: r._hit
        )

    def has_request(self, rid):
        return rid in self.requests

    def admit(self, ctl, r):
        v = ctl.verdict(self, r)
        if v is None:
            self.requests[r.request_id] = r
        return v


def ctl(s=40960, l=2, cutoff=65536):
    return A.AdmissionP26(s, l, cutoff)


def test_all_off_by_default(monkeypatch):
    for n in ("SHORT_MAX_TOKENS", "LONG_MAX_INFLIGHT", "LONG_CUTOFF_TOKENS"):
        monkeypatch.delenv(f"VLLM_ADMIT_{n}", raising=False)
    c = A.AdmissionP26.from_env()
    assert not c.enabled
    sch = FakeSched()
    assert c.verdict(sch, FakeReq("x", 10**6)) is None


def test_env_parse(monkeypatch):
    monkeypatch.setenv("VLLM_ADMIT_SHORT_MAX_TOKENS", "123")
    monkeypatch.setenv("VLLM_ADMIT_LONG_MAX_INFLIGHT", "junk")
    c = A.AdmissionP26.from_env()
    assert (c.short_max_tokens, c.long_max_inflight, c.cutoff) == (123, 0, 65536)


def test_lone_short_always_admitted():
    sch, c = FakeSched(), ctl(s=1000)
    assert sch.admit(c, FakeReq("a", 60000)) is None  # > S but nothing queued


def test_short_budget_reject_and_release():
    sch, c = FakeSched(), ctl(s=40960)
    assert sch.admit(c, FakeReq("a", 30000)) is None
    assert sch.admit(c, FakeReq("b", 10000)) is None  # 40000 <= 40960
    assert sch.admit(c, FakeReq("c", 1000)) == A.REASON_SHORT  # 41000 > 40960
    sch.requests["a"].num_computed_tokens = 20000  # progress frees budget
    assert sch.admit(c, FakeReq("c", 1000)) is None
    del sch.requests["b"]  # abort / disconnect
    assert sch.admit(c, FakeReq("d", 10000)) is None


def test_finished_awaiting_lease_not_counted():
    sch, c = FakeSched(), ctl(s=40960)
    sch.requests["a"] = FakeReq("a", 40000, finished=True)
    assert sch.admit(c, FakeReq("b", 40000)) is None
    assert sch.admit(c, FakeReq("c", 40000)) == A.REASON_SHORT


def test_prefill_complete_not_counted():
    sch, c = FakeSched(), ctl(s=40960)
    sch.requests["a"] = FakeReq("a", 40000, computed=40000)
    assert sch.admit(c, FakeReq("b", 40000)) is None
    assert sch.admit(c, FakeReq("c", 1000)) == A.REASON_SHORT


def test_long_gate_and_longs_dont_block_shorts():
    sch, c = FakeSched(), ctl(l=2)
    assert sch.admit(c, FakeReq("L1", 172000)) is None
    assert sch.admit(c, FakeReq("L2", 172000)) is None
    assert sch.admit(c, FakeReq("L3", 172000)) == A.REASON_LONG
    # in-flight longs never count toward the short budget
    assert sch.admit(c, FakeReq("s1", 30000)) is None
    assert sch.admit(c, FakeReq("s2", 10000)) is None
    del sch.requests["L1"]  # finish
    assert sch.admit(c, FakeReq("L3", 172000)) is None


def test_cached_long_is_short():
    sch, c = FakeSched(), ctl(l=1)
    assert sch.admit(c, FakeReq("L1", 172000)) is None
    # 172k prompt with 150k cached -> 22k uncached -> short class
    assert sch.admit(c, FakeReq("c", 172000, hit=150000)) is None
    assert sch.requests["c"].p26_long is False
    assert sch.requests["c"].p26_uncached == 22000
    # short queue counts uncached, not the full prompt
    # second cached short: queued 22000 + own 22000 = 44000 > 40960 -> reject
    assert sch.admit(c, FakeReq("d", 172000, hit=150000)) == A.REASON_SHORT


def test_cutoff_boundary():
    sch, c = FakeSched(), ctl(l=1, cutoff=65536)
    assert sch.admit(c, FakeReq("a", 65536)) is None
    assert sch.requests["a"].p26_long is False  # <= cutoff is short
    assert sch.admit(c, FakeReq("b", 65537)) is None
    assert sch.requests["b"].p26_long is True
    assert sch.admit(c, FakeReq("c", 65537)) == A.REASON_LONG


def test_decode_side_never_shed_or_counted():
    sch, c = FakeSched(), ctl(s=10, l=1)
    kvp = {"do_remote_prefill": True}
    for i in range(5):
        assert sch.admit(c, FakeReq(f"d{i}", 172000, kvp=kvp)) is None
    assert c.backlog(sch) == (0, 0)


def test_do_remote_decode_is_prefill_side():
    sch, c = FakeSched(), ctl(l=1)
    kvp = {"do_remote_decode": True}
    assert sch.admit(c, FakeReq("a", 100000, kvp=kvp)) is None
    assert sch.admit(c, FakeReq("b", 100000, kvp=kvp)) == A.REASON_LONG


def test_preempted_counts_full_remaining():
    sch, c = FakeSched(), ctl(s=40960)
    r = FakeReq("a", 30000, hit=10000)
    assert sch.admit(c, r) is None
    assert c.backlog(sch) == (20000, 0)  # first admission: uncached estimate
    r.num_computed_tokens = 0
    r.num_preemptions = 1  # preempted, cache evicted: recompute everything
    r._hit = 0
    assert c.backlog(sch) == (30000, 0)
    r._hit = 10000  # preempted but prefix still cached: uncached only
    assert c.backlog(sch) == (20000, 0)


def test_short_only_and_long_only_knobs():
    sch = FakeSched()
    c = ctl(s=0, l=1)
    assert sch.admit(c, FakeReq("a", 100000)) is None
    assert sch.admit(c, FakeReq("b", 100000)) == A.REASON_LONG
    for i in range(20):
        assert sch.admit(c, FakeReq(f"s{i}", 60000)) is None


def test_probe_failure_fails_open():
    sch, c = FakeSched(), ctl(s=1, l=1)
    sch.kv_cache_manager = SimpleNamespace(
        get_num_cached_tokens=lambda r: 1 / 0
    )
    assert c.verdict(sch, FakeReq("a", 100)) is None


def test_cached_estimate_refreshed_at_decision_point():
    sch, c = FakeSched(), ctl(s=40960, l=1)
    a = FakeReq("a", 100000)  # admitted as long (nothing cached yet)
    assert sch.admit(c, a) is None and a.p26_long is True
    assert c.backlog(sch) == (0, 1)
    a._hit = 90000  # cache filled while a waits: 10k uncached -> short
    assert c.backlog(sch) == (10000, 0)
    assert a.p26_long is False
    assert sch.admit(c, FakeReq("L", 100000)) is None  # long slot is free again
    a._hit = 0  # cache evicted: long again, not stale-short
    assert c.backlog(sch) == (0, 2)


def test_long_state_lifecycle_admit_prefill_done_preempt_abort():
    sch, c = FakeSched(), ctl(s=0, l=1)
    r = FakeReq("L1", 100000)
    assert sch.admit(c, r) is None and r.p26_long is True  # set on admit
    assert sch.admit(c, FakeReq("L2", 100000)) == A.REASON_LONG
    r.num_computed_tokens = 50000  # mid-prefill: still in flight
    assert c.backlog(sch) == (0, 1)
    r.num_computed_tokens = 100000  # prefill done: cleared, slot freed
    assert c.backlog(sch) == (0, 0) and r.p26_long is None
    assert sch.admit(c, FakeReq("L2", 100000)) is None
    r.num_computed_tokens, r.num_preemptions = 0, 1  # preempted: re-prefill
    assert c.backlog(sch) == (0, 2)
    del sch.requests["L1"]  # abort
    assert c.backlog(sch) == (0, 1)


def test_yield_long_rejected_short_accepted():  # (a)
    sch, c = FakeSched(), A.AdmissionP26(40960, 2, 65536, 20480)
    assert sch.admit(c, FakeReq("s1", 30000)) is None  # backlog 30000 > 20480
    assert sch.admit(c, FakeReq("L", 172000)) == A.REASON_LONG_YIELD
    assert sch.admit(c, FakeReq("s2", 5000)) is None  # short limit not hit
    assert c.rejected[A.REASON_LONG_YIELD] == 1
    assert c.rejected[A.REASON_SHORT] == 0
    sch.requests["s1"].num_computed_tokens = 30000  # backlog drained
    del sch.requests["s2"]
    assert sch.admit(c, FakeReq("L", 172000)) is None


def test_two_cold_longs_do_not_block_shorts():  # (b)
    sch, c = FakeSched(), A.AdmissionP26(40960, 2, 65536, 20480)
    assert sch.admit(c, FakeReq("L1", 172000)) is None
    assert sch.admit(c, FakeReq("L2", 172000)) is None
    assert sch.admit(c, FakeReq("L3", 172000)) == A.REASON_LONG
    for i in range(2):
        assert sch.admit(c, FakeReq(f"s{i}", 15000)) is None
    assert c.backlog(sch) == (30000, 2)  # longs never in the short backlog


def test_big_prompt_mostly_cached_is_short():  # (c)
    sch, c = FakeSched(), A.AdmissionP26(40960, 1, 65536, 20480)
    assert sch.admit(c, FakeReq("L1", 172000)) is None
    assert sch.admit(c, FakeReq("w", 172000, hit=160000)) is None
    assert sch.requests["w"].p26_long is False
    assert c.backlog(sch) == (12000, 1)
    # waiting-queue loop also uses uncached: a long that got cached becomes short
    assert sch.admit(c, FakeReq("x", 172000)) == A.REASON_LONG


def test_yield_knob_zero_is_old_behaviour():  # (d)
    sch, c = FakeSched(), A.AdmissionP26(40960, 2, 65536, 0)
    assert sch.admit(c, FakeReq("s1", 40000)) is None
    assert sch.admit(c, FakeReq("L1", 172000)) is None  # no yield
    assert sch.admit(c, FakeReq("L2", 172000)) is None
    assert sch.admit(c, FakeReq("L3", 172000)) == A.REASON_LONG
    assert sch.admit(c, FakeReq("s2", 5000)) == A.REASON_SHORT


def test_yield_only_enables(monkeypatch):
    monkeypatch.setenv("VLLM_ADMIT_LONG_YIELD_SHORT_TOKENS", "20480")
    c = A.AdmissionP26.from_env()
    assert c.enabled and c.long_yield_short_tokens == 20480


def test_role_gate(monkeypatch):
    monkeypatch.setenv("VLLM_ADMIT_SHORT_MAX_TOKENS", "100")
    monkeypatch.delenv("VLLM_ADMIT_ROLE", raising=False)

    def sched(role):
        kv = None if role is None else SimpleNamespace(kv_role=role)
        return SimpleNamespace(vllm_config=SimpleNamespace(kv_transfer_config=kv))

    assert A.AdmissionP26.from_env(sched("kv_producer")).enabled
    for role in ("kv_consumer", "kv_both", None):
        assert not A.AdmissionP26.from_env(sched(role)).enabled
    monkeypatch.setenv("VLLM_ADMIT_ROLE", "prefill")
    assert A.AdmissionP26.from_env(sched("kv_both")).enabled
    monkeypatch.setenv("VLLM_ADMIT_ROLE", "off")
    assert not A.AdmissionP26.from_env(sched("kv_producer")).enabled


def test_exempt_resumable_and_existing():
    sch, c = FakeSched(), ctl(s=1)
    sch.requests["x"] = FakeReq("x", 100)
    r = FakeReq("y", 100)
    r.resumable = True
    assert c.verdict(sch, r) is None
    assert c.verdict(sch, FakeReq("x", 100)) is None  # session continuation


# VLLM_ADMIT_LONG_REJECT_ALL (patch 30).
def _reject_all_env(monkeypatch, **over):
    for k in list(os.environ):
        if k.startswith("VLLM_ADMIT_"):
            monkeypatch.delenv(k)
    env = dict(
        VLLM_ADMIT_ROLE="prefill",
        VLLM_ADMIT_LONG_CUTOFF_TOKENS="1024",
        VLLM_ADMIT_LONG_REJECT_ALL="1",
        VLLM_ADMIT_SHORT_MAX_TOKENS="4096",
    )
    env.update(over)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return A.AdmissionP26.from_env(FakeSched())


def test_reject_all_refuses_cold_long_when_idle(monkeypatch):
    c = _reject_all_env(monkeypatch)
    assert c.enabled
    assert c.verdict(FakeSched(), FakeReq("a", 30000)) == A.REASON_LONG


def test_reject_all_admits_warm_followup(monkeypatch):
    c = _reject_all_env(monkeypatch)
    assert c.verdict(FakeSched(), FakeReq("a", 30000, hit=29500)) is None
    assert c.verdict(FakeSched(), FakeReq("a", 30000, hit=28976)) is None
    assert c.verdict(FakeSched(), FakeReq("a", 30000, hit=28975)) == A.REASON_LONG


def test_reject_all_exempts_pd_requests(monkeypatch):
    c = _reject_all_env(monkeypatch)
    r = FakeReq("a", 60000, kvp={"do_remote_prefill": True})
    assert c.verdict(FakeSched(), r) is None


def test_reject_all_keeps_short_budget(monkeypatch):
    c = _reject_all_env(monkeypatch)
    sch = FakeSched()
    for i in range(4):
        assert sch.admit(c, FakeReq(f"q{i}", 1000)) is None
    assert c.verdict(sch, FakeReq("n", 500)) == A.REASON_SHORT
    del sch.requests["q3"]
    assert c.verdict(sch, FakeReq("n", 500)) is None


def test_reject_all_off_keeps_limits(monkeypatch):
    c = _reject_all_env(
        monkeypatch, VLLM_ADMIT_LONG_REJECT_ALL="0", VLLM_ADMIT_SHORT_MAX_TOKENS="0"
    )
    assert not c.enabled
    assert c.verdict(FakeSched(), FakeReq("a", 30000)) is None
    c = _reject_all_env(
        monkeypatch, VLLM_ADMIT_LONG_REJECT_ALL="0", VLLM_ADMIT_LONG_MAX_INFLIGHT="1"
    )
    assert c.verdict(FakeSched(), FakeReq("a", 30000)) is None


def test_reject_all_off_on_decode_role(monkeypatch):
    c = _reject_all_env(monkeypatch, VLLM_ADMIT_ROLE="off")
    assert not c.enabled and not c.long_reject_all


@pytest.mark.parametrize(
    "val,expect",
    [("1", True), ("true", True), ("TRUE", True), ("Yes", True), ("on", True),
     ("0", False), ("false", False), ("off", False), ("garbage", False), ("", False)],
)
def test_reject_all_boolean_parsing(monkeypatch, val, expect):
    assert _reject_all_env(monkeypatch, VLLM_ADMIT_LONG_REJECT_ALL=val).long_reject_all is expect
