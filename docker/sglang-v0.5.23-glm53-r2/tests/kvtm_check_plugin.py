"""pytest plugin for test-cpu.sh step 9: run the upstream unified radix cache tests on CPU with the KV
tier recorder forced on in strict mode (a hook error fails the test), then fail the session unless the
recorder existed and its VRAM-eviction and load-back hooks fired.

Most of that test file builds CUDA tensors. In a CPU-only container those tests stop at torch's lazy
CUDA init ("Found no NVIDIA driver"); they are reported as skipped here, and any other failure still
fails. MIN_PASSED guards against everything being skipped (about 490 pass on CPU).

MOCK_ARENA_TESTS are deselected: they drive the tree with a Mock in place of its node arena, which the
strict-mode gauge walk cannot iterate. Outside strict mode that error is logged and they pass."""

import pytest

MIN_PASSED = 400
_NO_CUDA = "Found no NVIDIA driver"
_passed = set()  # node ids; unittest subtests report under their parent test
MOCK_ARENA_TESTS = (
    "TestUnifiedTreeCoreLoadBackPending::test_auxiliary_load_does_not_reuse_full_pending_pin",
    "TestUnifiedTreeCoreLoadBackPending::test_write_back_pending_blocks_reclaim_until_ack",
    "TestUnifiedTreeCoreLoadBackPending::test_write_through_different_anchors_track_duplicate_without_pending",
)


def pytest_configure(config):
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()


def pytest_collection_modifyitems(config, items):
    keep, drop = [], []
    for item in items:
        (drop if item.nodeid.endswith(MOCK_ARENA_TESTS) else keep).append(item)
    if len(drop) != len(MOCK_ARENA_TESTS):
        raise pytest.UsageError(f"expected {len(MOCK_ARENA_TESTS)} Mock-arena tests, found {len(drop)}")
    config.hook.pytest_deselected(items=drop)
    items[:] = keep


def _needs_cuda(excinfo) -> bool:
    exc, seen = excinfo.value, set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if _NO_CUDA in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed and call.excinfo is not None and _needs_cuda(call.excinfo):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, "Skipped: needs a CUDA device")
    elif report.when == "call" and report.passed:
        _passed.add(report.nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    from prometheus_client import REGISTRY

    from sglang.srt.observability import kv_tier_metrics

    totals = {}
    for metric in REGISTRY.collect():
        if not metric.name.startswith("sglang:kv_tier"):
            continue
        for sample in metric.samples:
            if sample.name.endswith("_created") or sample.labels.get("within_s", "inf") != "inf":
                continue
            labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()) if k != "model_name")
            totals[f"{sample.name}{{{labels}}}"] = sample.value
    for key, value in sorted(totals.items()):
        print(f"KVTM {key} {value:g}")

    def total(prefix):
        return sum(v for k, v in totals.items() if k.startswith(prefix + "{"))

    problems = []
    if kv_tier_metrics._recorder is None:
        problems.append("the recorder was never created")
    for name in ("sglang:kv_tier_vram_evicted_tokens_total", "sglang:kv_tier_load_back_tokens_total"):
        if total(name) <= 0:
            problems.append(f"{name} is 0")
    if len(_passed) < MIN_PASSED:
        problems.append(f"only {len(_passed)} tests passed (expected at least {MIN_PASSED})")
    if problems:
        print("KVTM check FAILED: " + "; ".join(problems))
        session.exitstatus = 1
    else:
        print(
            f"KVTM check OK: {len(_passed)} upstream tests passed with the recorder on in strict mode; "
            f"{total('sglang:kv_tier_vram_evicted_tokens_total'):g} VRAM-evicted and "
            f"{total('sglang:kv_tier_load_back_tokens_total'):g} loaded-back tokens recorded"
        )
