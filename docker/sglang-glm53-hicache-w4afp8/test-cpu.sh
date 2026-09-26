#!/usr/bin/env bash
set -euo pipefail
# Resolved before the cd: steps 7 and 8 run test files shipped next to this script.
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /sgl-workspace/sglang
# This pinned upstream test utility indexes the first visible-device digit.
# Use an unavailable ordinal, with the CPU container's runtime set to runc.
export CUDA_VISIBLE_DEVICES=9

# 1. INVARIANT: add_chunked_req must never set an extend length larger than what the KV allocator
# can actually serve. Every failure case below was observed in production or the gpu31 lab repro.
python3 - <<'EOF'
import sys
import types

from sglang.srt.managers.schedule_policy import PrefillAdder

PAGE, CHUNK = 64, 4096


class Range:
    def __init__(self, a, b):
        self.start, self.end = a, b

    @property
    def length(self):
        return self.end - self.start


class Req:
    def __init__(self, total, matched):
        self.full_untruncated_fill_ids = [0] * total
        self.prefix_indices = [0] * matched
        self.extend_range = Range(matched, matched)
        self.retracted_stain = False
        self.sampling_params = types.SimpleNamespace(max_new_tokens=4096)

    def set_extend_range(self, a, b):
        self.extend_range = Range(a, b)


class Adder:
    """Minimal stand-in exposing only what add_chunked_req touches."""

    def __init__(self, avail, evictable, offset, waiting=1, need=256):
        self.dllm_config = None
        self.page_size, self.rem_chunk_tokens = PAGE, CHUNK
        self._avail, self._evict = avail, evictable
        self.rem_total_token_offset = offset
        self.waiting_queue_len, self.first_waiting_req_tokens = waiting, need
        self.is_hybrid_swa = self.is_all_swa = False
        self.is_hybrid_ssm_cache = True
        self.rem_swa_tokens = 10 ** 9
        self.prefill_delayer_single_pass = None
        self.can_run_list = []

    @property
    def _pool_available_and_evictable(self):
        return self._avail + self._evict

    @property
    def rem_total_tokens(self):
        return self._pool_available_and_evictable - self.rem_total_token_offset

    def _update_prefill_budget(self, *a, **k):
        pass

    def _mamba_gap_budget_for_req(self, req):
        return 0


CASES = [
    ("gpu02 prod crash 1", 2368, 0, 41000),
    ("gpu02 prod crash 2", 3136, 0, 41000),
    ("gpu02 prod crash 3", 3584, 0, 41000),
    ("gpu02 prod crash 4", 3904, 0, 41000),
    ("gpu31 lab repro 1", 2304, 0, 2328),
    ("gpu31 lab repro 2", 1088, 2304, 3418),
    ("pool completely exhausted", 32, 0, 4096),
]

failures = 0
for label, avail, evictable, offset in CASES:
    adder = Adder(avail, evictable, offset)
    req = Req(total=142000, matched=40000)
    PrefillAdder.add_chunked_req(adder, req)
    asked, servable = req.extend_range.length, avail + evictable
    if asked > servable:
        print(
            f"FAILED {label}: extend={asked} but the allocator can serve {servable} "
            f"(available={avail} evictable={evictable}); alloc_extend would raise",
            file=sys.stderr,
        )
        failures += 1
    else:
        how = "parked for the next pass" if asked == 0 else f"extend={asked}"
        print(f"  ok {label}: {how} (servable {servable})")

adder = Adder(3136, 900000, 60000)
req = Req(total=142000, matched=40000)
PrefillAdder.add_chunked_req(adder, req)
if req.extend_range.length < CHUNK - PAGE:
    print(
        f"FAILED healthy path regressed: extend={req.extend_range.length}, expected about {CHUNK}",
        file=sys.stderr,
    )
    failures += 1
else:
    print(f"  ok healthy prefix cache: extend={req.extend_range.length}")

if failures:
    print(f"step 1 FAILED: {failures} pool-clamp violation(s)", file=sys.stderr)
    sys.exit(1)
print("step 1 OK: add_chunked_req never exceeds what the allocator can serve")
EOF

# 2. The admission-reserve behaviour inherited from the base image is unchanged: the reserve is
# still inert unless SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE is set.
python3 - <<'EOF'
import sglang.srt.managers.schedule_policy as sp
from sglang.srt.managers.scheduler import _admission_reserve_need

assert sp.CHUNKED_PREFILL_ADMISSION_RESERVE == 0, sp.CHUNKED_PREFILL_ADMISSION_RESERVE
assert _admission_reserve_need([], 64, 4096) is None
print("step 2 OK: the inherited admission reserve is still opt-in and inert by default")
EOF

# 3. The W4AFP8 checkpoint exclusion list must reach the actual quantization config API under all
# supported key spellings. Stock SGLang ignores these keys and leaves ignored_layers empty.
python3 - <<'EOF'
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config

excluded = ["model.layers.0.self_attn.b_proj"]
for key in ("modules_to_not_convert", "ignored_layers", "ignore"):
    config = W4AFp8Config.from_config({"quant_method": "w4afp8", key: excluded})
    assert config.ignored_layers == excluded, (key, config.ignored_layers)

default = W4AFp8Config.from_config({"quant_method": "w4afp8"})
assert default.ignored_layers == [], default.ignored_layers
print("step 3 OK: W4AFP8 loader propagates all supported exclusion keys into ignored_layers")
EOF

# 4. The DSA indexer query split is opt-in: inert unless SGLANG_DSA_INDEXER_QSPLIT=1, and never used
# below the minimum row count even when enabled.
python3 - <<'EOF'
import importlib
import os

import sglang.srt.layers.attention.dsa.dsa_indexer_kpool as kpool

assert kpool._QSPLIT is False, kpool._QSPLIT
assert kpool._qsplit_group(1_000_000) is None
assert hasattr(kpool.IndexerKPool, "_get_topk_ragged_kpool_plan_qsplit")
os.environ["SGLANG_DSA_INDEXER_QSPLIT"] = "1"
os.environ["SGLANG_DSA_INDEXER_QSPLIT_MIN_ROWS"] = "2048"
kpool = importlib.reload(kpool)
assert kpool._QSPLIT is True and kpool._QSPLIT_MIN_ROWS == 2048
assert kpool._qsplit_group(2047) is None
print("step 4 OK: the DSA indexer query split is opt-in and respects its minimum row count")
EOF

# 5. Request preprocessing runs off the event loop (upstream sglang PR #30771): chat template
# rendering and tokenization in _convert_to_internal_request, and the regular-tokenizer path of
# _tokenize_texts, run in the tokenizer manager's single request-preprocessor thread, so the uvicorn
# loop keeps answering /health during multi-MB requests. The pytest files are the PR's own tests,
# applied under test/registered/unit; on the stock files three of them fail and two are skipped.
python3 - <<'EOF'
import inspect

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

assert "run_in_request_preprocessor" in inspect.getsource(OpenAIServingBase.handle_request)
assert "run_in_request_preprocessor" in inspect.getsource(TokenizerManager._tokenize_texts)
assert "self.init_request_preprocessor()" in inspect.getsource(TokenizerManager.__init__)
EOF
python3 -m pytest -q -p no:cacheprovider \
  test/registered/unit/entrypoints/openai/test_serving_base_event_loop.py \
  test/registered/unit/entrypoints/test_http_server_liveness.py \
  test/registered/unit/managers/test_tokenizer_manager_event_loop.py
echo "step 5 OK: chat template rendering and tokenization run off the event loop (PR #30771 tests)"

# 6. The dispatch-time /dev/shm copy of multimodal features (wrap_shm_features: posix_fallocate +
# copy_, ~141 MiB per large image) runs on its own sglang-mm-shm thread while the event loop keeps
# running, and a request cancelled during the copy still has its segments discarded afterwards.
python3 - <<'EOF'
import asyncio
import inspect
import threading
import time
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.tokenizer_manager as tm  # noqa: E402

send = inspect.getsource(tm.TokenizerManager._send_one_request)
assert "await self._wrap_shm_features_off_loop(tokenized_obj)" in send, send
assert "= wrap_shm_features(" not in send, send

manager = tm.TokenizerManager.__new__(tm.TokenizerManager)
manager.init_request_preprocessor()
calls, release = [], threading.Event()


def blocking_wrap(obj):
    calls.append(("wrap", threading.current_thread().name))
    if not release.wait(10):
        raise TimeoutError("test did not release the copy")
    return obj


def recording_discard(obj):
    calls.append(("discard", threading.current_thread().name))


# _wrap_shm_features_off_loop resolves both names in the module at call time.
tm.wrap_shm_features, tm.discard_shm_features = blocking_wrap, recording_discard


async def wait_for(condition):
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting; calls={calls}")


async def copy_while_loop_ticks():
    obj = SimpleNamespace(mm_inputs=object())
    task = asyncio.create_task(manager._wrap_shm_features_off_loop(obj))
    await wait_for(lambda: calls)
    ticks, end = 0, time.monotonic() + 0.5
    while time.monotonic() < end:
        await asyncio.sleep(0.01)
        ticks += 1
    release.set()
    assert await task is obj
    return ticks


ticks = asyncio.run(copy_while_loop_ticks())
assert ticks >= 20, f"the event loop ran only {ticks} times in 0.5 s while the copy was blocked"
assert [c[0] for c in calls] == ["wrap"] and calls[0][1].startswith("sglang-mm-shm"), calls

calls.clear()
release.clear()


async def cancel_during_copy():
    task = asyncio.create_task(manager._wrap_shm_features_off_loop(SimpleNamespace(mm_inputs=object())))
    await wait_for(lambda: calls)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation did not reach the caller")
    release.set()
    await wait_for(lambda: len(calls) == 2)


asyncio.run(cancel_during_copy())
manager._shm_wrap_executor.shutdown(wait=True)
manager._request_preprocessor_executor.shutdown(wait=True)
assert [c[0] for c in calls] == ["wrap", "discard"], calls
assert all(name.startswith("sglang-mm-shm") for _, name in calls), calls
print("step 6 OK: the multimodal shm copy runs off the event loop and a cancelled copy is discarded")
EOF

# 7. The event-loop stall dump is armed from the HTTP server lifespan, before the warmup thread and
# before the lifespan yields, and importing it arms nothing. test_stall_dump.py then runs each
# scenario in its own child process: a loop blocked in time.sleep (uvloop and asyncio), a CPU spin,
# a C call that keeps the GIL (faulthandler's dump), fork() and a fork-context process pool,
# uvicorn + FastAPI with /health polled during a blocked request, the detector disabled with 0, and
# an invalid value. In the image it is on by default at 30 s; SGLANG_EVENT_LOOP_STALL_DUMP_SECS=0
# disables it.
python3 - <<'EOF'
import inspect

import sglang.srt.entrypoints.http_server as hs
import sglang.srt.utils.event_loop_stall_dump as stall

assert list(inspect.signature(stall.install).parameters) == [
    "loop",
    "stall_seconds",
    "repeat_seconds",
    "log_prefix",
]
assert hs.install_stall_dump is stall.install
lifespan = inspect.getsource(hs.lifespan)
hook, warmup, handoff = (
    lifespan.index(marker)
    for marker in (
        "install_stall_dump(asyncio.get_running_loop())",
        "warmup_thread = threading.Thread(",
        "yield",
    )
)
assert hook < warmup < handoff, (hook, warmup, handoff)
assert stall._detector is None, "importing the module must not arm the detector"
EOF
stall_report="$(mktemp)"
if ! python3 -W ignore::DeprecationWarning "$RECIPE_DIR/test_stall_dump.py" \
    --module "$PWD/python/sglang/srt/utils/event_loop_stall_dump.py" \
    --out "$stall_report" > /dev/null; then
  cat "$stall_report"
  echo "step 7 FAILED: event-loop stall dump functional test" >&2
  exit 1
fi
sed -n '/^SUMMARY/,$p' "$stall_report"
echo "step 7 OK: the event-loop stall dump reports blocked-loop stacks and recoveries, and stays off at 0"

# 8. The ghost prefix cache is hooked into release_kv_cache and inert by default. test_ghost_cache.py
# then checks the installed modules: predicted hit tokens equal a brute-force LRU simulation at every
# cache size with sampling off, 1/16 sampling stays within 5 points, compulsory misses equal
# never-seen pages, only 16-byte keyed digests are retained, the tracked set is bounded, the hook
# does nothing without SGLANG_GHOST_CACHE=1; racing replicas share one key file; the aggregator's
# pooled counts equal a brute-force two-replica simulation with a failover; and two recorders feed a
# running aggregator over a real unix socket whose /metrics match the direct computation.
python3 - <<'EOF'
import inspect

import sglang.srt.mem_cache.common as common
import sglang.srt.observability.ghost_cache as ghost

release = inspect.getsource(common.release_kv_cache)
assert release.index("observe_finished_req(req)") < release.index("assert (not req.kv.holds_kv)"), release
assert common.observe_finished_req is ghost.observe_finished_req
assert ghost._ENABLED is False and ghost._recorder is None
EOF
python3 "$RECIPE_DIR/test_ghost_cache.py" \
  --module "$PWD/python/sglang/srt/observability/ghost_cache.py" \
  --aggregator "$PWD/python/sglang/srt/observability/ghost_aggregator.py"
echo "step 8 OK: the ghost prefix cache and its pooled aggregator match exact LRU, store only keyed digests, and are off by default"

echo "GLM-5.3 W4AFP8 combined-image CPU checks passed"
