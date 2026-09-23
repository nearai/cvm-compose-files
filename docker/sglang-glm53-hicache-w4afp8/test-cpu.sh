#!/usr/bin/env bash
set -euo pipefail
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

echo "GLM-5.3 W4AFP8 combined-image CPU checks passed"
