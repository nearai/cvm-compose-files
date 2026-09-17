#!/usr/bin/env bash
set -euo pipefail
cd /sgl-workspace/sglang
# This pinned upstream test utility indexes the first visible-device digit.
# Use an unavailable ordinal, with the CPU container's runtime set to runc.
export CUDA_VISIBLE_DEVICES=9

# 1. Reserve disabled by default (env unset): the module-level constant is 0
#    and the admission helper is a no-op for both an empty and a non-empty
#    waiting queue.
python3 - <<'EOF'
import types

import sglang.srt.managers.schedule_policy as sp

assert sp.CHUNKED_PREFILL_ADMISSION_RESERVE == 0, sp.CHUNKED_PREFILL_ADMISSION_RESERVE

from sglang.srt.managers.scheduler import _admission_reserve_need

assert _admission_reserve_need([], 64, 4096) is None

fake_queue = [types.SimpleNamespace(seqlen=300, prefix_indices=None)]
assert _admission_reserve_need(fake_queue, 64, 4096) is None

print("step 1 OK: admission-reserve is a no-op with the env unset")
EOF

# 2. A negative reserve is rejected at import time.
SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=-1 python3 - <<'EOF'
import sys

try:
    import sglang.srt.managers.schedule_policy  # noqa: F401
except ValueError as exc:
    print(f"step 2 OK: negative reserve rejected at import ({exc})")
else:
    print("step 2 FAILED: negative SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE was not rejected", file=sys.stderr)
    sys.exit(1)
EOF

# 3. Sizing and the 75% fairness cap. cap = min(4096, int(4096*0.75)=3072,
#    4096-64=4032) = 3072. Waiter 1 (seqlen 300, no prefix) needs 320
#    page-aligned tokens; waiter 2 (seqlen 20000, 19700-token cached prefix)
#    has a 300-token extend, also 320 tokens; waiter 3 (seqlen 20000, no
#    prefix) needs 20000 tokens, over the cap, and is skipped. Total: 640.
SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096 SGLANG_ADMISSION_RESERVE_MAX_FRACTION=0.75 python3 - <<'EOF'
import types

from sglang.srt.managers.scheduler import _admission_reserve_need

waiters = [
    types.SimpleNamespace(seqlen=300, prefix_indices=None),
    types.SimpleNamespace(seqlen=20000, prefix_indices=list(range(19700))),
    types.SimpleNamespace(seqlen=20000, prefix_indices=None),
]
need = _admission_reserve_need(waiters, 64, 4096)
assert need == 640, need

# Cap arithmetic: a single waiter exactly at the 3072-token cap fits; one
# page over it does not.
fits = _admission_reserve_need(
    [types.SimpleNamespace(seqlen=3072, prefix_indices=None)], 64, 4096
)
assert fits == 3072, fits

does_not_fit = _admission_reserve_need(
    [types.SimpleNamespace(seqlen=3136, prefix_indices=None)], 64, 4096
)
assert does_not_fit is None, does_not_fit

print("step 3 OK: admission-reserve sizing and the 75% fairness cap")
EOF

# 4. The removed wall-clock gate is rejected at call time (rank-determinism
#    guard): it only fires once there is a non-empty waiting queue and a
#    positive reserve, not at import.
SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096 SGLANG_ADMISSION_RESERVE_MIN_WAIT_S=1 python3 - <<'EOF'
import sys
import types

from sglang.srt.managers.scheduler import _admission_reserve_need

waiters = [types.SimpleNamespace(seqlen=300, prefix_indices=None)]
try:
    _admission_reserve_need(waiters, 64)
except RuntimeError as exc:
    print(f"step 4 OK: SGLANG_ADMISSION_RESERVE_MIN_WAIT_S rejected at call time ({exc})")
else:
    print("step 4 FAILED: SGLANG_ADMISSION_RESERVE_MIN_WAIT_S did not raise", file=sys.stderr)
    sys.exit(1)
EOF

# 5. No import/syntax regression in the patched scheduler module.
python3 -c "import sglang.srt.managers.scheduler"
echo "step 5 OK: sglang.srt.managers.scheduler imports cleanly"

echo "admission-reserve CPU checks passed"
