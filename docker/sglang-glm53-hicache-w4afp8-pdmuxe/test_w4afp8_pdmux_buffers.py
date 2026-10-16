"""v7-port addition: W4A8 MoE grouped-GEMM metadata must not be shared between
the PDMux prefill and decode streams (not part of upstream PR #42411)."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _method():
    m = W4AFp8MoEMethod.__new__(W4AFp8MoEMethod)
    for name in (
        "a_strides1", "b_strides1", "c_strides1", "a_strides2", "b_strides2",
        "c_strides2", "s_strides13", "s_strides2",
    ):
        setattr(m, name, torch.zeros(4, 3, dtype=torch.int64))
    m.expert_offsets = torch.zeros(5, dtype=torch.int32)
    m.problem_sizes1 = torch.zeros(4, 3, dtype=torch.int32)
    m.problem_sizes2 = torch.zeros(4, 3, dtype=torch.int32)
    m.moe_runner_config = SimpleNamespace(routed_scaling_factor=None)
    return m


def _apply(method, pdmux):
    layer = SimpleNamespace(
        w13_weight=None, w2_weight=None, w13_weight_scale_inv=None,
        w2_weight_scale_inv=None, w13_input_scale=None, w2_input_scale=None,
    )
    out = SimpleNamespace(hidden_states=None, topk_output=(None, None, None))
    seen = {}

    def fake(*args, **kwargs):
        seen["bufs"] = args[15:18]
        return torch.zeros(1)

    with (
        # The real module needs sgl_kernel CUDA ops; stub it for the CPU run.
        patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.moe.cutlass_w4a8_moe": SimpleNamespace(
                    cutlass_w4a8_moe=fake
                )
            },
        ),
        patch(
            "sglang.srt.layers.quantization.w4afp8.get_disagg",
            return_value=SimpleNamespace(enable_pdmux=pdmux),
        ),
    ):
        method.apply(layer, out)
    return seen["bufs"]


class TestW4AFp8PDMuxBuffers(unittest.TestCase):
    def test_pdmux_uses_call_local_metadata(self):
        m = _method()
        bufs = _apply(m, True)
        for got, shared in zip(
            bufs, (m.expert_offsets, m.problem_sizes1, m.problem_sizes2)
        ):
            self.assertIsNot(got, shared)
            self.assertEqual(got.shape, shared.shape)
            self.assertEqual(got.dtype, shared.dtype)

    def test_without_pdmux_keeps_shared_metadata(self):
        m = _method()
        bufs = _apply(m, False)
        self.assertIs(bufs[0], m.expert_offsets)
        self.assertIs(bufs[1], m.problem_sizes1)
        self.assertIs(bufs[2], m.problem_sizes2)


if __name__ == "__main__":
    unittest.main()
