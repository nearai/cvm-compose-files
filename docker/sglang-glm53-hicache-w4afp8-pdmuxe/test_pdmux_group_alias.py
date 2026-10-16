"""v7-port addition: groups that alias the decode TP group follow the PDMux
prefill duplicate (not part of upstream PR #42411)."""

import unittest
from unittest.mock import patch

from sglang.srt.distributed import parallel_state as ps
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPDMuxGroupAlias(unittest.TestCase):
    def setUp(self):
        self.tp, self.dup, self.other = object(), object(), object()

    def _patched(self, flag, attn, moe_ep, moe_tp):
        return patch.multiple(
            ps,
            _TP=self.tp,
            _PDMUX_PREFILL_TP_GROUP=self.dup,
            _ENABLE_PDMUX_P_TP=flag,
            _ATTN_TP=attn,
            _MOE_EP=moe_ep,
            _MOE_TP=moe_tp,
        )

    def test_aliased_groups_follow_prefill_duplicate(self):
        with self._patched(True, self.tp, self.tp, self.other):
            self.assertIs(ps.get_tp_group(), self.dup)
            self.assertIs(ps.get_attn_tp_group(), self.dup)
            self.assertIs(ps.get_moe_ep_group(), self.dup)
            self.assertIs(ps.get_moe_tp_group(), self.other)

    def test_decode_side_unchanged(self):
        with self._patched(False, self.tp, self.tp, self.tp):
            self.assertIs(ps.get_tp_group(), self.tp)
            self.assertIs(ps.get_attn_tp_group(), self.tp)
            self.assertIs(ps.get_moe_ep_group(), self.tp)
            self.assertIs(ps.get_moe_tp_group(), self.tp)

    def test_independent_attn_group_is_not_remapped(self):
        with self._patched(True, self.other, self.other, self.other):
            self.assertIs(ps.get_attn_tp_group(), self.other)


if __name__ == "__main__":
    unittest.main()
