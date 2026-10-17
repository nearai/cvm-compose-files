"""Adaptive backlog mode of the PDMux scheduler (policy, config, loop wiring)."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.multiplex import pdmux_context
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.multiplex.pdmux_context import PDMuxBacklogPolicy, PDMuxConfig
from sglang.test.ci.ci_register import register_cpu_ci

# Shared fakes/stubs from the HiCache event-pump test (same directory).
from test_pdmux_hicache_events import (
    _FakeScheduler,
    _LoopFinished,
    _stubbed_cuda,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BASE_YAML = """
sm_group_num: 3
manual_divisions: [[96, 36, 1]]
split_forward_token_budget: 65536
max_split_forward_layers: 4
overlap_decode_full_sm: true
"""


def load_config(body):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pdmux.yaml"
        path.write_text(body)
        return pdmux_context.load_pdmux_config(str(path))


class TestBacklogConfig(unittest.TestCase):
    def test_default_is_off_and_old_yaml_unchanged(self):
        cfg = load_config(BASE_YAML)
        self.assertFalse(cfg.backlog_enabled)
        self.assertEqual(cfg, PDMuxConfig(**{
            k: getattr(cfg, k) for k in cfg.__dataclass_fields__
        }))
        self.assertEqual(cfg.max_split_forward_layers, 4)
        self.assertEqual(cfg.backlog_decode_every_n, 1)
        self.assertTrue(cfg.backlog_full_device_prefill)
        self.assertFalse(PDMuxConfig().backlog_enabled)
        self.assertFalse(pdmux_context.load_pdmux_config("").backlog_enabled)

    def test_yaml_with_backlog_fields(self):
        cfg = load_config(
            BASE_YAML
            + """
backlog_queue_on: 8
backlog_queue_off: 2
backlog_pending_tokens_on: 98304
backlog_pending_tokens_off: 16384
backlog_max_split_forward_layers: 24
backlog_full_device_prefill: false
backlog_decode_every_n: 2
backlog_log_interval_s: 2.5
"""
        )
        self.assertTrue(cfg.backlog_enabled)
        self.assertEqual(
            (cfg.backlog_queue_on, cfg.backlog_queue_off), (8, 2)
        )
        self.assertEqual(
            (cfg.backlog_pending_tokens_on, cfg.backlog_pending_tokens_off),
            (98304, 16384),
        )
        self.assertEqual(cfg.backlog_max_split_forward_layers, 24)
        self.assertFalse(cfg.backlog_full_device_prefill)
        self.assertEqual(cfg.backlog_decode_every_n, 2)
        self.assertEqual(cfg.backlog_log_interval_s, 2.5)
        # Existing fields untouched.
        self.assertEqual(cfg.max_split_forward_layers, 4)
        self.assertTrue(cfg.overlap_decode_full_sm)

    def test_off_defaults_to_quarter_of_on(self):
        cfg = load_config(BASE_YAML + "backlog_queue_on: 12\n")
        self.assertEqual(cfg.backlog_queue_off, 3)
        self.assertEqual(cfg.backlog_pending_tokens_on, 0)

    def test_validation(self):
        bad = [
            "backlog_queue_on: 4\nbacklog_queue_off: 4\n",  # off must be < on
            "backlog_queue_on: 4\nbacklog_queue_off: 9\n",
            "backlog_queue_off: 2\n",  # off without on
            "backlog_pending_tokens_on: 100\nbacklog_pending_tokens_off: 100\n",
            "backlog_queue_on: -1\n",
            "backlog_queue_on: true\n",
            "backlog_queue_on: 2.5\n",
            "backlog_decode_every_n: 0\n",
            "backlog_max_split_forward_layers: -1\n",
            "backlog_full_device_prefill: 1\n",
            "backlog_log_interval_s: -1\n",
        ]
        for extra in bad:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                load_config(BASE_YAML + extra)


class TestBacklogPolicy(unittest.TestCase):
    @staticmethod
    def _policy(**kw):
        return PDMuxBacklogPolicy(PDMuxConfig(**kw))

    def test_queue_hysteresis(self):
        p = self._policy(backlog_queue_on=8, backlog_queue_off=2)
        seq = [(0, False), (7, False), (8, True), (5, True), (3, True),
               (2, False), (5, False), (8, True), (9, True), (1, False)]
        got = []
        for q, _ in seq:
            p.update(q)
            got.append(p.active)
        self.assertEqual(got, [e for _, e in seq])

    def test_update_reports_flips_only(self):
        p = self._policy(backlog_queue_on=4, backlog_queue_off=1)
        self.assertEqual(
            [p.update(q) for q in (0, 4, 4, 3, 1, 1, 4)],
            [False, True, False, False, True, False, True],
        )

    def test_tokens_or_queue_enters_and_both_must_clear_to_exit(self):
        p = self._policy(
            backlog_queue_on=8,
            backlog_queue_off=2,
            backlog_pending_tokens_on=1000,
            backlog_pending_tokens_off=200,
        )
        p.update(0, 1000)
        self.assertTrue(p.active)  # tokens alone enter
        p.update(0, 500)
        self.assertTrue(p.active)  # between off and on: stay
        p.update(5, 100)
        self.assertTrue(p.active)  # queue still above off
        p.update(2, 200)
        self.assertFalse(p.active)  # both at/below off
        p.update(8, 0)
        self.assertTrue(p.active)  # queue alone enters

    def test_disabled_criterion_is_ignored(self):
        p = self._policy(backlog_queue_on=3, backlog_queue_off=1)
        p.update(0, 10**9)
        self.assertFalse(p.active)
        self.assertFalse(p.uses_tokens)
        p = self._policy(backlog_pending_tokens_on=100, backlog_pending_tokens_off=10)
        p.update(10**6, 0)
        self.assertFalse(p.active)

    def test_never_enabled_never_active(self):
        p = self._policy()
        for q in (0, 100):
            self.assertFalse(p.update(q, 10**9))
        self.assertFalse(p.active)


class _Helper:
    @staticmethod
    @contextmanager
    def _stubbed_stream_idx():
        state = {"idx": 0}
        with (
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.set_current_stream_idx",
                lambda idx: state.update(idx=idx),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_current_stream_idx",
                lambda: state["idx"],
            ),
        ):
            yield

    @staticmethod
    def _make_stream_group_scheduler(*, manual_divisions, group_num):
        return SimpleNamespace(
            split_prefill_batch=object(),
            pdmux_config=SimpleNamespace(
                manual_divisions=manual_divisions, decode_bs_divisor=36
            ),
            real_sm_group_num=group_num,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(update_decode_attn_backend=Mock())
            ),
            stream_groups=[(f"p{i}", f"d{i}") for i in range(group_num)],
        )


class TestBacklogSelection(unittest.TestCase):
    """Stream group and layer cap selection."""

    helper = _Helper()

    def _sched(self, *, active, full_device=True, cap=0, group_num=3, layers_done=0):
        sched = self.helper._make_stream_group_scheduler(
            manual_divisions=[[96, 36, 1]], group_num=group_num
        )
        sched.pdmux_config = PDMuxConfig(
            manual_divisions=[[96, 36, 1]],
            backlog_queue_on=8,
            backlog_max_split_forward_layers=cap,
            backlog_full_device_prefill=full_device,
            max_split_forward_layers=4,
            overlap_decode_full_sm=True,
        )
        sched._pdmux_backlog_active = active
        sched.model_config = SimpleNamespace(num_hidden_layers=61)
        sched.split_prefill_batch = SimpleNamespace(
            split_index=layers_done,
            extend_num_tokens=8192,
            scheduler_global_num_tokens=None,
        )
        return sched

    @staticmethod
    def _decode(n=4):
        return SimpleNamespace(is_empty=lambda: False, batch_size=lambda: n)

    def test_stream_group_normal_vs_backlog(self):
        with self.helper._stubbed_stream_idx():
            normal, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                self._sched(active=False), self._decode()
            )
            backlog, group = SchedulerMultiplexMixin.adjust_stream_groups(
                self._sched(active=True), self._decode()
            )
            kept, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                self._sched(active=True, full_device=False), self._decode()
            )
        self.assertEqual(normal, 1)  # green-context capped group
        self.assertEqual(backlog, 2)  # decode-only group: plain streams
        self.assertEqual(group, ("p2", "d2"))
        self.assertEqual(kept, 1)

    def test_backlog_without_decode_still_uses_prefill_only_group(self):
        with self.helper._stubbed_stream_idx():
            idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                self._sched(active=True), None
            )
        self.assertEqual(idx, 0)

    def test_backlog_without_split_prefill_uses_decode_group(self):
        sched = self._sched(active=True)
        sched.split_prefill_batch = None
        with self.helper._stubbed_stream_idx():
            idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(sched, self._decode())
        self.assertEqual(idx, 2)

    def test_default_off_matches_missing_attribute(self):
        """A scheduler that never heard of backlog mode behaves identically."""
        legacy = self._sched(active=False)
        del legacy._pdmux_backlog_active
        decode = SimpleNamespace(scheduler_global_num_tokens=[4])
        with self.helper._stubbed_stream_idx():
            a, _ = SchedulerMultiplexMixin.adjust_stream_groups(legacy, self._decode())
            b, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                self._sched(active=False), self._decode()
            )
        self.assertEqual(a, b)
        self.assertEqual(
            SchedulerMultiplexMixin._get_split_forward_count(legacy, decode),
            SchedulerMultiplexMixin._get_split_forward_count(
                self._sched(active=False), decode
            ),
        )
        self.assertEqual(
            SchedulerMultiplexMixin._get_split_forward_count(legacy, decode), 4
        )

    def test_layer_count(self):
        decode = SimpleNamespace(scheduler_global_num_tokens=[4])
        count = SchedulerMultiplexMixin._get_split_forward_count
        self.assertEqual(count(self._sched(active=False), decode), 4)
        self.assertEqual(count(self._sched(active=True), decode), 61)
        self.assertEqual(count(self._sched(active=True, layers_done=50), decode), 11)
        self.assertEqual(count(self._sched(active=True, cap=16), decode), 16)
        self.assertEqual(count(self._sched(active=True, cap=16, layers_done=55), decode), 6)
        # Independent of decode work (rank-consistent: no per-rank input).
        self.assertEqual(count(self._sched(active=True), None), 61)


class TestBacklogTick(unittest.TestCase):
    @staticmethod
    def _sched(**kw):
        cfg = PDMuxConfig(**kw)
        sched = SimpleNamespace(
            pdmux_config=cfg,
            split_prefill_batch=None,
            waiting_queue=[],
            tp_rank=0,
        )
        for name in (
            "_init_pdmux_backlog",
            "_pdmux_backlog_log_transition",
            "_pdmux_backlog_heartbeat",
            "_pdmux_backlog_skip_decode",
        ):
            setattr(sched, name, getattr(SchedulerMultiplexMixin, name).__get__(sched))
        sched._init_pdmux_backlog()
        return sched

    @staticmethod
    def _req(n):
        return SimpleNamespace(origin_input_ids=[0] * n)

    def test_unconfigured_is_inert(self):
        sched = self._sched()
        self.assertIsNone(sched._pdmux_backlog_policy)
        sched.waiting_queue = [self._req(10)] * 100
        self.assertFalse(SchedulerMultiplexMixin._pdmux_backlog_tick(sched))
        self.assertFalse(sched._pdmux_backlog_active)

    def test_tick_flips_and_counts(self):
        sched = self._sched(backlog_queue_on=3, backlog_queue_off=1)
        results = []
        for n in (0, 3, 3, 2, 1, 5):
            sched.waiting_queue = [self._req(10)] * n
            results.append(SchedulerMultiplexMixin._pdmux_backlog_tick(sched))
        self.assertEqual(results, [False, True, False, False, True, True])
        stats = sched._pdmux_backlog_stats
        self.assertEqual((stats["entered"], stats["exited"]), (2, 1))
        self.assertTrue(sched._pdmux_backlog_active)

    def test_pending_tokens_counted_from_waiting_queue(self):
        sched = self._sched(
            backlog_pending_tokens_on=1000, backlog_pending_tokens_off=100
        )
        sched.waiting_queue = [self._req(400), self._req(500)]
        self.assertFalse(SchedulerMultiplexMixin._pdmux_backlog_tick(sched))
        sched.waiting_queue.append(self._req(100))
        self.assertTrue(SchedulerMultiplexMixin._pdmux_backlog_tick(sched))

    def test_transition_log_is_rate_limited_but_counters_are_not(self):
        sched = self._sched(
            backlog_queue_on=2, backlog_queue_off=0, backlog_log_interval_s=3600
        )
        with self.assertLogs(
            "sglang.srt.multiplex.multiplexing_mixin", level="INFO"
        ) as logs:
            for n in (2, 0, 2, 0, 2):
                sched.waiting_queue = [self._req(1)] * n
                SchedulerMultiplexMixin._pdmux_backlog_tick(sched)
        lines = [r for r in logs.output if "pdmux backlog mode on" in r]
        self.assertEqual(len(lines), 1)
        self.assertEqual(sched._pdmux_backlog_stats["entered"], 3)
        self.assertEqual(sched._pdmux_backlog_suppressed, 4)

    def test_transition_log_line_content(self):
        sched = self._sched(backlog_queue_on=2, backlog_queue_off=0)
        sched.waiting_queue = [self._req(1)] * 2
        with self.assertLogs("sglang.srt.multiplex.multiplexing_mixin", "INFO") as logs:
            SchedulerMultiplexMixin._pdmux_backlog_tick(sched)
            sched.waiting_queue = []
            SchedulerMultiplexMixin._pdmux_backlog_tick(sched)
        text = "\n".join(logs.output)
        self.assertIn("pdmux backlog mode on: queue=2", text)
        self.assertIn("entered=1", text)

    def test_skip_decode_only_in_backlog_with_split_in_flight(self):
        sched = self._sched(
            backlog_queue_on=2, backlog_queue_off=0, backlog_decode_every_n=3
        )
        skip = lambda: SchedulerMultiplexMixin._pdmux_backlog_skip_decode(sched)
        self.assertFalse(skip())  # not active
        sched._pdmux_backlog_active = True
        self.assertFalse(skip())  # no split in flight
        sched.split_prefill_batch = object()
        self.assertEqual([skip() for _ in range(6)], [True, True, False] * 2)
        sched.split_prefill_batch = None
        self.assertFalse(skip())
        sched.split_prefill_batch = object()
        self.assertTrue(skip())  # counter restarted
        self.assertEqual(sched._pdmux_backlog_stats["decode_steps_skipped"], 5)

    def test_every_n_one_never_skips(self):
        sched = self._sched(backlog_queue_on=2, backlog_queue_off=0)
        sched._pdmux_backlog_active = True
        sched.split_prefill_batch = object()
        self.assertFalse(any(
            SchedulerMultiplexMixin._pdmux_backlog_skip_decode(sched)
            for _ in range(5)
        ))

    def test_dp_attention_disables_backlog(self):
        cfg = PDMuxConfig(backlog_queue_on=2, backlog_queue_off=0)
        sched = SimpleNamespace(pdmux_config=cfg, enable_dp_attention=True)
        SchedulerMultiplexMixin._init_pdmux_backlog(sched)
        self.assertIsNone(sched._pdmux_backlog_policy)


class _BacklogLoopScheduler(_FakeScheduler):
    def __init__(self, *, config, queue_by_iteration, **kw):
        super().__init__(**kw)
        self.pdmux_config = config
        self.waiting_queue = []
        self.queue_by_iteration = queue_by_iteration
        self.adjust_calls = []
        self.decode_runs = []
        self.pending_split_batch = None
        self.enable_dp_attention = False
        self._init_pdmux_backlog()

    def process_input_requests(self, recv_reqs):
        n = self.queue_by_iteration.get(self.iteration, None)
        if n is not None:
            self.waiting_queue = [
                SimpleNamespace(origin_input_ids=[0] * 10) for _ in range(n)
            ]

    def adjust_stream_groups(self, decode_batch):
        self.adjust_calls.append((self.iteration, decode_batch))
        return 0, self.stream_groups[0]

    def run_batch(self, batch):
        if batch is not self.split_prefill_batch:
            self.decode_runs.append(self.iteration)
        return object()


class TestBacklogLoop(unittest.TestCase):
    @staticmethod
    def _run(sched):
        with _stubbed_cuda():
            try:
                sched.event_loop_pdmux()
            except _LoopFinished:
                pass
        return sched

    @staticmethod
    def _cfg(**kw):
        return PDMuxConfig(
            split_forward_token_budget=1000, backlog_queue_on=2,
            backlog_queue_off=0, **kw,
        )

    def test_mode_flip_forces_stream_group_adjustment(self):
        sched = self._run(_BacklogLoopScheduler(
            config=self._cfg(), queue_by_iteration={2: 3, 4: 0},
            max_iterations=6, query_results=[],
        ))
        # Without a flip the loop never re-adjusts (idx 0, decode present, no
        # prefill created); each flip adjusts exactly once.
        self.assertEqual([i for i, _ in sched.adjust_calls], [2, 4])
        self.assertFalse(sched._pdmux_backlog_active)
        self.assertEqual(sched._pdmux_backlog_stats["entered"], 1)
        self.assertEqual(sched._pdmux_backlog_stats["exited"], 1)

    def test_default_off_loop_never_adjusts_or_skips(self):
        sched = self._run(_BacklogLoopScheduler(
            config=PDMuxConfig(split_forward_token_budget=1000),
            queue_by_iteration={2: 50}, max_iterations=6, query_results=[],
        ))
        self.assertEqual(sched.adjust_calls, [])
        self.assertEqual(sched.decode_runs, [0, 1, 2, 3, 4, 5])
        self.assertIsNone(sched._pdmux_backlog_policy)

    def _run_with_split(self, config, queue, **kw):
        from test_pdmux_hicache_events import _SplitBatch

        sched = _BacklogLoopScheduler(
            config=config, queue_by_iteration={0: queue}, max_iterations=5,
            query_results=[False, False, False], **kw,
        )
        sched.pending_split_batch = _SplitBatch()
        sched.forward_counts = []
        orig = sched.run_batch

        def run_batch(batch):
            if batch is sched.split_prefill_batch:
                sched.forward_counts.append(batch.split_forward_count)
            return orig(batch)

        sched.run_batch = run_batch
        return self._run(sched)

    def test_backlog_submits_all_layers_and_skips_decode(self):
        sched = self._run_with_split(self._cfg(backlog_decode_every_n=2), queue=3)
        # NUM_LAYERS=3 and a busy decode batch: normally one layer per step.
        self.assertEqual(sched.forward_counts, [3])
        # Prefill stays in flight for iterations 0-3 (finish event not ready
        # until then): decode runs on every 2nd of them, then every iteration
        # once the batch has merged.
        self.assertEqual(sched.decode_runs, [1, 3, 4])
        self.assertEqual(sched._pdmux_backlog_stats["decode_steps_skipped"], 2)

    def test_same_scenario_default_off_is_layerwise_with_decode_every_step(self):
        sched = self._run_with_split(
            PDMuxConfig(split_forward_token_budget=1000), queue=3
        )
        self.assertEqual(sched.forward_counts, [1, 1, 1])
        self.assertEqual(sched.decode_runs, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
