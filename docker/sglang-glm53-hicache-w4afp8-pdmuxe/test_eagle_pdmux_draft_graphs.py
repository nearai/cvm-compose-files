"""Per-stream-group EAGLE draft CUDA graphs under PDMux (mocked CUDA, CPU only)."""

import os
import sys
import types
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.model_executor.runner import DecodeCudaGraphRunner, ShapeKey
from sglang.srt.multiplex import pdmux_context as pc
from sglang.srt.speculative import eagle_draft_cuda_graph_runner as dmod
from sglang.srt.speculative import eagle_draft_extend_cuda_graph_runner as emod
from sglang.srt.speculative import eagle_worker_v2 as wmod
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
    EagleDraftInputBuffers,
    share_cuda_graph_state_across_groups,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ENV_KEYS = (
    "SGLANG_PDMUX_EAGLE_DRAFT_GRAPHS",
    "SGLANG_PDMUX_EAGLE_DRAFT_GRAPH_GROUPS",
    "SGLANG_PDMUX_EAGLE_DRAFT_SHARE_PAGE_TABLE",
)


class EnvCleanTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in ENV_KEYS}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


class TestStreamIndexSelection(EnvCleanTest):
    def test_decode_never_uses_group_zero(self):
        # sm_group_num=3 (prod ov96): [prefill-only normal, 96-SM division, decode normal]
        self.assertEqual(pc.get_pdmux_decode_stream_indices(3), [1, 2])
        self.assertEqual(pc.get_pdmux_decode_stream_indices(2), [1])
        self.assertEqual(pc.get_pdmux_decode_stream_indices(5), [1, 2, 3, 4])

    def test_draft_graph_groups_default_and_override(self):
        self.assertEqual(pc.get_pdmux_eagle_draft_graph_stream_indices(3), [1, 2])
        os.environ["SGLANG_PDMUX_EAGLE_DRAFT_GRAPH_GROUPS"] = "auto"
        self.assertEqual(pc.get_pdmux_eagle_draft_graph_stream_indices(3), [1, 2])
        os.environ["SGLANG_PDMUX_EAGLE_DRAFT_GRAPH_GROUPS"] = "2"
        self.assertEqual(pc.get_pdmux_eagle_draft_graph_stream_indices(3), [2])
        os.environ["SGLANG_PDMUX_EAGLE_DRAFT_GRAPH_GROUPS"] = "2, 1"
        self.assertEqual(pc.get_pdmux_eagle_draft_graph_stream_indices(3), [1, 2])

    def test_draft_graph_groups_rejects_unused_or_missing_groups(self):
        for bad in ("0", "3", "1,7"):
            os.environ["SGLANG_PDMUX_EAGLE_DRAFT_GRAPH_GROUPS"] = bad
            with self.assertRaises(ValueError):
                pc.get_pdmux_eagle_draft_graph_stream_indices(3)

    def test_env_switch(self):
        self.assertTrue(pc.pdmux_eagle_draft_graphs_enabled())
        os.environ["SGLANG_PDMUX_EAGLE_DRAFT_GRAPHS"] = "1"
        self.assertTrue(pc.pdmux_eagle_draft_graphs_enabled())
        os.environ["SGLANG_PDMUX_EAGLE_DRAFT_GRAPHS"] = "0"
        self.assertFalse(pc.pdmux_eagle_draft_graphs_enabled())


def fake_multistep_backend(num_steps=4, page_numel=8):
    steps = [
        SimpleNamespace(
            decode_cuda_graph_metadata={"page_table": torch.zeros(page_numel, dtype=torch.int32)}
        )
        for _ in range(num_steps - 1)
    ]
    backend = Mock(name="multistep")
    backend.attn_backends = steps
    return backend


class TestSharePageTable(unittest.TestCase):
    def test_aliases_matching_tensors_and_reports_bytes(self):
        ref = [SimpleNamespace(decode_cuda_graph_metadata={"page_table": torch.zeros(4, 8, dtype=torch.int32), "real_page_table": None})]
        dst = [SimpleNamespace(decode_cuda_graph_metadata={"page_table": torch.zeros(4, 8, dtype=torch.int32), "real_page_table": None})]
        freed = share_cuda_graph_state_across_groups(ref, dst)
        self.assertEqual(freed, 4 * 8 * 4)
        self.assertIs(dst[0].decode_cuda_graph_metadata["page_table"], ref[0].decode_cuda_graph_metadata["page_table"])

    def test_skips_shape_mismatch_and_non_dsa_backends(self):
        ref = [SimpleNamespace(decode_cuda_graph_metadata={"page_table": torch.zeros(4, 8)})]
        dst = [SimpleNamespace(decode_cuda_graph_metadata={"page_table": torch.zeros(2, 8)})]
        self.assertEqual(share_cuda_graph_state_across_groups(ref, dst), 0)
        self.assertEqual(share_cuda_graph_state_across_groups([object()], [object()]), 0)


class TestDraftDecodeRunnerPerStream(EnvCleanTest):
    def make_runner(self, group_idx=(1, 2), steps=4):
        runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
        n = 3
        self.group = [None] * n
        for i in group_idx:
            self.group[i] = fake_multistep_backend(steps)
        runner.draft_attn_backend = fake_multistep_backend(steps)  # eager instance
        runner.draft_attn_backend_group = self.group
        runner.captured_stream_indices = ()
        runner.enable_pdmux = True
        runner.max_bs = 64
        runner.max_num_token = 64
        runner.speculative_num_steps = steps
        runner.captured_req_width = 1
        runner.capture_bs = [1, 2, 4, 8]
        runner.require_mlp_tp_gather = False
        runner.require_mlp_sync = False
        runner.require_gathered_buffer = False
        runner.disable_padding = False
        runner.backend = Mock()
        return runner

    def test_maybe_init_pdmux_inits_only_captured_groups_not_eager_backend(self):
        runner = self.make_runner((1, 2))
        with patch.object(dmod, "get_stream_groups", return_value=[object()] * 3):
            runner.maybe_init_pdmux()
        self.assertEqual(runner.captured_stream_indices, (1, 2))
        runner.draft_attn_backend.init_cuda_graph_state.assert_not_called()
        self.assertIsNone(self.group[0])
        for i in (1, 2):
            self.group[i].init_cuda_graph_state.assert_called_once_with(64, 64)

    def test_maybe_init_pdmux_subset_and_share(self):
        runner = self.make_runner((1, 2))
        os.environ["SGLANG_PDMUX_EAGLE_DRAFT_SHARE_PAGE_TABLE"] = "1"
        with patch.object(dmod, "get_stream_groups", return_value=[object()] * 3):
            runner.maybe_init_pdmux()
        for a, b in zip(self.group[1].attn_backends, self.group[2].attn_backends):
            self.assertIs(
                a.decode_cuda_graph_metadata["page_table"],
                b.decode_cuda_graph_metadata["page_table"],
            )

    def test_maybe_init_pdmux_rejects_length_mismatch(self):
        runner = self.make_runner((1, 2))
        with patch.object(dmod, "get_stream_groups", return_value=[object()] * 4):
            with self.assertRaises(AssertionError):
                runner.maybe_init_pdmux()

    def test_graph_key_carries_stream_idx(self):
        runner = self.make_runner()
        self.assertEqual(runner._make_graph_key(8, 2), ShapeKey(size=8, stream_idx=2))
        self.assertNotEqual(runner._make_graph_key(8, 1), runner._make_graph_key(8, 2))
        self.assertEqual(runner._make_graph_key(8), ShapeKey(size=8, stream_idx=None))

    def test_backend_selection_is_per_group(self):
        runner = self.make_runner((1, 2))
        self.assertIs(runner._draft_backend_for(1), self.group[1])
        self.assertIs(runner._draft_backend_for(2), self.group[2])
        self.assertIsNot(runner._draft_backend_for(1), runner._draft_backend_for(2))
        self.assertIs(runner._draft_backend_for(None), runner.draft_attn_backend)
        with self.assertRaises(AssertionError):
            runner._draft_backend_for(0)

    def test_non_pdmux_runner_keeps_single_backend_and_key(self):
        runner = self.make_runner()
        runner.enable_pdmux = False
        runner.draft_attn_backend_group = None
        self.assertIsNone(runner._current_stream_idx())
        self.assertIs(runner._draft_backend_for(None), runner.draft_attn_backend)
        self.assertEqual(runner._make_graph_key(4, None), ShapeKey(size=4))

    def fb(self, bs=2):
        return SimpleNamespace(
            spec_info=SimpleNamespace(num_tokens_per_req=1),
            batch_size=bs,
        )

    def test_can_run_graph_requires_captured_current_group(self):
        runner = self.make_runner((1, 2))
        runner.captured_stream_indices = (1, 2)
        for idx, expected in ((0, False), (1, True), (2, True)):
            with patch.object(dmod, "get_current_stream_idx", return_value=idx):
                self.assertEqual(runner.can_run_graph(self.fb()), expected, idx)

    def test_can_run_graph_only_restricted_group(self):
        runner = self.make_runner((2,))
        runner.captured_stream_indices = (2,)
        with patch.object(dmod, "get_current_stream_idx", return_value=1):
            self.assertFalse(runner.can_run_graph(self.fb()))
        with patch.object(dmod, "get_current_stream_idx", return_value=2):
            self.assertTrue(runner.can_run_graph(self.fb()))
            self.assertFalse(runner.can_run_graph(self.fb(bs=65)))  # over max_bs

    def test_can_run_graph_disable_padding_uses_stream_keyed_lookup(self):
        runner = self.make_runner((1, 2))
        runner.captured_stream_indices = (1, 2)
        runner.disable_padding = True
        runner.backend.can_run.return_value = True
        with patch.object(dmod, "get_current_stream_idx", return_value=2):
            self.assertTrue(runner.can_run_graph(self.fb(3)))
        key = runner.backend.can_run.call_args[0][1]
        self.assertEqual(key, ShapeKey(size=3, stream_idx=2))

    def test_capture_loop_skips_uncaptured_groups(self):
        runner = self.make_runner((1, 2))
        runner.captured_stream_indices = (1, 2)
        runner.device = "cpu"
        seen = []
        with patch.object(
            DecodeCudaGraphRunner,
            "_capture_one_stream",
            lambda self, stream_idx=None: seen.append(stream_idx),
        ), patch.object(torch.cpu, "mem_get_info", create=True, return_value=(10, 10)):
            for i in range(3):  # what DecodeCudaGraphRunner.capture() does under pdmux
                runner._capture_one_stream(i)
        self.assertEqual(seen, [1, 2])

    def test_execute_replays_current_groups_graph_with_its_backend(self):
        for idx in (1, 2):
            runner = self.make_runner((1, 2))
            runner.captured_stream_indices = (1, 2)
            bs, steps = 2, 4
            runner.deepep_adapter = Mock()
            runner.model_runner = SimpleNamespace(
                spec_algorithm=Mock(),
                model_config=SimpleNamespace(vocab_size=32),
                server_args=SimpleNamespace(speculative_use_rejection_sampling=False),
                device_timer=None,
            )
            runner.seq_len_fill_value = 1
            runner.buffers = EagleDraftInputBuffers(
                input_ids=torch.zeros(8, dtype=torch.int64),
                req_pool_indices=torch.zeros(8, dtype=torch.int64),
                out_cache_loc=torch.zeros(8 * steps, dtype=torch.int64),
                positions=torch.zeros(8, dtype=torch.int64),
                mrope_positions=torch.zeros(3, 8, dtype=torch.int64),
                rids_int=None,
                bootstrap_room_ids_int=None,
                seq_lens=torch.ones(8, dtype=torch.int64),
                seq_lens_cpu=torch.ones(8, dtype=torch.int64),
                extend_seq_lens=torch.ones(8, dtype=torch.int32),
                topk_p=torch.zeros(8, 1),
                topk_index=torch.zeros(8, 1, dtype=torch.int64),
                draft_probs=None,
                hidden_states=torch.zeros(8, 4),
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                dsa_seed_topk=None,
            )
            out = (torch.zeros(bs), torch.zeros(bs), torch.zeros(bs), None)
            runner.backend.replay.return_value = out
            fb = SimpleNamespace(
                batch_size=bs,
                out_cache_loc=torch.arange(bs * steps),
                seq_lens=torch.full((bs,), 5),
                positions=torch.zeros(bs, dtype=torch.int64),
                req_pool_indices=torch.arange(bs),
                rids_int=None,
                bootstrap_room_ids_int=None,
                seq_lens_sum=10,
                seq_lens_cpu=torch.full((bs,), 5),
                sampling_info=None,
                spec_info=SimpleNamespace(
                    topk_p=torch.zeros(bs, 1),
                    topk_index=torch.zeros(bs, 1, dtype=torch.int64),
                    draft_probs=None,
                    hidden_states=torch.zeros(bs, 4),
                    dsa_topk_indices=None,
                ),
            )
            with patch.object(dmod, "get_current_stream_idx", return_value=idx):
                result = runner.execute(fb)
            self.assertIs(result, out)
            chosen, other = self.group[idx], self.group[3 - idx]
            chosen.init_forward_metadata_out_graph.assert_called_once()
            other.init_forward_metadata_out_graph.assert_not_called()
            runner.draft_attn_backend.init_forward_metadata_out_graph.assert_not_called()
            shape_key, _ = runner.backend.replay.call_args[0]
            self.assertEqual(shape_key, ShapeKey(size=2, stream_idx=idx))


class TestDraftExtendRunnerPerStream(EnvCleanTest):
    def make_runner(self, group_idx=(1, 2)):
        runner = EAGLEDraftExtendCudaGraphRunner.__new__(EAGLEDraftExtendCudaGraphRunner)
        self.group = [None] * 3
        for i in group_idx:
            self.group[i] = Mock(name=f"extend{i}")
        runner.draft_extend_attn_backend = Mock(name="eager_extend")
        runner.draft_extend_attn_backend_group = self.group
        runner.captured_stream_indices = ()
        runner.enable_pdmux = True
        runner.max_bs = 64
        runner.max_num_token = 320
        runner.captured_req_width = 5
        runner.capture_bs = [1, 2, 4, 8]
        runner.require_mlp_tp_gather = False
        runner.require_mlp_sync = False
        runner.disable_padding = False
        runner.backend = Mock()
        return runner

    def fb(self, n=3):
        return SimpleNamespace(
            spec_info=SimpleNamespace(num_tokens_per_req=5),
            seq_lens=torch.zeros(n),
        )

    def test_init_and_selection(self):
        runner = self.make_runner((1, 2))
        with patch.object(emod, "get_stream_groups", return_value=[object()] * 3):
            runner.maybe_init_pdmux()
        self.assertEqual(runner.captured_stream_indices, (1, 2))
        runner.draft_extend_attn_backend.init_cuda_graph_state.assert_not_called()
        for i in (1, 2):
            self.group[i].init_cuda_graph_state.assert_called_once_with(64, 320)
            self.assertIs(runner._extend_backend_for(i), self.group[i])
        self.assertIs(runner._extend_backend_for(None), runner.draft_extend_attn_backend)
        self.assertEqual(runner._make_graph_key(4, 1), ShapeKey(size=4, stream_idx=1))

    def test_can_run_graph_keyed_by_stream(self):
        runner = self.make_runner((2,))
        runner.captured_stream_indices = (2,)
        for idx, expected in ((0, False), (1, False), (2, True)):
            with patch.object(emod, "get_current_stream_idx", return_value=idx):
                self.assertEqual(runner.can_run_graph(self.fb()), expected, idx)
        runner.disable_padding = True
        runner.backend.can_run.return_value = True
        with patch.object(emod, "get_current_stream_idx", return_value=2):
            runner.can_run_graph(self.fb(3))
        self.assertEqual(
            runner.backend.can_run.call_args[0][1], ShapeKey(size=3, stream_idx=2)
        )


class TestWorkerWiring(EnvCleanTest):
    def make_worker(self, steps=4):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.speculative_num_steps = steps
        worker.topk = 1
        worker.speculative_num_draft_tokens = 5
        worker._specialized_graph_memory_usage = {}
        worker._specialized_graph_time_usage = {}
        worker.device = "cuda"
        worker.gpu_id = 0
        factory = Mock()
        factory.create_decode_backend.side_effect = lambda: Mock(name="decode_be")
        factory.create_draft_extend_backend.side_effect = lambda: Mock(name="extend_be")
        worker._draft_backend_factory = factory
        worker.draft_attn_backend = Mock(name="eager_decode")
        worker.draft_extend_attn_backend = Mock(name="eager_extend")
        worker.draft_runner = SimpleNamespace(tp_group=object())
        worker.target_worker = SimpleNamespace(device="cuda")
        return worker, factory

    def test_build_groups_one_instance_per_decode_group(self):
        worker, factory = self.make_worker()
        with patch.object(wmod, "get_stream_groups", return_value=[object()] * 3):
            decode, extend = worker._build_pdmux_draft_backend_groups()
        self.assertIsNone(decode[0])
        self.assertIsNone(extend[0])
        self.assertEqual(len(decode), 3)
        self.assertEqual(len({id(b) for b in decode[1:]}), 2)
        self.assertEqual(len({id(b) for b in extend[1:]}), 2)
        self.assertNotIn(worker.draft_attn_backend, decode)
        self.assertNotIn(worker.draft_extend_attn_backend, extend)
        self.assertEqual(factory.create_decode_backend.call_count, 2)
        self.assertEqual(factory.create_draft_extend_backend.call_count, 2)

    def test_build_groups_restricted_and_single_step(self):
        os.environ["SGLANG_PDMUX_EAGLE_DRAFT_GRAPH_GROUPS"] = "2"
        worker, factory = self.make_worker()
        with patch.object(wmod, "get_stream_groups", return_value=[object()] * 3):
            decode, extend = worker._build_pdmux_draft_backend_groups()
        self.assertEqual([b is not None for b in decode], [False, False, True])
        worker, factory = self.make_worker(steps=1)
        with patch.object(wmod, "get_stream_groups", return_value=[object()] * 3):
            decode, extend = worker._build_pdmux_draft_backend_groups()
        self.assertIsNone(decode)  # 1-step draft has no multi-step decode backend
        self.assertIsNotNone(extend)

    def run_capture(self, *, pdmux=True):
        worker, factory = self.make_worker()
        decode_cls, extend_cls = MagicMock(name="DecodeRunner"), MagicMock(name="ExtendRunner")
        # The CUDA-only attention modules need sgl_kernel/libcuda; stub them.
        class StubDsa:
            pass

        stubs = {}
        for name, attr in (
            ("sglang.srt.layers.attention.dsa_backend", "DeepseekSparseAttnBackend"),
            ("sglang.srt.layers.attention.deepseek_v4_backend", "DeepseekV4AttnBackend"),
            ("sglang.srt.layers.attention.flashmla_backend", "FlashMLABackend"),
        ):
            mod = types.ModuleType(name)
            setattr(mod, attr, StubDsa if attr == "DeepseekSparseAttnBackend" else type(attr, (), {}))
            stubs[name] = mod
        worker.draft_extend_attn_backend = StubDsa()
        with (
            patch.dict(sys.modules, stubs),
            patch.object(wmod, "get_disagg", return_value=SimpleNamespace(enable_pdmux=pdmux)),
            patch.object(wmod, "_is_cpu", False),
            patch.object(wmod, "_is_cuda", True),
            patch.object(wmod, "_is_hip", False),
            patch.object(wmod, "_is_npu", False),
            patch.object(wmod, "_is_xpu", False),
            patch.object(wmod, "check_cuda_graph_backend", return_value=False),
            patch.object(wmod, "get_model", return_value=SimpleNamespace(model_impl="sglang")),
            patch.object(wmod, "get_exec", return_value=MagicMock()),
            patch.object(wmod, "get_batch_sizes_to_capture", return_value=([1, 2], [])),
            patch.object(wmod, "get_available_gpu_memory", return_value=10.0),
            patch.object(wmod, "log_info_on_rank0"),
            patch.object(wmod, "EAGLEDraftCudaGraphRunner", decode_cls),
            patch.object(wmod, "EAGLEDraftExtendCudaGraphRunner", extend_cls),
            patch.object(wmod, "get_stream_groups", return_value=[object()] * 3),
        ):
            worker._capture_cuda_graphs()
        return worker, decode_cls, extend_cls

    def test_capture_under_pdmux_builds_group_runners(self):
        worker, decode_cls, extend_cls = self.run_capture()
        decode_cls.assert_called_once()
        extend_cls.assert_called_once()
        self.assertIs(
            decode_cls.call_args.kwargs["draft_attn_backend_group"],
            worker.draft_attn_backend_group,
        )
        self.assertIs(
            extend_cls.call_args.kwargs["draft_extend_attn_backend_group"],
            worker.draft_extend_attn_backend_group,
        )
        self.assertIs(worker.cuda_graph_runner, decode_cls.return_value)
        self.assertIs(worker.cuda_graph_runner_for_draft_extend, extend_cls.return_value)

    def test_capture_env_switch_forces_eager_under_pdmux(self):
        os.environ["SGLANG_PDMUX_EAGLE_DRAFT_GRAPHS"] = "0"
        worker, decode_cls, extend_cls = self.run_capture()
        decode_cls.assert_not_called()
        extend_cls.assert_not_called()
        self.assertIsNone(worker.cuda_graph_runner)
        self.assertIsNone(worker.cuda_graph_runner_for_draft_extend)
        self.assertIsNone(worker.draft_attn_backend_group)

    def test_capture_without_pdmux_is_unchanged_single_instance(self):
        worker, decode_cls, extend_cls = self.run_capture(pdmux=False)
        self.assertEqual(decode_cls.call_args.kwargs, {})
        self.assertEqual(extend_cls.call_args.kwargs, {})
        self.assertIsNone(worker.draft_attn_backend_group)


class TestDraftForwardBackendOverride(unittest.TestCase):
    def test_steps_use_the_passed_groups_backend(self):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.topk = 1
        worker.speculative_num_steps = 3
        worker.speculative_num_draft_tokens = 4
        worker.hot_token_id = None
        worker.index_share_for_mtp_iteration = False
        worker.seed_dsa_topk_from_draft_extend = False
        worker._topk1_parents_prealloc = torch.zeros(8, 3, dtype=torch.int64)
        worker._topk1_score_indices_prealloc = torch.zeros(8, 1, dtype=torch.int64)
        eager = SimpleNamespace(attn_backends=[Mock(name="e0"), Mock(name="e1")])
        group = SimpleNamespace(attn_backends=[Mock(name="g0"), Mock(name="g1")])
        worker.draft_attn_backend = eager
        seen = []

        def fwd(fb):
            from sglang.srt.model_executor.forward_context import get_attn_backend

            seen.append(get_attn_backend())
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=torch.randn(2, 16),
                    hidden_states=torch.zeros(2, 4),
                )
            )

        worker.draft_runner = SimpleNamespace(
            forward=fwd,
            canary_manager=None,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["X"])),
        )
        spec = SimpleNamespace(
            topk_p=torch.ones(2, 1),
            topk_index=torch.zeros(2, 1, dtype=torch.int64),
            hidden_states=torch.zeros(2, 4),
            draft_probs=None,
        )
        fb = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            spec_info=spec,
            out_cache_loc=torch.arange(2 * 3),
            batch_size=2,
            positions=torch.zeros(2, dtype=torch.int64),
            sampling_info=None,
            input_ids=None,
        )
        spec_cfg = SimpleNamespace(speculative_use_rejection_sampling=False)
        with (
            patch.object(wmod, "get_spec", return_value=spec_cfg),
            patch.object(wmod, "_is_cuda", False),
            patch.object(wmod, "_is_hip", False),
            patch.object(wmod, "per_step_draft_out_cache_loc", return_value=torch.arange(6).reshape(3, 2)),
            patch.object(wmod, "draft_topk1_postprocess", create=True),
            patch.object(wmod, "organize_draft_results", return_value=(None, None, None)),
        ):
            worker.draft_forward(fb, draft_attn_backend=group)
            self.assertEqual(seen, [group.attn_backends[0], group.attn_backends[1]])
            seen.clear()
            worker.draft_forward(fb)
            self.assertEqual(seen, [eager.attn_backends[0], eager.attn_backends[1]])


if __name__ == "__main__":
    unittest.main()
