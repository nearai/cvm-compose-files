"""Exercise the exact pinned methods without importing or initializing CUDA."""
import ast
import asyncio
import concurrent.futures
from enum import Enum
import logging
import re
from types import SimpleNamespace as NS
import unittest

from patch_runtime import PREIMAGES, ROOT, patched_sources

BASE = 'python/sglang/srt/multimodal/processors/base_processor.py'
VIDEO = 'python/sglang/srt/utils/video_decoder.py'
ORIGINAL = {p: (ROOT / p).read_text() for p in PREIMAGES}
PATCHED = patched_sources(ORIGINAL)


class Modality(Enum):
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3


def method(source, class_name, name, namespace):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    exec('from __future__ import annotations\n' + ast.unparse(node), namespace)
    return namespace[name]


def context(enabled=True):
    class Tensor:
        pass

    class BaseImageProcessor:
        pass

    return dict(
        envs=NS(SGLANG_MM_CPU_PREPROCESS=NS(get=lambda: enabled)),
        Modality=Modality, re=re, torch=NS(Tensor=Tensor),
        BaseImageProcessor=BaseImageProcessor,
        _is_cpu=False, _is_xpu=False, _is_npu=False,
        get_server_args=lambda: NS(base_gpu_id=0, rl_on_policy_target=None),
        asyncio=asyncio, logger=logging.getLogger('test'),
        BaseMultiModalProcessorOutput=lambda **kw: NS(**kw),
    )


class PatchTests(unittest.TestCase):
    def test_unknown_base_rejected(self):
        changed = dict(ORIGINAL)
        changed[BASE] += '\n'
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            patched_sources(changed)

    def test_all_outputs_compile(self):
        for name, value in PATCHED.items():
            compile(value, name, 'exec')

    def processor_device(self, source, enabled):
        ns = context(enabled)
        captured = {}
        class Processor:
            image_processor = ns['BaseImageProcessor']()
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return {}
        self_obj = NS(_processor=Processor(), image_config={'device': 'cuda:4'},
                      video_config={'device': 'cuda:5'}, audio_config={},
                      disable_fast_image_processor=False,
                      _tokenizer_auto_adds_specials=False,
                      keep_mm_feature_on_device=False, FEATURE_NAMES=[])
        run = method(source, 'BaseMultimodalProcessor', 'process_mm_data', ns)
        run(self_obj, 'synthetic', images=['image'], videos=['video'])
        return captured

    def test_default_preprocessor_is_cuda(self):
        self.assertEqual(self.processor_device(ORIGINAL[BASE], False)['device'], 'cuda:0')

    def test_cpu_overrides_both_nested_modalities(self):
        result = self.processor_device(PATCHED[BASE], True)
        self.assertEqual(result['device'], 'cpu')
        self.assertNotIn('device', result['images_kwargs'])
        self.assertNotIn('device', result['videos_kwargs'])

    def test_disabled_mitigation_preserves_existing_options(self):
        self.assertEqual(self.processor_device(PATCHED[BASE], False),
                         self.processor_device(ORIGINAL[BASE], False))

    def loader(self, enabled=True, image=None, video=None, source=None):
        ns = context(enabled)
        ns.update(load_image=image, load_video=video, load_audio=lambda *a: None)
        cls = NS(_is_preprocessed_input=lambda data: False, gpu_image_decode=True)
        run = method(source or PATCHED[BASE], 'BaseMultimodalProcessor', '_load_single_item', ns)
        return lambda data, modality, **kw: run(cls, data, modality, **kw)

    def test_cpu_disables_nvjpeg(self):
        observed = []
        image = NS(mode='RGB', load=lambda: None)
        load = self.loader(image=lambda data, gpu: (observed.append(gpu) or image, None))
        self.assertIs(load('synthetic', Modality.IMAGE), image)
        self.assertEqual(observed, [False])

    def test_cpu_disables_video_cuda(self):
        observed = []
        load = self.loader(video=lambda data, **kw: observed.append(kw) or 'decoded')
        self.assertEqual(load('synthetic', Modality.VIDEO, frame_count_limit=128), 'decoded')
        self.assertEqual(observed, [{'use_gpu': False}])

    def test_nonpositive_frame_dimensions_become_client_error(self):
        for dimension in ('height', 'width'):
            for value in ('0', '-1'):
                def invalid(*a, **kw):
                    raise RuntimeError(f'FrameDims.{dimension} must be > 0, got: {value}')
                with self.subTest(dimension=dimension, value=value):
                    with self.assertRaisesRegex(ValueError, '^Invalid video: non-positive frame dimensions$'):
                        self.loader(video=invalid)('synthetic', Modality.VIDEO)

    def test_cuda_and_oom_are_not_client_errors(self):
        for message in ('CUDA error: an illegal memory access', 'CUDA out of memory',
                        'FrameDims.height must be > 0, got: 0.5'):
            def broken(*a, **kw):
                raise RuntimeError(message)
            with self.subTest(message=message):
                with self.assertRaises(RuntimeError):
                    self.loader(video=broken)('synthetic', Modality.VIDEO)

    def test_invalid_input_does_not_echo_media(self):
        def invalid(*a):
            raise ValueError('synthetic-private-payload')
        with self.assertRaisesRegex(ValueError, '^Invalid image input$') as caught:
            self.loader(image=invalid)('synthetic-private-payload', Modality.IMAGE)
        self.assertTrue(caught.exception.__suppress_context__)

    def run_async_loader(self, source):
        ns = context()
        future = concurrent.futures.Future()
        future.set_exception(ValueError('Invalid video: non-positive frame dimensions'))
        instance = NS(_submit_mm_data_loading_tasks_simple=lambda data, modality, *a:
                      [(modality, 0, future)] if data else [])
        run = method(source, 'BaseMultimodalProcessor', 'fast_load_mm_data', ns)
        return asyncio.run(run(instance, prompt='synthetic', multimodal_tokens=NS(),
                               video_data=['synthetic']))

    def test_original_async_wrapper_causes_500_class(self):
        with self.assertLogs('test', level='ERROR'):
            with self.assertRaises(RuntimeError):
                self.run_async_loader(ORIGINAL[BASE])

    def test_async_loader_preserves_400_class(self):
        with self.assertRaises(ValueError):
            self.run_async_loader(PATCHED[BASE])

    def test_legacy_loader_preserves_value_error(self):
        ns = context()
        run = method(PATCHED[BASE], 'BaseMultimodalProcessor', 'legacy_load_mm_data', ns)
        # Exercise the real method's loading catch, not a rewritten stand-in.
        tokens = NS(get_combined_regex=lambda: re.compile('(VIDEO)'),
                    get_modality_of_token=lambda text: Modality.VIDEO if text == 'VIDEO' else None,
                    video_token='VIDEO', image_token=None, audio_token=None)
        future = concurrent.futures.Future()
        future.set_exception(ValueError('Invalid video: non-positive frame dimensions'))
        instance = NS(submit_data_loading_tasks=lambda **kw:
                      ([future], [(Modality.VIDEO, 'synthetic', None)]))
        with self.assertRaisesRegex(ValueError, '^Invalid video: non-positive frame dimensions$'):
            asyncio.run(run(instance, prompt='VIDEO', multimodal_tokens=tokens,
                            video_data=['synthetic']))

    def test_video_constructor_forces_cpu_without_changing_decoder(self):
        ns = context()
        seen = []
        ns.update(_BACKEND='torchcodec', VideoDecoder=lambda source, **kw: seen.append(kw) or object(),
                  _try_cuda_backend=lambda: self.fail('CUDA decoder must not be initialized'))
        run = method(PATCHED[VIDEO], 'VideoDecoderWrapper', '__init__', ns)
        run(NS(), b'synthetic', device='cuda')
        self.assertEqual(seen, [{'dimension_order': 'NHWC'}])

    def test_cpu_video_frames_do_not_pin_memory_or_initialize_cuda(self):
        ns = context()
        ns['_BACKEND'] = 'torchcodec'
        tensor = NS(pin_memory=lambda: self.fail('CPU path must not allocate pinned CUDA memory'))
        instance = NS(_num_decode_threads=1,
                      _decoder=NS(get_frames_at=lambda indices: NS(data=tensor)))
        run = method(PATCHED[VIDEO], 'VideoDecoderWrapper', 'get_frames_as_tensor', ns)
        self.assertIs(run(instance, [0]), tensor)


if __name__ == '__main__':
    unittest.main()
