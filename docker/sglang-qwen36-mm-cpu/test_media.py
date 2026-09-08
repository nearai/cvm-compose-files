"""Real image/video preprocessing smoke, deliberately run with no GPU devices.

Downloads only the pinned public processor/tokenizer, never model weights.
It does not claim to qualify model inference, signatures, or a production CVM.
"""
import ast
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from PIL import Image
import torch
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata
import sglang.srt.multimodal.processors.base_processor as base
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.srt.environ import envs

MODEL = 'Qwen/Qwen3.6-35B-A3B-FP8'
REVISION = '95a723d08a9490559dae23d0cff1d9466213d989'
assert envs.SGLANG_MM_CPU_PREPROCESS.get()
assert not torch.cuda.is_available(), 'This smoke must run without GPU devices'
assert not torch.cuda.is_initialized()
processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
args = NS(base_gpu_id=0, rl_on_policy_target=None)
instance = NS(_processor=processor, image_config={}, video_config={}, audio_config={},
              disable_fast_image_processor=False, _tokenizer_auto_adds_specials=False,
              keep_mm_feature_on_device=False, use_cuda_ipc=False,
              FEATURE_NAMES=base.BaseMultimodalProcessor.FEATURE_NAMES if hasattr(base.BaseMultimodalProcessor, 'FEATURE_NAMES') else ['pixel_values', 'pixel_values_videos'])
rows = []
for fmt in ('PNG', 'JPEG'):
    for color in ('red', 'blue'):
        buf = BytesIO()
        Image.new('RGB', (224, 224), color).save(buf, format=fmt)
        image = base.BaseMultimodalProcessor._load_single_item(buf.getvalue(), Modality.IMAGE)
        text = processor.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': 'Name the color.'}]}],
            tokenize=False, add_generation_prompt=True)
        with patch.object(base, 'get_server_args', return_value=args):
            result = base.BaseMultimodalProcessor.process_mm_data(instance, text, images=[image])
        pixels = result['pixel_values']
        assert pixels.device.type == 'cpu' and torch.isfinite(pixels).all()
        reference = processor(text=[text], images=[image], padding=True,
                              return_tensors='pt', device='cpu')
        torch.testing.assert_close(pixels, reference['pixel_values'], rtol=0, atol=0)
        assert torch.equal(result['image_grid_thw'], reference['image_grid_thw'])
        rows.append({'kind': fmt, 'color': color, 'shape': list(pixels.shape), 'device': pixels.device.type})

# Reuse the reviewed, wholly synthetic blue H264 clip; do not execute its module.
tree = ast.parse(Path('/repo/scripts/qwen_qualify.py').read_text())
encoded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'BLUE_VIDEO' for t in n.targets))
clip = base64.b64decode(encoded)
def decode(_):
    with VideoDecoderWrapper(clip, device='cuda') as decoder:
        assert len(decoder) == 4
        frames = decoder.get_frames_as_tensor([0, 1, 2, 3])
        assert frames.device.type == 'cpu' and tuple(frames.shape) == (4, 224, 224, 3)
        assert frames[..., 2].float().mean() > 240
        assert frames[..., :2].float().mean() < 10
        return True
assert decode(0)
with ThreadPoolExecutor(max_workers=4) as pool:
    assert all(pool.map(decode, range(16)))
with VideoDecoderWrapper(clip) as decoder:
    frames = decoder.get_frames_as_tensor([0, 1, 2, 3])
    text = processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'video'}, {'type': 'text', 'text': 'Name the color.'}]}],
        tokenize=False, add_generation_prompt=True)
    instance.video_config = {'do_sample_frames': False}
    with patch.object(base, 'get_server_args', return_value=args):
        result = base.BaseMultimodalProcessor.process_mm_data(
            instance, text, videos=[frames.numpy()], video_metadata=[VideoMetadata(
                total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])])
    pixels = result['pixel_values_videos']
    assert pixels.device.type == 'cpu' and torch.isfinite(pixels).all()
    rows.append({'kind': 'video_processor', 'shape': list(pixels.shape), 'device': pixels.device.type})
assert not torch.cuda.is_initialized(), 'Media processing initialized CUDA'
print(json.dumps({'images': rows, 'video_serial': 1, 'video_concurrent': 16,
                  'cuda_initialized': False, 'model_weights_loaded': False, 'status': 'ok'}))
