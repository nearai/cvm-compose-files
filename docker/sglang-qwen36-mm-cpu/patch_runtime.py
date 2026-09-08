"""Apply a narrow, opt-in multimodal CPU mitigation to the pinned image only."""
import hashlib
from pathlib import Path

ROOT = Path('/sgl-workspace/sglang')
PREIMAGES = {
    'python/sglang/srt/environ.py': '60d9d0230013fda286f3f0fc9c2c9f9d332ad04424a306c8ce8a2cb035ed1675',
    'python/sglang/srt/multimodal/processors/base_processor.py': '5b05a9be3d54c539e0fde4272b2d2dbc2f2c1450b6c19eca1b1c4862eeff55b5',
    'python/sglang/srt/utils/video_decoder.py': 'c797bc40320a0485736c5f44df2dc1745e925ca2dce5e4eac4ca5375b9baff16',
}


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Pinned source anchor mismatch')
    return source.replace(before, after, 1)


def patched_sources(sources):
    for path, digest in PREIMAGES.items():
        if hashlib.sha256(sources[path].encode()).hexdigest() != digest:
            raise ValueError('Pinned source digest mismatch: ' + path)
    out = dict(sources)
    env, base, video = PREIMAGES
    out[env] = replace_once(out[env], '    # VLM\n', '''    # VLM
    # Keep media decode and preprocessing out of the tokenizer's CUDA context.
    # Model/vision-encoder inference still runs on its assigned GPU.
    SGLANG_MM_CPU_PREPROCESS = EnvBool(False)
''')
    out[base] = replace_once(out[base], '''        # Avoid double BOS when the chat template already wrote one.
''', '''        if envs.SGLANG_MM_CPU_PREPROCESS.get():
            # Transformers rejects duplicate common/nested device options.
            # Keep only the common CPU selection; retain all other options.
            kwargs["device"] = "cpu"
            for key in ("images_kwargs", "videos_kwargs"):
                if key in kwargs:
                    kwargs[key] = {k: v for k, v in kwargs[key].items() if k != "device"}

        # Avoid double BOS when the chat template already wrote one.
''')
    out[base] = replace_once(out[base], '                img, _ = load_image(data, cls.gpu_image_decode)\n', '''                img, _ = load_image(
                    data, cls.gpu_image_decode and not envs.SGLANG_MM_CPU_PREPROCESS.get()
                )
''')
    out[base] = replace_once(out[base], '                return load_video(data, frame_count_limit)\n', '''                return load_video(
                    data,
                    use_gpu=False if envs.SGLANG_MM_CPU_PREPROCESS.get() else frame_count_limit,
                )
''')
    before = '''        except ValueError as e:
            # Bad input (e.g. invalid base64) -> 400, not 500.
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            raise ValueError(f"Error while loading data {data_str}: {e}") from e
        except Exception as e:
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            raise RuntimeError(f"Error while loading data {data_str}: {e}") from e
'''
    out[base] = replace_once(out[base], before, '''        except ValueError:
            # Do not echo media bytes, URLs, or decoder exception payloads.
            raise ValueError(f"Invalid {modality.name.lower()} input") from None
        except Exception as e:
            # Narrow decoder rejection: do not turn CUDA/OOM/server failures
            # into client errors. The bad-dimension signature is deterministic.
            if modality == Modality.VIDEO and re.search(
                r"FrameDims\\.(?:height|width) must be > 0, got: (?:0|-[0-9]+)(?![0-9.])",
                str(e),
            ):
                raise ValueError("Invalid video: non-positive frame dimensions") from None
            raise RuntimeError(f"Failed to load {modality.name.lower()} input") from e
''')
    out[base] = replace_once(out[base], '''                result = await asyncio.wrap_future(future)
            except Exception as e:
''', '''                result = await asyncio.wrap_future(future)
            except ValueError:
                # Preserve the client-error classification through the async hop.
                raise
            except Exception as e:
''')
    out[base] = replace_once(out[base], '''            except StopIteration as e:
''', '''            except ValueError:
                # The legacy/dynamic-frame path must preserve it as well.
                raise
            except StopIteration as e:
''')
    out[video] = replace_once(out[video], 'import numpy as np\n', '''import numpy as np
from sglang.srt.environ import envs
''')
    out[video] = replace_once(out[video], '''        self._source = source
''', '''        if envs.SGLANG_MM_CPU_PREPROCESS.get():
            device = "cpu"
        self._source = source
''')
    # Pinned host memory allocation can initialize CUDA too. CPU preprocessing
    # returns ordinary CPU tensors; the GPU inference consumer handles transfer.
    out[video] = replace_once(out[video], '            return batch.data.pin_memory()\n', '            return batch.data if envs.SGLANG_MM_CPU_PREPROCESS.get() else batch.data.pin_memory()\n')
    out[video] = replace_once(out[video], '            return torch.from_numpy(arr).pin_memory()\n', '''            tensor = torch.from_numpy(arr)
            return tensor if envs.SGLANG_MM_CPU_PREPROCESS.get() else tensor.pin_memory()
''')
    out[video] = replace_once(out[video], '        return torch.cat(results, dim=0).pin_memory()\n', '''        tensor = torch.cat(results, dim=0)
        return tensor if envs.SGLANG_MM_CPU_PREPROCESS.get() else tensor.pin_memory()
''')
    for path, source in out.items():
        compile(source, path, 'exec')
    return out


if __name__ == '__main__':
    originals = {path: (ROOT / path).read_text() for path in PREIMAGES}
    for path, source in patched_sources(originals).items():
        (ROOT / path).write_text(source)
