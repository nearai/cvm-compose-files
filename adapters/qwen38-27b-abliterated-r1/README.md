# Qwen3.8 27B abliterated rank-one LoRA

`adapter_model.safetensors` and `adapter_config.json` are a rank-one PEFT
approximation of the changed output-projection matrices between
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
at `739e3c5b89849f6c238ce1e5b70008612ae42cdd`. Both upstream model
repositories identify their license as Apache-2.0; its text is included here.

## Attribution

This adapter is a derivative work of two Apache-2.0 models, and is
redistributed under the same license:

- Qwen3.8-27B, copyright 2026 Alibaba Cloud (the Qwen team).
- Huihui-Qwen3.8-27B-abliterated, published by huihui-ai, itself derived from
  Qwen3.8-27B.

The adapter weights were changed from both: they approximate the difference
between the two checkpoints above.

The adapter contains 70 changed `o_proj`, `out_proj`, and `down_proj` matrices
from language-model layers, represented by 140 LoRA tensors. MTP and vision
weights were excluded. The aggregate relative weight reconstruction error
against the BF16 deltas was 0.0821. This is an approximation of the upstream
abliterated checkpoint, not the full model. One isolated serving probe showed
a different response to a historical question; no systematic quality, safety,
or load evaluation has been completed.

SHA-256:

```text
3dd325373e6ad720196ce6d30a5d7302d245da87ec45362c95ffcc60a36eb934  adapter_model.safetensors
c2f78fe464c7c24d742e20d0fab6477647f848525736d2e5661f5bb4c66d0faf  adapter_config.json
```

The Compose downloader in `prod/small-models.yaml` fetches these files from this repository and
checks both hashes before mounting the adapter into SGLang. The FP8 base model
has its own pinned Hugging Face revision in the Compose file.
