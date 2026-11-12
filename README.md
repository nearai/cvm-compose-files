# cvm-compose-files

Docker Compose files for confidential inference, and the build contexts of the images they pin.

## Status

Every deployable file, the served model names it serves, and whether it is `live` or a `candidate`.

<!-- status:begin -->
| file | serves | status |
|---|---|---|
| `cleanup-hf-model.yaml` | - | live |
| `prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml` | deepseek-ai/DeepSeek-V4.1-Flash-1017 | live |
| `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml` | z-ai/glm-5.3-flash | live |
| `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` | z-ai/glm-5.3-flash | live |
| `prod/qwen38-27b-abliterated-cc-1gpu.yaml` | Qwen/Qwen3.8-27B | live |
| `prod/rotate-compose-manager-token.yaml` | - | live |
| `prod/sage-pull.yaml` | - | live |
| `prod/sage.yaml` | - | live |
| `prod/small-models.yaml` | Qwen/Qwen3-Embedding-0.6B,Qwen/Qwen3-Reranker-0.6B,Qwen/Qwen3-VL-30B-A3B-Instruct,Qwen/Qwen3.6-35B-A3B-FP8,Qwen/Qwen3.8-27B,black-forest-labs/FLUX.2-klein-4B,openai/whisper-large-v3,z-ai/glm-5.3-flash | live |
<!-- status:end -->
