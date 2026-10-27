# vLLM v0.32.0 for DeepSeek-V4.1-Flash-1017 with a LoRA adapter

The image serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` (rev `1bae0e10`) as the base model and as an abliterated variant: `UnderTides/DeepSeek-V4.1-Flash-1017-heretic-lora`, rev `196c06e3`, rank 1 on `wo_b`. The request's `model` field selects the adapter, which is set by `--enable-lora --lora-modules`. It is the image of tee-bench exp 51 arm l9, Pierre, 2026-10-26: the patched files are byte-identical to the soaked `lab/vllm:v0.32.0-dsv41-lora-rows`.

## Patches against `vllm/vllm-openai:v0.32.0`

Each patch is applied with `patch -p1` only when the upstream file has the expected sha256.

| File | What | Why |
|---|---|---|
| `01-dsv41-supports-lora-wo-b.patch` | `DeepseekV41ForCausalLM` declares `SupportsLoRA`. It fails closed: only `target_modules=['wo_b']`, no vision or aligner LoRA, and MTP is skipped. | The 1017 checkpoint's multimodal class has no LoRA support upstream. |
| `02-moe-lora-only-when-targeted.patch` | MoE layers use the LoRA kernels only when `target_modules` reaches the experts. Backport of vllm#61165. | Without it, any LoRA moves every MoE layer from Humming W4A8 to Marlin, for base requests too. |
| `03-decoder-replay-lora-rows.patch` | SWA bounded decoder replay remaps the punica LoRA token metadata to the replayed rows, then restores it. | Replay runs layers 20-39 on a gathered sub-batch. The LoRA kernels indexed the full batch, which caused illegal memory access and adapters applied to the wrong rows. The other fix, `--no-swa-bounded-replay`, costs prefill speed. |
| `04-cc-disable-uva.patch` | With `VLLM_NEAR_DISABLE_UVA=1`, `is_uva_available()` returns False, so the V2 model runner copies its per-step inputs to the device instead of letting kernels read pinned host memory. | Under TDX + GPU PPCIe the GPU reads a stale snapshot of pinned host memory. Block tables and positions went stale, decode produced NaN logits, and output was garbage with DSpark acceptance 0. Found on the gpu02 dev CVM on 10-27 (tee-bench #16, `syncs/2026-10-27-dsv41-gpu02-uva-fix.md`). |
| `05-engram-cpu-gather.patch` | With `VLLM_NEAR_ENGRAM_CPU_GATHER=1` and `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'`, one Engram table copy in `/dev/shm` is shared by every DP group on the host (fixed file name, flock-elected loader, `.ready` markers, reused on restart) and mapped without `cudaHostRegister`. Rows are gathered on the CPU and copied to the GPU; the result is bit-identical to `_engram_lookup_kernel`. Needs `cudagraph_mode` PIECEWISE. | It makes the 2 × DEP4 host-table layout safe in a CVM. `cudaHostRegister` fails with 801 under PPCIe, and GPU reads of host memory there are stale or panic the guest (`nvidia_uvm` CC fault-decrypt, gpu02 at 15:02 on 10-27). |

Patches 04 and 05 change nothing unless their environment variable is set.

`security-upgrades.txt` lists hash-pinned upgrades that clear the base image's fixable CRITICAL advisories:
- certifi (GHSA-6q48-6c7w-9gw2), both the installed copy and pip's vendored copy;
- uv (GHSA-hq85-5cpj-mr62).

A Trivy scan of the gpu31 build on 2026-10-26 found 0 fixable CRITICAL and 18 HIGH.

## Serving flags (exp 50 best stack + LoRA, per DP rank)

```text
--tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-rank <r> --enable-expert-parallel
--max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.85
--speculative-config '{"method":"dspark","num_speculative_tokens":3}'
--enable-lora --max-loras 1 --max-cpu-loras 1 --max-lora-rank 8 --lora-target-modules wo_b
--lora-modules '{"name":"deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated","path":"<adapter dir>","base_model_name":"deepseek-ai/DeepSeek-V4.1-Flash-1017"}'
env VLLM_HUMMING_INPUT_QUANT_CONFIG={"dtype":"float8e4m3","quant_mode":"dynamic_token"}   # W4A8
```

Publish with `.github/workflows/publish-vllm-dsv41-lora.yaml`, from a merged commit on main. Production pins the resulting digest.
