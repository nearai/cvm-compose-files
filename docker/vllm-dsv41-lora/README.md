# vLLM v0.32.0 for DeepSeek-V4.1-Flash-1017 with a LoRA adapter

The image serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` (rev `1bae0e10`) as the base model and as an abliterated variant: `UnderTides/DeepSeek-V4.1-Flash-1017-heretic-lora`, rev `196c06e3`, rank 1 on `wo_b`. The request's `model` field selects the adapter, which is set by `--enable-lora --lora-modules`.

## Patches against `vllm/vllm-openai:v0.32.0`

Each patch is applied with `patch -p1` only when the upstream file has the expected sha256.

| File | What | Why |
|---|---|---|
| `01-dsv41-supports-lora-wo-b.patch` | `DeepseekV41ForCausalLM` declares `SupportsLoRA`. It fails closed: only `target_modules=['wo_b']`, no vision or aligner LoRA, and MTP is skipped. | The 1017 checkpoint's multimodal class has no LoRA support upstream. |
| `02-moe-lora-only-when-targeted.patch` | MoE layers use the LoRA kernels only when `target_modules` reaches the experts. Backport of vllm#61165. | Without it, any LoRA moves every MoE layer from Humming W4A8 to Marlin, for base requests too. |
| `03-decoder-replay-lora-rows.patch` | SWA bounded decoder replay remaps the punica LoRA token metadata to the replayed rows, then restores it. | Replay runs layers 20-39 on a gathered sub-batch. The LoRA kernels indexed the full batch, which caused illegal memory access and adapters applied to the wrong rows. The other fix, `--no-swa-bounded-replay`, costs prefill speed. |
| `04-cc-disable-uva.patch` | With `VLLM_NEAR_DISABLE_UVA=1`, `is_uva_available()` returns False, so the V2 model runner copies its per-step inputs to the device instead of letting kernels read pinned host memory. | Under TDX + GPU PPCIe the GPU reads a stale snapshot of pinned host memory. Block tables and positions go stale, decode produces NaN logits, and output is garbage with DSpark acceptance 0. |
| `05-engram-cpu-gather.patch` | With `VLLM_NEAR_ENGRAM_CPU_GATHER=1` and `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'`, one Engram table copy in `/dev/shm` is shared by every DP group on the host (fixed file name, flock-elected loader, `.ready` markers, reused on restart) and mapped without `cudaHostRegister`. Rows are gathered on the CPU and copied to the GPU; the result is bit-identical to `_engram_lookup_kernel`. Needs `cudagraph_mode` PIECEWISE. | It makes the 2 × DEP4 host-table layout safe in a CVM. `cudaHostRegister` fails with 801 under PPCIe, and GPU reads of host memory there are stale or fault in `nvidia_uvm` (CC fault-decrypt). |
| `06-engram-fast-gather.patch` | With `VLLM_NEAR_ENGRAM_FAST_GATHER=1` on top of patch 05, the model copies the step's Engram hash ids to the host once for every Engram layer (patch 05 synced once per layer), each layer gathers its rows into its own preallocated pinned buffers, and the bytes reach the GPU with non-blocking copies before dequantization there. The host sync runs in an `@eager_break_during_capture` function, so it stays outside the V2 runner's breakable PIECEWISE graphs. The GPU still never reads host memory. Use with `OMP_NUM_THREADS=8` per rank: vLLM otherwise sets one torch thread per worker, which makes the CPU gather (and weight loading) single-threaded. | Patch 05 syncs once per layer; one sync per step and pinned non-blocking copies cut that host work. |
| `07-cc-explicit-copies.patch` | With `VLLM_CONFIDENTIAL_COMPUTE=1`, the V2 model runner keeps its non-blocking copies non-blocking under CC. It is a port of the open upstream PR vllm#59981 ("V2 runner decode optimizations for NVIDIA Confidential Computing", head `0340d9a7`), adapted to this image, not a backport of merged code. It adds `vllm/v1/conf_compute_utils.py` and changes `vllm/v1/worker/gpu/{buffer_utils,async_utils,spec_decode/utils}.py`. With the variable set: V2 runner buffers use explicit copies from pinned host memory instead of UVA. Torch's non-blocking H2D copies (`copy_`, `.to`, `.cuda`) go through an idle staging stream. Output and draft-token D2H copies are issued from a worker thread on streams created outside torch's stream pool. The EP-fault `query_fault()` kernel runs on the main stream. Copies of 32 MiB or more, and copies during graph capture or torch.compile tracing, keep the original path. Not ported: the PR's `model_runner.py`, `adaptive_verification.py` and encoder-model hunks (the same effect comes from `async_utils`/`buffer_utils`; the rest is unused here). | Under bounce-buffer CC, `cudaMemcpyAsync` returns only after the copy and all earlier work on its stream finish, so every "non-blocking" copy on the compute stream waits for the GPU to drain and async scheduling stops hiding host work. |

Patches 04, 05, 06 and 07 change nothing unless their environment variable is set.

Patch 07 is off unless `VLLM_CONFIDENTIAL_COMPUTE=1` is set explicitly. Unlike the upstream PR it does not auto-detect CC through NVML: unset, `0` or any other value is off. When the variable is off, the code paths match the image without patch 07 in effect: UVA buffers as before, `NonUvaBuffer` with pageable host memory (for example under `VLLM_NEAR_DISABLE_UVA=1`), plain `torch.cuda.Stream` copy streams, `query_fault()` on the copy stream, and no torch patches installed. The only addition is an import of `conf_compute_utils`, which saves references to torch methods and has no side effects. When it is on, look for `NVIDIA Confidential Computing copy paths enabled by VLLM_CONFIDENTIAL_COMPUTE='1'` and `issuing non-blocking H2D copies on a staging stream` in the worker logs. `VLLM_CONFIDENTIAL_COMPUTE_KEEP_UVA=1` (bare metal only, never under real CC) keeps UVA buffers with the CC copy paths on, to measure their cost separately.

With patch 07 on, keep `VLLM_NEAR_DISABLE_UVA=1` set as well: 07 does not disable the global `is_uva_available()`, so Engram UVA views stay governed by patch 04. With CC on, in-place uploads into `NonUvaBuffer` rely on the pool slot-retirement invariant (the host waits on `AsyncOutput.copy_event`, which is recorded before propose). To turn it off, unset `VLLM_CONFIDENTIAL_COMPUTE` and restart.

`security-upgrades.txt` lists hash-pinned upgrades that clear the base image's fixable CRITICAL advisories:
- certifi (GHSA-6q48-6c7w-9gw2), both the installed copy and pip's vendored copy;
- uv (GHSA-hq85-5cpj-mr62).

## Serving flags (per DP rank)

```text
--tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-rank <r> --enable-expert-parallel
--max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.85
--speculative-config '{"method":"dspark","num_speculative_tokens":3}'
--enable-lora --max-loras 1 --max-cpu-loras 1 --max-lora-rank 8 --lora-target-modules wo_b
--lora-modules '{"name":"deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated","path":"<adapter dir>","base_model_name":"deepseek-ai/DeepSeek-V4.1-Flash-1017"}'
env VLLM_HUMMING_INPUT_QUANT_CONFIG={"dtype":"float8e4m3","quant_mode":"dynamic_token"}   # W4A8
```

Publish with `.github/workflows/publish-vllm-dsv41-lora.yaml`, from a merged commit on main. Production pins the resulting digest.
