# vLLM v0.33.0 for DeepSeek-V4.1-Flash-1017 with a LoRA adapter

The base is `vllm/vllm-openai:v0.33.0` (upstream tag `4ba787ec9`), pinned by digest in the Dockerfile. Patches 01-06 and 08 are ported from `docker/vllm-dsv41-lora/` (v0.32.0); patches 07 and 09 are new. `SHA256SUMS` lists each patch with its sha256.

| patch | what | against v0.33.0 |
|---|---|---|
| 01-dsv41-supports-lora-wo-b | DS4.1 declares LoRA support for wo_b | not upstream; same lines as v0.32.0 |
| 02-moe-lora-only-when-targeted | MoE LoRA wrappers only when targeted | not upstream; same as v0.32.0 |
| 03-decoder-replay-lora-rows | map LoRA token indices to decoder-replay rows | **adapted** to v0.33's `num_padded` row padding at the same spot (vllm#62384): the padded rows get LoRA index -1 (no adapter), so the mapping length matches the padded batch |
| 04-cc-disable-uva | `VLLM_NEAR_DISABLE_UVA=1` makes `is_uva_available()` False (TDX+PPCIe stale pinned reads) | same as v0.32.0. The new v0.33 UVA users (DS4.1 `model_state.py` replay pools) go through `UvaBufferPool`, which falls back to `NonUvaBuffer`, so they are covered |
| 05-engram-cpu-gather | CC-safe host Engram (CPU gather) | **adapted** to upstream peer-HBM Engram (vllm#62091): keeps the upstream peer logic and adds the UVA bypass on the offload check. Inert with `cpu_offload:false` |
| 06-engram-fast-gather | single sync + pinned async H2D for 05 | same as v0.32.0. Inert with `cpu_offload:false` |
| 07-engram-peer-reads-kill-switch | `VLLM_NEAR_ENGRAM_PEER_READS=0` keeps the v0.32 DP-collective lookup | new; default on (upstream behaviour) |
| 08-cc-explicit-copies | `VLLM_CONFIDENTIAL_COMPUTE=1` keeps non-blocking H2D/D2H non-blocking under NVIDIA CC | same lines as v0.32.0: the three files it edits are unchanged in v0.33.0 within its context, and `vllm/v1/conf_compute_utils.py` is new |
| 09-cc-retain-ep-fault-flag | the async D2H keeps a reference to the EP fault flag until the copy has run | new; under CC the flag is written on the main stream, so without this a freed block can be reused before the copy stream reads it |

**Confidential Computing copy paths:** under bounce-buffer CC, `cudaMemcpyAsync` is host-synchronous, so the V2 runner's non-blocking copies serialise against the compute stream. Patch 08 (port of vllm#59981) runs H2D on a per-device staging stream and issues D2H from a worker thread, and makes the `NonUvaBuffer` fallback use pinned host memory with explicit copies. Every behaviour change is behind `is_confidential_compute()`, which is True only when `VLLM_CONFIDENTIAL_COMPUTE` is exactly `1`; there is no NVML auto-detection, so an engine that does not set the variable runs the same paths as `vllm-dsv41-v033`. `VLLM_CONFIDENTIAL_COMPUTE_KEEP_UVA=1` keeps UVA buffers with the copy paths on, for bare-metal attribution only.

**Peer-HBM Engram reads:** with vllm#62091, when HBM Engram runs with TP1 per rank and NVLink fully connected, each rank reads Engram rows from its peers' HBM through IPC pointers instead of using DP collectives. That is GPU-to-GPU traffic under TDX+PPCIe. `VLLM_NEAR_ENGRAM_PEER_READS=0` restores the v0.32 path without an image change.

**Security:** the v0.33.0 base needs no Python package upgrades; the Dockerfile upgrades `linux-libc-dev` to the exact noble-security fix.
