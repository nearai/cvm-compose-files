# vLLM v0.33.0 for DeepSeek-V4.1-Flash-1017 with a LoRA adapter

The base is `vllm/vllm-openai:v0.33.0` (upstream tag `4ba787ec9`), pinned by digest in the Dockerfile. Patches 01-06 are ported from the earlier v0.32.0 build dir `docker/vllm-dsv41-lora/`, which is in release tags up to `v0.0.582`; patch 07 is new. `PROVENANCE` lists each patch with its sha256.

| patch | what | against v0.33.0 |
|---|---|---|
| 01-dsv41-supports-lora-wo-b | DS4.1 declares LoRA support for wo_b | not upstream; same lines as v0.32.0 |
| 02-moe-lora-only-when-targeted | MoE LoRA wrappers only when targeted | not upstream; same as v0.32.0 |
| 03-decoder-replay-lora-rows | map LoRA token indices to decoder-replay rows | **adapted** to v0.33's `num_padded` row padding at the same spot (vllm#62384): the padded rows get LoRA index -1 (no adapter), so the mapping length matches the padded batch |
| 04-cc-disable-uva | `VLLM_NEAR_DISABLE_UVA=1` makes `is_uva_available()` False (TDX+PPCIe stale pinned reads) | same as v0.32.0. The new v0.33 UVA users (DS4.1 `model_state.py` replay pools) go through `UvaBufferPool`, which falls back to `NonUvaBuffer`, so they are covered |
| 05-engram-cpu-gather | CC-safe host Engram (CPU gather) | **adapted** to upstream peer-HBM Engram (vllm#62091): keeps the upstream peer logic and adds the UVA bypass on the offload check. Inert with `cpu_offload:false` |
| 06-engram-fast-gather | single sync + pinned async H2D for 05 | same as v0.32.0. Inert with `cpu_offload:false` |
| 07-engram-peer-reads-kill-switch | `VLLM_NEAR_ENGRAM_PEER_READS=0` keeps the v0.32 DP-collective lookup | new; default on (upstream behaviour) |

**Peer-HBM Engram reads:** with vllm#62091, when HBM Engram runs with TP1 per rank and NVLink fully connected, each rank reads Engram rows from its peers' HBM through IPC pointers instead of using DP collectives. That is GPU-to-GPU traffic under TDX+PPCIe. `VLLM_NEAR_ENGRAM_PEER_READS=0` restores the v0.32 path without an image change.

**Security:** the v0.33.0 base needs no Python package upgrades; the Dockerfile upgrades `linux-libc-dev` (see `PROVENANCE`).
