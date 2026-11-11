# DS4.1 vLLM: patch triage v0.32.0 -> v0.33.0

The base is `vllm/vllm-openai:v0.32.0` (befccf0e) plus patches 01-06 from `docker/vllm-dsv41-lora/`. The target is v0.33.0 (tag 4ba787ec9, 10-31, final on PyPI and Docker Hub). The port is branch `near-033` in a scratch clone.

| patch | what | v0.33.0 | verdict |
|---|---|---|---|
| 01-dsv41-supports-lora-wo-b | DS4.1 declares LoRA support for wo_b | not upstream | ported, clean (same lines) |
| 02-moe-lora-only-when-targeted | MoE LoRA wrappers only when targeted | not upstream | ported, clean |
| 03-decoder-replay-lora-rows | map LoRA token indices to decoder-replay rows | not upstream; v0.33 adds `num_padded` row padding at the same spot (#62384) | **adapted**: the padded rows get LoRA index -1 (no adapter), so the mapping length matches the padded batch |
| 04-cc-disable-uva | `VLLM_NEAR_DISABLE_UVA=1` makes `is_uva_available()` False (TDX+PPCIe stale pinned reads) | not upstream | ported, clean. The new v0.33 UVA users (DS4.1 `model_state.py` replay pools) go through `UvaBufferPool`, which falls back to `NonUvaBuffer`, so they are covered |
| 05-engram-cpu-gather | CC-safe host Engram (CPU gather) | not upstream; conflicts with #62091 (peer-HBM Engram) | **adapted** (kept upstream peer logic; NEAR UVA bypass on the offload check). Inert in prod: DEP8 runs `cpu_offload:false` |
| 06-engram-fast-gather | single sync + pinned async H2D for 05 | not upstream | ported, clean. Inert in prod (same reason) |
| 07-engram-peer-reads-kill-switch (NEW) | `VLLM_NEAR_ENGRAM_PEER_READS=0` keeps the v0.32 DP-collective lookup | n/a | new; default on (upstream behaviour) |

**Behaviour change to validate in a CVM:** upstream #62091 (v0.33). With HBM Engram on DEP8 / TP1 and NVLink fully connected (our gpu02/gpu04 layout), each rank now reads Engram rows from its peers' HBM through IPC pointers instead of using DP collectives. That is new GPU-to-GPU traffic under TDX+PPCIe. If it misbehaves, `VLLM_NEAR_ENGRAM_PEER_READS=0` restores the v0.32 path without an image change.

**Security:** only three advisories still affect v0.32.0 (GHSA-q9v9, -5r3c, -qrj2), and none is reachable in our setup (no `--skip-tokenizer-init`, text-only, pinned snapshot). Moving to v0.33.0 is hygiene, not urgent.
