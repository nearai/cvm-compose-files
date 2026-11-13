FROM vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e

LABEL org.opencontainers.image.title="vLLM v0.33.0 DeepSeek-V4.1-Flash LoRA production derivative"
LABEL org.opencontainers.image.description="vLLM v0.33.0 with wo_b-only LoRA on DeepseekV41ForCausalLM, MoE kept on Humming under attention-only LoRA, LoRA-aware (and padding-aware) SWA decoder replay, the TDX+PPCIe UVA switch, the CC-safe Engram CPU gather, a kill switch for v0.33's peer-HBM Engram reads, and the Confidential Computing copy paths"
LABEL org.opencontainers.image.version="v0.33.0"
LABEL nearai.vllm.base="vllm/vllm-openai:v0.33.0"
LABEL nearai.vllm.base.linux_amd64.digest="sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
LABEL nearai.vllm.upstream.revision="4ba787ec9"
LABEL nearai.vllm.ported_from="docker/vllm-dsv41-lora (v0.32.0 patches 01-06 and 07-cc-explicit-copies)"
LABEL nearai.vllm.dsv41_supports_lora_patch_sha256="a4ff8a08bd1aade5bb108af3b15fb163a0edd3f80af1dff81eca414c33bd988f"
LABEL nearai.vllm.moe_lora_only_when_targeted_patch_sha256="60374d690ace180c8084ef22f489419dfc98edb720f224afd5c648c41ba07899"
LABEL nearai.vllm.decoder_replay_lora_rows="nearai; adapted to v0.33 num_padded replay rows: padding rows carry LoRA index -1"
LABEL nearai.vllm.decoder_replay_lora_rows_patch_sha256="065be72658ef5a9b58f2f76b5d7aef49c63aa0e51ba785b2854411dd11ae6640"
LABEL nearai.vllm.cc_disable_uva="nearai; VLLM_NEAR_DISABLE_UVA=1: under TDX + GPU PPCIe the GPU reads stale pinned host memory; v0.33 UvaBufferPool users fall back to explicit copies"
LABEL nearai.vllm.cc_disable_uva_patch_sha256="e728ad59c10180a95de6ea4f12b9bb2c8f53f616ede6840913a7ea89369bcaa5"
LABEL nearai.vllm.engram_cpu_gather_patch_sha256="9de6c5d0ce8b839ccd47339382da5e6236ff40bb9d099c099ad5980b49de0d30"
LABEL nearai.vllm.engram_fast_gather_patch_sha256="7d52dbaef6651a83fee9d3d4d72d0d4cccdec7c81a89e73228b45cbc8dfa4ba6"
LABEL nearai.vllm.engram_peer_reads_kill_switch="nearai; VLLM_NEAR_ENGRAM_PEER_READS=0 keeps the v0.32 DP-collective Engram lookup instead of v0.33 peer-HBM reads (#62091)"
LABEL nearai.vllm.engram_peer_reads_kill_switch_patch_sha256="694a75c29aae8b6ee5b4a096599694bcd044d80e73e65093484ab569a261c516"
LABEL nearai.vllm.cc_explicit_copies="port of vllm-project/vllm#59981 (head 0340d9a7); VLLM_CONFIDENTIAL_COMPUTE=1 only, no NVML auto-detect: V2 runner H2D on a staging stream, D2H from a worker thread"
LABEL nearai.vllm.cc_explicit_copies_patch_sha256="fb20ee5be7e0a07a0c1ba31d47d03c032d3c52567821d8b58a8be739464b6564"
LABEL nearai.vllm.cc_retain_ep_fault_flag="nearai; under CC the EP fault flag is written on the main stream, so the async D2H keeps a reference until the copy has run"
LABEL nearai.vllm.cc_retain_ep_fault_flag_patch_sha256="2039df25e33e55c3a8b2cdc62d5dc90764e08816fd882481ff2c23572d60e606"
LABEL nearai.vllm.engram_shm_table_fingerprint="nearai; a reused /dev/shm Engram table is checked against the fingerprint recorded when it was loaded"
LABEL nearai.vllm.engram_shm_table_fingerprint_patch_sha256="505a33aa7282be09d49057f8781e3e4857449fccf4a23ab9313814dcf4596c51"
LABEL nearai.security.upgrades="python packages: none needed (the v0.33.0 base ships certifi 2026.10.23, pip 26.2.2 and uv 0.13.6); OS: linux-libc-dev 6.8.0-149.149"
LABEL nearai.security.linux_libc_dev="6.8.0-149.149"

COPY patches /tmp/near/patches

# The patches are against the image's own files: refuse to build if any upstream file differs.
RUN cd /usr/local/lib/python3.12/dist-packages \
    && printf '%s\n' \
       "a4ff8a08bd1aade5bb108af3b15fb163a0edd3f80af1dff81eca414c33bd988f  /tmp/near/patches/01-dsv41-supports-lora-wo-b.diff" \
       "60374d690ace180c8084ef22f489419dfc98edb720f224afd5c648c41ba07899  /tmp/near/patches/02-moe-lora-only-when-targeted.diff" \
       "065be72658ef5a9b58f2f76b5d7aef49c63aa0e51ba785b2854411dd11ae6640  /tmp/near/patches/03-decoder-replay-lora-rows.diff" \
       "e728ad59c10180a95de6ea4f12b9bb2c8f53f616ede6840913a7ea89369bcaa5  /tmp/near/patches/04-cc-disable-uva.diff" \
       "9de6c5d0ce8b839ccd47339382da5e6236ff40bb9d099c099ad5980b49de0d30  /tmp/near/patches/05-engram-cpu-gather.diff" \
       "7d52dbaef6651a83fee9d3d4d72d0d4cccdec7c81a89e73228b45cbc8dfa4ba6  /tmp/near/patches/06-engram-fast-gather.diff" \
       "694a75c29aae8b6ee5b4a096599694bcd044d80e73e65093484ab569a261c516  /tmp/near/patches/07-engram-peer-reads-kill-switch.diff" \
       "fb20ee5be7e0a07a0c1ba31d47d03c032d3c52567821d8b58a8be739464b6564  /tmp/near/patches/08-cc-explicit-copies.diff" \
       "2039df25e33e55c3a8b2cdc62d5dc90764e08816fd882481ff2c23572d60e606  /tmp/near/patches/09-cc-retain-ep-fault-flag.diff" \
       "505a33aa7282be09d49057f8781e3e4857449fccf4a23ab9313814dcf4596c51  /tmp/near/patches/10-engram-shm-table-fingerprint.diff" \
       "c9af2eed017387380ec43ac1aa41dc11a05c42bcbab1fc4fa1cc8971bde0bebd  vllm/model_executor/layers/fused_moe/layer.py" \
       "720923ccd796dfdeab4b396eeeea79cd4c11a9dcd2e6ef8d57c0390fa4cf7ab2  vllm/models/deepseek_v41/common/engram.py" \
       "0038dd2ee11cfd3234c89226b772db958388216d5f3a7bd74b33c8d22dd03880  vllm/models/deepseek_v41/decoder_replay_layers.py" \
       "166e5cea128dd7ae36dcac71ec2121952cf818484c20a9768a78aeeaa8130a71  vllm/models/deepseek_v41/nvidia/model.py" \
       "517f135eb8fb75b112d335c60f94175e1622c254f339d7ef854f6051b29d78f9  vllm/models/deepseek_v41/nvidia/vl_model.py" \
       "9501968a99185c29094610e252d00569afae20e59313c8e2e197e07370500aef  vllm/utils/platform_utils.py" \
       "af54328bad11b63b09ea8fff7756354c59c58a6713499e7e94b566309331896c  vllm/v1/worker/gpu/async_utils.py" \
       "e8789b87f4185c0805bd8e2a39b50581e882102bc852ab7e6e1963af7115a475  vllm/v1/worker/gpu/buffer_utils.py" \
       "7a7e245ecdf5041e309813b940dd8a2c67b7c658924ac31726dec86442a11d19  vllm/v1/worker/gpu/spec_decode/utils.py" \
       | sha256sum --check --strict \
    && test ! -e vllm/v1/conf_compute_utils.py \
    && for p in /tmp/near/patches/[0-9][0-9]-*.diff; do patch -p1 --dry-run -s < "$p" && patch -p1 -s < "$p" || exit 1; done \
    && printf '%s\n' \
       "debd88ab5e8d8153ea9f335ba50f47054ca79f395204844b4f78332f27064d16  vllm/model_executor/layers/fused_moe/layer.py" \
       "43558d106f82bd76fb06ebdf44fcc69c526d543f41beb93a2956425cae44c3e8  vllm/models/deepseek_v41/common/engram.py" \
       "3e36b391d50c0db1c1705eb67812d363c25d177c69259ad2d99f35a2a4770395  vllm/models/deepseek_v41/decoder_replay_layers.py" \
       "6ed98852e2f411a9927110b23d0318a34d497fb499d558ff5f26eb873de6ef0b  vllm/models/deepseek_v41/nvidia/model.py" \
       "03366dab71dc7c88e394da27a03649c075bbd34ab0d8a3cd4ab205f3622a77d1  vllm/models/deepseek_v41/nvidia/vl_model.py" \
       "54679239ccd729c43247adb64a34843828f01cd727d6013994afdd14415814d8  vllm/utils/platform_utils.py" \
       "56cc09df35cd3f5e50c20b161431d7e640e86a47d657962e334ee03163b43281  vllm/v1/conf_compute_utils.py" \
       "d57286c26f3d4d374cf96f3d594ef46e43afa75214a4d24c4651f7805c0308d2  vllm/v1/worker/gpu/async_utils.py" \
       "32bb197462e6ca5af4c6db676ef181ddfb7c7bc7f627feee4baf45234f86efee  vllm/v1/worker/gpu/buffer_utils.py" \
       "514fa2b412d475319fc814857bf50d37fd021bc6b66e0804f6265877e108415e  vllm/v1/worker/gpu/spec_decode/utils.py" \
       | sha256sum --check --strict \
    && grep -Fq 'SupportsLoRA,' vllm/models/deepseek_v41/nvidia/vl_model.py \
    && grep -Fq 'def _is_moe_lora_target' vllm/model_executor/layers/fused_moe/layer.py \
    && grep -Fq 'replay graphs hold no LoRA mapping' vllm/models/deepseek_v41/decoder_replay_layers.py \
    && grep -Fq 'VLLM_NEAR_DISABLE_UVA' vllm/utils/platform_utils.py \
    && grep -Fq 'def _near_cpu_lookup' vllm/models/deepseek_v41/common/engram.py \
    && grep -Fq 'def near_stage_engram_rows' vllm/models/deepseek_v41/common/engram.py \
    && grep -Fq 'near_stage_engram_rows(gathered_hashes)' vllm/models/deepseek_v41/nvidia/model.py \
    && grep -Fq 'VLLM_NEAR_ENGRAM_PEER_READS' vllm/models/deepseek_v41/common/engram.py \
    && grep -Fq 'def is_confidential_compute' vllm/v1/conf_compute_utils.py \
    && grep -Fq 'cc_disables_uva()' vllm/v1/worker/gpu/buffer_utils.py \
    && grep -Fq 'output_copy_stream(main_stream.device)' vllm/v1/worker/gpu/async_utils.py \
    && grep -Fq 'new_d2h_stream(device)' vllm/v1/worker/gpu/spec_decode/utils.py \
    && grep -Fq '_has_fault_gpu = has_fault' vllm/v1/worker/gpu/async_utils.py \
    && grep -Fq 'def _near_verify' vllm/models/deepseek_v41/common/engram.py \
    && python3 -m compileall -q \
        vllm/models/deepseek_v41/nvidia/vl_model.py \
        vllm/model_executor/layers/fused_moe/layer.py \
        vllm/models/deepseek_v41/decoder_replay_layers.py \
        vllm/utils/platform_utils.py \
        vllm/models/deepseek_v41/common/engram.py \
        vllm/models/deepseek_v41/nvidia/model.py \
        vllm/v1/conf_compute_utils.py \
        vllm/v1/worker/gpu/async_utils.py \
        vllm/v1/worker/gpu/buffer_utils.py \
        vllm/v1/worker/gpu/spec_decode/utils.py \
    && (cd / && python3 -c "import vllm.v1.conf_compute_utils, vllm.v1.worker.gpu.buffer_utils, vllm.v1.worker.gpu.async_utils, vllm.v1.worker.gpu.spec_decode.utils") \
    && rm -rf /tmp/near

# Base-image finding with a fix available: linux-libc-dev (kernel UAPI headers, not executed in containers)
# upgraded to the exact noble-security fix.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        linux-libc-dev=6.8.0-149.149 \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(dpkg-query -W -f='${Version}' linux-libc-dev)" = "6.8.0-149.149"
