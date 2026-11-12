FROM vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e

LABEL org.opencontainers.image.title="vLLM v0.33.0 DeepSeek-V4.1-Flash LoRA production derivative"
LABEL org.opencontainers.image.description="vLLM v0.33.0 with wo_b-only LoRA on DeepseekV41ForCausalLM, MoE kept on Humming under attention-only LoRA, LoRA-aware (and padding-aware) SWA decoder replay, the TDX+PPCIe UVA switch, the CC-safe Engram CPU gather, and a kill switch for v0.33's peer-HBM Engram reads"
LABEL org.opencontainers.image.version="v0.33.0"
LABEL nearai.vllm.base="vllm/vllm-openai:v0.33.0"
LABEL nearai.vllm.base.linux_amd64.digest="sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
LABEL nearai.vllm.upstream.revision="4ba787ec9"
LABEL nearai.vllm.ported_from="docker/vllm-dsv41-lora (v0.32.0 patches 01-06)"
LABEL nearai.vllm.dsv41_supports_lora_patch_sha256="a4ff8a08bd1aade5bb108af3b15fb163a0edd3f80af1dff81eca414c33bd988f"
LABEL nearai.vllm.moe_lora_only_when_targeted_patch_sha256="60374d690ace180c8084ef22f489419dfc98edb720f224afd5c648c41ba07899"
LABEL nearai.vllm.decoder_replay_lora_rows="nearai; adapted to v0.33 num_padded replay rows: padding rows carry LoRA index -1"
LABEL nearai.vllm.decoder_replay_lora_rows_patch_sha256="065be72658ef5a9b58f2f76b5d7aef49c63aa0e51ba785b2854411dd11ae6640"
LABEL nearai.vllm.cc_disable_uva="nearai; VLLM_NEAR_DISABLE_UVA=1: under TDX + GPU PPCIe the GPU reads stale pinned host memory; v0.33 UvaBufferPool users fall back to explicit copies"
LABEL nearai.vllm.cc_disable_uva_patch_sha256="27d2b1bcd616950629a438ad15292bebc86b63280a6591282b626b11a914c1b7"
LABEL nearai.vllm.engram_cpu_gather_patch_sha256="b2cdaf320f7d19213e6e0b71f63d3cf46ffe2093d00ad29f0e483f75ae4a2f30"
LABEL nearai.vllm.engram_fast_gather_patch_sha256="9f8a001ac8121f43f7cf99e9b506133961b81cf35f99e7c05e6a5cb201bdb111"
LABEL nearai.vllm.engram_peer_reads_kill_switch="nearai; VLLM_NEAR_ENGRAM_PEER_READS=0 keeps the v0.32 DP-collective Engram lookup instead of v0.33 peer-HBM reads (#62091)"
LABEL nearai.vllm.engram_peer_reads_kill_switch_patch_sha256="3b2e41d848877c4556b84eb46ff91ffdd20f05ef25f64ada9e5753799681f3c7"
LABEL nearai.security.upgrades="none needed: the v0.33.0 base ships certifi 2026.10.23, pip 26.2.2 and uv 0.13.6"
LABEL nearai.security.linux_libc_dev="6.8.0-149.149"

COPY 01-dsv41-supports-lora-wo-b.patch 02-moe-lora-only-when-targeted.patch 03-decoder-replay-lora-rows.patch \
     04-cc-disable-uva.patch 05-engram-cpu-gather.patch 06-engram-fast-gather.patch \
     07-engram-peer-reads-kill-switch.patch /tmp/near/

# The patches are against the image's own files: refuse to build if any upstream file differs.
RUN cd /usr/local/lib/python3.12/dist-packages \
    && printf '%s\n' \
       "a4ff8a08bd1aade5bb108af3b15fb163a0edd3f80af1dff81eca414c33bd988f  /tmp/near/01-dsv41-supports-lora-wo-b.patch" \
       "60374d690ace180c8084ef22f489419dfc98edb720f224afd5c648c41ba07899  /tmp/near/02-moe-lora-only-when-targeted.patch" \
       "065be72658ef5a9b58f2f76b5d7aef49c63aa0e51ba785b2854411dd11ae6640  /tmp/near/03-decoder-replay-lora-rows.patch" \
       "27d2b1bcd616950629a438ad15292bebc86b63280a6591282b626b11a914c1b7  /tmp/near/04-cc-disable-uva.patch" \
       "b2cdaf320f7d19213e6e0b71f63d3cf46ffe2093d00ad29f0e483f75ae4a2f30  /tmp/near/05-engram-cpu-gather.patch" \
       "9f8a001ac8121f43f7cf99e9b506133961b81cf35f99e7c05e6a5cb201bdb111  /tmp/near/06-engram-fast-gather.patch" \
       "3b2e41d848877c4556b84eb46ff91ffdd20f05ef25f64ada9e5753799681f3c7  /tmp/near/07-engram-peer-reads-kill-switch.patch" \
       "c9af2eed017387380ec43ac1aa41dc11a05c42bcbab1fc4fa1cc8971bde0bebd  vllm/model_executor/layers/fused_moe/layer.py" \
       "720923ccd796dfdeab4b396eeeea79cd4c11a9dcd2e6ef8d57c0390fa4cf7ab2  vllm/models/deepseek_v41/common/engram.py" \
       "0038dd2ee11cfd3234c89226b772db958388216d5f3a7bd74b33c8d22dd03880  vllm/models/deepseek_v41/decoder_replay_layers.py" \
       "166e5cea128dd7ae36dcac71ec2121952cf818484c20a9768a78aeeaa8130a71  vllm/models/deepseek_v41/nvidia/model.py" \
       "517f135eb8fb75b112d335c60f94175e1622c254f339d7ef854f6051b29d78f9  vllm/models/deepseek_v41/nvidia/vl_model.py" \
       "9501968a99185c29094610e252d00569afae20e59313c8e2e197e07370500aef  vllm/utils/platform_utils.py" \
       | sha256sum --check --strict \
    && for p in /tmp/near/0*.patch; do patch -p1 --dry-run -s < "$p" && patch -p1 -s < "$p" || exit 1; done \
    && printf '%s\n' \
       "debd88ab5e8d8153ea9f335ba50f47054ca79f395204844b4f78332f27064d16  vllm/model_executor/layers/fused_moe/layer.py" \
       "57d02ddd66c39f0154dcf8baceea59c5cc0fa6bdf4f8c6bb3e939eea4bbfbfb5  vllm/models/deepseek_v41/common/engram.py" \
       "3e36b391d50c0db1c1705eb67812d363c25d177c69259ad2d99f35a2a4770395  vllm/models/deepseek_v41/decoder_replay_layers.py" \
       "6ed98852e2f411a9927110b23d0318a34d497fb499d558ff5f26eb873de6ef0b  vllm/models/deepseek_v41/nvidia/model.py" \
       "03366dab71dc7c88e394da27a03649c075bbd34ab0d8a3cd4ab205f3622a77d1  vllm/models/deepseek_v41/nvidia/vl_model.py" \
       "ba5c76b276e89595dd3bd35784524ef9da3fe8d2608fd64ccb2ca0066e281e1b  vllm/utils/platform_utils.py" \
       | sha256sum --check --strict \
    && grep -Fq 'SupportsLoRA,' vllm/models/deepseek_v41/nvidia/vl_model.py \
    && grep -Fq 'def _is_moe_lora_target' vllm/model_executor/layers/fused_moe/layer.py \
    && grep -Fq 'replay graphs hold no LoRA mapping' vllm/models/deepseek_v41/decoder_replay_layers.py \
    && grep -Fq 'VLLM_NEAR_DISABLE_UVA' vllm/utils/platform_utils.py \
    && grep -Fq 'def _near_cpu_lookup' vllm/models/deepseek_v41/common/engram.py \
    && grep -Fq 'def near_stage_engram_rows' vllm/models/deepseek_v41/common/engram.py \
    && grep -Fq 'near_stage_engram_rows(gathered_hashes)' vllm/models/deepseek_v41/nvidia/model.py \
    && grep -Fq 'VLLM_NEAR_ENGRAM_PEER_READS' vllm/models/deepseek_v41/common/engram.py \
    && python3 -m compileall -q \
        vllm/models/deepseek_v41/nvidia/vl_model.py \
        vllm/model_executor/layers/fused_moe/layer.py \
        vllm/models/deepseek_v41/decoder_replay_layers.py \
        vllm/utils/platform_utils.py \
        vllm/models/deepseek_v41/common/engram.py \
        vllm/models/deepseek_v41/nvidia/model.py \
    && rm -rf /tmp/near

# Base-image finding with a fix available: linux-libc-dev (kernel UAPI headers, not executed in containers)
# upgraded to the exact noble-security fix.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        linux-libc-dev=6.8.0-149.149 \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(dpkg-query -W -f='${Version}' linux-libc-dev)" = "6.8.0-149.149"
