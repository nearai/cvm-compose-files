FROM lmsysorg/sglang@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c

LABEL org.opencontainers.image.title="SGLang v0.5.16 DS4F-0731 production canary"
LABEL org.opencontainers.image.description="Qualified pre-stream admission handling and official DSV4 reasoning-effort profile"
LABEL org.opencontainers.image.revision="fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"
LABEL nearai.sglang.admission_patch="pr28175"
LABEL nearai.sglang.reasoning_backport="059269594c5f245f77dad711631843c299d7713f"
LABEL nearai.sglang.patch.sha256="0666a644e0791a92606c73d91725b0b52908aa347880973ffa1d3e9dc47282f4"
LABEL nearai.security.cve_2025_68121="removed-unused-nsight-efa-samplers"

COPY sglang-v0.5.16-dsv4-0731-runtime.patch /tmp/sglang-v0.5.16-dsv4-0731-runtime.patch

RUN echo "0666a644e0791a92606c73d91725b0b52908aa347880973ffa1d3e9dc47282f4  /tmp/sglang-v0.5.16-dsv4-0731-runtime.patch" | sha256sum --check --strict \
    && cd /sgl-workspace/sglang \
    && git apply --check /tmp/sglang-v0.5.16-dsv4-0731-runtime.patch \
    && git apply /tmp/sglang-v0.5.16-dsv4-0731-runtime.patch \
    && git diff --check \
    && python3 -m compileall -q \
        python/sglang/srt/entrypoints/openai/chat_encoding.py \
        python/sglang/srt/entrypoints/openai/encoding_dsv4.py \
        python/sglang/srt/entrypoints/openai/serving_base.py \
        python/sglang/srt/entrypoints/openai/serving_chat.py \
        python/sglang/srt/entrypoints/openai/serving_completions.py \
    && test -f /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && test -f /opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && rm \
        /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
        /opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && rm /tmp/sglang-v0.5.16-dsv4-0731-runtime.patch
