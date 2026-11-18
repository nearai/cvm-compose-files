FROM vllm/vllm-openai@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1

LABEL org.opencontainers.image.title="vLLM v0.34.0 DeepSeek-V4.1-Flash LoRA production derivative"
LABEL org.opencontainers.image.description="vLLM v0.34.0 with wo_b-only LoRA on DeepseekV41ForCausalLM, attention-only MoE LoRA, LoRA-aware decoder replay, the explicit-copy switch, CC-safe Engram gathers, explicit copy paths for confidential computing, bounded request handling, and pinned OpenSSL and glibc versions"
LABEL org.opencontainers.image.version="v0.34.0"
LABEL nearai.vllm.base="vllm/vllm-openai:v0.34.0"
LABEL nearai.vllm.base.linux_amd64.digest="sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484"
LABEL nearai.vllm.upstream.revision="62d4bc3c7e33b1b6c906e233e6af2d3506c482ca"
LABEL nearai.build.recipe="docker/vllm-v0.34.0-dsv41-r1/Dockerfile"
LABEL nearai.vllm.patch_series="01-17"
LABEL nearai.vllm.patched_files_sha256="8f68aa5e002b96fa5c69d89e22e69b0c8d228961cd446189306a9b692663c6ed"
LABEL nearai.os.linux_libc_dev="6.8.0-149.149"
LABEL nearai.os.packages="openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 linux-libc-dev=6.8.0-149.149"

COPY PROVENANCE SHA256SUMS PATCHED-FILES.sha256 apply-patches.py /usr/share/nearai/vllm-v0.34.0-dsv41-r1/
COPY patches/ /usr/share/nearai/vllm-v0.34.0-dsv41-r1/patches/

RUN cd /usr/share/nearai/vllm-v0.34.0-dsv41-r1 \
    && sha256sum --check --strict SHA256SUMS \
    && test -e /usr/local/lib/python3.12/dist-packages/vllm/v1/conf_compute_utils.py \
    && python3 apply-patches.py \
    && test -z "$(find /usr/local/lib/python3.12/dist-packages/vllm \( -name '*.orig' -o -name '*.rej' \) -print -quit)" \
    && python3 -m compileall -q $(awk '{print "/usr/local/lib/python3.12/dist-packages/" $2}' PATCHED-FILES.sha256) \
    && (cd / && CUDA_VISIBLE_DEVICES= python3 -c "import vllm.connections, vllm.sampling_params, vllm.v1.conf_compute_utils, vllm.v1.worker.gpu.buffer_utils, vllm.v1.worker.gpu.async_utils, vllm.v1.worker.gpu.spec_decode.utils")

# libssl and libc6 back the engine's media fetches (TLS, DNS); every version here is pinned exactly.
RUN pkgs="openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 linux-libc-dev=6.8.0-149.149" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 \
        libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 \
        linux-libc-dev=6.8.0-149.149 \
    && rm -rf /var/lib/apt/lists/* \
    && for pv in $pkgs; do test "$(dpkg-query -W -f='${Version}' "${pv%%=*}")" = "${pv#*=}" || exit 1; done
