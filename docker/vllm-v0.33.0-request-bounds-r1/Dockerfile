FROM vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e

ARG BUILD_SOURCE_REVISION

LABEL org.opencontainers.image.title="NEAR AI vLLM v0.33.0 runtime with bounded request handling"
LABEL org.opencontainers.image.description="Upstream vLLM v0.33.0 with bounded request handling (patches 01-13) and pinned OpenSSL and glibc versions"
LABEL org.opencontainers.image.vendor="NEAR AI"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.version="v0.33.0"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/vllm-v0.33.0-request-bounds-r1"
LABEL org.opencontainers.image.base.name="docker.io/vllm/vllm-openai"
LABEL org.opencontainers.image.base.digest="sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e"
LABEL nearai.vllm.base="vllm/vllm-openai:v0.33.0"
LABEL nearai.vllm.base.linux_amd64.digest="sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
LABEL nearai.vllm.upstream.revision="4ba787ec9959e566d3b587a796dff07673324da5"
LABEL nearai.build.repository="https://github.com/nearai/cvm-compose-files"
LABEL nearai.build.recipe="docker/vllm-v0.33.0-request-bounds-r1/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.vllm.patch_series="01-13"
LABEL nearai.os.packages="openssl=3.0.13-0ubuntu3.16 libc6=2.39-0ubuntu8.12"

COPY PROVENANCE SHA256SUMS PATCHED-FILES.sha256 apply-patches.py test-cpu.sh /usr/share/nearai/vllm-v0.33.0-request-bounds-r1/
COPY patches/ /usr/share/nearai/vllm-v0.33.0-request-bounds-r1/patches/
COPY tests/ /usr/share/nearai/vllm-v0.33.0-request-bounds-r1/tests/

RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/vllm-v0.33.0-request-bounds-r1 \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py \
    && test -z "$(find /usr/local/lib/python3.12/dist-packages/vllm \( -name '*.orig' -o -name '*.rej' \) -print -quit)" \
    && python3 -m compileall -q $(awk '{print "/usr/local/lib/python3.12/dist-packages/" $2}' PATCHED-FILES.sha256) \
    && CUDA_VISIBLE_DEVICES= python3 -c "import vllm.connections, vllm.sampling_params, vllm.v1.structured_output.utils, vllm.multimodal.media.audio, vllm.entrypoints.openai.chat_completion.protocol, vllm.entrypoints.pooling.scoring.protocol"

# libssl and libc6 back the engine's media fetches (TLS, DNS); every version here is pinned exactly.
RUN pkgs="openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 \
        libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 \
    && rm -rf /var/lib/apt/lists/* \
    && for pv in $pkgs; do test "$(dpkg-query -W -f='${Version}' "${pv%%=*}")" = "${pv#*=}" || exit 1; done
