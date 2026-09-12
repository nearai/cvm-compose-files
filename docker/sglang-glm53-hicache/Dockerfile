# syntax=docker/dockerfile:1.7
FROM docker.io/nearaidev/sglang@sha256:a7b7136abcf5e07522289d96e96fec9b42a1a30f9dfda57e957f142680d2d67b

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash HiCache pooled-transfer canary"
LABEL org.opencontainers.image.description="Pinned fc91d24 with complete hybrid HiCache restore, pooled DMA, and CUDA-owned HCC host memory"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/nearaidev/sglang"
LABEL org.opencontainers.image.base.digest="sha256:a7b7136abcf5e07522289d96e96fec9b42a1a30f9dfda57e957f142680d2d67b"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-hicache"
LABEL nearai.build.recipe="docker/sglang-glm53-hicache/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_revision="fc91d2403cc210a86bd5fc715c6609f58c932e5a"
LABEL nearai.sglang.hicache_fix_revision="230102db838cbbe6eca7952c729518ee6d93c3a9"
LABEL nearai.sglang.pooled_transfers="v1-opt-in"
LABEL nearai.sglang.hcc_managed_memory="v3-host-prefetch-opt-in"
LABEL nearai.sglang.hcc_host_memory="v4-cuda-hostalloc-opt-in"

COPY . /usr/share/nearai/glm53-hicache/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-hicache \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# The control image and HiCache-off behavior remain available. Enable pooling
# explicitly in the canary service, never globally in the image environment.
