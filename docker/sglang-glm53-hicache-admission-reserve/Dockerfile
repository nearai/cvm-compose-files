# syntax=docker/dockerfile:1.7
FROM docker.io/nearaidev/sglang@sha256:01f44c790d43b3eae136f5f3bc930599947bd37fcea081d2dbd3271ff4fc3f8a

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash HiCache admission-reserve engine"
LABEL org.opencontainers.image.description="Signed HiCache canary base with the opt-in chunked-prefill admission-reserve v10 patch layered on top"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/nearaidev/sglang"
LABEL org.opencontainers.image.base.digest="sha256:01f44c790d43b3eae136f5f3bc930599947bd37fcea081d2dbd3271ff4fc3f8a"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-hicache-admission-reserve"
LABEL nearai.build.recipe="docker/sglang-glm53-hicache-admission-reserve/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_revision="fc91d2403cc210a86bd5fc715c6609f58c932e5a"
LABEL nearai.sglang.admission_reserve="v10"
LABEL nearai.sglang.admission_reserve_revision="fb944723035d6c8b8135fd3c6ffbdc21a4615409"
LABEL nearai.sglang.hicache_fix_revision="230102db838cbbe6eca7952c729518ee6d93c3a9"
LABEL nearai.sglang.pooled_transfers="v1-opt-in"
LABEL nearai.sglang.hcc_host_memory="v4-cuda-hostalloc-opt-in"
LABEL nearai.sglang.hcc_managed_memory="v3-host-prefetch-opt-in"

COPY . /usr/share/nearai/glm53-hicache-admission-reserve/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-hicache-admission-reserve \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# The patch is inert unless SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE > 0.
# Enable it explicitly in the compose service, never globally in the image.
