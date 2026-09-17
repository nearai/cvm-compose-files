# syntax=docker/dockerfile:1.7
FROM docker.io/nearaidev/sglang@sha256:a7b7136abcf5e07522289d96e96fec9b42a1a30f9dfda57e957f142680d2d67b

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash admission-reserve engine"
LABEL org.opencontainers.image.description="Pinned fc91d24 production runtime with the opt-in chunked-prefill admission-reserve v10 patch"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/nearaidev/sglang"
LABEL org.opencontainers.image.base.digest="sha256:a7b7136abcf5e07522289d96e96fec9b42a1a30f9dfda57e957f142680d2d67b"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-admission-reserve"
LABEL nearai.build.recipe="docker/sglang-glm53-admission-reserve/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_revision="fc91d2403cc210a86bd5fc715c6609f58c932e5a"
LABEL nearai.sglang.admission_reserve="v10"
LABEL nearai.sglang.admission_reserve_revision="fb944723035d6c8b8135fd3c6ffbdc21a4615409"

COPY . /usr/share/nearai/glm53-admission-reserve/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-admission-reserve \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# The patch is inert unless SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE > 0.
# Enable it explicitly in the compose service, never globally in the image.
