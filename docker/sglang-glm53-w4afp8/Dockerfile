# syntax=docker/dockerfile:1.7
FROM docker.io/nearaidev/sglang@sha256:e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash W4AFP8 engine with the chunked-prefill pool clamp"
LABEL org.opencontainers.image.description="Pinned admission-reserve v10 engine with the W4AFP8 loader fix and unconditional chunked-prefill pool clamp"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/nearaidev/sglang"
LABEL org.opencontainers.image.base.digest="sha256:e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-w4afp8"
LABEL nearai.build.recipe="docker/sglang-glm53-w4afp8/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_revision="fc91d2403cc210a86bd5fc715c6609f58c932e5a"
LABEL nearai.sglang.admission_reserve="v10"
LABEL nearai.sglang.admission_reserve_revision="fb944723035d6c8b8135fd3c6ffbdc21a4615409"
LABEL nearai.sglang.w4afp8_loader_patch="modules-to-not-convert-v1"
LABEL nearai.sglang.chunked_prefill_pool_clamp="v1"

COPY Dockerfile \
     PROVENANCE \
     README.md \
     SHA256SUMS \
     apply-patches.py \
     chunked-prefill-pool-clamp.diff \
     modules-to-not-convert.diff \
     source-manifest.json \
     test-cpu.sh \
     /usr/share/nearai/glm53-w4afp8/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-w4afp8 \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# Both patches are required for the W4AFP8 image. The pool clamp is unconditional; the inherited
# admission-reserve v10 behavior remains opt-in via SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE.
