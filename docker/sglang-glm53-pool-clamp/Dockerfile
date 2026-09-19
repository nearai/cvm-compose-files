# syntax=docker/dockerfile:1.7
FROM docker.io/nearaidev/sglang@sha256:e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash admission-reserve engine with the chunked-prefill pool clamp"
LABEL org.opencontainers.image.description="Deployed admission-reserve v10 engine plus the upstream fix for add_chunked_req requesting a full chunk when rem_total_tokens goes negative"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/nearaidev/sglang"
LABEL org.opencontainers.image.base.digest="sha256:e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-pool-clamp"
LABEL nearai.build.recipe="docker/sglang-glm53-pool-clamp/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_revision="fc91d2403cc210a86bd5fc715c6609f58c932e5a"
LABEL nearai.sglang.admission_reserve="v10"
LABEL nearai.sglang.admission_reserve_revision="fb944723035d6c8b8135fd3c6ffbdc21a4615409"
LABEL nearai.sglang.chunked_prefill_pool_clamp="v1"

COPY . /usr/share/nearai/glm53-pool-clamp/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-pool-clamp \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# This patch is unconditional: it is a correctness fix in the chunked-prefill path and has no
# opt-in switch. The admission-reserve knobs it inherits from the base image are unchanged and
# still opt-in via SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE.
