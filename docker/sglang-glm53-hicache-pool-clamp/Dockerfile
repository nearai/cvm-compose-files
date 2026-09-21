# syntax=docker/dockerfile:1.7
FROM docker.io/nearaidev/sglang@sha256:3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash HiCache admission-reserve engine with the chunked-prefill pool clamp"
LABEL org.opencontainers.image.description="Deployed HiCache admission-reserve v10 engine plus the upstream fix for add_chunked_req requesting a full chunk when rem_total_tokens goes negative"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/nearaidev/sglang"
LABEL org.opencontainers.image.base.digest="sha256:3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-hicache-pool-clamp"
LABEL nearai.build.recipe="docker/sglang-glm53-hicache-pool-clamp/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_revision="fc91d2403cc210a86bd5fc715c6609f58c932e5a"
LABEL nearai.sglang.admission_reserve="v10"
LABEL nearai.sglang.admission_reserve_revision="fb944723035d6c8b8135fd3c6ffbdc21a4615409"
LABEL nearai.sglang.hicache_fix_revision="230102db838cbbe6eca7952c729518ee6d93c3a9"
LABEL nearai.sglang.chunked_prefill_pool_clamp="v1"

COPY . /usr/share/nearai/glm53-hicache-pool-clamp/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-hicache-pool-clamp \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# This patch is unconditional: it is a correctness fix in the chunked-prefill path and has no
# opt-in switch. The admission-reserve and HiCache knobs inherited from the base image are
# unchanged.
