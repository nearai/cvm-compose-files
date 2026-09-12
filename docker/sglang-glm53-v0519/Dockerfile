# syntax=docker/dockerfile:1.7

FROM lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9

ARG BUILD_SOURCE_REVISION

LABEL org.opencontainers.image.title="NEAR AI SGLang runtime for GLM-5.3"
LABEL org.opencontainers.image.description="Unmodified, provenance-wrapped SGLang v0.5.19 runtime qualified for GLM-5.3 W4AFP8 on H200"
LABEL org.opencontainers.image.vendor="NEAR AI"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"
LABEL org.opencontainers.image.revision="0bcd822377da7b5718e674eaf9c870d349424dd1"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-v0519"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang"
LABEL org.opencontainers.image.base.digest="sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
LABEL nearai.sglang.base.linux_amd64.digest="sha256:37bbbd3444732a464bbc68dee4fb0164e0ce9e18e2f027f3fc967f1152d3c262"
LABEL nearai.build.repository="https://github.com/nearai/cvm-compose-files"
LABEL nearai.build.recipe="docker/sglang-glm53-v0519/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.upstream.build="https://github.com/sgl-project/sglang/actions/runs/33912440803"
LABEL nearai.runtime.overlay="provenance-only"
LABEL nearai.hicache.experimental_allocator="absent"

COPY PROVENANCE /usr/share/nearai/glm53-sglang/PROVENANCE

RUN test -n "$BUILD_SOURCE_REVISION" \
    && test "$(grep '^source_revision=' /usr/share/nearai/glm53-sglang/PROVENANCE | cut -d= -f2)" = \
      "0bcd822377da7b5718e674eaf9c870d349424dd1" \
    && test "$(grep '^runtime_overlay=' /usr/share/nearai/glm53-sglang/PROVENANCE | cut -d= -f2)" = \
      "provenance-only"
