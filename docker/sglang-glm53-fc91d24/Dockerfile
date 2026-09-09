# syntax=docker/dockerfile:1.7

FROM lmsysorg/sglang@sha256:eb090e39434085f713d39421cb2e5b6d342424aa95c3d9ecdfe8cadd4289e99b

ARG BUILD_SOURCE_REVISION

LABEL org.opencontainers.image.title="NEAR AI SGLang runtime for GLM-5.3 Flash"
LABEL org.opencontainers.image.description="Pinned SGLang fc91d24 runtime used by the qualified GLM-5.3 Flash H200 deployment"
LABEL org.opencontainers.image.vendor="NEAR AI"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/PierreLeGuen/sglang-upstream"
LABEL org.opencontainers.image.revision="fc91d2403cc210a86bd5fc715c6609f58c932e5a"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-fc91d24"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang"
LABEL org.opencontainers.image.base.digest="sha256:eb090e39434085f713d39421cb2e5b6d342424aa95c3d9ecdfe8cadd4289e99b"
LABEL nearai.build.repository="https://github.com/nearai/cvm-compose-files"
LABEL nearai.build.recipe="docker/sglang-glm53-fc91d24/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_archive.sha256="1e4340ed7e476696f36f4094e027f2c7648bcca98934f0df3220b307bdcd1673"
LABEL nearai.security.cve_2025_68121="removed-unused-nsight-efa-samplers"
LABEL nearai.security.nltk_version="3.10.3"

COPY PROVENANCE /usr/share/nearai/glm53-sglang/PROVENANCE

ADD --checksum=sha256:1e4340ed7e476696f36f4094e027f2c7648bcca98934f0df3220b307bdcd1673 \
    https://github.com/PierreLeGuen/sglang-upstream/archive/fc91d2403cc210a86bd5fc715c6609f58c932e5a.tar.gz \
    /tmp/sglang.tar.gz

RUN rm -rf /sgl-workspace/sglang \
    && mkdir -p /sgl-workspace/sglang \
    && tar -xzf /tmp/sglang.tar.gz --strip-components=1 -C /sgl-workspace/sglang \
    && rm /tmp/sglang.tar.gz

WORKDIR /sgl-workspace/sglang

RUN python3 -m pip install --no-cache-dir \
      setuptools-rust==1.13.0 \
      setuptools-scm==10.2.3 \
      pytest-asyncio==1.4.0 \
      nltk==3.10.3

RUN --mount=type=cache,target=/sgl-workspace/sglang/rust/target \
    --mount=type=cache,target=/sgl-workspace/sglang/rust/sglang-radix-tree/target \
    CARGO_BUILD_JOBS=8 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.5.16.dev20260904 \
    python3 -m pip install --no-deps --no-build-isolation -e python

RUN python3 -c "import sglang; from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest; print(sglang.__file__)" \
    && test "$(cat /usr/share/nearai/glm53-sglang/PROVENANCE | grep '^source_revision=' | cut -d= -f2)" = \
      "fc91d2403cc210a86bd5fc715c6609f58c932e5a" \
    && test -n "$BUILD_SOURCE_REVISION" \
    && test -f /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && test -f /opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && rm \
      /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
      /opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/plugins/efa_metrics/nic_sampler
