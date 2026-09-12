# syntax=docker/dockerfile:1.7
FROM lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 non-Flash HiCache HCC canary"
LABEL org.opencontainers.image.description="Pinned SGLang 0bcd822 with opt-in host-preferred CUDA managed memory for HiCache"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang"
LABEL org.opencontainers.image.base.digest="sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-hicache"
LABEL nearai.build.recipe="docker/sglang-glm53-hicache/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_revision="0bcd822377da7b5718e674eaf9c870d349424dd1"
LABEL nearai.sglang.hcc_managed_memory="v2-host-preferred-opt-in"

COPY . /usr/share/nearai/glm53-hicache/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-hicache \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# The pinned upstream image contains an old NLTK wheel and an unused Nsight EFA
# metrics helper that trip the fixable-critical release gate. Upgrade NLTK from
# a hash-pinned wheel without resolving new dependencies, and remove only the
# affected non-serving helper binary.
RUN printf '%s\n' \
      'nltk==3.10.3 --hash=sha256:ff9598a8e20518ee0d557745890cc4435b9578489e2dcbc69c4f81fa060caf7c' \
      > /tmp/nltk-requirement.txt \
    && python3 -m pip install --no-cache-dir --no-deps --require-hashes \
      --requirement /tmp/nltk-requirement.txt \
    && test -x /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && rm -f /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && rm -f /tmp/nltk-requirement.txt
