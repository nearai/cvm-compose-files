# syntax=docker/dockerfile:1.7
FROM docker.io/nearaidev/sglang@sha256:3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash HiCache W4AFP8 engine"
LABEL org.opencontainers.image.description="HCC-safe HiCache engine with admission reserve v10, the W4AFP8 loader fix, the unconditional chunked-prefill pool clamp and the opt-in DSA indexer query split"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/nearaidev/sglang"
LABEL org.opencontainers.image.base.digest="sha256:3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-hicache-w4afp8"
LABEL nearai.build.recipe="docker/sglang-glm53-hicache-w4afp8/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.source_revision="fc91d2403cc210a86bd5fc715c6609f58c932e5a"
LABEL nearai.sglang.admission_reserve="v10"
LABEL nearai.sglang.admission_reserve_revision="fb944723035d6c8b8135fd3c6ffbdc21a4615409"
LABEL nearai.sglang.hicache_fix_revision="230102db838cbbe6eca7952c729518ee6d93c3a9"
LABEL nearai.sglang.pooled_transfers="v1-opt-in"
LABEL nearai.sglang.hcc_host_memory="v4-cuda-hostalloc-opt-in"
LABEL nearai.sglang.hcc_managed_memory="v3-host-prefetch-opt-in"
LABEL nearai.sglang.w4afp8_loader_patch="modules-to-not-convert-v1"
LABEL nearai.sglang.chunked_prefill_pool_clamp="v1"
LABEL nearai.sglang.dsa_indexer_qsplit="v1-opt-in"

COPY Dockerfile \
     PROVENANCE \
     README.md \
     SHA256SUMS \
     apply-patches.py \
     chunked-prefill-pool-clamp.diff \
     dsa-indexer-qsplit.diff \
     modules-to-not-convert.diff \
     source-manifest.json \
     test-cpu.sh \
     /usr/share/nearai/glm53-hicache-w4afp8/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-hicache-w4afp8 \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# The loader and pool-clamp patches are required for the W4AFP8 image and are unconditional. The DSA
# indexer query split is inert unless SGLANG_DSA_INDEXER_QSPLIT=1. Everything inherited from
# the base stays opt-in: admission reserve v10 via SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE, and
# HiCache via --enable-hierarchical-cache plus SGLANG_HICACHE_*. On an HCC/PPCIe (TEE) host
# SGLANG_HICACHE_CUDA_HOST_MEMORY must stay 1; 0 selects the anonymous-mmap plus cudaHostRegister
# path, which fails with CUDA error 801.
