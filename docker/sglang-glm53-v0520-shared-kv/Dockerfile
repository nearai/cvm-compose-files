# syntax=docker/dockerfile:1.7
FROM docker.io/lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash engine with an in-CVM shared KV cache"
LABEL org.opencontainers.image.description="SGLang v0.5.20 + the NEAR AI GLM-5.3 long-tier port + HiCache L3 storage for compressed DSA, so replicas in one CVM share a CPU-RAM KV store"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang"
LABEL org.opencontainers.image.base.digest="sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-v0520-shared-kv"
LABEL nearai.build.recipe="docker/sglang-glm53-v0520-shared-kv/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.upstream_revision="94602c9c2b7cbdb8efd5c52802dac6a1c180089e"
LABEL nearai.sglang.port_revision="4fb82813b260e0da805db2d75be672307f54f0ca"
LABEL nearai.sglang.shared_kv="v1-compressed-dsa-l3-file"

COPY Dockerfile \
     PROVENANCE \
     README.md \
     SHA256SUMS \
     apply-patches.py \
     shared-kv.diff \
     source-manifest.json \
     test-cpu.sh \
     v0520-port.diff \
     /usr/share/nearai/glm53-v0520-shared-kv/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-v0520-shared-kv \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py \
    && cd /sgl-workspace/sglang \
    && python3 -c "import compileall, sys; sys.exit(0 if compileall.compile_dir('python/sglang/srt', quiet=1) else 1)"

# Sharing is opt-in: it needs --enable-hierarchical-cache --hicache-storage-backend file and
# SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR pointing at a directory every replica in the CVM mounts
# (tmpfs, so the store is guest RAM). With SGLANG_HICACHE_RAM_BUDGET also set, the budget needs
# SGLANG_HICACHE_SHARED_STORE_BUDGET and a per-replica SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE of at most
# budget/replicas. On an HCC/PPCIe (TEE) host SGLANG_HICACHE_CUDA_HOST_MEMORY must stay 1.
