# syntax=docker/dockerfile:1.7
FROM docker.io/nearaidev/sglang@sha256:47aff791090003a37f893e998c44794c410d3f7bdfc7fdd2dfab5eb5592b30bb

ARG BUILD_SOURCE_REVISION
LABEL org.opencontainers.image.title="GLM-5.3 Flash HiCache W4AFP8 engine with self-profiling hook"
LABEL org.opencontainers.image.description="The production GLM-5.3 Flash HiCache W4AFP8 engine plus an env-gated one-shot self-profiler: NEAR_SELF_PROFILE=1 profiles a few decode steps once and prints a compact NEAR_PROFILE summary to stdout, then deletes the trace"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/nearaidev/sglang"
LABEL org.opencontainers.image.base.digest="sha256:47aff791090003a37f893e998c44794c410d3f7bdfc7fdd2dfab5eb5592b30bb"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-hicache-w4afp8-selfprof"
LABEL nearai.build.recipe="docker/sglang-glm53-hicache-w4afp8-selfprof/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.self_profile="v1-opt-in"

COPY Dockerfile \
     PROVENANCE \
     README.md \
     SHA256SUMS \
     apply-patches.py \
     assert-inert.py \
     near-self-profile.diff \
     source-manifest.json \
     /usr/share/nearai/glm53-hicache-w4afp8-selfprof/
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-hicache-w4afp8-selfprof \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py \
    && CUDA_VISIBLE_DEVICES=9 python3 assert-inert.py

# The base is the production image; this recipe changes only how the scheduler can profile itself.
# Everything is inert unless NEAR_SELF_PROFILE=1 (assert-inert.py checks that at build time). With it
# set, tp_rank 0 profiles NEAR_SELF_PROFILE_STEPS (default 50) scheduler steps once, starting at the
# first decode batch after NEAR_SELF_PROFILE_AFTER_S (default 600) seconds. A niced child process then
# prints lines prefixed "NEAR_PROFILE " to stdout and deletes /tmp/near_selfprof.
# NEAR_SELF_PROFILE_ACTIVITIES=cpu forces the CPU-only fallback.
