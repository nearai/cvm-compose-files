FROM lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f

ARG BUILD_SOURCE_REVISION

LABEL org.opencontainers.image.title="SGLang v0.5.23 for GLM-5.3 Flash"
LABEL org.opencontainers.image.description="Upstream SGLang v0.5.23 with the GLM-5.3 Flash patch series (W4AFP8 loader, FP8 NoPE DSA KV and Q8 prefill, HiCache host-memory runtime, admission reserve and pool clamp, off-loop preprocessing), exact-rid aborts, credential and request-content log redaction, and request validation"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang"
LABEL org.opencontainers.image.base.digest="sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
LABEL nearai.sglang.base.linux_amd64.digest="sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5"
LABEL nearai.sglang.upstream.revision="297e54fd105e974cdf401e5b4d20253a22c16f90"
LABEL nearai.sglang.upstream.build="https://github.com/sgl-project/sglang/actions/runs/40978390987"
LABEL nearai.build.repository="https://github.com/nearai/cvm-compose-files"
LABEL nearai.build.recipe="docker/sglang-v0.5.23-glm53-r3/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.patch_series="01-38"
LABEL nearai.python.packages="nltk=3.10.3 defusedxml=0.7.1"
LABEL nearai.os.linux_libc_dev="6.8.0-149.149"

COPY PROVENANCE SHA256SUMS SERIES apply-patches.py test-cpu.sh requirements.txt \
     /usr/share/nearai/sglang-v0.5.23-glm53-r3/
COPY patches/ /usr/share/nearai/sglang-v0.5.23-glm53-r3/patches/
COPY tests/ /usr/share/nearai/sglang-v0.5.23-glm53-r3/tests/

RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/sglang-v0.5.23-glm53-r3 \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py \
    && python3 -c "import sglang.srt.entrypoints.http_server, sglang.srt.managers.scheduler, sglang.srt.mem_cache.memory_pool_host" \
    && python3 -m pip install --no-cache-dir --no-deps --require-hashes --upgrade \
       -r /usr/share/nearai/sglang-v0.5.23-glm53-r3/requirements.txt \
    && python3 -c 'from importlib.metadata import version; import defusedxml, nltk; assert version("defusedxml") == "0.7.1"; assert version("nltk") == "3.10.3"' \
    && rm -f /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
             /opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && test ! -e /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && test ! -e /opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/plugins/efa_metrics/nic_sampler

# linux-libc-dev holds kernel headers only and is pinned to an exact version. The image builds no Rust
# code (SGLANG_RUST_BUILD_MODE=never), so the cargo cache and the radix-tree crate's build files go.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        linux-libc-dev=6.8.0-149.149 \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(dpkg-query -W -f='${Version}' linux-libc-dev)" = "6.8.0-149.149" \
    && test "$SGLANG_RUST_BUILD_MODE" = "never" \
    && rm -rf /root/.cargo/registry \
              /sgl-workspace/sglang/rust/sglang-radix-tree/Cargo.lock \
              /sgl-workspace/sglang/rust/sglang-radix-tree/target \
    && python3 -c "import sglang.srt.mem_cache.rust_tree_core.extension as e; assert e.RustUnifiedTreeCoreBinding"
