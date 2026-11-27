FROM vllm/vllm-openai@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1

LABEL org.opencontainers.image.title="vLLM v0.34.0 DeepSeek-V4.1-Flash LoRA and GLM-5.3-Flash production derivative"
LABEL org.opencontainers.image.description="vLLM v0.34.0 with wo_b-only LoRA on DeepseekV41ForCausalLM, attention-only MoE LoRA, LoRA-aware decoder replay, the DP-global prefill cadence, quantized GLM-5.3-Flash MTP draft loading, opt-in GLM-5.3-Flash KDA fp8 projections and contiguous slices, the explicit-copy switch, an option to turn off the NIXL UCX progress thread, opt-in class-aware engine admission that answers overload with HTTP 429 and Retry-After, two prefix-cache fixes for hybrid models that use a prefix match unit (mid-block state caching and copy-on-write deferral), opt-in one-pass Mamba prefix checkpoints and relaxed Mamba chunk alignment (both off by default), a FlashInfer 0.7.3 workspace check that accepts the unpinned host scratch buffer under confidential computing, CC-safe Engram gathers, explicit copy paths for confidential computing, bounded request handling, and pinned OpenSSL and glibc versions"
LABEL org.opencontainers.image.version="v0.34.0"
LABEL nearai.vllm.base="vllm/vllm-openai:v0.34.0"
LABEL nearai.vllm.base.linux_amd64.digest="sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484"
LABEL nearai.vllm.upstream.revision="62d4bc3c7e33b1b6c906e233e6af2d3506c482ca"
LABEL nearai.build.recipe="docker/vllm-v0.34.0-dsv41-r11/Dockerfile"
LABEL nearai.vllm.patch_series="01-29"
LABEL nearai.vllm.patched_files_sha256="07fc18d7b8eacb8f5f7bf583049519bb9df60bb704d6af59daeb4807d2147287"
LABEL nearai.os.linux_libc_dev="6.8.0-151.151"
LABEL nearai.os.packages="openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 linux-libc-dev=6.8.0-151.151"

COPY PROVENANCE SHA256SUMS PATCHED-FILES.sha256 apply-patches.py /usr/share/nearai/vllm-v0.34.0-dsv41-r11/
COPY patches/ /usr/share/nearai/vllm-v0.34.0-dsv41-r11/patches/

RUN cd /usr/share/nearai/vllm-v0.34.0-dsv41-r11 \
    && grep -v '  tests/' SHA256SUMS | sha256sum --check --strict - \
    && test -e /usr/local/lib/python3.12/dist-packages/vllm/v1/conf_compute_utils.py \
    && python3 apply-patches.py \
    && test -z "$(find /usr/local/lib/python3.12/dist-packages/vllm /usr/local/lib/python3.12/dist-packages/flashinfer \( -name '*.orig' -o -name '*.rej' \) -print -quit)" \
    && python3 -m compileall -q $(awk '{print "/usr/local/lib/python3.12/dist-packages/" $2}' PATCHED-FILES.sha256) \
    && (cd / && CUDA_VISIBLE_DEVICES= python3 -c "import vllm.connections, vllm.v1.engine.core, vllm.sampling_params, vllm.v1.conf_compute_utils, vllm.v1.worker.gpu.buffer_utils, vllm.v1.worker.gpu.async_utils, vllm.v1.worker.gpu.spec_decode.utils, vllm.models.deepseek_v41.decoder_replay_layers, vllm.models.deepseek_v41.nvidia.model, vllm.models.glm5next.common.mtp, vllm.models.glm5next.common.kda, vllm.v1.core.sched.admission_p26") \
    && (cd / && CUDA_VISIBLE_DEVICES= python3 -c "import inspect; from vllm.distributed.nixl_utils import nixl_agent_config, nixl_thread_sync_t; assert nixl_thread_sync_t is not None; assert {'sync_mode', 'enable_prog_thread'} <= set(inspect.signature(nixl_agent_config).parameters)")

# libssl and libc6 back the engine's media fetches (TLS, DNS); every version here is pinned exactly.
RUN pkgs="openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 linux-libc-dev=6.8.0-151.151" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 \
        libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 \
        linux-libc-dev=6.8.0-151.151 \
    && rm -rf /var/lib/apt/lists/* \
    && for pv in $pkgs; do test "$(dpkg-query -W -f='${Version}' "${pv%%=*}")" = "${pv#*=}" || exit 1; done
