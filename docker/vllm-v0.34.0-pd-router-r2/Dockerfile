FROM vllm/vllm-openai@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1

ARG BUILD_SOURCE_REVISION

LABEL org.opencontainers.image.title="vLLM prefill/decode sidecar router"
LABEL org.opencontainers.image.description="Single-file router for one vLLM NixlConnector prefill/decode pair, on the stock vLLM v0.34.0 runtime"
LABEL org.opencontainers.image.vendor="NEAR AI"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/nearai/cvm-compose-files"
LABEL org.opencontainers.image.revision="${BUILD_SOURCE_REVISION}"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/vllm-v0.34.0-pd-router-r2"
LABEL org.opencontainers.image.base.name="docker.io/vllm/vllm-openai"
LABEL org.opencontainers.image.base.digest="sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1"
LABEL nearai.vllm.base="vllm/vllm-openai:v0.34.0"
LABEL nearai.vllm.base.linux_amd64.digest="sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484"
LABEL nearai.build.recipe="docker/vllm-v0.34.0-pd-router-r2/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.os.packages="openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 linux-libc-dev=6.8.0-151.151"

# Tests stay in the repository (hashed in SHA256SUMS) and are not copied into the image.
# The router uses only fastapi, aiohttp and uvicorn, all of which the vLLM image already ships; orjson is used when present.
COPY PROVENANCE SHA256SUMS router.py /opt/pd-router/

RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /opt/pd-router \
    && grep -v ' tests/' SHA256SUMS | sha256sum --check --strict - \
    && python3 -m py_compile router.py \
    && python3 -c "import fastapi, aiohttp, uvicorn" \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin pdrouter

# Pin the system packages to the same versions as the other vLLM images built from this repository.
RUN pkgs="openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 linux-libc-dev=6.8.0-151.151" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssl=3.0.13-0ubuntu3.16 libssl3t64=3.0.13-0ubuntu3.16 \
        libc6=2.39-0ubuntu8.12 libc-bin=2.39-0ubuntu8.12 libc6-dev=2.39-0ubuntu8.12 libc-dev-bin=2.39-0ubuntu8.12 \
        linux-libc-dev=6.8.0-151.151 \
    && rm -rf /var/lib/apt/lists/* \
    && for pv in $pkgs; do test "$(dpkg-query -W -f='${Version}' "${pv%%=*}")" = "${pv#*=}" || exit 1; done

USER pdrouter
EXPOSE 31150
# Arguments: --prefill host:port ... --decode host:port ... [--host 0.0.0.0] [--port 31150] [--max-imbalance N]
ENTRYPOINT ["python3", "/opt/pd-router/router.py"]
