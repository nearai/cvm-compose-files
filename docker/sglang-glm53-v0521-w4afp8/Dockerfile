# syntax=docker/dockerfile:1.7

FROM lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469

ARG BUILD_SOURCE_REVISION

LABEL org.opencontainers.image.title="NEAR AI SGLang runtime for GLM-5.3 PD (v0.5.21 + W4AFP8 loader fix)"
LABEL org.opencontainers.image.description="Stock SGLang v0.5.21 with exactly one patch (W4AFp8Config propagates modules_to_not_convert), security-hardened, for GLM-5.3 Flash W4AFP8 prefill/decode on H200"
LABEL org.opencontainers.image.vendor="NEAR AI"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"
LABEL org.opencontainers.image.revision="e00930c5489053f26d86b179cee0d087f846acbb"
LABEL org.opencontainers.image.documentation="https://github.com/nearai/cvm-compose-files/tree/main/docker/sglang-glm53-v0521-w4afp8"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang"
LABEL org.opencontainers.image.base.digest="sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
LABEL nearai.sglang.base.linux_amd64.digest="sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75"
LABEL nearai.build.repository="https://github.com/nearai/cvm-compose-files"
LABEL nearai.build.recipe="docker/sglang-glm53-v0521-w4afp8/Dockerfile"
LABEL nearai.build.source_revision="${BUILD_SOURCE_REVISION}"
LABEL nearai.sglang.upstream.build="https://github.com/sgl-project/sglang/actions/runs/36824467195"
LABEL nearai.sglang.source_revision="e00930c5489053f26d86b179cee0d087f846acbb"
LABEL nearai.sglang.w4afp8_loader_patch="modules-to-not-convert-v1"
LABEL nearai.runtime.overlay="w4afp8-loader-patch-and-security-hardening"
LABEL nearai.hicache.experimental_allocator="absent"
LABEL nearai.security.nltk.version="3.10.3"
LABEL nearai.security.cve_2025_68121="removed-unused-nsight-efa-samplers"

COPY PROVENANCE /usr/share/nearai/glm53-sglang/PROVENANCE
COPY Dockerfile \
     PROVENANCE \
     README.md \
     SHA256SUMS \
     apply-patches.py \
     modules-to-not-convert.diff \
     security-requirements.txt \
     source-manifest.json \
     /usr/share/nearai/glm53-w4afp8/

# 1. Verify every recipe input, then apply the single loader patch to the exact upstream bytes.
RUN test -n "$BUILD_SOURCE_REVISION" \
    && cd /usr/share/nearai/glm53-w4afp8 \
    && sha256sum --check --strict SHA256SUMS \
    && python3 apply-patches.py

# 2. Security hardening, identical in intent to the v0.5.19 recipe: hash-pinned nltk/defusedxml and
#    removal of the unused Nsight EFA nic_sampler executables (CVE-2025-68121, Go stdlib).
RUN python3 -m pip install \
      --no-cache-dir \
      --no-deps \
      --require-hashes \
      --upgrade \
      -r /usr/share/nearai/glm53-w4afp8/security-requirements.txt \
    && python3 -c 'from importlib.metadata import version; import defusedxml, nltk; assert version("defusedxml") == "0.7.1"; assert version("nltk") == "3.10.3"' \
    && find /opt/nvidia -type f -path '*/plugins/efa_metrics/nic_sampler' -print -delete \
    && test -z "$(find /opt/nvidia -path '*/plugins/efa_metrics/nic_sampler' -print)"

# 3. Build-time assertions: the patch is live in the imported module, and provenance is intact.
RUN python3 - <<'PY'
import inspect
import sglang
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config

assert sglang.__file__.startswith("/sgl-workspace/sglang/python/"), sglang.__file__
cfg = W4AFp8Config.from_config(
    {"quant_method": "w4afp8", "modules_to_not_convert": ["model.layers.0.linear_attn.b_proj"]}
)
assert cfg.ignored_layers == ["model.layers.0.linear_attn.b_proj"], cfg.ignored_layers
assert "modules_to_not_convert" in inspect.getsource(W4AFp8Config.from_config)
print("W4AFp8 loader patch verified:", cfg.ignored_layers)
PY
RUN test "$(grep '^source_revision=' /usr/share/nearai/glm53-sglang/PROVENANCE | cut -d= -f2)" = \
      "e00930c5489053f26d86b179cee0d087f846acbb" \
    && test "$(grep '^runtime_overlay=' /usr/share/nearai/glm53-sglang/PROVENANCE | cut -d= -f2)" = \
      "w4afp8-loader-patch-and-security-hardening"
