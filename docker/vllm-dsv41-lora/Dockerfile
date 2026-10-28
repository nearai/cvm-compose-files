FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

LABEL org.opencontainers.image.title="vLLM v0.32.0 DeepSeek-V4.1-Flash LoRA production derivative"
LABEL org.opencontainers.image.description="vLLM v0.32.0 with wo_b-only LoRA on DeepseekV41ForCausalLM, MoE kept on Humming under attention-only LoRA, and LoRA-aware SWA decoder replay"
LABEL org.opencontainers.image.version="v0.32.0"
LABEL nearai.vllm.base="vllm/vllm-openai:v0.32.0"
LABEL nearai.vllm.dsv41_supports_lora_patch_sha256="b5d39b0cb9ef4664549af4b623087cc12d366da60801c6531adf090b1d307e23"
LABEL nearai.vllm.moe_lora_only_when_targeted="backport vllm-project/vllm#61165"
LABEL nearai.vllm.moe_lora_only_when_targeted_patch_sha256="df8789380a4c3265bc3087c1fcf4f5217c873e2741ec7636d48846049b7d5adc"
LABEL nearai.vllm.decoder_replay_lora_rows="nearai; fixes the illegal memory access of SWA bounded replay under LoRA (cf. vllm-project/vllm#57245)"
LABEL nearai.security.upgrades="certifi 2026.10.18, pip 26.2.2 (vendored certifi), uv 0.13.4"
LABEL nearai.vllm.decoder_replay_lora_rows_patch_sha256="cef23d110278d712ad3aa13bdd2f88e5c8aa72afbc148905b42e771387f2b22e"
LABEL nearai.vllm.cc_disable_uva="nearai; VLLM_NEAR_DISABLE_UVA=1: under TDX + GPU PPCIe the GPU reads stale pinned host memory, so V2 runner UVA buffers corrupt decode"
LABEL nearai.vllm.cc_disable_uva_patch_sha256="c1372a40c99bac3774f38756f32b696bb3e0324badb527f857e3ff88285f4ba5"
LABEL nearai.vllm.engram_cpu_gather="nearai; VLLM_NEAR_ENGRAM_CPU_GATHER=1: host Engram tables shared in /dev/shm, CPU gather + H2D copy instead of GPU reads of host memory"
LABEL nearai.vllm.engram_cpu_gather_patch_sha256="2d8c7e4c47ee6ed72a456e053768811a0f11d82d1ace2ee4b4246879ece53575"

COPY 01-dsv41-supports-lora-wo-b.patch /tmp/01-dsv41-supports-lora-wo-b.patch
COPY 02-moe-lora-only-when-targeted.patch /tmp/02-moe-lora-only-when-targeted.patch
COPY 03-decoder-replay-lora-rows.patch /tmp/03-decoder-replay-lora-rows.patch
COPY 04-cc-disable-uva.patch /tmp/04-cc-disable-uva.patch
COPY 05-engram-cpu-gather.patch /tmp/05-engram-cpu-gather.patch
COPY security-upgrades.txt /tmp/security-upgrades.txt

# The patches are against the image's own files: refuse to build if any upstream file differs.
RUN cd /usr/local/lib/python3.12/dist-packages \
    && echo "b5d39b0cb9ef4664549af4b623087cc12d366da60801c6531adf090b1d307e23  /tmp/01-dsv41-supports-lora-wo-b.patch" | sha256sum --check --strict \
    && echo "df8789380a4c3265bc3087c1fcf4f5217c873e2741ec7636d48846049b7d5adc  /tmp/02-moe-lora-only-when-targeted.patch" | sha256sum --check --strict \
    && echo "cef23d110278d712ad3aa13bdd2f88e5c8aa72afbc148905b42e771387f2b22e  /tmp/03-decoder-replay-lora-rows.patch" | sha256sum --check --strict \
    && echo "ff2c16614dc694f0c07a45929def3c76f30cb547f897b57c18c70bb99587171c  vllm/models/deepseek_v41/nvidia/vl_model.py" | sha256sum --check --strict \
    && echo "ddc92bf76eabaf0465e2dc515b844293c16691bb6e66f92a45e47c4956b6e328  vllm/model_executor/layers/fused_moe/layer.py" | sha256sum --check --strict \
    && echo "2fedc0a656a57cde7bc6780904e17484e2a4b9ac7aa7429ea392b1762908d1a8  vllm/models/deepseek_v41/decoder_replay_layers.py" | sha256sum --check --strict \
    && echo "c1372a40c99bac3774f38756f32b696bb3e0324badb527f857e3ff88285f4ba5  /tmp/04-cc-disable-uva.patch" | sha256sum --check --strict \
    && echo "2d8c7e4c47ee6ed72a456e053768811a0f11d82d1ace2ee4b4246879ece53575  /tmp/05-engram-cpu-gather.patch" | sha256sum --check --strict \
    && echo "9501968a99185c29094610e252d00569afae20e59313c8e2e197e07370500aef  vllm/utils/platform_utils.py" | sha256sum --check --strict \
    && echo "80d6f14973c99be431029e0fba4a6a784a428501bd6ba1726cff0879e0e1627f  vllm/models/deepseek_v41/common/engram.py" | sha256sum --check --strict \
    && patch -p1 --dry-run < /tmp/01-dsv41-supports-lora-wo-b.patch \
    && patch -p1 < /tmp/01-dsv41-supports-lora-wo-b.patch \
    && patch -p1 --dry-run < /tmp/02-moe-lora-only-when-targeted.patch \
    && patch -p1 < /tmp/02-moe-lora-only-when-targeted.patch \
    && patch -p1 --dry-run < /tmp/03-decoder-replay-lora-rows.patch \
    && patch -p1 < /tmp/03-decoder-replay-lora-rows.patch \
    && patch -p1 --dry-run < /tmp/04-cc-disable-uva.patch \
    && patch -p1 < /tmp/04-cc-disable-uva.patch \
    && patch -p1 --dry-run < /tmp/05-engram-cpu-gather.patch \
    && patch -p1 < /tmp/05-engram-cpu-gather.patch \
    && grep -Fq 'SupportsLoRA,' vllm/models/deepseek_v41/nvidia/vl_model.py \
    && grep -Fq 'def _is_moe_lora_target' vllm/model_executor/layers/fused_moe/layer.py \
    && grep -Fq 'replay graphs hold no LoRA mapping' vllm/models/deepseek_v41/decoder_replay_layers.py \
    && grep -Fq 'VLLM_NEAR_DISABLE_UVA' vllm/utils/platform_utils.py \
    && grep -Fq 'def _near_cpu_lookup' vllm/models/deepseek_v41/common/engram.py \
    && python3 -m compileall -q \
        vllm/models/deepseek_v41/nvidia/vl_model.py \
        vllm/model_executor/layers/fused_moe/layer.py \
        vllm/models/deepseek_v41/decoder_replay_layers.py \
        vllm/utils/platform_utils.py \
        vllm/models/deepseek_v41/common/engram.py \
    && echo "00772204fb79e8dd9e83f82bd042ac3898075603d480ef7f171a28d9d48760b4  /tmp/security-upgrades.txt" | sha256sum --check --strict \
    && python3 -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/security-upgrades.txt \
    && python3 -c "import certifi; assert certifi.__version__ == '2026.10.18'" \
    && python3 -c "from pip._vendor import certifi; assert certifi.__version__ >= '2026.10.18'" \
    && uv --version | grep -Fq "uv 0.13.4" \
    && rm /tmp/01-dsv41-supports-lora-wo-b.patch /tmp/02-moe-lora-only-when-targeted.patch /tmp/03-decoder-replay-lora-rows.patch \
        /tmp/04-cc-disable-uva.patch /tmp/05-engram-cpu-gather.patch /tmp/security-upgrades.txt
