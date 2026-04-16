# SGLang image for GLM-5/5.1 on Hopper (H100/H200)
#
# Produces a self-contained image with:
#   - SGLang + sglang-kernel + FlashInfer + DeepEP
#   - Model weights baked in
#   - Autotuned DeepGemm kernels (not heuristic)
#   - Optimal serve parameters discovered on the build host
#
# The image is ready to drop into docker-compose with zero config:
#   services:
#     glm51:
#       image: nearaidev/sglang:glm5.1-tuned
#       runtime: nvidia
#       ipc: host
#
# Build with build.sh (must run on a GPU host):
#   ./build.sh zai-org/GLM-5.1-FP8 nearaidev/sglang:glm5.1-tuned
#
# What changed vs the old lmsysorg/sglang:glm5-hopper image:
#   SGLang        main@9ebffef1e (pre-v0.5.10)  →  main@ce31934ca80e (post-v0.5.10.post1)
#   sglang-kernel 0.3.21 (was sgl-kernel)        →  0.4.1
#   FlashInfer    0.6.4                           →  0.6.7.post3
#   transformers  5.2.0 (pinned)                  →  5.3.0 (now required by SGLang)
#   NCCL          2.27.3                          →  2.28.3
#   CUDA toolkit  12.9.1                          →  12.8.1 (matches host driver)
#   Fixes: KV pool retract-tail leak (#22862), HiCache memory free (#22767)

ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

ARG CUDA_VERSION=12.8.1
ARG SGL_KERNEL_VERSION=0.4.1
ARG FLASHINFER_VERSION=0.6.7.post3
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG MODEL_PATH=zai-org/GLM-5.1-FP8

# cu128 for 12.8.x
ARG CU_INDEX=128

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FLASHINFER_VERSION=${FLASHINFER_VERSION} \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget software-properties-common ca-certificates curl git git-lfs lsof unzip \
        build-essential cmake patchelf ccache \
        libopenmpi-dev libnuma1 libnuma-dev numactl ffmpeg \
        libibverbs-dev libibverbs1 libibumad3 librdmacm1 \
        libnl-3-200 libnl-route-3-200 libnl-route-3-dev libnl-3-dev \
        ibverbs-providers infiniband-diags \
        libssl-dev libcurl4-openssl-dev pybind11-dev \
        locales kmod \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.12 ──────────────────────────────────────────────────────────────
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12-full python3.12-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py --break-system-packages && rm get-pip.py \
    && python3 -m pip config set global.break-system-packages true \
    && rm -rf /var/lib/apt/lists/*

# ── SGLang from source ───────────────────────────────────────────────────────
# Build context = sglang repo root checked out at the desired commit.
WORKDIR /sgl-workspace
COPY . /sgl-workspace/sglang

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install sglang-kernel==${SGL_KERNEL_VERSION} \
    && cd sglang \
    && python3 -m pip install -e "python[all]" \
        --extra-index-url https://download.pytorch.org/whl/cu${CU_INDEX} \
    && python3 -m pip install flashinfer-jit-cache==${FLASHINFER_VERSION} \
        --index-url https://flashinfer.ai/whl/cu${CU_INDEX} \
    && FLASHINFER_CUBIN_DOWNLOAD_THREADS=8 FLASHINFER_LOGGING_LEVEL=warning \
        python3 -m flashinfer --download-cubin

# ── DeepEP (MoE expert parallelism kernels) ──────────────────────────────────
RUN curl -fsSL -o /tmp/deepep.zip \
        https://github.com/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip \
    && unzip -q /tmp/deepep.zip -d /sgl-workspace \
    && rm /tmp/deepep.zip \
    && mv /sgl-workspace/DeepEP-${DEEPEP_COMMIT} /sgl-workspace/DeepEP \
    && cd /sgl-workspace/DeepEP \
    && sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh \
    && sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' csrc/kernels/configs.cuh \
    && TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 \
        pip install --no-build-isolation .

# ── NCCL / cuDNN upgrades ────────────────────────────────────────────────────
RUN python3 -m pip install \
        nvidia-nccl-cu12==2.28.3 \
        nvidia-cudnn-cu12==9.16.0.29 \
    --force-reinstall --no-deps

# ── Download model weights ───────────────────────────────────────────────────
ARG HF_TOKEN=""
RUN python3 -m pip install "huggingface_hub[hf_xet]" \
    && HF_TOKEN="${HF_TOKEN}" python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_PATH}', local_dir='/models/${MODEL_PATH}')" \
    && echo "${MODEL_PATH}" > /models/.model_path

# ── Warmup + entrypoint scripts ──────────────────────────────────────────────
COPY warmup.py /sgl-workspace/warmup.py
COPY entrypoint.sh /sgl-workspace/entrypoint.sh
RUN chmod +x /sgl-workspace/entrypoint.sh

# ── Verify ────────────────────────────────────────────────────────────────────
RUN python3 -c "import sglang; print(f'SGLang: {sglang.__version__}')" \
    && python3 -c "import sglang_kernel; print('sglang-kernel: OK')" \
    && python3 -c "import flashinfer; print(f'FlashInfer: {flashinfer.__version__}')" \
    && python3 -c "import transformers; print(f'transformers: {transformers.__version__}')" \
    && python3 -c "import deep_ep; print('DeepEP: OK')" \
    && ls /models/${MODEL_PATH}/config.json

ENV HF_HUB_OFFLINE=1
WORKDIR /sgl-workspace/sglang
CMD ["/sgl-workspace/entrypoint.sh"]
