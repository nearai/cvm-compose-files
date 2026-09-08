FROM lmsysorg/sglang@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c

LABEL org.opencontainers.image.title="Qwen3.6 CPU multimodal preprocessing mitigation"
LABEL org.opencontainers.image.revision="fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"
LABEL nearai.sglang.multimodal_patch="qwen36-mm-cpu-v1"

COPY patch_runtime.py test_patch.py /opt/nearai-mm/
RUN test "$(git -C /sgl-workspace/sglang rev-parse HEAD)" = "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1" \
    && python3 /opt/nearai-mm/test_patch.py \
    && python3 /opt/nearai-mm/patch_runtime.py \
    && python3 -m compileall -q \
        /sgl-workspace/sglang/python/sglang/srt/environ.py \
        /sgl-workspace/sglang/python/sglang/srt/multimodal/processors/base_processor.py \
        /sgl-workspace/sglang/python/sglang/srt/utils/video_decoder.py

# This derivative is opt-in by image, with an explicit runtime rollback switch.
# It changes neither checkpoint, scheduler/speculation settings, nor GPU ownership.
ENV SGLANG_MM_CPU_PREPROCESS=1
