FROM lmsysorg/sglang@sha256:51e576f02368480c055c7aadb67590d82b172e2392123ce4cf4cc8251b2d8caf

LABEL org.opencontainers.image.title="SGLang nightly c0b6474 DS4F-0731 production derivative"
LABEL org.opencontainers.image.description="Post-PR-33587 SGLang with bounded chunked-prefill admission and pre-stream abort handling"
LABEL org.opencontainers.image.revision="c0b6474b43363c2f4bc60fe3d7817d393fb51d32"
LABEL nearai.sglang.war_fence_fix="sgl-project/sglang#33587"
LABEL nearai.sglang.chunked_prefill_admission_patch="nearai-v2-nightly-c0b6474"
LABEL nearai.sglang.chunked_prefill_admission_patch_sha256="3f7111c472b583b0921246b38d54d09ab6bb41090efe8a6301b750612b98fc89"
LABEL nearai.sglang.chunked_prefill_admission_reserve="runtime-env"
LABEL nearai.sglang.streaming_abort_patch="nearai-v2-nightly-c0b6474"
LABEL nearai.sglang.streaming_abort_patch_sha256="28bc45ba377e3616d7221a609e584272c765f4fc3e69e0620d1ed0003bd608e4"
LABEL nearai.sglang.dsv4_reasoning_profile="upstream"
LABEL nearai.security.cve_2025_68121="removed-unused-nsight-efa-samplers"

COPY sglang-nightly-c0b6474-streaming-abort.patch /tmp/sglang-nightly-c0b6474-streaming-abort.patch
COPY sglang-nightly-c0b6474-chunked-prefill-admission-reserve.patch /tmp/sglang-nightly-c0b6474-chunked-prefill-admission-reserve.patch

RUN echo "28bc45ba377e3616d7221a609e584272c765f4fc3e69e0620d1ed0003bd608e4  /tmp/sglang-nightly-c0b6474-streaming-abort.patch" | sha256sum --check --strict \
    && echo "3f7111c472b583b0921246b38d54d09ab6bb41090efe8a6301b750612b98fc89  /tmp/sglang-nightly-c0b6474-chunked-prefill-admission-reserve.patch" | sha256sum --check --strict \
    && cd /sgl-workspace/sglang \
    && test "$(git rev-parse HEAD)" = "c0b6474b43363c2f4bc60fe3d7817d393fb51d32" \
    && grep -Fq 'SGLANG_ENABLE_PREFILL_WAR_READ_DONE' python/sglang/srt/environ.py \
    && grep -Fq 'SGLANG_DSV4_COMPRESS_STATE_DTYPE' python/sglang/srt/environ.py \
    && grep -Fq 'REASONING_EFFORT_PROFILES = {' python/sglang/srt/entrypoints/openai/encoding_dsv4.py \
    && grep -Fq 'broker_endpoint = f"tcp://127.0.0.1:{server_args.broker_port}"' python/sglang/multimodal_gen/runtime/scheduler_client.py \
    && ! grep -Fq 'broker_endpoint = f"tcp://*:{server_args.broker_port}"' python/sglang/multimodal_gen/runtime/scheduler_client.py \
    && grep -Fq 'from sglang.srt.utils.common import safe_pickle_loads' python/sglang/srt/disaggregation/encode_receiver.py \
    && ! grep -Fq 'pickle.loads(parts[0])' python/sglang/srt/disaggregation/encode_receiver.py \
    && git apply --check /tmp/sglang-nightly-c0b6474-streaming-abort.patch \
    && git apply /tmp/sglang-nightly-c0b6474-streaming-abort.patch \
    && git apply --check /tmp/sglang-nightly-c0b6474-chunked-prefill-admission-reserve.patch \
    && git apply /tmp/sglang-nightly-c0b6474-chunked-prefill-admission-reserve.patch \
    && git diff --check \
    && python3 -m compileall -q \
        python/sglang/srt/entrypoints/openai/serving_base.py \
        python/sglang/srt/entrypoints/openai/serving_chat.py \
        python/sglang/srt/entrypoints/openai/serving_completions.py \
        python/sglang/srt/managers/schedule_policy.py \
        python/sglang/srt/managers/scheduler.py \
    && test -f /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && test -f /opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && rm \
        /opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler \
        /opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/plugins/efa_metrics/nic_sampler \
    && rm \
        /tmp/sglang-nightly-c0b6474-streaming-abort.patch \
        /tmp/sglang-nightly-c0b6474-chunked-prefill-admission-reserve.patch
