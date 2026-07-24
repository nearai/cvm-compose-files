FROM lmsysorg/sglang@sha256:6bb5fee34b6c4537c09a4775e2292ac40350d5ad1218fcc835b2692142f443b1

LABEL org.opencontainers.image.title="SGLang DS4F EAGLE prefill-fairness experiment"
LABEL org.opencontainers.image.description="GPU02-only experiment: force one decode turn between chunked EAGLE prefills"
LABEL nearai.experiment.patch="eagle-prefill-fairness-v1"

COPY sglang-eagle-prefill-fairness.patch /tmp/sglang-eagle-prefill-fairness.patch

RUN cd /sgl-workspace/sglang \
    && patch -p1 --forward --batch < /tmp/sglang-eagle-prefill-fairness.patch \
    && python3 -m py_compile python/sglang/srt/managers/scheduler.py \
    && rm /tmp/sglang-eagle-prefill-fairness.patch
