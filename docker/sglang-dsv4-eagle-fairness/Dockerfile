FROM lmsysorg/sglang@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c

LABEL org.opencontainers.image.title="SGLang v0.5.16 DS4F EAGLE prefill-fairness canary"
LABEL org.opencontainers.image.description="Force one decode turn between chunked EAGLE prefills when active decodes are runnable"
LABEL org.opencontainers.image.revision="fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"
LABEL nearai.sglang.patch="eagle-prefill-fairness-v2"
LABEL nearai.sglang.patch.sha256="1f8e133308716f84d3301cb2636c217d0bddef7f85527b9cda0d06760235fcb4"

COPY sglang-v0.5.16-eagle-prefill-fairness.patch /tmp/sglang-v0.5.16-eagle-prefill-fairness.patch

RUN cd /sgl-workspace/sglang \
    && patch -p1 --forward --batch < /tmp/sglang-v0.5.16-eagle-prefill-fairness.patch \
    && python3 -m py_compile python/sglang/srt/managers/scheduler.py \
    && rm /tmp/sglang-v0.5.16-eagle-prefill-fairness.patch
