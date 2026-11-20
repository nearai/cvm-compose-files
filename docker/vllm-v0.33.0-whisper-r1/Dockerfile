FROM vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e

COPY audio-requirements.txt patches/01-language-fallback-english.diff /tmp/near/

# vLLM v0.33.0 decodes audio with av, scipy, soundfile and soxr (its `audio` extra). The four wheels go in
# alone, hash-pinned and without dependencies, so the image's own package pins stay as they are.
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/near/audio-requirements.txt \
    && python3 -c 'import av, scipy.signal, soundfile, soxr'

# The patch is against the image's own file: refuse to build if it differs before or after.
RUN cd /usr/local/lib/python3.12/dist-packages \
    && echo "5180807089d70d7036a5402b4ed016b4ef147e7c593aad0a57101ce40fddcd02  vllm/model_executor/models/whisper.py" \
       | sha256sum --check --strict \
    && patch -p1 --dry-run -s < /tmp/near/01-language-fallback-english.diff \
    && patch -p1 -s < /tmp/near/01-language-fallback-english.diff \
    && echo "ea83c959c6860660bf457ac1197147bdf2e0e8475233d3f735d8606a39430c2c  vllm/model_executor/models/whisper.py" \
       | sha256sum --check --strict \
    && grep -qxF '        return lang_code if lang_code in cls.supported_languages else "en"' \
       vllm/model_executor/models/whisper.py \
    && rm -rf /tmp/near

# linux-libc-dev holds kernel UAPI headers, which containers never execute; pinned to the noble-security version.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        linux-libc-dev=6.8.0-149.149 \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(dpkg-query -W -f='${Version}' linux-libc-dev)" = "6.8.0-149.149"
