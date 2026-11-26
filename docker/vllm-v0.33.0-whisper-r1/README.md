# vLLM v0.33.0 for Whisper

The upstream `vllm/vllm-openai:v0.33.0` image, pinned by digest, with the Python packages vLLM uses to
decode audio (`av`, `scipy`, `soundfile`, `soxr`), installed hash-pinned from `audio-requirements.txt`
without dependencies.

| patch | what |
|---|---|
| 01-language-fallback-english | Whisper language detection returns English when the detected token is not a language vLLM supports |

`linux-libc-dev` (kernel UAPI headers, not executed in containers) is upgraded to the pinned
noble-security version.
