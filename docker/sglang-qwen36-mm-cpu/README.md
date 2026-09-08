# Qwen3.6 multimodal CPU mitigation

Related: #239's per-replica image/video qualification and guarded recovery.
This build context changes **no production Compose file or running service**.

## What this fixes

- `--mm-feature-transport cpu` moves finished tensors; it does not prevent CUDA
  image preprocessing or nvJPEG decoding in SGLang v0.5.16.
- `SGLANG_MM_CPU_PREPROCESS=1` keeps image decode, image/video preprocessing,
  video decode and staging tensors on CPU. GPU model/vision-encoder inference,
  weights, speculation, scheduling and allocation are unchanged.
- TorchCodec's explicit non-positive frame-dimension rejection becomes a
  sanitized `ValueError`. Both asynchronous loader paths preserve that type
  instead of wrapping it as `RuntimeError`; the existing OpenAI error handler
  can therefore classify it as a bad request. CUDA/OOM/unknown runtime failures
  remain server errors. New client-error responses do not echo media bytes or URLs.

This is containment of the preprocessing CUDA path, **not proof of the kernel
that caused a prior illegal-address fault**. NVIDIA documents CUDA error700 as
leaving a process inconsistent; changing a setting cannot repair an already
poisoned process. A replacement process is necessary, with survivor/drain gates.
Zero-sized decoder metadata also does not, by itself, prove the supplied media
was malformed rather than exposing a decoder bug.

## Provenance and checks

- Base image: `lmsysorg/sglang@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c`.
- Source: `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` (v0.5.16).
- Actual base packages: torch2.11.0+cu130, torchvision0.26.0+cu130,
  torchcodec0.11.1+cu130, transformers5.12.1. Do not infer CUDA ABI from a tag name.
- Every edited source has a checked SHA256 preimage; changed/already-patched
  inputs fail the build. All 16 method regressions run before applying the patch.
- The pinned `load_video(video_file, use_gpu=True)` has no frame-limit argument.
  Its existing caller passes a variable named `frame_count_limit` in that slot.
  This mitigation leaves that call unchanged and forces CPU in the decoder
  wrapper; tests verify original arguments in both modes. It does not introduce
  or claim to fix the runtime's separate frame-sampling policy.
- The real-media smoke uses the pinned Qwen processor, synthetic PNG/JPEG/H264,
  concurrent video decoders and **no GPU devices**. It must leave CUDA uninitialized.
  This verifies preprocessing, not inference performance or real-CVM serving.

```bash
docker build --network none --iidfile /tmp/qwen-mm-candidate.id \
  -t nearai-qwen36-mm-cpu:local docker/sglang-qwen36-mm-cpu
docker run --rm --read-only --cpus 4 --memory 8g \
  --tmpfs /tmp:rw,size=1g --tmpfs /root/.cache:rw,size=1g \
  --cap-drop ALL --security-opt no-new-privileges:true \
  -e TRITON_CACHE_DIR=/tmp/triton \
  -e HF_HOME=/tmp/huggingface -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  -v "$PWD:/repo:ro" \
  --entrypoint python3 "$(< /tmp/qwen-mm-candidate.id)" \
  /repo/docker/sglang-qwen36-mm-cpu/test_media.py
```

The smoke downloads only a public processor/tokenizer at the pinned checkpoint;
no model weights, credentials or customer media are needed. No image is pushed
by this test. The dedicated image enables the mitigation by default; explicit
`SGLANG_MM_CPU_PREPROCESS=0` preserves the original device-selection behavior.
The bad-media error classification remains fixed in either mode.

## Rollout gates

1. Current-head human approval and green CI, then build/publish/scan the exact
   reviewed source and use its resulting immutable image digest.
2. Qualify an isolated **real GPU/CVM** replica on that image: image/video
   JSON+SSE, normal completion termination, text/tools, signatures, encrypted
   requests, attestation, GPU health, CPU pressure, latency and concurrency.
   CPU smoke tests do not replace this gate or justify reducing capacity.
3. Keep the healthy off-host source and all unrelated models. Never restart
   the shared ingress, registrar, CVM, or GPU. Do not silently bypass a failed
   local-survivor check in the existing handover procedure.
4. After an admitted survivor and natural connection/engine drain, replace one
   engine through compose-manager. Require full per-replica qualification and
   a30minute representative soak before touching another engine.
5. Do not consolidate/evacuate source replicas until every replacement and
   registry peer passes. Roll back the candidate if correctness/latency regresses.

References: [pinned preprocessing source](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/multimodal/processors/base_processor.py),
[pinned video decoder](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/utils/video_decoder.py),
[CUDA error semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html).
