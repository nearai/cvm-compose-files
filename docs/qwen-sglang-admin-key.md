# Qwen SGLang admin keys

The derived Qwen SGLang images require an admin key on control endpoints and keep keys out of readbacks, logs and dumps. A container running `sglang serve` generates a random key on each start unless one is already supplied. The entrypoint does not store or export the generated key; it is visible in the process arguments inside the container.

The Qwen stack does not call a gated endpoint at runtime. LoRA adapters load at startup, while inference uses chat, model discovery, load and health routes. The images cover the Qwen3.6 and Qwen3.8 SGLang engines only; other SGLang engines are not covered. The optional `SGLANG_RUST_SERVER` ingress in the c4271c3f base is not verified by these patches because its native implementation is not supplied with the reviewed code.

After publishing and verifying both image digests, pin them in a follow-up PR for the Qwen engines. Roll out one engine at a time and verify its readiness and inference before continuing. To roll back, restore the current pinned base digest for the affected engine through a separate deployment change. Merging the image recipe alone does not publish or deploy it.
