# Qwen SGLang admin key images

These derived images require an admin key on SGLang control endpoints and keep keys out of readbacks, logs and dumps. Each `sglang serve` container generates a fresh random admin key at startup unless an explicit `--admin-api-key` is supplied. The key is passed as a process argument and is not written to a file or exported by the entrypoint.

The manually dispatched publish workflow builds either `qwen36-v0516` or `qwen38-c4271c3f` from its pinned base. It verifies the build context, tests the imported code during the build, and publishes an immutable digest with provenance and a signature. See [PROVENANCE](PROVENANCE) for the base images and patch details and [the operator guide](../../docs/qwen-sglang-admin-key.md) for rollout and rollback.

Publishing or merging this recipe does not change a running engine. Pin the resulting digest in a separate, reviewed change.

The optional `SGLANG_RUST_SERVER` ingress in the c4271c3f base is not verified by these patches because its native implementation is not supplied with the reviewed code.
