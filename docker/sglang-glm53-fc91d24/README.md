# GLM-5.3 Flash SGLang runtime

This public build context replaces the former host-local inline build used by
`prod/GLM-5.3-Flash-SGL-TP4.yaml`.

The recipe pins all source identities that define the engine:

- SGLang CUDA base image by registry digest;
- the exact `PierreLeGuen/sglang-upstream` source commit;
- the downloaded GitHub source archive by SHA-256;
- the editable package version used when compiling the Rust extension.

The base scan also identified fixable findings in NLTK and two unused NVIDIA
Nsight EFA sampler binaries. The recipe upgrades NLTK to 3.10.3 and removes
those samplers before the final digest is scanned. The workflow refuses to
attest or sign an image with a remaining fixable CRITICAL finding.

## Publishing

Only `.github/workflows/publish-glm53-fc91d24.yaml` publishes this image. The
workflow accepts an exact commit already merged into `main`, builds on the
self-hosted GitHub Actions infrastructure runner, and pushes to
`docker.io/nearaidev/sglang`.

The workflow publishes and verifies:

- an immutable OCI digest;
- BuildKit `mode=max` SLSA provenance;
- a BuildKit SPDX SBOM;
- a GitHub build-provenance attestation;
- a keyless cosign signature recorded through Sigstore.

The mutable publishing tag is only a discovery handle. Production compose
files must use the digest printed by the workflow.

## Verification

Replace `<digest>` with the digest pinned in the production compose file.

```bash
IMAGE=docker.io/nearaidev/sglang@sha256:<digest>

cosign verify \
  --certificate-identity \
  'https://github.com/nearai/cvm-compose-files/.github/workflows/publish-glm53-fc91d24.yaml@refs/heads/main' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  "$IMAGE"

gh attestation verify "oci://$IMAGE" --repo nearai/cvm-compose-files

docker buildx imagetools inspect \
  --format '{{json .Provenance.SLSA}}' "$IMAGE" | jq .
docker buildx imagetools inspect \
  --format '{{json .SBOM.SPDX}}' "$IMAGE" | jq .
```
