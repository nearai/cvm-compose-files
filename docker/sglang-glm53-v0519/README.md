# GLM-5.3 SGLang v0.5.19 runtime

This public build context wraps the exact upstream SGLang image qualified for
`PhalaCloud/GLM-5.3-W4AFP8` on eight H200 GPUs. It does not replace SGLang
source, install patches, or include the experimental managed-memory allocator.
The only filesystem addition is the machine-readable `PROVENANCE` record.

The recipe pins:

- upstream image `lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`;
- upstream SGLang commit `0bcd822377da7b5718e674eaf9c870d349424dd1`;
- upstream build run `sgl-project/sglang/actions/runs/33912440803`;
- the NEAR AI build recipe and exact merged source commit.

The publishing workflow independently verifies the upstream image labels and
embedded BuildKit provenance before wrapping it. It then publishes and checks:

- an immutable OCI digest;
- BuildKit `mode=max` SLSA provenance;
- a BuildKit SPDX SBOM;
- a GitHub build-provenance attestation;
- a vulnerability scan with no remaining fixable CRITICAL finding;
- a keyless cosign signature recorded through Sigstore.

## Publishing

Only `.github/workflows/publish-glm53-v0519.yaml` publishes this image. The
workflow accepts an exact commit already merged into `main` and publishes to
`docker.io/nearaidev/sglang`. The mutable tag is only a discovery handle;
production compose files must pin the resulting digest.

## Independent verification

Replace `<digest>` with the digest in the production compose file.

```bash
IMAGE=docker.io/nearaidev/sglang@sha256:<digest>

cosign verify \
  --certificate-identity \
  'https://github.com/nearai/cvm-compose-files/.github/workflows/publish-glm53-v0519.yaml@refs/heads/main' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  "$IMAGE"

gh attestation verify "oci://$IMAGE" --repo nearai/cvm-compose-files

docker buildx imagetools inspect \
  --format '{{json .Provenance.SLSA}}' "$IMAGE" | jq .
docker buildx imagetools inspect \
  --format '{{json .SBOM.SPDX}}' "$IMAGE" | jq .
```
