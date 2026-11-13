# Security

## Reporting
Report vulnerabilities privately through GitHub: Security tab -> "Report a vulnerability".
This is the only reporting channel. Do not open a public issue or pull request, and do not include
secrets or user data in the report.
If you find a credential in this repository, report it the same way and do not use it.

## Scope
In scope:
- Compose files in this repository: anything that could log, store or send request or response data,
  pull an unpinned image or model, widen host access, or expose a secret.
- Image build contexts in `docker/` and the workflows that publish and sign images, including any way
  to publish or sign an image from a branch other than `main`.
- The verification path in VERIFYING.md: any way to make a published image, signature or
  attestation look valid when it should not.
- Secrets or private data committed anywhere in this repository or its history.

Items listed under "Known gaps" in VERIFYING.md are already known; reports that show a gap is wider
than described are in scope.

Out of scope (report to the project concerned):
- Bugs in upstream inference engines, base images or models.
- Denial of service or load testing against live endpoints.
- Findings that need control of the deploying host beyond what the trust model in README.md assumes.

Please give us reasonable time to fix an issue before you disclose it.
