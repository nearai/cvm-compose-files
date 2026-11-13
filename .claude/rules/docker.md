---
paths:
  - "docker/**"
---

# docker/ build contexts

One dir per published image. The hard rules in AGENTS.md cover pins, freezing and the new-image PR.

- Build dir: `docker/<engine>-v<upstream>-<scope>-r<N>/`, e.g. `sglang-v0.5.23-glm53-r1`; CI checks it
  for new dirs (`layout.build-dir`), and dirs older than the rule keep their names.
- A rebuild is a new dir with `r<N+1>`, never an edit of a build input (`ci/checks/frozen.py`).
- Image tag = dir name without `<engine>-`.

## A new build dir (CI: `ci/checks/docker.py`)
```
docker/<engine>-v<upstream>-<scope>-r<N>/
  Dockerfile          FROM an upstream image by digest; never an image built here
  PROVENANCE          engine, upstream_version, base_image, base_image_linux_amd64,
                      source_repository, source_revision; exactly these keys, in this order
  SHA256SUMS          every other file except Dockerfile and README.md
  patches/NN-name.diff   applied in file-name order, numbered 01..N
  tests/, helpers     optional, hashed like every input
  README.md           optional; what the image is and one line per patch. Never copied into the image
```
- COPY names hashed files or dirs only: no `ADD`, no `COPY .`, no globs, no docs. A README that the
  build never reads stays editable for the life of the dir.
- No lab results, benchmarks, dates, hosts, experiment ids or incident notes in any file of the dir.
- The image's own digest is recorded in `ci/data/images.allow` and `docs/verifying.md`, never in the dir.
- A dir may leave `docker/` while its images are still pinned: verifiers read it at the source commit
  recorded in the image's provenance, and its rows in docs/verifying.md stay.

