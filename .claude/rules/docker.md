---
paths:
  - "docker/**"
---

# docker/ build contexts

One dir per published image. The hard rules in AGENTS.md cover pins, freezing and the new-image PR.

- Build dir: `docker/<engine>-<upstream>-<scope>-r<N>/`, e.g. `sglang-v0.5.23-glm53-r1`. A rebuild is a
  new dir with `r<N+1>`, never an edit of a build input (`ci/checks/frozen.py`). Image tag = dir name without `<engine>-`.
- Compose files pin the digest only, the one the `main` run of the dir's publish workflow produced,
  with a row in `ci/data/images.allow`; never a digest built anywhere else.
