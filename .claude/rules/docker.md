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
