---
paths:
  - ".github/workflows/*.yaml"
---

# .github/workflows/

The workflow set and the publish rules are in AGENTS.md (Layout; Hard rules). This file says how a
workflow is written and which rule checks it.

- `uses:` pins a 40-hex commit with the version as a trailing comment, or a `docker://` image by
  digest. Local `./` actions are allowed. (`layout.w3`)
- `runs-on` is a GitHub-hosted label listed under `runners` in `ci/data/public.json`. (`public.runner`)
- Workflow, job and step names carry no lifecycle or version tokens. (`public.lifecycle`)
- Publish jobs set `environment: publish`; no trigger is `pull_request*` or `workflow_run`.
  (`layout.w1`, `layout.w2`)
- `ci.yaml` has the jobs in `CI_JOBS` in `ci/checks/layout.py` plus `all-green`, which needs all of
  them. Adding a job is a change to that set. (`layout.all-green`)
- Checkout sets `persist-credentials: false`. Pass `${{ }}` values to `run:` through `env:`, never
  inline.
