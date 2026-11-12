---
paths:
  - ".github/workflows/*.yaml"
---

# .github/workflows/

The workflow set and the publish rules are in AGENTS.md. This file says how a workflow is written.

- `uses:` pins a 40-hex commit with the version as a trailing comment, or a `docker://` image by
  digest. Local `./` actions are allowed. (`layout.w3`)
- `runs-on` is a GitHub-hosted label listed under `runners` in `ci/data/public.json`. (`public.runner`)
- Workflow, job and step names carry no lifecycle or version tokens. (`public.lifecycle`)
- The publish rule means: every job of a publish workflow sets `environment: publish`, and no trigger
  is `pull_request*` or `workflow_run`. (`layout.w1`, `layout.w2`)
- In `ci.yaml`, `all-green` needs every other job and is the only job with that name; a new job also
  raises the job count its last step checks. (`layout.all-green`)
- Checkout sets `persist-credentials: false`. Pass `${{ }}` values to `run:` through `env:`, never
  inline. actionlint and zizmor check `ci.yaml`, `auto-tag.yaml` and publish workflows not yet at
  the base commit.
- A publish workflow is linted only in the PR that adds it and is frozen after merge; fix every
  finding before merge.
