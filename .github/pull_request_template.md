This repo is public. No hosts, IPs, internal links, people, customers, rollout state or secrets in this PR.

Title: `<prod|docker|ci|docs|repo>: <imperative summary>`, at most 72 chars with the ` (#N)` suffix. One concern per PR (AGENTS.md).

## What

<!-- What changes and why, in a few lines. -->

## Type

- [ ] new config
- [ ] new image
- [ ] image bump
- [ ] config tune
- [ ] rules or CI change
- [ ] deletion or promotion
- [ ] docs

## Evidence

- [ ] `python3 ci/run.py --base origin/main` is green
- [ ] `prod/` changes: rendered diff read (`python3 ci/run.py --render <file>`)
- [ ] comment-only or refactor changes: `python3 ci/run.py --noop origin/main` output (empty)
- [ ] deletions: deployment manager checked, nothing runs the deleted file

## Pins and identity

<!-- Image bumps: digest, the main publish run that produced it, signature verification output.
     Model pins: repo and 40-hex revision. New image: build dir and its publish workflow. -->

## Privacy and safety flags

<!-- Name every new sink, endpoint, port, mount or env var, every moved default and every removed
     safety line. Write "none" if there are none. -->

## Gates

<!-- Changes to ci/ outside ci/data/, ci.yaml or the workflow set are not allowed in the same PR as a
     change they check. List every ci/data/ file that rides along. -->

## Non-goals

<!-- What this PR deliberately does not change. -->
