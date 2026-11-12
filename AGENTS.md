# AGENTS.md

Rules for everyone who changes or reviews this repo, people and agents. CI enforces what it can;
the author and the reviewer cover the rest. `CLAUDE.md` only imports this file. The rule files under
Area rules are part of these rules. Each rule is stated in one file; other files link to it.

## Purpose
Deployable Docker Compose files for confidential inference, plus the build contexts of the images
they pin. The repo is public. From a release tag and a file path, a verifier can trace image digests,
build sources and model revisions, except where VERIFYING.md lists a known gap.

## Layout
- `prod/` every deployable compose file, flat: serving, pack (`pack-*`) and utility (`util-*`) files.
- `docker/<dir>/` build context of one published image. Its build inputs never change once merged.
- `ci/` the gate: `run.py`, `lib/`, `checks/` (one guard each), `config/` (tool configs), `templates/`;
  and `ci/data/`, data files that ride along with the changes they record. No other subdirs.
- `docs/` usage docs, at most one per deployable: `docs/<file stem>.md`.
- `.github/` `ci.yaml`, `auto-tag.yaml`, `publish-<docker dir>.yaml` workflows, PR template.
- `.claude/` area rules (`rules/`), read-only review agents (`agents/`), a deny-only `settings.json`.
- Legacy, listed in `ci/data/legacy.txt`, shrinking: older deployable and publish workflow names,
  `scripts/` (old validators, no new files), root `cleanup-hf-model.yaml`, and `adapters/`, kept
  until no release tag that may be redeployed for rollback still fetches from `main`.
- No other top-level dirs.

## Area rules
Read the rule file for a path before you change files under it; Claude Code also loads it by itself.
- `prod/`, `ci/templates/`: [.claude/rules/prod.md](.claude/rules/prod.md), names, labels, lifecycle.
- `docker/`: [.claude/rules/docker.md](.claude/rules/docker.md), build dirs and image tags.
- `ci/`: [.claude/rules/ci.md](.claude/rules/ci.md), guard code, data files, self-tests.
- `.github/workflows/`: [.claude/rules/workflows.md](.claude/rules/workflows.md), how a workflow is written.
- Every review: [.claude/rules/review.md](.claude/rules/review.md). Agents in `.claude/agents/` only
  read and report; the author or reviewer decides.

## Commands
`ci/run.py` renders with the pinned compose image in `ci/lib/common.py`, as CI does; never local compose.
- All guards: `python3 ci/run.py --base origin/main`. Self-tests: `python3 ci/run.py --self-test`.
- Findings on the real tree are pinned in `ci/data/expected.json`; update it with `--write-expected`.
- Render one file: `--render prod/<f>.yaml`. Show a change is a no-op: `--noop origin/main` (empty).

## Hard rules (CI fails)
- Paths follow the layout rules in `ci/checks/layout.py` or are listed in `ci/data/legacy.txt`, which
  only shrinks. No binaries, bytecode, notebooks, logs or secret files. Size cap per file.
- Agent instruction and config files (`CLAUDE*.md`, `AGENTS*.md`, `.claude/`, `.mcp.json`, `.cursorrules`
  and the like) exist only at the paths listed under Layout.
- Every image is `repo@sha256:<64 hex>` and has a row in `ci/data/images.allow`.
- Every model is pinned to a 40-hex revision. `--trust-remote-code` only with a pinned revision.
- No `build`, `dockerfile_inline`, `extends`, `include` or `env_file` in deployable files.
- No runtime fetch from a branch. URLs pin a 40-hex commit or revision.
- Every SGLang engine sets `--log-requests-level 0` and every vLLM engine `--no-enable-log-requests`,
  each exactly once. Engines without it are listed exceptions and appear in VERIFYING.md.
- No service on an SGLang or vLLM image, whatever its role, enables request logging, profiling,
  tracing or dumps by flag or env, or logs at debug.
- Privileged mode, devices, sockets, host mounts and published ports are listed per service in
  `ci/data/services.allow.json`. Host-provided variables are listed under `env` in the same file, each
  with `egress: true` (the value can choose where data goes) or `egress: false`.
- No IPs, internal domains, host or runner names or private repo names in files, paths, commit
  messages, branch names or PR text; only what `ci/data/public.json` lists as public is allowed.
  Comments follow the policy below. No inline linter ignores.
- Build inputs of merged build dirs, `adapters/` and existing publish workflows are only ever deleted,
  never edited. Build inputs: the Dockerfile, what it copies, files in `SHA256SUMS` or `SERIES`, files
  a publish workflow names. Other `.md` files there may be edited; the public-text rules scan them.
- A new image is a new `docker/<dir>/` plus `.github/workflows/publish-<dir>.yaml` in one PR. CI lints
  a publish workflow only until it merges; its jobs run only in the protected `publish` env, never on PRs.
- The README status table is generated from the rendered files (`--write-status`); only `status` is
  hand-written. At most one candidate per serving group. A model served by more than one live file
  has one note below the table saying how the files differ.
- New paths follow Naming. Service key == `container_name`. Served names never change in place.
- Waivers live only in `ci/data/exceptions.json`: `rule`, exact `path`, `sha256` of the matched text,
  `reason`, `remove_when`. Never copy the matched text into the entry or the PR. `claude.*`,
  `layout.claude-md` and `layout.agent-file` cannot be waived.

## Naming
Lowercase, `-` between tokens. No lifecycle words (new, old, test, canary, candidate, final, `v<N>`,
dates) and no host names in `prod/` or `docs/` paths or in workflow, job or step `name:` fields.
Version tokens are allowed in `docker/` dir names and so in `publish-<dir>.yaml` file names. Patterns
per area are in the area rules. CI checks paths, new build-dir names, service-key equality and
served-name stability;
review checks the rest against the templates.
- Served names are public API IDs and never change for a live model. A new model serves its upstream
  Hugging Face `org/repo` ID; quantization, engine and topology never appear in a served name.
- Config revision = release tag `v0.0.N` (auto-tag only) plus the file path; never in file names.

## Comments and docs
- Comments say why, at most 2 lines in a row. No history, dates, names, hosts, plans, run IDs, PR
  numbers, TODO/FIXME/HACK or commented-out config.
- No edits inside YAML block scalars (`command: |`, embedded config contents) to change comments;
  those bytes are config. Comment edits stop at the scalar.
- Docs describe what is. No plans, runbooks, rollout or incident notes, experiments or live state.
- Docs never claim more than CI checks. Exceptions and known gaps are listed in VERIFYING.md.

## Security and sensitive data
Everything here is public and copied by others: files, branches, tags, PRs, reviews, issues, CI logs.
- Never write: secrets (also expired or test ones); hostnames, IPs, internal domains or URLs, dashboards,
  account IDs; people, partners, customers; incidents, experiments, benchmarks, rollout state, plans;
  private repo names or links; deploy-time values. This includes `ci/` data files. If internal context
  matters, say "see the private ops repo" without naming it.
- Secrets arrive only as deploy-time env: `${NAME}` or `${NAME:?msg}`, never a literal or a default.
  Secret variable names end in `_TOKEN`, `_KEY`, `_SECRET`, `_PASSWORD` or `_CREDENTIALS`. They appear
  only under `environment:` of the service that needs them.
- Workflows that hold secrets never run PR code. `ci.yaml` has read-only permissions and no secrets.
- `.claude/settings.json` denies secret reads, outward actions (`gh` reviews, merges, comments, issues,
  gists and API calls; `curl`, `wget`, WebFetch) and history rewrites. CI requires every entry. The list
  is a guardrail, not a boundary; branch and tag protection is enforced by GitHub rulesets.
- If something sensitive was pushed: do not force-push or explain it publicly; rotate secrets first;
  tell maintainers privately; remove it in a neutral PR. Vulnerabilities: see SECURITY.md.

## Pull requests
- One concern: new config, new image, image bump, config tune, rules/CI change, deletion or promotion,
  or docs. Only the doc and table updates the change forces may ride along.
- Title and commit subjects: `<prod|docker|ci|docs|repo>: <summary>`, at most 72 chars counting the
  ` (#N)` squash adds; merge commits from updating a branch are exempt.
- Squash merge only. No force-push after review starts. Never move or delete tags or rewrite history.
- Gates: `ci/` outside `ci/data/`, `ci.yaml`, the workflow set, `AGENTS.md`, `CLAUDE.md` and `.claude/`.
- Merging needs `all-green` and one approval from a non-author, two for a gate change; a
  `ci/data/`-only change, `expected.json` included, needs one. Settings cannot enforce that per path.
  A new push dismisses approvals. Agents never approve their own work.
- Evidence in the PR body: `ci/run.py` green; rendered-diff note for `prod/` changes; `--noop` output
  for comment-only or refactor changes; for image bumps the `main` publish run and the signature
  verification; for deletions the deployment-manager check; every new sink, port, mount, env var,
  moved default or removed safety line named.
- Never change a gate in the same PR as a change it checks or reviews. Entries in `ci/data/` (waivers,
  allowlists, legacy line removals, expected findings) may ride along and are listed under Gates in the PR.
