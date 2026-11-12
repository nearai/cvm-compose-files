# AGENTS.md

Rules for everyone who changes or reviews this repo, people and agents. CI enforces what it can;
the author and the reviewer cover the rest. `CLAUDE.md` only imports this file.

## Purpose
Deployable Docker Compose files for confidential inference, plus the build contexts of the images
they pin. The repo is public. Anyone can go from a release tag and a file path to the exact image,
its build source and the model revision (VERIFYING.md, which also lists known gaps).

## Layout
- `prod/` every deployable compose file, flat: serving, pack (`pack-*`) and utility (`util-*`) files.
- `docker/<dir>/` build context of one published image. Never edited once published.
- `ci/` guards, their data files, tool configs and two templates. Flat.
- `docs/` usage docs, at most one per deployable: `docs/<file stem>.md`.
- `.github/` `ci.yaml`, `auto-tag.yaml`, `publish-<docker dir>.yaml` workflows, PR template.
- Legacy, listed in `ci/legacy.txt`, shrinking: older deployable and publish workflow names,
  `scripts/` (old validators, no new files), `adapters/` (removed after repinning), root
  `cleanup-hf-model.yaml`.
- No other top-level dirs, no subdirs in `prod/` or `ci/`.

## Commands
`ci/run.py` renders through the digest-pinned compose image set in `ci/common.py`, the same one CI
uses. Do not judge a change with a locally installed compose version.
- Render one file: `python3 ci/run.py --render prod/<f>.yaml`
- All guards: `python3 ci/run.py --base origin/main`. Self-tests: `python3 ci/run.py --self-test`.
- Findings on the real tree are pinned in `ci/expected.json`; update it with `--write-expected`.
- Show a change is a no-op: `python3 ci/run.py --noop origin/main` (empty output).

## Hard rules (CI fails)
- Paths follow the layout rules in `ci/check_layout.py` or are listed in `ci/legacy.txt`, which only
  shrinks. No binaries, bytecode, notebooks, logs or secret files. Size cap per file.
- Every image is `repo@sha256:<64 hex>` and has a row in `ci/images.allow`.
- Every model is pinned to a 40-hex revision. `--trust-remote-code` only with a pinned revision.
- No `build`, `dockerfile_inline`, `extends`, `include` or `env_file` in deployable files.
- No runtime fetch from a branch. URLs pin a 40-hex commit or revision.
- New SGLang engines set `--log-requests-level 0`. New vLLM engines keep the logging flags of
  `ci/template-vllm.yaml`, which sets an explicit disabling flag only if the pinned vLLM has one.
  Existing engines without an explicit flag are listed exceptions and appear in VERIFYING.md.
- No service on an SGLang or vLLM image, whatever its role, enables request logging, profiling,
  tracing or dumps by flag or env, or logs at debug.
- Privileged mode, devices, sockets, host mounts and published ports are listed per service in
  `ci/services.allow.json`. Host-provided variables are listed under `env` in the same file, each
  with `egress: true` (the value can choose where data goes) or `egress: false`.
- No IPs, internal domains, host or runner names or private repo names in files, paths, commit
  messages, branch names or PR text. Comments follow the policy below. No inline linter ignores.
- Existing build dirs, `adapters/` and existing publish workflows are only ever deleted, never edited.
- A new image is a new `docker/<dir>/` plus `.github/workflows/publish-<dir>.yaml` in one PR. Publish
  jobs run only in the protected publish environment and never on pull requests.
- The README status table lists every deployable once with the served names it `serves`; at most
  one candidate per serving group.
- New paths follow Naming. Service key == `container_name`. Served names never change in place.
- Waivers live only in `ci/exceptions.json`: `rule`, exact `path`, `sha256` of the matched text,
  `reason`, `remove_when`. Never copy the matched text into the entry or the PR.

## Naming and versioning
Lowercase, `-` between tokens. No lifecycle words (new, old, test, canary, candidate, final, v2, dates)
and no host names in file or workflow names. CI checks paths, service keys and served-name stability;
the rest is shown in the templates and checked in review.
- Serving file: `prod/<checkpoint>-<engine>-<topology>[-<tier>].yaml`. `checkpoint` = lowercased HF repo
  name; `engine` = `sglang` | `vllm`; `topology` = `tp<N>` | `dep<N>` | `dp<N>`, `x<R>` for replicas;
  `tier` only to tell two files apart (`base`, `long`).
- Pack: `prod/pack-<purpose>.yaml`. Utility or any non-serving project: `prod/util-<verb>-<object>.yaml`.
- Build dir: `docker/<engine>-<upstream>-<scope>-r<N>/`, e.g. `sglang-v0.5.23-glm53-r1`. A rebuild is a
  new dir with `r<N+1>`. Image tag = dir name without `<engine>-`. Compose files pin the digest only.
- Serving and pack files set no top-level `name:`; utility files set `name:` to the file stem.
- Service keys: `model-sg-*`, `model-vllm-*`, `proxy-*`, `model-proxy-registrar*`, `model-downloader*`,
  `nginx*`, `dcgm-*`, `sidecar-*`, `otelcol-contrib`, `hf-cleanup`.
- Served model name: `<family>-<size>[-long|-abliterated]`, no engine, layout or quantization tokens.
  Domain label = served name with `.` replaced by `-`.
- Engine labels: `served_model`, `model_path`, `deployment` (= file stem), `engine_image`, `instance`,
  `config_variant`; change `config_variant` when image, command or env changes.
- Model pin: the engine loads `models--<org>--<repo>/snapshots/<40hex>` offline; the downloader
  fetches the same repo at the same `--revision`. Env names `UPPER_SNAKE`.
- Config revision = release tag `v0.0.N` (auto-tag only) plus the file path. No version in names or files.
- New compose file: copy `ci/template-sglang.yaml` or `ci/template-vllm.yaml` and keep its line-1
  header form, key order and service order.

## Retention and lifecycle
- Status is `live` or `candidate`, only in the README table, never in a compose file.
- `live` runs in production. A `candidate` is being rolled out now: at most one per serving group
  (same served-name set), with the same service keys as the live file it replaces, or an announced
  change with an exception entry.
- Everything else not in production is deleted. Rollback = redeploy an older tag. Tags never move.
- Never rename a legacy path in place. A functional change (the rendered config differs) to a legacy
  file ships as a new standard-named candidate. Comment- or doc-only edits stay in place.
- Moving a deployed file takes two PRs: add the new file as candidate; the promotion PR deletes the
  old file, its table row, its legacy line and its doc.
- Before deleting a file a host may run, check the deployment manager and say so in the PR.

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
- If something sensitive was pushed: do not force-push or explain it publicly; rotate secrets first;
  tell maintainers privately; remove it in a neutral PR. Vulnerabilities: see SECURITY.md.

## Pull requests
- One concern: new config, new image, image bump, config tune, rules/CI change, deletion or promotion,
  or docs. Only the doc and table updates the change forces may ride along.
- Title and commit subjects: `<prod|docker|ci|docs|repo>: <summary>`, at most 72 chars counting the
  ` (#N)` squash adds; merge commits from updating a branch are exempt.
- Squash merge only. No force-push after review starts. Never move or delete tags or rewrite history.
- Merging needs `all-green` and one approval from a non-author (two for `ci/` changes; settings
  cannot enforce that per path). A new push dismisses approvals. Agents never approve their own work.
- Evidence in the PR body: `ci/run.py` green; rendered-diff note for `prod/` changes; `--noop` output
  for comment-only or refactor changes; for image bumps the `main` publish run and the signature
  verification; for deletions the deployment-manager check; every new sink, port, mount, env var,
  moved default or removed safety line named.
- Never change a gate (guard code, `ci/common.py`, `ci.yaml`, tool configs, the workflow set) in the
  same PR as a change it checks. Data entries (exceptions, allowlists, legacy line removals) may ride
  along and are listed under Gates in the PR.

## Reviewer checklist (every push, state the head SHA)
- [ ] Read the rendered diff, not only the YAML.
- [ ] Image bump: digest from a `main` publish run, signature verifies, no tag-only refs.
- [ ] New sink, endpoint, port, mount or env var; moved default; removed safety line.
- [ ] Gate edited together with the thing it gates; waiver text in the PR instead of `ci/exceptions.json`.
- [ ] Prose (README, VERIFYING, comments, PR text) says the same as the config and claims no more.
- [ ] Names of people, partners or customers and other leaks CI cannot see.
- [ ] New secret-like variable without a secret suffix.
- [ ] Naming, header, key and service order, labels follow the templates; `config_variant` changed if due.
- [ ] Docs: one per deployable; links resolve; `CLAUDE.md` is exactly `@AGENTS.md`; this file stays
      at most 140 lines; the signing-identity table in VERIFYING.md only gains rows.
- [ ] Changed files updated their references: README table, `ci/services.allow.json`, `ci/images.allow`,
      `ci/legacy.txt`, VERIFYING.md identities and known gaps.
