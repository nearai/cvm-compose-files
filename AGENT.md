# AGENT.md

Rules for editing this repo. See README.md for the layout.

## Scope

- `prod/` holds only what runs in production. Do not add experiments, canaries or variants; when a file stops
  running, delete it (and anything only it needed).
- This repo is public: no hostnames, IPs, secrets, people, incident details or private repo names, in files, commits
  or PRs. Keep PR and commit text to 1-3 sentences.

## Editing compose files

- Pin every image by digest. Change engine images only through the publish workflows (README.md, Images).
- Comments: one short line, only where a flag or setting is non-obvious. No history, measurements, rollout notes
  or dates.
- `container_name` equals the service key. Service names: `model-<sg|vllm>-<model>-<quant>-tp<N>[-r<i>]`,
  `proxy-<model>`, `dcgm-<model>`; fixed names `nginx`, `model-downloader`, `model-proxy-registrar`,
  `otelcol-contrib`. Renaming a service means updating every reference (backend URLs, nginx, scrape targets,
  `depends_on`, registrar checks).
- Never change a prod model's served name (`MODEL_NAME`, `--served-model-name`) or SNI domain: clients and usage
  billing key on them.
- Secrets come from deploy-time environment variables (`${VAR}` or `${VAR:?message}`); never commit values.

## Deploying

- Deploy only from a `v0.0.N` tag on `main`.
- Replace one engine replica at a time with a scoped services list, and wait for it to be healthy before the next.
  The GLM long-context file also defines TP4 services on the same GPUs as the TP2 replicas: never run an unscoped
  `up` with it.
- The `*-openrouter-baremetal.yaml` files are not deployed by compose-manager; they run with plain
  `docker compose -p openrouter` on their hosts.

## Before opening a PR

Run the validators listed in README.md (CI) and `docker compose -f <file> config` on every file you changed.
