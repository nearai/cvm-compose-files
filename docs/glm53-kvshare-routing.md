# Cache-aware routing for in-host KV sharing (GLM-5.3 base tier)

What routing has to do so the in-host shared KV tier (`docs/glm53-kvshare-ab.md`) pays off.

**How this was produced.**
- The repos were set up with Pierre's `nearai-inference-workspace` skill (knowledge-base PR #11), so the review read every routing layer at `origin/main` side by side: cloud-api, inference-proxy, model-proxy, cvm-compose-files, cvm-ansible-playbooks, compose-manager, infra-docs.
- Command used: `workspace.py setup --root ~/Documents/nearai --protocol ssh --no-docs`.
- File:line references are at `origin/main` on 2026-10-07.

## Where the cache is lost today (prod ghost cache, gpu03/gpu04, 3 h)

- About 63–64% of prompt tokens are served from cache.
- Only 1.2–1.6% are cached solely on another replica of the same host (`sglang:ghost_pool_other_replica_only_tokens_total`). The in-host proxy's conversation affinity already keeps turns on one replica.
- The rest of the gap is capacity, plus conversations whose turns land on different hosts.

**In-host sharing fixes replica moves. Host moves are fixed only by host-level routing.**

## The routing layers, as built

| Layer | Picks | Key | Stickiness | On a miss |
|---|---|---|---|---|
| cloud-api legacy (prod) | host index `-iN` | turn 1: first message; later: first two messages (`fleet.rs:502-508`, `prefix_router.rs`) | none (stateless `key % count`, `fleet.rs:518`) | the hash slot, plus a hot-prefix burst spill of 4 |
| cloud-api placement (staging only) | host and replica | system message + first user message (`completions/affinity.rs:167`) | Valkey pin; HRW home | lane/score; falls back to legacy on CapacityFull/LaneFull/stale |
| model-proxy | which host each `-iN` means | `N % count` over sorted routable backends (`src/router.rs:90-108`) | none | a health flap remaps every index |
| OpenRouter gateway (inference-proxy gateway mode, cpu01) | host | salted body key, salt regenerated per restart (`backend_affinity.rs:104`) | 1200 s idle pin | least-connections |
| in-host proxy (inference-proxy host mode) | replica | salted `conversation_key` (`vllm_dp_affinity.rs:82`) | 1200 s idle pin, `MAX_IMBALANCE` 8 (`backend_pool.rs:477`) | least-connections |

## What has to change to get the full benefit

| # | Change | Where | Effect | Status |
|---|---|---|---|---|
| 1 | Key every turn through the first user message | cloud-api `prefix_router.rs` + `fleet.rs::route_key` | Turn 2 lands on the host that built turn 1's cache, removing the planned turn-2 host change of PR #903. Hot system prompts no longer share one host on turn 1, but each host caches them once | **Built**, behind `NEARAI_PREFIX_ROUTE_THROUGH_FIRST_USER=1` (default off). cloud-api branch `pranavraja99/route-key-through-first-user`, with tests. Needs a decision against #903's rationale |
| 2 | Rendezvous (HRW) hashing over host IDs instead of `key % count` | cloud-api `fleet.rs::candidate_indices` | A health flap or scale change moves only that host's share, not every conversation | Proposed |
| 3 | Ship placement to prod (set `REPLICA_STATE_REDIS_URL` per host) | cvm-compose-files host files + cloud-api config | Already keys like #1, pins in Valkey, ranks by HRW. For a host with sharing, rank by host only and let the in-host proxy pick the replica | Proposed (merged code; gated per host) |
| 4 | In-host sharing groups (`VLLM_BACKEND_SHARING_GROUPS`, e.g. `0,1;2,3`) | inference-proxy `pick` | Within a group, route by load and drop the pin; across groups keep the `MAX_IMBALANCE` bound | Proposed for GPU-to-GPU sharing (~0.1 s moves). Keep affinity on while sharing is L3-only (a moved turn costs a restore, ~13 s at 220K) |
| 5 | Gateway: HRW first pick within slack of least-loaded, plus a stable salt from an env secret | inference-proxy gateway mode + cvm-ansible-playbooks vars | OpenRouter conversations keep their host across gateway restarts | Proposed (small gain) |

## Metrics for the A/B and for routing

- **Built:** `backend_affinity_followup_lookups_total{outcome=hit|miss}`, inference-proxy branch `pranavraja99/affinity-followup-metric`. It counts follow-up turns (the body already holds assistant or tool messages) that do or don't find this host's pin. A miss means the conversation's earlier turns were served on another host. This is the direct readout for changes 1–3, and it covers both cloud-api and gateway traffic.
- **Built:** `sglang:kvshare_storage_hit_tokens_total{source=self|peer}` and `sglang:kvshare_storage_written_tokens_total`, from the shared-KV startup patch.
- **Existing:** `sglang:ghost_*` (the prize and recovery), `sglang:kv_tier_*`, `backend_affinity_lookups_total`, `backend_affinity_selections_total{outcome}`, `placement_hint_honored_total` / `_overridden_total`, and `cloud_api.placement.*`.

## Recommended order

1. **Deploy the in-host A/B (shared L3 tier) with affinity unchanged.** Measure peer hits and TTFT per arm.
2. **Ship the follow-up metric to one host's proxy.** Measure the cross-host follow-up miss rate (the size of the prize for 1–3).
3. **Enable change 1 on cloud-api** (staging, then prod), and watch the follow-up miss rate fall. Then do change 2 or 3.
4. **Do change 4** only with GPU-to-GPU sharing (`docs/glm53-gpu-peer-kv-fetch-design.md`).
