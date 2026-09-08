# gpu02 maintenance handoff

- Move the existing Qwen configurations to another production GPU host before maintenance; do not change images, model arguments or replica counts as part of that move.
- `prod/qwen36-qwen38.yaml` retains two Qwen3.8 replicas on GPUs 2 and 3 and two Qwen3.6 replicas on GPUs 4 and 5, with their existing proxy pools, ports and caches.
- `prod/small-models.yaml` restores the pre-consolidation Qwen layout: two Qwen3.6 replicas on GPUs 1 and 2; utility models and GLM5.1 allocations are unchanged.
- Destination placement is undecided. Verify current allocations before choosing a host; gpu13 is only a candidate.
- The previous Qwen consolidation, handover, custom runtime and post-upgrade layout are withdrawn.
- This is a configuration cleanup, not a deployed migration. Read the live Compose Manager `projects.work.current`, containers and registry before any operation; repository contents are not proof of current host state.
- Do not apply a reduced recipe over a live stack: Compose Manager can remove omitted services as orphans. Withdraw and drain removed services against the deployed configuration first.
- Keep source serving until every replacement replica passes real inference, streaming, routing and capacity checks. Preserve unrelated models and existing client streams.
- Shut down the CVM only after its complete live inventory is served elsewhere and all source requests/connections have drained.
- Use Compose Manager with the auto-generated release tag containing the approved change. Do not deploy a raw commit as the tag.
