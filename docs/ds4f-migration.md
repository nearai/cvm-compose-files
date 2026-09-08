# gpu03 serving configuration

- Keep `prod/GLM-5.1-DSV4-Migration.yaml`: GLM5.1 r1 on GPUs 0–3; two DS4F TP2 replicas on GPUs 4–5 and 6–7.
- The Qwen cleanup does not change this configuration, its images, model arguments, routing or telemetry.
- Preserve the deployed `GLM51_BACKEND_URLS` single-replica override and the explicitly enabled `ds4f-migration-registrar`.
- Verify actual Compose Manager project state and registry health before operations. Keep the existing control-plane environment; do not copy credentials into this repository.
- Do not apply the original two-replica GLM file over this stack: its second replica would conflict with the occupied GPUs.
- Any later capacity change requires healthy replacement serving, withdrawal and graceful drain before selective removal. Never prune caches or volumes during a cutover.
