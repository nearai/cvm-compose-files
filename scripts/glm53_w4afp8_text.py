HEADER = (
    "# gpu02 long-context r1 W4AFP8 candidate generated from\n"
    "# prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml. r2 remains the deployed FP8\n"
    "# HiCache arm. r1 uses the Graphistry W4AFP8 checkpoint,\n"
    "# a 16384-token prefill chunk, and the gpu31/gpu32-verified loader source change.\n"
    "# r1 pins docker.io/nearaidev/sglang@sha256:8bce6a7cc872a80faded3bd1ef0a64873a1d7abae34c94e5358775ca21f133cc,\n"
    "# published by workflow run 35659748426 from recipe merge commit\n"
    "# 7c473970af2ac040afb233df8b274aa0cf8ebbcb. PR #278 and the combined-image\n"
    "# recipe PR are merged; publication reported zero critical vulnerabilities and\n"
    "# passed CPU, provenance, attestation, and cosign verification.\n"
    "# Admission reserve remains disabled on both long-context arms. Deploy only to\n"
    "# gpu02 with docs/gpu02-glm53-w4afp8-long-context.md. All operational services\n"
    "# require the w4afp8-long-context profile, so an unscoped default apply cannot\n"
    "# change the stack.\n"
    "# Do not hand-edit this file.\n"
)

HEADER_REPLACEMENTS = (
    (
        (
            "# Hand-derived from prod/GLM-5.3-Flash-SGL-TP4.yaml. Routing remains the dedicated\n"
            "# long-context contract below, while the engines intentionally form an r1 control / r2\n"
            "# HiCache experiment and therefore are not byte-identical to the canonical file.\n"
        ),
        (
            "# Generated from the deployed long-context file. Routing remains the dedicated\n"
            "# long-context contract below, while r1 is the W4AFP8 treatment and r2 remains\n"
            "# the deployed FP8 HiCache arm.\n"
        ),
    ),
    (
        (
            "#   behavior stay unchanged; only r2's engine and the two replicas' truthful telemetry\n"
            "#   variants differ for this experiment:\n"
        ),
        (
            "#   behavior stay unchanged; only r1's engine identity and the two replicas' truthful\n"
            "#   telemetry variants differ for this experiment:\n"
        ),
    ),
    (
        (
            "#   Non-HiCache engine flags remain identical between replicas: chunk 8192 OOMs in the\n"
            "#   DSA indexer under concurrent 400K+ contexts (exactly this tier's load), chunk 2048\n"
            "#   halves prefill speed, and no other per-replica flag moved the tail. Conversation\n"
            "#   affinity stays on because the prefix cache is worth ~3x in request capacity.\n"
        ),
        (
            "#   r1 intentionally raises its prefill chunk to 16384 for the W4A8 kernel. The deployed\n"
            "#   FP8 arm previously OOMed at 8192 under concurrent 400K+ contexts, so the mandatory\n"
            "#   pre-customer gate is a pool-clamped 100K-to-1M long-context replay. Conversation\n"
            "#   affinity stays on because the prefix cache is worth ~3x in request capacity.\n"
        ),
    ),
    (
        (
            "#   Experiment rollback: redeploy the prior long-context tag, or remove r2's HiCache\n"
            "#   image/command/environment override so it inherits the r1 control settings. Routing\n"
            "#   rollback remains LONG_TIER_ONLY=false + registrar restart (host rejoins the base\n"
            "#   pool while still serving the long domain), or remove the cloud-api long_context block.\n"
        ),
        (
            "#   Experiment rollback: stop the W4AFP8 r1 and restore r1 from the prior long-context\n"
            "#   tag; r2 stays on its deployed FP8 HiCache definition. Routing rollback remains\n"
            "#   LONG_TIER_ONLY=false + registrar restart, or remove the cloud-api long_context block.\n"
        ),
    ),
    (
        (
            "# DSA import-cycle fixes. r1 pins the published base engine as the ordinary GPU-prefix-\n"
            "# cache control; r2 pins the published HCC-safe HiCache derivative and uses an 80%\n"
            "# startup host-memory budget across all four TP ranks by default. The deployment may\n"
            "# override that percentage with GLM53_HICACHE_RAM_BUDGET.\n"
        ),
        (
            "# DSA import-cycle fixes. r1 pins the published signed loader-plus-pool-clamp combined\n"
            "# image; r2 keeps the published HCC-safe HiCache derivative and its deployment-\n"
            "# overridable 80% startup host-memory budget.\n"
        ),
    ),
    (
        (
            "# bounded 8-request queue, 4096-token prefill chunks, decode graphs capped at batch 32,\n"
            "# TileLang DSA, DeepGEMM, and adaptive EAGLE 5/1/6. This retains the full\n"
        ),
        (
            "# bounded 8-request queue, r1's 16384-token W4AFP8 chunk, r2's 4096-token FP8 chunk,\n"
            "# decode graphs capped at batch 32, TileLang DSA, and adaptive EAGLE 5/1/6. This retains the full\n"
        ),
    ),
)
