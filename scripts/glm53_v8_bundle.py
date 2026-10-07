"""Shared constants and release gate for the GLM-5.3 Flash "v8 bundle" canary slots.

The v8 bundle is one engine image: v6 + the FP8 KV patch + the preprocessing stall fix + a
profiling hook (off) + the tool-schema depth cap (docs/glm53-v8-bundle-canary.md). It canaries on
exactly one replica of the base file (gpu03 r4) and one TP2 replica of the long-context file
(gpu02 r2a). Both files are deployed by two hosts, so neither file hardcodes the bundle: each canary
slot reads a set of per-replica override variables that are empty (or the current prod value) by
default and are set only in the canary host's compose-manager env map. This module owns what the two
generators and the env-map printer share.

FILL AFTER PUBLISH: `V8_IMAGE_DIGEST` and `TOOL_SCHEMA_MAX_DEPTH`. Both are placeholders on purpose;
`release_errors()` fails on them (the unit tests and `scripts/glm53_v8_canary_env.py` call it), so the
change cannot be deployed with a placeholder by accident.
"""

import re
from typing import Final

IMAGE_REPO: Final = "docker.io/nearaidev/sglang"

# ---- FILL AFTER PUBLISH (the parallel image PR reports the digest and the depth cap) ----------------
IMAGE_DIGEST_PLACEHOLDER: Final = "sha256:REPLACE_WITH_V8_BUNDLE_DIGEST_AFTER_PUBLISH"
V8_IMAGE_DIGEST: Final = IMAGE_DIGEST_PLACEHOLDER
TOOL_SCHEMA_MAX_DEPTH_PLACEHOLDER: Final = "<tbd>"
# Final per the image PR (#345): both tool-schema knobs default to 0/off in the image; the node cap is the real guard
# (check_schema is linear in node count).
TOOL_SCHEMA_MAX_DEPTH: Final = "32"
TOOL_SCHEMA_MAX_NODES: Final = "25000"
PREPROCESS_LOG_SLOW_S: Final = "5"
# ------------------------------------------------------------------------------------------------------

# The preprocessing environment of the bundle, in the order the printer emits it. Names as reported by
# the image PR's author; the values for the first two are the assumed defaults and are easy to change.
PREPROCESS_WORKERS: Final = "4"
PREPROCESS_TIMEOUT_S: Final = "60"


# Self-profiling hook (#343, near-self-profile.diff): inert unless NEAR_SELF_PROFILE is exactly "1", TP rank 0 only.
# It is enabled on the two canary replicas only, through the same per-replica ENV_PREFIX as the rest of the bundle's
# environment, so every other replica and host renders without it (the validator forbids the name as a literal).
PROFILE_ENVIRONMENT: Final = {"NEAR_SELF_PROFILE": "1", "NEAR_SELF_PROFILE_AFTER_S": "900", "NEAR_SELF_PROFILE_STEPS": "50"}


def preprocess_environment() -> dict[str, str]:
    return {
        "SGLANG_PREPROCESS_WORKERS": PREPROCESS_WORKERS,
        "SGLANG_PREPROCESS_TIMEOUT_S": PREPROCESS_TIMEOUT_S,
        "SGLANG_TOOL_SCHEMA_MAX_DEPTH": TOOL_SCHEMA_MAX_DEPTH,
        "SGLANG_TOOL_SCHEMA_MAX_NODES": TOOL_SCHEMA_MAX_NODES,
        "SGLANG_PREPROCESS_LOG_SLOW_S": PREPROCESS_LOG_SLOW_S,
    }


# Telemetry: the suffix on config_variant, and the precision label the FP8 KV cache makes true.
VARIANT_SUFFIX: Final = "-v8bundle"
FP8_PRECISION: Final = "int4-weights-fp8-activations-fp8-kv"
KV_CACHE_DTYPE: Final = "fp8_e4m3"
DSA_BACKEND: Final = "flashmla_kv"
OVERLAP_FLAG: Final = "--disable-overlap-schedule"

# Variable name suffixes of one canary slot (the slot's prefix is prepended: GLM53_V8_R4_, GLM53_V8_R2A_).
# Each is read only by that one replica. The first group has a non-empty default (today's prod value);
# the second group is empty by default. Order is the order the printer emits them.
SLOT_VALUE_VARIABLES: Final = ("IMAGE", "IMAGE_LABEL", "PRECISION", "KV_DTYPE", "DSA_BACKEND", "MAX_RUNNING")
SLOT_EMPTY_VARIABLES: Final = ("VARIANT_SUFFIX", "EXTRA_ARGS", "ENV_PREFIX")


def expression(prefix: str, name: str, default: str) -> str:
    """Compose's ${VARIABLE:-default}: the default when the variable is unset or empty."""
    return "${" + prefix + name + ":-" + default + "}"


def image_reference(digest: str | None = None) -> str:
    return f"{IMAGE_REPO}@{digest or V8_IMAGE_DIGEST}"


def engine_image_label(digest: str | None = None) -> str:
    """The 12-hex label dashboards carry for an image (the first 12 characters of the digest hex)."""
    value = (digest or V8_IMAGE_DIGEST).split(":", 1)[-1]
    return value[:12]


def env_prefix_value() -> str:
    """Value of the slot's ENV_PREFIX: `env` followed by the bundle's environment.

    The engine environment cannot be overridden per replica without leaving a trace in the rendered
    file of every other replica (an empty list entry renders as an empty-named variable, and a
    mapping entry renders as an empty value). The command line can: ${..._ENV_PREFIX:-} is the first
    token, empty by default, so every other replica's argv is untouched, and when set it is
    `env NAME=value ... sglang serve ...`, which execs sglang with the variables in its environment.
    Values must therefore be single words (no spaces, quotes or shell syntax).
    """
    return "env " + " ".join(f"{name}={value}" for name, value in (preprocess_environment() | PROFILE_ENVIRONMENT).items())


def release_errors(digest: str | None = None, environment: dict[str, str] | None = None) -> list[str]:
    """What stops this bundle from being deployed. Empty means the placeholders have been filled in."""
    errors: list[str] = []
    digest = digest or V8_IMAGE_DIGEST
    if digest == IMAGE_DIGEST_PLACEHOLDER or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        errors.append(
            f"V8_IMAGE_DIGEST is {digest!r}: replace it with the published v8 bundle digest (sha256:<64 hex>) in "
            "scripts/glm53_v8_bundle.py (the generated compose files hold only today's value as the default and do not change)"
        )
    for name, value in (environment if environment is not None else preprocess_environment()).items():
        if name in ("SGLANG_TOOL_SCHEMA_MAX_DEPTH", "SGLANG_TOOL_SCHEMA_MAX_NODES"):
            continue  # checked below as integers
        if not re.fullmatch(r"[A-Za-z0-9_.:/=-]+", value) or "<" in value or "tbd" in value.lower():
            errors.append(f"{name}={value!r} is a placeholder or not a single plain word: set the real value in scripts/glm53_v8_bundle.py")
    depth = (environment if environment is not None else preprocess_environment()).get("SGLANG_TOOL_SCHEMA_MAX_DEPTH", "")
    if not re.fullmatch(r"[1-9][0-9]*", depth):
        errors.append(f"SGLANG_TOOL_SCHEMA_MAX_DEPTH={depth!r} must be a positive integer (the depth cap the image PR chose)")
    nodes = (environment if environment is not None else preprocess_environment()).get("SGLANG_TOOL_SCHEMA_MAX_NODES", "")
    if not re.fullmatch(r"[1-9][0-9]*", nodes):
        errors.append(f"SGLANG_TOOL_SCHEMA_MAX_NODES={nodes!r} must be a positive integer (the node cap is the real tool-schema guard)")
    return errors


def interpolate(text: str, environment: dict[str, str]) -> str:
    """Compose's ${NAME:-default} (the default when NAME is unset or empty), as the tests resolve a file.

    Only the form this repo's override variables use is supported; `docker compose config` is the reference
    and the tests compare against it whenever the docker compose plugin is available.
    """

    def resolve(match: "re.Match[str]") -> str:
        return environment.get(match.group(1)) or match.group(2)

    return re.sub(r"\$\{([A-Za-z0-9_]+):-([^}]*)\}", resolve, text)
