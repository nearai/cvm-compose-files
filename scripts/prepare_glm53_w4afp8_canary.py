#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run scripts/prepare_glm53_w4afp8_canary.py --write

import argparse
import difflib
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = Path("prod/GLM-5.3-Flash-SGL-TP4.yaml")
CANDIDATE = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-Canary.yaml")
PATCH = Path("overlays/glm53-w4afp8/modules-to-not-convert.diff")
CHECKPOINT = "graphistry/GLM-5.3-Flash-W4AFP8"
CHECKPOINT_REVISION = "99f1fa70408c52b007d4fd69e02e5a522422e755"
CONTROL_SERVICE = "model-sg-glm53-fp8-tp4-r2"
CANDIDATE_SERVICE = "model-sg-glm53-w4afp8-tp4-r2"
CANARY_PROFILE = "w4afp8-canary"
BASE_SOURCE_SHA256 = "21e9c527c9b83e350cdc35ce2bc62891cda1550934b2a5d302f0f807f752f125"
PATCH_SHA256 = "29764baa3e464d2272ea85f2e254392c2a61a3fc61a51f8d33b5910ce0cd8d00"
PATCHED_SOURCE_SHA256 = "039316192fb40a2aefe425102734d821c98e4c6c22a32ee51df21e47c315603d"
CONTROL_VARIANT = "fc91d24-admission-reserve-v10-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
CANDIDATE_VARIANT = "fc91d24-w4afp8-c16384-admission-reserve-v10-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
HEADER = (
    "# gpu04 r2-only W4AFP8 canary generated from prod/GLM-5.3-Flash-SGL-TP4.yaml.\n"
    "# r1 remains the production FP8 control. r2 uses the Graphistry W4AFP8 checkpoint,\n"
    "# a 16384-token prefill chunk, and the gpu31/gpu32-verified loader source change.\n"
    "# BLOCKED: this file still inherits the admission-reserve-only image, which lacks\n"
    "# the mandatory chunked-prefill pool clamp. Do not start the candidate until a\n"
    "# signed W4AFP8 image containing both patches is published and pinned here.\n"
    "# The current admission-reserve scheduler stays enabled on both arms, making this the\n"
    "# required interaction canary rather than a fleet-wide replacement. Deploy only to\n"
    "# gpu04 with docs/gpu04-glm53-w4afp8-canary.md. All operational services require\n"
    "# the w4afp8-canary profile, so an unscoped default apply cannot change the stack.\n"
    "# Do not hand-edit this file.\n"
)


class GenerationError(ValueError):
    pass


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    actual = text.count(old)
    if actual != expected:
        raise GenerationError(f"{label}: expected {expected} matches, found {actual}")
    return text.replace(old, new)


def section(text: str, start_marker: str, end_marker: str, label: str) -> tuple[int, int, str]:
    try:
        start = text.index(start_marker)
        end = text.index(end_marker, start + len(start_marker))
    except ValueError as error:
        raise GenerationError(f"{label}: canonical structure changed") from error
    return start, end, text[start:end]


def patch_bootstrap(
    patch_target: str = "/usr/share/nearai/glm53-w4afp8/modules-to-not-convert.diff",
    source_target: str = "python/sglang/srt/layers/quantization/w4afp8.py",
    patch_sha256: str = PATCH_SHA256,
    base_sha256: str = BASE_SOURCE_SHA256,
    patched_sha256: str = PATCHED_SOURCE_SHA256,
) -> list[str]:
    patch_check = f"{patch_sha256}  {patch_target}"
    after_check = f"{patched_sha256}  {source_target}"
    return [
        f"        printf '%s\\n' '{patch_check}' | sha256sum --check --strict",
        f"        source_sha256=$$(sha256sum {source_target} | cut -d' ' -f1)",
        '        case "$$source_sha256" in', f"          {base_sha256})",
        f"            git apply --check {patch_target}", f"            git apply {patch_target}", "            ;;",
        f"          {patched_sha256})", '            echo "W4AFP8 loader patch already applied"', "            ;;",
        "          *)",
        f"            echo \"Unexpected SHA256 for {source_target}: $$source_sha256\" >&2",
        "            exit 1",
        "            ;;",
        "        esac",
        f"        printf '%s\\n' '{after_check}' | sha256sum --check --strict",
    ]


def candidate_command(common: str) -> str:
    _, _, command = section(common, "  command: >\n", "  volumes:\n", "engine command")
    arguments = [line.strip() for line in command.splitlines()[1:] if line.strip()]
    if not arguments or arguments[0] != "sglang serve":
        raise GenerationError("engine command must start with sglang serve")

    model_path = (
        "/root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/"
        f"snapshots/{CHECKPOINT_REVISION}"
    )
    model_option = (
        "--model-path /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/"
        "snapshots/84c6a6aa9497188e15a635ba793b0f95a79b1033"
    )
    replacements = {
        model_option: f"--model-path {model_path}",
        "--chunked-prefill-size 4096": "--chunked-prefill-size 16384",
    }
    forbidden = {
        "--revision 84c6a6aa9497188e15a635ba793b0f95a79b1033",
        "--moe-runner-backend deep_gemm",
    }
    transformed = [replacements.get(argument, argument) for argument in arguments if argument not in forbidden]
    chunk_index = transformed.index("--chunked-prefill-size 16384")
    transformed.insert(chunk_index + 1, "--max-prefill-tokens 32768")

    lines = [
        "    command:",
        "      - /bin/bash",
        "      - -lc",
        "      - |",
        "        set -euo pipefail",
        '        echo "BLOCKED: signed two-patch W4AFP8 image digest is not pinned" >&2',
        "        exit 78",
        "        cd /sgl-workspace/sglang",
        *patch_bootstrap(),
    ]
    for index, argument in enumerate(transformed):
        prefix = "exec " if index == 0 else "  "
        suffix = " \\" if index + 1 < len(transformed) else ""
        lines.append(f"        {prefix}{argument}{suffix}")
    return "\n".join(lines) + "\n"


def generate(canonical: str, patch: str) -> str:
    digest = hashlib.sha256(patch.encode()).hexdigest()
    if digest != PATCH_SHA256:
        raise GenerationError(f"loader patch checksum mismatch: {digest}")
    if CANDIDATE_SERVICE in canonical or CHECKPOINT in canonical:
        raise GenerationError("canonical compose already contains the W4AFP8 canary")

    download_marker = "        echo \"Download complete.\"\n"
    download = (
        f"        echo \"Downloading {CHECKPOINT}...\"\n"
        "        uvx --from 'huggingface_hub[hf_xet]' hf download "
        f"{CHECKPOINT} --revision {CHECKPOINT_REVISION}\n"
    )
    updated = replace_exact(
        canonical,
        download_marker,
        download + download_marker,
        1,
        "model downloader",
    )
    updated = replace_exact(updated, CONTROL_SERVICE, CANDIDATE_SERVICE, 8, "r2 service identity")
    perception_loop = (
        "        for replica in (1, 2):\n"
        "            base = f\"http://model-sg-glm53-fp8-tp4-r{replica}:8000\"\n"
    )
    candidate_perception_loop = (
        "        for replica, service in ((1, \"model-sg-glm53-fp8-tp4-r1\"), "
        f"(2, \"{CANDIDATE_SERVICE}\")):\n"
        "            base = f\"http://{service}:8000\"\n"
    )
    updated = replace_exact(
        updated,
        perception_loop,
        candidate_perception_loop,
        1,
        "perception-check replica routing",
    )

    _, _, common = section(
        updated,
        "x-sg-glm53-flash-common: &sg-glm53-flash-common\n",
        "\nx-dcgm-common: &dcgm-common\n",
        "shared engine anchor",
    )
    command = candidate_command(common)

    service_start, service_end, service = section(
        updated,
        f"  {CANDIDATE_SERVICE}:\n",
        "\n  # Explicit operator-only semantic check;",
        "candidate service",
    )
    service_header = f"  {CANDIDATE_SERVICE}:\n"
    container = f"    container_name: {CANDIDATE_SERVICE}\n"
    config_mount = (
        "    configs:\n"
        "      - source: glm53_w4afp8_patch\n"
        "        target: /usr/share/nearai/glm53-w4afp8/modules-to-not-convert.diff\n"
        "        mode: 0444\n"
    )
    service = replace_exact(
        service,
        service_header,
        service_header + f'    profiles: ["{CANARY_PROFILE}"]\n    restart: "no"\n',
        1,
        "candidate safety profile",
    )
    service = replace_exact(service, container, container + command + config_mount, 1, "candidate override")
    service = replace_exact(service, "zai-org/GLM-5.3-Flash", CHECKPOINT, 2, "candidate model path labels")
    service = replace_exact(
        service,
        "precision:fp8-weights-bf16-kv",
        "precision:int4-weights-fp8-activations-bf16-kv",
        1,
        "candidate precision label",
    )
    service = replace_exact(service, CONTROL_VARIANT, CANDIDATE_VARIANT, 2, "candidate variant labels")
    updated = updated[:service_start] + service + updated[service_end:]

    scrape_start, scrape_end, scrape = section(
        updated,
        f"              - job_name: sglang-{CANDIDATE_SERVICE}\n",
        "              - job_name:",
        "candidate scrape job",
    )
    scrape = replace_exact(scrape, "zai-org/GLM-5.3-Flash", CHECKPOINT, 1, "scrape model path")
    scrape = replace_exact(
        scrape,
        'precision: "fp8-weights-bf16-kv"',
        'precision: "int4-weights-fp8-activations-bf16-kv"',
        1,
        "scrape precision",
    )
    scrape = replace_exact(scrape, CONTROL_VARIANT, CANDIDATE_VARIANT, 1, "scrape variant")
    updated = updated[:scrape_start] + scrape + updated[scrape_end:]

    profiled_service_markers = {
        "model-downloader": "    image: ghcr.io/astral-sh/uv:",
        "model-proxy-registrar": "    image: curlimages/curl@",
        "proxy-glm53": "    <<: *vllm-proxy-common",
        "model-sg-glm53-fp8-tp4-r1": "    <<: *sg-glm53-flash-common",
        "dcgm-glm53": "    <<: *dcgm-common",
        "otelcol-contrib": "    image: otel/opentelemetry-collector-contrib@",
        "nginx": "    image: nginx@",
    }
    for service_name, following_line in profiled_service_markers.items():
        marker = f"  {service_name}:\n{following_line}"
        replacement = f'  {service_name}:\n    profiles: ["{CANARY_PROFILE}"]\n{following_line}'
        updated = replace_exact(updated, marker, replacement, 1, f"{service_name} safety profile")

    configs_marker = "\nconfigs:\n  glm53_soak_nginx_conf:\n"
    indented_patch = "".join(f"      {line}" for line in patch.splitlines(keepends=True))
    patch_config = "\nconfigs:\n  glm53_w4afp8_patch:\n    content: |\n" + indented_patch + "  glm53_soak_nginx_conf:\n"
    updated = replace_exact(updated, configs_marker, patch_config, 1, "top-level patch config")
    return HEADER + updated


def main() -> int:
    class Arguments(argparse.Namespace):
        write: bool = False
        check: bool = False

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--write", action="store_true")
    _ = parser.add_argument("--check", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    if args.write and args.check:
        parser.error("--write and --check are mutually exclusive")

    expected = generate((ROOT / COMPOSE).read_text(), (ROOT / PATCH).read_text())
    actual = (ROOT / CANDIDATE).read_text() if (ROOT / CANDIDATE).exists() else ""
    if args.check:
        if actual == expected:
            return 0
        print(
            "".join(
                difflib.unified_diff(
                    actual.splitlines(keepends=True),
                    expected.splitlines(keepends=True),
                    fromfile=f"a/{CANDIDATE}",
                    tofile=f"b/{CANDIDATE}",
                )
            ),
            end="",
        )
        return 1

    if args.write:
        _ = (ROOT / CANDIDATE).write_text(expected)
        return 0

    print(expected, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
