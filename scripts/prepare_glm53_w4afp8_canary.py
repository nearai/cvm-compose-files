#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run scripts/prepare_glm53_w4afp8_canary.py --write

import argparse
import difflib
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.glm53_w4afp8_text import HEADER, HEADER_REPLACEMENTS

COMPOSE = Path("prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml")
CANDIDATE = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-Canary.yaml")
CHECKPOINT = "graphistry/GLM-5.3-Flash-W4AFP8"
CHECKPOINT_REVISION = "99f1fa70408c52b007d4fd69e02e5a522422e755"
PROMOTED_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:8bce6a7cc872a80faded3bd1ef0a64873a1d7abae34c94e5358775ca21f133cc"
CONTROL_SERVICE = "model-sg-glm53-fp8-tp4-r1"
CANDIDATE_SERVICE = "model-sg-glm53-w4afp8-tp4-r1"
CANARY_PROFILE = "w4afp8-long-context"
CONTROL_VARIANT = "fc91d24-long-context-admission-reserve-disabled-hicache-disabled-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
CANDIDATE_VARIANT = "fc91d24-long-context-w4afp8-c4096-admission-reserve-disabled-pool-clamp-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"


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
    replacements = {model_option: f"--model-path {model_path}"}
    forbidden = {
        "--revision 84c6a6aa9497188e15a635ba793b0f95a79b1033",
        "--moe-runner-backend deep_gemm",
    }
    transformed = [replacements.get(argument, argument) for argument in arguments if argument not in forbidden]
    chunk_index = transformed.index("--chunked-prefill-size 4096")
    transformed.insert(chunk_index + 1, "--max-prefill-tokens 32768")

    lines = ["    command: >"]
    lines.extend(f"        {argument}" for argument in transformed)
    return "\n".join(lines) + "\n"


def generate(canonical: str) -> str:
    snapshot_dir = f"models--{CHECKPOINT.replace('/', '--')}/"
    if CANDIDATE_SERVICE in canonical or snapshot_dir in canonical:
        raise GenerationError("canonical compose already contains the W4AFP8 canary")

    # The long-context source's model-downloader already pre-stages the W4AFP8 snapshot
    # (as every dedicated GLM-5.3 Flash TP4 file does); the canary reuses that download.
    download = (
        f"        echo \"Downloading {CHECKPOINT}...\"\n"
        "        uvx --from 'huggingface_hub[hf_xet]' hf download "
        f"{CHECKPOINT} --revision {CHECKPOINT_REVISION}\n"
        "        echo \"Download complete.\"\n"
    )
    updated = replace_exact(canonical, download, download, 1, "model downloader W4AFP8 pre-stage")
    for old, new in HEADER_REPLACEMENTS:
        updated = replace_exact(updated, old, new, 1, "long-context header")
    updated = replace_exact(updated, CONTROL_SERVICE, CANDIDATE_SERVICE, 8, "r1 service identity")
    perception_loop = (
        "        for replica in (1, 2):\n"
        "            base = f\"http://model-sg-glm53-fp8-tp4-r{replica}:8000\"\n"
    )
    candidate_perception_loop = (
        f"        for replica, service in ((1, \"{CANDIDATE_SERVICE}\"), "
        "(2, \"model-sg-glm53-fp8-tp4-r2\")):\n"
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
        "\n  model-sg-glm53-fp8-tp4-r2:\n",
        "candidate service",
    )
    service_header = f"  {CANDIDATE_SERVICE}:\n"
    container = f"    container_name: {CANDIDATE_SERVICE}\n"
    service = replace_exact(
        service,
        service_header,
        service_header + f'    profiles: ["{CANARY_PROFILE}"]\n    restart: unless-stopped\n',
        1,
        "candidate safety profile",
    )
    candidate_override = f"    image: {PROMOTED_IMAGE}\n" + command
    service = replace_exact(service, container, container + candidate_override, 1, "candidate override")
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
        "model-sg-glm53-fp8-tp4-r2": "    <<: *sg-glm53-flash-common",
        "dcgm-glm53": "    <<: *dcgm-common",
        "otelcol-contrib": "    image: otel/opentelemetry-collector-contrib@",
        "nginx": "    image: nginx@",
    }
    for service_name, following_line in profiled_service_markers.items():
        marker = f"  {service_name}:\n{following_line}"
        replacement = f'  {service_name}:\n    profiles: ["{CANARY_PROFILE}"]\n{following_line}'
        updated = replace_exact(updated, marker, replacement, 1, f"{service_name} safety profile")

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

    expected = generate((ROOT / COMPOSE).read_text())
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
