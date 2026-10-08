#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_glm53_v7_canary (needs ruby for the validator cases)
"""The two generated v7 bundle canary files: confined to the canary replica, exact flags/environment/labels, validator contract.

Imported by test_glm53_w4afp8_tp2x4.py and test_glm53_w4afp8_long_context.py so the existing CI steps run it.
"""

import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_glm53_v7_canary as generator

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = Path("scripts/validate_glm53_prod_config.rb")
DCGM_VALIDATOR = Path("scripts/validate_glm53_dcgm_metrics.rb")
OVERLAP = "--disable-overlap-schedule"
LABEL_KEYS = ("precision", "engine_image", "config_variant")
V6 = generator.V6_IMAGE


def block(text: str, name: str) -> str:
    start, end = generator.service_span(text, name)
    return text[start:end]


def job(text: str, name: str) -> str:
    start, end = generator.job_span(text, name)
    return text[start:end]


def without_canary(text: str, name: str) -> str:
    """The file minus the canary service block, its scrape job and the generated header."""
    text = text.replace(block(text, name), "")
    text = text.replace(job(text, name), "")
    return text


def folded_argv(service_block: str, anchor_text: str | None = None) -> list[str]:
    """argv of a service block's own `command: >` (compose joins the folded lines and splits like a shell)."""
    body = service_block.split("    command: >\n", 1)[1].splitlines()
    lines = []
    for line in body:
        if line.strip() and len(line) - len(line.lstrip(" ")) < 6:
            break
        lines.append(line.strip())
    return shlex.split(" ".join(line for line in lines if line))


def environment(service_block: str) -> list[str]:
    env = service_block.split("    environment:\n", 1)[1].split("\n    depends_on:", 1)[0]
    return [line.strip()[2:] for line in env.splitlines() if line.strip().startswith("- ")]


def labels_of(job_text: str) -> dict[str, str]:
    return dict(re.findall(r'^ {22}([a-z_]+): "([^"]*)"$', job_text, re.MULTILINE))


def all_jobs(text: str) -> dict[str, str]:
    return {m.group(1): m.group(0) for m in re.finditer(r"- job_name: (\S+)\n(?:(?! {14}- job_name:).*\n)*", text)}


class CanaryFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = {}
        for kind, spec in generator.KINDS.items():
            self.cases[kind] = (spec, (ROOT / spec["source"]).read_text(), (ROOT / spec["target"]).read_text())

    def each(self):
        return [(kind, spec, source, target) for kind, (spec, source, target) in self.cases.items()]

    def test_committed_files_match_the_generator_and_check_passes(self) -> None:
        for kind, spec, source, target in self.each():
            self.assertEqual(generator.generate(kind, source), target)
        result = subprocess.run(["python3", str(ROOT / "scripts/prepare_glm53_v7_canary.py"), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:])

    def test_generator_refuses_its_own_output_and_a_drifted_source(self) -> None:
        for kind, spec, source, target in self.each():
            with self.assertRaises(generator.GenerationError):
                generator.generate(kind, target)
            with self.assertRaises(generator.GenerationError):
                generator.generate(kind, source.replace("--kv-cache-dtype bfloat16", "--kv-cache-dtype fp8_e5m2"))

    def test_the_source_files_are_untouched_by_the_bundle(self) -> None:
        for kind, spec, source, target in self.each():
            for needle in ("fa730e6e62b2", "NEAR_SELF_PROFILE", "SGLANG_PREPROCESS", "SGLANG_TOOL_SCHEMA", "fp8_e4m3", "flashmla_kv", OVERLAP, "GLM53_V"):
                self.assertNotIn(needle, "\n".join(l for l in source.splitlines() if not l.lstrip().startswith("#")), needle)

    def test_the_diff_is_confined_to_the_canary_service_its_scrape_job_and_the_header(self) -> None:
        for kind, spec, source, target in self.each():
            name = spec["service"]
            header = generator.header(kind, spec)
            self.assertTrue(target.startswith(header))
            self.assertEqual(without_canary(target[len(header):], name), without_canary(source, name))
            # And inside those two blocks only the intended lines differ.
            changed = [l for l in difflib.unified_diff(source.splitlines(), target[len(header):].splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
            bad = [l for l in changed if not any(token in l for token in ("image:", "command: >", "sglang serve", "--", "- SGLANG_", "- NEAR_", "# ", "precision", "engine_image", "config_variant", "datadoghq"))]
            self.assertEqual(bad, [])

    def test_canary_argv_is_the_source_argv_with_exactly_the_requested_edits_and_each_flag_once(self) -> None:
        for kind, spec, source, target in self.each():
            name = spec["service"]
            if kind == "base":  # r4 inherits the candidate anchor in the source
                anchor = source.split("x-sg-glm53-flash-candidate: &sg-glm53-flash-candidate\n", 1)[1].split("\nx-", 1)[0]
                original = folded_argv("    command: >\n" + anchor.split("  command: >\n", 1)[1])
            else:
                original = folded_argv(block(source, name))
            argv = folded_argv(block(target, name))
            expected = list(original)
            for flag, (old, new) in spec["flags"].items():
                self.assertEqual(expected[expected.index(flag) + 1], old, flag)
                expected[expected.index(flag) + 1] = new
            self.assertEqual(argv, expected + [OVERLAP])
            for flag in (*spec["flags"], OVERLAP, "--kv-cache-dtype"):
                self.assertEqual(argv.count(flag), 1, flag)
            self.assertNotIn("", argv)
            value = lambda flag: argv[argv.index(flag) + 1]
            self.assertEqual((value("--kv-cache-dtype"), value("--dsa-prefill-backend"), value("--dsa-decode-backend")), ("fp8_e4m3", "flashmla_kv", "flashmla_kv"))
            self.assertEqual(value("--max-running-requests"), value("--cuda-graph-max-bs-decode"))
            self.assertGreaterEqual(int(value("--max-mamba-cache-size")), 5 * int(value("--max-running-requests")))

    def test_exact_values_requested(self) -> None:
        base, long = (folded_argv(block(self.cases[k][2], self.cases[k][0]["service"])) for k in ("base", "long"))
        get = lambda argv, flag: argv[argv.index(flag) + 1]
        self.assertEqual([get(base, f) for f in ("--max-running-requests", "--cuda-graph-max-bs-decode", "--max-mamba-cache-size")], ["64", "64", "380"])
        self.assertEqual([get(long, f) for f in ("--max-running-requests", "--max-queued-requests", "--cuda-graph-max-bs-decode", "--max-mamba-cache-size")], ["16", "4", "16", "330"])
        self.assertEqual(self.cases["base"][0]["service"], "model-sg-glm53-w4afp8-tp2-r4")
        self.assertEqual(self.cases["long"][0]["service"], "model-sg-glm53-w4afp8-tp2-r2a")

    def test_canary_environment_image_and_labels(self) -> None:
        wanted = [f"{n}={v}" for n, v in generator.V7_ENVIRONMENT]
        self.assertEqual([n for n, _ in generator.V7_ENVIRONMENT][-3:], ["NEAR_SELF_PROFILE", "NEAR_SELF_PROFILE_AFTER_S", "NEAR_SELF_PROFILE_STEPS"])
        for kind, spec, source, target in self.each():
            name = spec["service"]
            canary_env = environment(block(target, name))
            original_env = environment(block(source, name))
            self.assertEqual(canary_env, original_env + wanted)
            self.assertEqual(sum(1 for e in canary_env if e.startswith("NEAR_SELF_PROFILE=")), 1)
            for other in re.findall(r"^  (model-sg-glm53-[a-z0-9-]+):$", target, re.MULTILINE):
                if other != name:
                    self.assertNotIn("NEAR_SELF_PROFILE", block(target, other))
                    self.assertNotIn(generator.V7_IMAGE_DIGEST, block(target, other))
            canary = block(target, name)
            self.assertIn(f"    image: {generator.V7_IMAGE}\n", canary)
            self.assertEqual(target.count(generator.V7_IMAGE), 1 if kind == "long" else 1)
            self.assertIn(f'"precision:{generator.V7_PRECISION}"', canary)
            self.assertIn(f'"engine_image:{generator.V7_IMAGE_LABEL}"', canary)
            self.assertIn('nearai.otel.deployment: "', canary)
            self.assertEqual(canary.count(spec["suffix"] + '"'), 2)

    def test_otel_scrape_jobs_equal_the_source_except_three_labels_on_the_canary_job(self) -> None:
        for kind, spec, source, target in self.each():
            name = spec["service"]
            src, new = all_jobs(source), all_jobs(target)
            self.assertEqual(list(new), list(src))
            for key in src:
                if key != f"sglang-{name}":
                    self.assertEqual(new[key], src[key], key)
                    continue
                a, b = labels_of(src[key]), labels_of(new[key])
                self.assertEqual({k for k in a if a[k] != b[k]}, set(LABEL_KEYS))
                self.assertEqual(b["precision"], generator.V7_PRECISION)
                self.assertEqual(b["engine_image"], generator.V7_IMAGE_LABEL)
                self.assertEqual(b["config_variant"], a["config_variant"] + spec["suffix"])
                for kept in ("deployment", "host_machine", "host", "model", "service", "instance", "container_name"):
                    self.assertEqual(a[kept], b[kept])

    def test_dashboard_selectors_are_unchanged_for_every_service(self) -> None:
        # GLM-5.3 Flash production (glm53-flash-prod) selects on model, deployment, service, host_machine, server_address.
        for kind, spec, source, target in self.each():
            for pattern in (r'deployment: "[^"]*"', r'nearai\.otel\.deployment: "[^"]*"', r"host_machine: [^\n]*", r'service: "[^"]*"', r'model: "[^"]*"', r'nearai\.otel\.service: "[^"]*"'):
                self.assertEqual(re.findall(pattern, target), re.findall(pattern, source), pattern)

    def test_no_variables_no_placeholders_and_a_real_digest(self) -> None:
        self.assertRegex(generator.V7_IMAGE_DIGEST, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(generator.V7_IMAGE_LABEL, generator.V7_IMAGE_DIGEST[7:19])
        for kind, spec, source, target in self.each():
            self.assertNotRegex(target, r"GLM53_V\d|REPLACE_WITH|<tbd>")

    @unittest.skipUnless(shutil.which("docker"), "docker CLI not available")
    def test_docker_compose_render_confined_to_the_canary(self) -> None:
        if subprocess.run(["docker", "compose", "version"], capture_output=True, check=False).returncode != 0:
            self.skipTest("docker compose plugin not available")

        def render(path: Path) -> dict:
            env = {k: v for k, v in os.environ.items()}
            env.update({"HOST_IP": "127.0.0.1", "CVM_NAME": "compose-manager", "CVM_HOST": "gpu03", "ENV": "prod", "HUGGING_FACE_HUB_TOKEN": "x"})
            text = path.read_text()
            for secret in set(re.findall(r"\$\{([A-Z0-9_]+):?\?", text)):
                env.setdefault(secret, "ci-dummy")
            result = subprocess.run(["docker", "compose", "-p", "v7proof", "-f", str(path), "config", "--format", "json"], capture_output=True, text=True, env=env, check=False)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            return json.loads(result.stdout)

        for kind, spec, source, target in self.each():
            old, new = render(ROOT / spec["source"]), render(ROOT / spec["target"])
            self.assertEqual(list(old["services"]), list(new["services"]))
            for name in old["services"]:
                if name != spec["service"]:
                    self.assertEqual(old["services"][name], new["services"][name], name)
            canary = new["services"][spec["service"]]
            self.assertEqual(canary["image"], generator.V7_IMAGE)
            self.assertEqual(canary["command"].count(OVERLAP), 1)
            self.assertNotIn("", canary["command"])
            self.assertEqual({k: v for k, v in old.items() if k != "services" and k != "configs"}, {k: v for k, v in new.items() if k != "services" and k != "configs"})


class V7ReleaseGateTest(unittest.TestCase):
    def test_digest_is_a_real_sha256_not_a_placeholder(self) -> None:
        self.assertRegex(generator.V7_IMAGE_DIGEST, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(generator.V7_IMAGE_DIGEST, V6.split("@")[1])


class ValidatorContractTest(unittest.TestCase):
    """Each mutation of a committed canary file must fail the production validator with its reason."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        names = [VALIDATOR, DCGM_VALIDATOR, Path("prod/GLM-5.3-Flash-SGL-TP4.yaml"), Path("prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml"),
                 Path("prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml"), Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml"),
                 Path("docker/sglang-glm53-hicache/RELEASED_IMAGE")]
        names += [spec[k] for spec in generator.KINDS.values() for k in ("source", "target")]
        for name in names:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.root / name)
        self.files = {kind: self.root / spec["target"] for kind, spec in generator.KINDS.items()}
        self.valid = {kind: path.read_text() for kind, path in self.files.items()}

    def run_ruby(self, script: Path = VALIDATOR) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["ruby", str(self.root / script)], capture_output=True, text=True, check=False)

    def assert_fails(self, kind: str, mutated: str, message: str, script: Path = VALIDATOR) -> None:
        self.assertNotEqual(mutated, self.valid[kind])
        self.files[kind].write_text(mutated)
        try:
            result = self.run_ruby(script)
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 1, output)
            self.assertIn(message, output)
            self.assertNotRegex(output, r"\.rb:\d+:in")
        finally:
            self.files[kind].write_text(self.valid[kind])

    def test_committed_files_pass(self) -> None:
        for script in (VALIDATOR, DCGM_VALIDATOR):
            result = self.run_ruby(script)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("V7Canary.yaml", result.stdout)

    def test_rejects_drift_outside_the_canary_and_unsafe_canary_values(self) -> None:
        for kind, spec in generator.KINDS.items():
            valid, name = self.valid[kind], spec["service"]
            other = "model-sg-glm53-w4afp8-tp2-r1" if kind == "base" else "model-sg-glm53-w4afp8-tp2-r2b"
            other_block = block(valid, other)
            canary_block = block(valid, name)
            job_text = job(valid, name)
            cases = [
                # the v7 image or a flag leaking onto a neighbouring replica, or any other service drifting
                (valid.replace(other_block, other_block.replace("    depends_on:", "    restart: \"no\"\n    depends_on:", 1)), f"{other} must be identical to the source file"),
                (valid.replace("    container_name: proxy-glm53\n", "    container_name: proxy-glm53\n    mem_limit: 1g\n", 1), "proxy-glm53 must be identical to the source file"),
                # canary flags
                (valid.replace(f"{'      ' if kind == 'base' else '        '}{OVERLAP}\n", "", 1), "argv must be the source argv"),
                (valid.replace(f"{'      ' if kind == 'base' else '        '}{OVERLAP}\n", f"{'      ' if kind == 'base' else '        '}{OVERLAP}\n{'      ' if kind == 'base' else '        '}{OVERLAP}\n", 1), "argv must be the source argv"),
                (valid.replace(canary_block, canary_block.replace("--dsa-prefill-backend flashmla_kv", "--dsa-prefill-backend tilelang")), "argv must be the source argv"),
                (valid.replace(canary_block, canary_block.replace("fp8_e4m3", "bfloat16")), "argv must be the source argv"),
                (valid.replace(canary_block, canary_block.replace("--dsa-decode-backend flashmla_kv", "--dsa-decode-backend tilelang")), "argv must be the source argv"),
                (valid.replace(canary_block, canary_block.replace("- NEAR_SELF_PROFILE=1", "- NEAR_SELF_PROFILE=2")), "environment must be the source environment plus"),
                (valid.replace(canary_block, canary_block.replace("      - SGLANG_TOOL_SCHEMA_MAX_NODES=25000\n", "")), "environment must be the source environment plus"),
                (valid.replace(f"    image: {generator.V7_IMAGE}\n", f"    image: {V6}\n", 1), "image must be"),
                (valid.replace(generator.V7_IMAGE, "docker.io/nearaidev/sglang@sha256:" + "0" * 63, 1), "image must be"),
                (valid + "\n# GLM53_V7_R4_IMAGE\n" + "x: ${GLM53_V7_R4_IMAGE:-y}\n", "must not contain GLM53_V7_"),
                # telemetry: the dashboard labels must stay identical
                (valid.replace(canary_block, canary_block.replace('nearai.otel.deployment: "', 'nearai.otel.deployment: "v7-')), "may change only the telemetry labels"),
                (valid.replace(canary_block, canary_block.replace(f'"engine_image:{generator.V7_IMAGE_LABEL}"', f'"engine_image:{generator.V6_IMAGE_LABEL}"')), "log tags must be the source tags"),
                (valid.replace(canary_block, canary_block.replace(spec["suffix"] + '"', '"', 1)), "log tags must be the source tags"),
                (valid.replace(canary_block, canary_block.replace(f'nearai.otel.config_variant: "', 'nearai.otel.config_variant: "x', 1)), "nearai.otel.config_variant must be"),
                (valid.replace(job_text, job_text.replace('deployment: "', 'deployment: "v7-', 1)), "scrape job sglang-" + name + " labels must equal the source's"),
                (valid.replace(job_text, job_text.replace(f'engine_image: "{generator.V7_IMAGE_LABEL}"', f'engine_image: "{generator.V6_IMAGE_LABEL}"')), "scrape job sglang-" + name + " labels must equal the source's"),
                (valid.replace(job_text, job_text.replace('host_machine: "${CVM_HOST}"', 'host_machine: "x"', 1)), "scrape job sglang-" + name + " labels must equal the source's"),
            ]
            for mutated, message in cases:
                if mutated is None:
                    continue
                with self.subTest(kind=kind, expect=message, mutation=hash(mutated) % 10000):
                    self.assert_fails(kind, mutated, message)

    def test_a_missing_cap_flag_is_reported_not_a_crash(self) -> None:
        # Dropping either flag the mamba-slot rule reads must produce a validation error, not a Ruby NoMethodError
        # (assert_fails also rejects any "<file>.rb:<line>:in" stack frame in the output).
        for kind, spec in generator.KINDS.items():
            valid = self.valid[kind]
            canary_block = block(valid, spec["service"])
            for flag in ("--max-mamba-cache-size", "--max-running-requests"):
                lines = [line for line in canary_block.splitlines(keepends=True) if line.strip().startswith(flag + " ")]
                self.assertEqual(len(lines), 1, f"{kind} {flag}")
                with self.subTest(kind=kind, flag=flag):
                    self.assert_fails(kind, valid.replace(canary_block, canary_block.replace(lines[0], "", 1)),
                                      "argv must carry --max-mamba-cache-size and --max-running-requests")

    def test_dcgm_validator_rejects_a_dcgm_exporter_change_in_each_canary_file(self) -> None:
        dcgm_image = "nvcr.io/nvidia/k8s/dcgm-exporter@sha256:613ab03c11d442fd960ff515f547e9921537454a712d08160bc8f677f89f1c35"
        for kind, spec in generator.KINDS.items():
            valid = self.valid[kind]
            self.assertEqual(valid.count(dcgm_image), 1, kind)
            with self.subTest(kind=kind):
                self.assert_fails(kind, valid.replace(dcgm_image, dcgm_image[:-1] + "0", 1),
                                  f"GLM-5.3 DCGM telemetry contract failed ({spec['target']})", script=DCGM_VALIDATOR)

    def test_rejects_an_unrelated_scrape_job_or_dcgm_drift(self) -> None:
        for kind, spec in generator.KINDS.items():
            valid = self.valid[kind]
            jobs = all_jobs(valid)
            dcgm = next(text for key, text in jobs.items() if key.startswith("dcgm"))
            self.assert_fails(kind, valid.replace(dcgm, dcgm.replace("service:", "service_x:", 1)) if "service:" in dcgm else valid.replace(dcgm, dcgm + "# x\n").replace("- job_name: dcgm", "- job_name: dcgmx", 1), "scrape job")


class RunbookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runbook = (ROOT / "docs/glm53-v7-canary.md").read_text()

    def test_it_names_files_services_hosts_and_the_scoped_calls(self) -> None:
        for spec in generator.KINDS.values():
            for text in (spec["target"].name, spec["host"], spec["service"]):
                self.assertIn(text, self.runbook)
            self.assertIn(f'"services": ["{spec["service"]}"]', self.runbook)
        self.assertIn('"services": ["otelcol-contrib"]', self.runbook)
        self.assertNotIn("services: []", self.runbook)
        self.assertNotIn('"services": []', self.runbook)

    def test_it_carries_the_final_config_and_no_stale_mechanics(self) -> None:
        for text in (generator.V7_IMAGE_DIGEST, "glm53-hicache-w4afp8-v7", "37693106399", "29a7db9", "-v7-mr16q4", "fa730e6e62b2", "dry_run",
                     "--max-running-requests 64", "380", "16/4", "deployment", "glm53-flash-prod", "Evidence", "exp 29", "exp 26", "exp 27", "long-tp2-wedge-rca",
                     "SGLANG_PREPROCESS_WORKERS=4", "NEAR_SELF_PROFILE_AFTER_S=900", "NEAR_PROFILE", "action log", "cannot be recalled", "Rollback"):
            self.assertIn(text, self.runbook)
        for stale in ("KMS", "ENV_PREFIX", "EXTRA_ARGS", "GLM53_V7", "GLM53_V8", "v8", "mr16q6", "16/6 caps", "REPLACE_WITH", "env map to set", "does not exist yet"):
            self.assertNotIn(stale, self.runbook.replace("(16/6 vs 12/4)", ""), stale)

    def test_the_user_rules_and_preregistered_abort_criteria_are_in_the_runbook(self) -> None:
        lowered = self.runbook.lower()
        for text in ("one replica at a time", "never more than 2 base replicas", "ignores `dry_run`", "more than 20% worse", "xid"):
            self.assertIn(text.lower(), lowered)
