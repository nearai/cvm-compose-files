"""Confinement tests for the PD-Multiplexing + EAGLE canary file (scripts/prepare_glm53_pdmux_canary.py) and its image recipe.

Run: python3 -m scripts.test_glm53_pdmux_canary
"""

import difflib
import json
import re
import shlex
import unittest
from pathlib import Path

from scripts import prepare_glm53_pdmux_canary as gen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / gen.SOURCE).read_text()
TARGET = (ROOT / gen.TARGET).read_text()
RECIPE = ROOT / "docker/sglang-glm53-hicache-w4afp8-pdmuxe"
PDMUX_FLAGS = ["--enable-pdmux", "--sm-group-num", "3", "--pdmux-config-path", "/etc/glm53/pdmux-ov96-l4.yaml"]


def block(text: str, start_marker: str, end_pattern: str) -> str:
    start = text.index(start_marker)
    return text[start : start + len(start_marker) + re.search(end_pattern, text[start + len(start_marker) :]).start()]


def command_tokens(text: str, start_marker: str, end_pattern: str) -> list[str]:
    section = block(text, start_marker, end_pattern)
    folded = re.search(r"\n\s+command: >\n((?:\s{6,}\S.*\n)+)", section).group(1)
    return shlex.split(folded)


def changed_lines():
    body = TARGET.split("#\n", 1)[1] if TARGET.startswith("# GLM-5.3 Flash PD-Multiplexing") else TARGET
    removed, added = [], []
    for line in difflib.unified_diff(SOURCE.splitlines(), body.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        (removed if line.startswith("-") else added).append(line[1:])
    return removed, added


class PdmuxCanaryTest(unittest.TestCase):
    def test_committed_target_is_fresh(self):
        self.assertEqual(TARGET, gen.generate(SOURCE), "run scripts/prepare_glm53_pdmux_canary.py --write")

    def test_digest_is_published(self):
        self.assertRegex(gen.PDMUX_IMAGE_DIGEST, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(gen.PDMUX_IMAGE_DIGEST, gen.V7_IMAGE_DIGEST)
        if gen.is_placeholder():
            self.skipTest("PDMUX_IMAGE_DIGEST is still the placeholder: publish the image and fill it in before deploying")

    def test_only_r4_image_command_and_labels_change(self):
        removed, added = changed_lines()
        # 5 label lines change (log tags, otel engine_image, otel config_variant, scrape engine_image and config_variant).
        self.assertEqual(len(removed), 5, removed)
        relabelled = [a for a in added if (gen.PDMUX_LABEL in a or gen.SUFFIX in a) and not a.startswith("    image:")]
        self.assertEqual(len(relabelled), 5, relabelled)
        for old, new in zip(removed, relabelled):
            self.assertEqual(old.replace(gen.V7_LABEL, gen.PDMUX_LABEL).replace('-v7"', f'-v7{gen.SUFFIX}"').replace("-v7,", f"-v7{gen.SUFFIX},"), new)
        rest = [a for a in added if a not in relabelled]
        self.assertIn(f"    image: {gen.PDMUX_IMAGE}", rest)
        self.assertIn("    command: >", rest)
        # everything else added is a comment, the image, or the command body
        self.assertTrue(all(a.lstrip().startswith(("#", "image:", "command:", "sglang", "--")) or a.startswith("        ") for a in rest), rest)

    def test_image_is_only_on_r4(self):
        self.assertEqual(TARGET.count(gen.PDMUX_IMAGE), 1)
        section = block(TARGET, f"\n  {gen.SERVICE}:\n", r"\n  [^ \n]")
        self.assertIn(gen.PDMUX_IMAGE, section)
        # r1-r3 and the ghost aggregator still get the v7 image from the anchor / their own line.
        self.assertEqual(TARGET.count(gen.V7_IMAGE_DIGEST), SOURCE.count(gen.V7_IMAGE_DIGEST))

    def test_argv_is_the_fleet_argv_plus_the_three_pdmux_flags(self):
        fleet = command_tokens(SOURCE, f"\n{gen.ANCHOR}: &sg-glm53-flash-candidate\n", r"\n\S")
        canary = command_tokens(TARGET, f"\n  {gen.SERVICE}:\n", r"\n  [^ \n]")
        self.assertEqual(canary, fleet + PDMUX_FLAGS)
        # the pinned v7 fleet shape the controls run, spelled out so a fleet edit cannot silently change the canary
        for flag, value in {
            "--speculative-algorithm": "EAGLE", "--speculative-num-steps": "4", "--speculative-eagle-topk": "1",
            "--speculative-num-draft-tokens": "5", "--chunked-prefill-size": "8192", "--prefill-decode-interval": "2",
            "--max-queued-requests": "32", "--mem-fraction-static": "0.86", "--max-running-requests": "64",
            "--max-mamba-cache-size": "380",
        }.items():
            self.assertEqual(canary[canary.index(flag) + 1], value, flag)
        self.assertIn("--disable-overlap-schedule", canary)
        self.assertNotIn("--enable-hierarchical-cache", canary)
        self.assertNotIn("--enable-mixed-chunk", canary)

    def test_argv_equals_the_recipe_argv_that_test_cpu_validates(self):
        recipe = json.loads((RECIPE / "pdmux-canary-argv.json").read_text())
        canary = command_tokens(TARGET, f"\n  {gen.SERVICE}:\n", r"\n  [^ \n]")
        self.assertEqual(canary, ["sglang", "serve"] + recipe)

    def test_environment_gpus_and_other_services_unchanged(self):
        removed, added = changed_lines()
        for line in removed + added:
            self.assertNotRegex(line, r"^\s+- [A-Z_]+=|device_ids|container_name", line)
        self.assertEqual(TARGET.count("--enable-pdmux"), 1)

    def test_missing_label_is_a_generation_error(self):
        broken = SOURCE.replace('nearai.otel.config_variant: "', 'nearai.otel.config_variant_x: "')
        with self.assertRaises(gen.GenerationError):
            gen.generate(broken)

    def test_dashboard_selectors_unchanged(self):
        for key in ("deployment", "host_machine", "server_address", "container_name", "instance", "service"):
            self.assertEqual(
                re.findall(rf"{key}: \"[^\"]*\"", SOURCE), re.findall(rf"{key}: \"[^\"]*\"", TARGET), key
            )

    def test_baked_config_path_matches_the_recipe(self):
        dockerfile = (RECIPE / "Dockerfile").read_text()
        self.assertIn("/etc/glm53/pdmux-ov96-l4.yaml", dockerfile)
        self.assertEqual(
            (RECIPE / "pdmux-ov96-l4.yaml").read_text(),
            "sm_group_num: 3\nmanual_divisions: [[96, 36, 1]]\nsplit_forward_token_budget: 65536\n"
            "max_split_forward_layers: 4\noverlap_decode_full_sm: true\n",
        )


if __name__ == "__main__":
    unittest.main()
