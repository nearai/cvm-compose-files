"""Confinement tests for the v8 zero-sync KDA canary file (scripts/prepare_glm53_v8_kda_canary.py).

Run: python3 -m scripts.test_glm53_v8_kda_canary
"""

import difflib
import re
import unittest
from pathlib import Path

from scripts import prepare_glm53_v8_kda_canary as gen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / gen.SOURCE).read_text()
TARGET = (ROOT / gen.TARGET).read_text()


def changed_lines():
    body = TARGET.split("#\n", 1)[1] if TARGET.startswith("# GLM-5.3 Flash v8") else TARGET
    removed, added = [], []
    for line in difflib.unified_diff(SOURCE.splitlines(), body.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        (removed if line.startswith("-") else added).append(line[1:])
    return removed, added


class V8KdaCanaryTest(unittest.TestCase):
    def test_committed_target_is_fresh(self):
        self.assertEqual(TARGET, gen.generate(SOURCE), "run scripts/prepare_glm53_v8_kda_canary.py --write")

    def test_digest_is_published(self):
        self.assertRegex(gen.V8_IMAGE_DIGEST, r"^sha256:[0-9a-f]{64}$", "fill in the published v8 digest")
        self.assertNotEqual(gen.V8_IMAGE_DIGEST, gen.V7_IMAGE_DIGEST)

    def test_only_r4_image_and_labels_change(self):
        removed, added = changed_lines()
        # 5 label lines change (log tags, otel engine_image, otel config_variant, scrape engine_image and config_variant)
        # and 2 lines are added (comment + image).
        self.assertEqual(len(removed), 5, removed)
        self.assertEqual(len(added), 7, added)
        for old, new in zip(removed, [a for a in added if "image: docker.io" not in a and not a.lstrip().startswith("#")]):
            self.assertEqual(old.replace(gen.V7_LABEL, gen.V8_LABEL).replace('-v7"', '-v7-kda"').replace("-v7,", "-v7-kda,"), new)
        self.assertIn(f"    image: {gen.V8_IMAGE}", added)

    def test_image_is_only_on_r4(self):
        self.assertEqual(TARGET.count(gen.V8_IMAGE), 1)
        start = TARGET.index(f"\n  {gen.SERVICE}:\n")
        end = re.search(r"\n  [^ \n]", TARGET[start + 3:]).start() + start + 3
        self.assertIn(gen.V8_IMAGE, TARGET[start:end])
        # r1-r3 and the ghost aggregator still get the v7 image from the anchor / their own line.
        self.assertEqual(TARGET.count(gen.V7_IMAGE_DIGEST), SOURCE.count(gen.V7_IMAGE_DIGEST))

    def test_argv_and_environment_unchanged(self):
        removed, added = changed_lines()
        for line in removed + added:
            self.assertNotRegex(line, r"^\s+--|^\s+- [A-Z_]+=", line)

    def test_missing_label_is_a_generation_error(self):
        broken = SOURCE.replace('nearai.otel.config_variant: "', 'nearai.otel.config_variant_x: "')
        with self.assertRaises(gen.GenerationError):
            gen.generate(broken)

    def test_dashboard_selectors_unchanged(self):
        for key in ("deployment", "host_machine", "server_address", "container_name", "instance", "service"):
            self.assertEqual(
                re.findall(rf"{key}: \"[^\"]*\"", SOURCE), re.findall(rf"{key}: \"[^\"]*\"", TARGET), key
            )


if __name__ == "__main__":
    unittest.main()
