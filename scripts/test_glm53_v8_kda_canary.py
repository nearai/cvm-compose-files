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
OLD_PROXY_IMAGE = "nearaidev/vllm-proxy-rs@sha256:d61357da39918a57126864a451eaf054f06a6989c03fe9a1666f7e6374ba6907"
NEW_PROXY_IMAGE = "nearaidev/vllm-proxy-rs@sha256:295f16f892d7e2f7c59e54354007762b5f0054d00467d42183f95d1596d8c916"


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

    def test_only_r4_image_labels_and_proxy_change(self):
        removed, added = changed_lines()
        # 5 label lines change (log tags, otel engine_image, otel config_variant, scrape engine_image and config_variant)
        # plus one proxy anchor line; 2 lines are added for r4 (comment + image).
        self.assertEqual(len(removed), 6, removed)
        self.assertEqual(len(added), 8, added)
        self.assertEqual(removed.count(f"  image: {OLD_PROXY_IMAGE}"), 1)
        self.assertEqual(added.count(f"  image: {NEW_PROXY_IMAGE}"), 1)
        old_labels = [line for line in removed if line != f"  image: {OLD_PROXY_IMAGE}"]
        new_labels = [line for line in added if line != f"  image: {NEW_PROXY_IMAGE}" and "image: docker.io" not in line and not line.lstrip().startswith("#")]
        for old, new in zip(old_labels, new_labels):
            self.assertEqual(old.replace(gen.V7_LABEL, gen.V8_LABEL).replace('-v7"', '-v7-kda"').replace("-v7,", "-v7-kda,"), new)
        self.assertIn(f"    image: {gen.V8_IMAGE}", added)

    def test_proxy_image_is_new_only_in_canary(self):
        self.assertEqual(SOURCE.count(f"  image: {OLD_PROXY_IMAGE}"), 1)
        self.assertNotIn(NEW_PROXY_IMAGE, SOURCE)
        self.assertEqual(TARGET.count(f"  image: {NEW_PROXY_IMAGE}"), 1)
        self.assertNotIn(OLD_PROXY_IMAGE, TARGET)

    def test_source_with_new_proxy_is_a_generation_error(self):
        broken = SOURCE.replace(OLD_PROXY_IMAGE, NEW_PROXY_IMAGE)
        with self.assertRaises(gen.GenerationError):
            gen.generate(broken)

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
