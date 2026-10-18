"""Confinement tests for the long-tier Q8 + 16K-chunk canary generator (scripts/prepare_glm53_long_q8_canary.py).

The tests run the generator on the committed long fleet file with a stand-in digest, so they hold before the v9
image is published. Once V9_IMAGE_DIGEST is filled in and the target is committed, the freshness test also
checks the committed file.

Run: python3 -m scripts.test_glm53_long_q8_canary
"""

import difflib
import re
import unittest
from pathlib import Path

from scripts import prepare_glm53_long_q8_canary as gen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / gen.SOURCE).read_text()
DIGEST = gen.V9_IMAGE_DIGEST or "sha256:" + "0123456789abcdef" * 4
IMAGE = f"docker.io/nearaidev/sglang@{DIGEST}"
LABEL = DIGEST.split(":")[1][:12]
OUT = gen.generate(SOURCE, DIGEST)


def changed_lines():
    body = OUT.split("#\n", 1)[1]
    removed, added = [], []
    for line in difflib.unified_diff(SOURCE.splitlines(), body.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        (removed if line.startswith("-") else added).append(line[1:])
    return removed, added


def service_block(text):
    start = text.index(f"\n  {gen.SERVICE}:\n")
    end = re.search(r"\n  [^ \n]", text[start + 3:]).start() + start + 3
    return text[start:end]


class LongQ8CanaryTest(unittest.TestCase):
    def test_committed_target_is_fresh(self):
        target = ROOT / gen.TARGET
        if not gen.V9_IMAGE_DIGEST:
            self.assertFalse(target.exists(), "commit the target only together with the published v9 digest")
            return
        self.assertEqual(target.read_text(), OUT, "run scripts/prepare_glm53_long_q8_canary.py --write")

    def test_no_digest_is_a_generation_error(self):
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE, "")

    def test_only_r2b_image_two_flags_and_labels_change(self):
        removed, added = changed_lines()
        # image, chunk, prefill backend, and 4 label lines (log tags, otel config_variant, otel engine_image,
        # scrape config_variant, scrape engine_image) are replaced; one comment line is added.
        self.assertEqual(len(removed), 8, removed)
        self.assertEqual(len(added), 9, added)
        self.assertIn(f"    image: {IMAGE}", added)
        self.assertIn("        --chunked-prefill-size 16384", added)
        self.assertIn("        --dsa-prefill-backend flashmla_sparse_q8", added)
        argv = [l for l in removed + added if l.lstrip().startswith("--")]
        self.assertEqual(
            sorted(argv),
            sorted([
                "        --chunked-prefill-size 8192",
                "        --chunked-prefill-size 16384",
                "        --dsa-prefill-backend flashmla_kv",
                "        --dsa-prefill-backend flashmla_sparse_q8",
            ]),
        )

    def test_image_and_flags_are_only_on_r2b(self):
        self.assertEqual(OUT.count(IMAGE), 1)
        block = service_block(OUT)
        self.assertIn(IMAGE, block)
        body = OUT.split("#\n", 1)[1]  # past the generated header comment
        self.assertEqual(body.count("flashmla_sparse_q8"), 1)
        self.assertEqual(body.count("--chunked-prefill-size 16384"), SOURCE.count("--chunked-prefill-size 16384") + 1)
        self.assertIn("--dsa-decode-backend flashmla_kv", block)

    def test_environment_unchanged_and_split_kept(self):
        removed, added = changed_lines()
        for line in removed + added:
            self.assertNotRegex(line, r"^\s+- [A-Z_]+=", line)
        self.assertIn("      - SGLANG_DSA_INDEXER_QSPLIT=1\n", service_block(OUT))

    def test_split_missing_is_a_generation_error(self):
        block = service_block(SOURCE)
        broken = SOURCE.replace(block, block.replace("      - SGLANG_DSA_INDEXER_QSPLIT=1\n", ""))
        with self.assertRaises(gen.GenerationError):
            gen.generate(broken, DIGEST)

    def test_labels_carry_the_new_image_and_suffix(self):
        block = service_block(OUT)
        self.assertIn(f'nearai.otel.engine_image: "{LABEL}"', block)
        self.assertRegex(block, rf'nearai\.otel\.config_variant: "[^"]+{re.escape(gen.SUFFIX)}"')

    def test_dashboard_selectors_unchanged(self):
        for key in ("deployment", "host_machine", "server_address", "container_name", "instance", "service"):
            self.assertEqual(re.findall(rf"{key}: \"[^\"]*\"", SOURCE), re.findall(rf"{key}: \"[^\"]*\"", OUT), key)


if __name__ == "__main__":
    unittest.main()
