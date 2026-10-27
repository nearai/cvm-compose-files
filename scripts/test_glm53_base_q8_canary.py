"""Confinement tests for the base-tier v12 Q8-prefill canary file (scripts/prepare_glm53_base_q8_canary.py).

The canary is gpu04 r4 only: image v12, its own command (anchor argv with the prefill backend swapped), and its
telemetry labels, all equal to r4 of the v12 fleet file. Run: python3 -m scripts.test_glm53_base_q8_canary
"""

import difflib
import re
import unittest
from pathlib import Path

from scripts import prepare_glm53_base_q8_canary as gen
from scripts import prepare_glm53_base_v12_q8_fleet as fleet

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / gen.SOURCE).read_text()
TARGET = (ROOT / gen.TARGET).read_text()
BODY = TARGET.split("#\n", 1)[1]  # past the generated header comment
FLEET = (ROOT / fleet.TARGET).read_text()
OLD_BACKEND = "--dsa-prefill-backend flashmla_kv"
NEW_BACKEND = "--dsa-prefill-backend flashmla_sparse_q8"


def service_block(text, name):
    start = text.index(f"\n  {name}:\n")
    end = re.search(r"\n  [^ \n]", text[start + 3:]).start() + start + 3
    return text[start:end]


def anchor_block(text):
    start = text.index(f"\n{gen.ANCHOR}: &sg-glm53-flash-candidate\n")
    end = re.search(r"\n[^ \n#]", text[start + 3:]).start() + start + 3
    return text[start:end]


def scrape_job(text, name):
    start = text.index(f"\n              - job_name: sglang-{name}\n")
    end = re.search(r"\n              - job_name:", text[start + 3:]).start() + start + 3
    return text[start:end]


def command_body(text):
    """Lines of the `command: >` block (header at 2 or 4 spaces): every line indented 6 spaces after the header."""
    match = re.search(r"\n {2,4}command: >\n", text)
    if match is None:
        return []
    body = []
    for line in text[match.end():].splitlines():
        if not line.startswith("      "):
            break
        body.append(line)
    return body


def r4_argv_tokens(text):
    return " ".join(command_body(service_block(text, gen.SERVICE))).split()


def changed_lines():
    removed, added = [], []
    for line in difflib.unified_diff(SOURCE.splitlines(), BODY.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        (removed if line.startswith("-") else added).append(line[1:])
    return removed, added


class BaseQ8CanaryTest(unittest.TestCase):
    def test_committed_target_is_fresh(self):
        self.assertEqual(TARGET, gen.generate(SOURCE), "run scripts/prepare_glm53_base_q8_canary.py --write")

    def test_digest_is_the_long_tier_v12(self):
        self.assertRegex(gen.V12_IMAGE_DIGEST, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(gen.V12_IMAGE_DIGEST, gen.V7_IMAGE_DIGEST)
        self.assertEqual(gen.V12_LABEL, "da8d4e4b5ec6")
        long_file = (ROOT / "prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml").read_text()
        self.assertIn(f"image: {gen.V12_IMAGE}", long_file)
        self.assertNotIn("c15a33672ade", TARGET)  # the superseded v10 canary image is gone

    def test_only_r4_image_command_and_labels_differ(self):
        # Undo exactly the generator's additions (comment, image, command block) and its label edits; the result
        # must equal the source byte for byte. Everything else in the file is therefore identical.
        r4 = service_block(BODY, gen.SERVICE)
        start = r4.index("    # Base-tier v12 Q8 canary (docs/glm53-base-v12-q8-rollout.md)")
        argv = "\n".join(command_body(r4)) + "\n"
        end = r4.index(argv, start) + len(argv)
        restored = BODY.replace(r4, r4[:start] + r4[end:])
        variant = re.search(r'nearai\.otel\.config_variant: "([^"]+)"', SOURCE).group(1)
        self.assertTrue(variant.endswith(gen.V7_VARIANT_SUFFIX))
        restored = restored.replace(variant[: -len(gen.V7_VARIANT_SUFFIX)] + gen.V12_VARIANT_SUFFIX, variant)
        restored = restored.replace(f"engine_image:{gen.V12_LABEL}", f"engine_image:{gen.V7_LABEL}")
        restored = restored.replace(f'engine_image: "{gen.V12_LABEL}"', f'engine_image: "{gen.V7_LABEL}"')
        self.assertEqual(restored, SOURCE)

    def test_changed_lines_are_r4_only(self):
        removed, added = changed_lines()
        # 5 label lines are replaced (datadog tags, otel config_variant/engine_image, scrape config_variant/engine_image);
        # nothing else is removed.
        self.assertEqual(len(removed), 5, removed)
        for line in removed:
            self.assertRegex(line, r"config_variant|engine_image")
        self.assertIn(f"    image: {gen.V12_IMAGE}", added)
        self.assertIn("    command: >", added)

    def test_image_is_only_on_r4(self):
        self.assertEqual(TARGET.count(gen.V12_IMAGE), 1)
        self.assertIn(gen.V12_IMAGE, service_block(TARGET, gen.SERVICE))
        self.assertEqual(TARGET.count(gen.V7_IMAGE_DIGEST), SOURCE.count(gen.V7_IMAGE_DIGEST))

    def test_r4_argv_is_anchor_argv_with_only_backend_swapped(self):
        anchor_tokens = " ".join(command_body(anchor_block(TARGET))).split()
        r4_tokens = r4_argv_tokens(TARGET)
        self.assertEqual(len(r4_tokens), len(anchor_tokens))
        diffs = [(a, r) for a, r in zip(anchor_tokens, r4_tokens) if a != r]
        self.assertEqual(diffs, [("flashmla_kv", "flashmla_sparse_q8")])
        self.assertEqual(r4_tokens[r4_tokens.index("--dsa-prefill-backend") + 1], "flashmla_sparse_q8")
        self.assertEqual(r4_tokens[r4_tokens.index("--dsa-decode-backend") + 1], "flashmla_kv")
        self.assertNotIn(NEW_BACKEND, anchor_block(TARGET))
        self.assertEqual(anchor_block(TARGET), anchor_block(SOURCE))

    def test_chunk_stays_8192(self):
        tokens = r4_argv_tokens(TARGET)
        self.assertEqual(tokens[tokens.index("--chunked-prefill-size") + 1], "8192")
        self.assertNotIn("16384", tokens)
        self.assertNotIn("--chunked-prefill-size 16384", TARGET)

    def test_r1_to_r3_and_anchor_unchanged(self):
        for replica in ("r1", "r2", "r3"):
            name = f"model-sg-glm53-w4afp8-tp2-{replica}"
            self.assertEqual(service_block(TARGET, name), service_block(SOURCE, name), name)
            self.assertEqual(scrape_job(TARGET, name), scrape_job(SOURCE, name), name)
        self.assertEqual(anchor_block(TARGET), anchor_block(SOURCE))
        self.assertEqual(r4_argv_tokens(SOURCE), [])  # the source r4 has no command of its own

    def test_r4_labels_carry_v12_and_suffix(self):
        block = service_block(TARGET, gen.SERVICE)
        self.assertIn(f'"engine_image:{gen.V12_LABEL}"', block)
        self.assertIn(f'nearai.otel.engine_image: "{gen.V12_LABEL}"', block)
        self.assertRegex(block, r'nearai\.otel\.config_variant: "[^"]+-obs-v1-v12-q8"')
        job = scrape_job(TARGET, gen.SERVICE)
        self.assertIn(f'engine_image: "{gen.V12_LABEL}"', job)
        self.assertRegex(job, r'config_variant: "[^"]+-obs-v1-v12-q8"')
        self.assertNotIn(gen.V7_LABEL, block + job)
        self.assertNotIn("-v7-q8", TARGET)  # the v10 canary's suffix

    def test_r4_is_exactly_r4_of_the_v12_fleet_file(self):
        # Same image, argv, labels and scrape job as the fleet file's r4, so a pass on the canary qualifies the fleet's r4.
        self.assertEqual(r4_argv_tokens(TARGET), " ".join(command_body(anchor_block(FLEET))).split())
        self.assertEqual(scrape_job(TARGET, gen.SERVICE), scrape_job(FLEET, gen.SERVICE))
        canary_labels = service_block(TARGET, gen.SERVICE).split("    labels:\n", 1)[1]
        fleet_labels = service_block(FLEET, gen.SERVICE).split("    labels:\n", 1)[1]
        self.assertEqual(canary_labels, fleet_labels)
        self.assertIn(f"\n  image: {gen.V12_IMAGE}\n", FLEET)  # the fleet's r4 inherits v12 from the common anchor

    def test_generator_refuses_source_already_carrying_q8(self):
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE.replace(OLD_BACKEND, NEW_BACKEND, 1))
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE + f"\n# {gen.V12_IMAGE}\n")

    def test_generator_refuses_missing_or_duplicated_tokens(self):
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE.replace(f"  {gen.SERVICE}:\n", f"  {gen.SERVICE}-x:\n", 1))
        with self.assertRaises(gen.GenerationError):
            anchor = anchor_block(SOURCE)  # the chunk in the common anchor is overridden by this one, so edit this one
            gen.generate(SOURCE.replace(anchor, anchor.replace("--chunked-prefill-size 8192", "--chunked-prefill-size 4096", 1)))
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE.replace(OLD_BACKEND, OLD_BACKEND + "\n      " + OLD_BACKEND, 1))
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE.replace('nearai.otel.config_variant: "', 'nearai.otel.config_variant_x: "'))

    def test_generator_refuses_a_broken_anchor_command_block(self):
        # A blank or stray line inside the anchor's folded command must fail loudly, never truncate the argv.
        backend = "      --dsa-prefill-backend flashmla_kv\n"
        start = SOURCE.index(f"\n{gen.ANCHOR}: &sg-glm53-flash-candidate\n")
        at = SOURCE.index(backend, start)
        for inserted in ("\n", "      # comment\n", "    --misindented flag\n"):
            broken = SOURCE[:at] + inserted + SOURCE[at:]
            with self.subTest(inserted=inserted), self.assertRaises(gen.GenerationError):
                gen.generate(broken)

    def test_header_names_host_service_and_runbook(self):
        header = TARGET.split("#\n", 1)[0]
        for needle in (gen.HOST, gen.SERVICE, "docs/glm53-base-v12-q8-rollout.md", "Deploy it ONLY"):
            self.assertIn(needle, header)
        self.assertTrue((ROOT / "docs/glm53-base-v12-q8-rollout.md").exists())

    def test_dashboard_selectors_unchanged(self):
        for key in ("deployment", "host_machine", "server_address", "container_name", "instance", "service"):
            self.assertEqual(
                re.findall(rf"{key}: \"[^\"]*\"", SOURCE), re.findall(rf"{key}: \"[^\"]*\"", TARGET), key
            )


if __name__ == "__main__":
    unittest.main()
