"""Confinement tests for the base-tier v12 + Q8-prefill fleet file (scripts/prepare_glm53_base_v12_q8_fleet.py).

Against the v7 fleet file only the engine image (four replicas and the ghost aggregator), the prefill backend token and
the replicas' engine_image / config_variant labels may change. The files gpu03 and gpu04 are recorded as running
(docs/glm53-proxy-emudhra.md) are compared too, so a scoped compose/up of one replica moves nothing else.
Run: python3 -m scripts.test_glm53_base_v12_q8_fleet
"""

import difflib
import re
import unittest
from pathlib import Path

from scripts import prepare_glm53_base_v12_q8_fleet as gen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / gen.SOURCE).read_text()
TARGET = (ROOT / gen.TARGET).read_text()
BODY = TARGET.split("#\n", 1)[1]  # past the generated header comment
GPU03_LIVE = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V8KdaCanary.yaml")
GPU04_LIVE = Path("prod/GLM-5.3-Flash-SGL-TP2x3-W4AFP8-gpu04.yaml")
CANARY = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-Q8Canary.yaml")
RUNBOOK = ROOT / "docs/glm53-base-v12-q8-rollout.md"
V8_LABEL = "59700363c0b6"
# Engine-side blocks a per-replica rollout replaces; the aggregator and the collector config move only in their own per-host step.
ENGINE_SIDE = {*gen.REPLICAS, gen.AGGREGATOR, "otelcol_app_config"}


def block(text, name):
    start = text.index(f"\n  {name}:\n")
    match = re.search(r"\n  [^ \n]|\n[^ \n#]", text[start + 3:])
    return text[start:match.start() + start + 3 if match else len(text)]


def blocks(text):
    names = set(re.findall(r"\n  ([a-z0-9_-]+):\n", text))
    return {name: block(text, name) for name in names if text.count(f"\n  {name}:\n") == 1}


def anchor_block(text):
    start = text.index(f"\n{gen.ANCHOR}: &sg-glm53-flash-candidate\n")
    end = re.search(r"\n[^ \n#]", text[start + 3:]).start() + start + 3
    return text[start:end]


def anchor_tokens(text):
    body = anchor_block(text).split("\n  command: >\n", 1)[1]
    return " ".join(line for line in body.splitlines() if line.startswith("      ")).split()


def changed_lines(old, new):
    removed, added = [], []
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        (removed if line.startswith("-") else added).append(line[1:])
    return removed, added


def is_comment(line):
    return line.lstrip().startswith("#")


class BaseV12Q8FleetTest(unittest.TestCase):
    def test_committed_target_is_fresh(self):
        self.assertEqual(TARGET, gen.generate(SOURCE), "run scripts/prepare_glm53_base_v12_q8_fleet.py --write")

    def test_source_is_still_the_v7_fleet_file(self):
        # The v7 file is the rollback target; this change must not touch it.
        self.assertIn(f"\n  image: {gen.V7_IMAGE}\n", SOURCE)
        self.assertIn("      --dsa-prefill-backend flashmla_kv\n", anchor_block(SOURCE))
        self.assertNotIn(gen.V12_IMAGE_DIGEST, SOURCE)
        self.assertNotIn(gen.Q8_BACKEND, SOURCE)

    def test_only_image_backend_and_labels_change(self):
        removed, added = changed_lines(SOURCE, BODY)
        self.assertEqual([line for line in removed if is_comment(line)], [])
        # 2 images (common anchor, aggregator) + 1 backend + 4 replicas x 3 label lines + 4 scrape jobs x 2 label lines.
        self.assertEqual(len(removed), 23, removed)
        added = [line for line in added if not is_comment(line)]
        self.assertEqual(len(added), len(removed))
        to_v12 = lambda line: (
            line.replace(gen.V7_IMAGE_DIGEST, gen.V12_IMAGE_DIGEST)
            .replace(gen.V7_LABEL, gen.V12_LABEL)
            .replace("-obs-v1-v7\"", "-obs-v1-v12-q8\"")
            .replace("-obs-v1-v7\",", "-obs-v1-v12-q8\",")
            .replace("--dsa-prefill-backend flashmla_kv", "--dsa-prefill-backend flashmla_sparse_q8")
        )
        self.assertEqual([to_v12(line) for line in removed], added)
        for line in removed:
            self.assertRegex(line, r"image: .*fa730e6e|config_variant|engine_image|--dsa-prefill-backend flashmla_kv")

    def test_argv_differs_from_v7_only_in_the_prefill_backend(self):
        # Pinned on purpose: a lever added to ARGV_EDITS / ARGV_APPENDED later must update this line deliberately.
        v7, v12 = anchor_tokens(SOURCE), anchor_tokens(TARGET)
        self.assertEqual(len(v7), len(v12))
        self.assertEqual([(a, b) for a, b in zip(v7, v12) if a != b], [("flashmla_kv", "flashmla_sparse_q8")])
        value = lambda flag: v12[v12.index(flag) + 1]
        self.assertEqual(value("--dsa-prefill-backend"), "flashmla_sparse_q8")
        self.assertEqual(value("--dsa-decode-backend"), "flashmla_kv")
        self.assertEqual(value("--kv-cache-dtype"), "fp8_e4m3")
        self.assertEqual(value("--chunked-prefill-size"), "8192")
        self.assertEqual(value("--max-queued-requests"), "32")
        self.assertEqual(value("--max-running-requests"), "64")

    def test_every_replica_inherits_the_one_anchor(self):
        for name in gen.REPLICAS:
            service = block(TARGET, name)
            self.assertIn("    <<: *sg-glm53-flash-candidate\n", service, name)
            self.assertNotIn("\n    image:", service, name)
            self.assertNotIn("\n    command:", service, name)
        self.assertEqual(TARGET.count(gen.V12_IMAGE), 2)  # common anchor + ghost aggregator
        self.assertIn(gen.V12_IMAGE, block(TARGET, gen.AGGREGATOR))
        live = "".join(line + "\n" for line in TARGET.splitlines() if not is_comment(line))
        self.assertNotIn(gen.V7_LABEL, live)

    def test_replica_labels(self):
        for name in gen.REPLICAS:
            service = block(TARGET, name)
            self.assertIn(f'"engine_image:{gen.V12_LABEL}"', service)
            self.assertIn(f'nearai.otel.engine_image: "{gen.V12_LABEL}"', service)
            self.assertEqual(len(re.findall(r'config_variant(?::|: ")[^"]*-obs-v1-v12-q8"', service)), 2, name)
        self.assertEqual(TARGET.count('engine_image: "da8d4e4b5ec6"'), 8)
        self.assertEqual(TARGET.count("-obs-v1-v12-q8\""), 12)

    def test_dashboard_selectors_unchanged(self):
        for key in ("deployment", "host_machine", "server_address", "container_name", "instance", "service", "precision"):
            self.assertEqual(re.findall(rf"{key}: \"[^\"]*\"", SOURCE), re.findall(rf"{key}: \"[^\"]*\"", TARGET), key)

    def test_generator_refuses_bad_sources(self):
        with self.assertRaises(gen.GenerationError):
            gen.generate(TARGET)  # already v12
        anchor = anchor_block(SOURCE)
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE.replace(anchor, anchor.replace("--chunked-prefill-size 8192", "--chunked-prefill-size 16384", 1)))
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE.replace(anchor, anchor.replace("--max-queued-requests 32", "--max-queued-requests 64", 1)))
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE.replace(anchor, anchor.replace("--dsa-prefill-backend flashmla_kv", "--dsa-prefill-backend tilelang", 1)))
        r2 = block(SOURCE, gen.REPLICAS[1])
        with self.assertRaises(gen.GenerationError):
            gen.generate(SOURCE.replace(r2, r2.replace("    container_name:", f"    image: {gen.V7_IMAGE}\n    container_name:", 1)))

    def test_argv_appended_refuses_a_flag_already_set(self):
        original = gen.ARGV_APPENDED
        try:
            gen.ARGV_APPENDED = ("--speculative-num-steps 3",)
            with self.assertRaises(gen.GenerationError):
                gen.generate(SOURCE)
        finally:
            gen.ARGV_APPENDED = original

    def test_header_names_rollback_and_runbook(self):
        header = TARGET.split("#\n", 1)[0]
        for needle in (str(gen.SOURCE), "docs/glm53-base-v12-q8-rollout.md", "ROLLBACK", "scoped compose/up"):
            self.assertIn(needle, header)
        self.assertTrue(RUNBOOK.exists())


class LiveFileCoverageTest(unittest.TestCase):
    """What each base host is recorded as running, and what a scoped per-replica deploy of the fleet file changes there."""

    def setUp(self):
        self.fleet = blocks(TARGET)

    def differing(self, path):
        live = blocks((ROOT / path).read_text())
        return set(live) ^ set(self.fleet), {name for name in live.keys() & self.fleet.keys() if live[name] != self.fleet[name]}

    def test_gpu03_v8_kda_file_differs_only_engine_side(self):
        only, differ = self.differing(GPU03_LIVE)
        self.assertEqual(only, set())
        self.assertEqual(differ, ENGINE_SIDE)
        # gpu03 r4 moves v8 -> v12: v12 carries v8's zero-sync KDA patch, so nothing it runs is reverted.
        live = (ROOT / GPU03_LIVE).read_text()
        self.assertIn(V8_LABEL, block(live, gen.REPLICAS[3]))
        for name in gen.REPLICAS[:3]:
            self.assertEqual(block(live, name), block(SOURCE, name), name)

    def test_gpu04_three_replica_file_also_differs_in_the_proxy_pool(self):
        # The recorded gpu04 file has no r4 and pools r1-r3 only. A scoped engine deploy never touches proxy-glm53, so
        # if gpu04's live proxy came from that file the r4 canary gets no traffic: the runbook's pre-check covers it.
        only, differ = self.differing(GPU04_LIVE)
        self.assertEqual(only, {gen.REPLICAS[3]})
        self.assertEqual(differ, (ENGINE_SIDE - {gen.REPLICAS[3]}) | {"proxy-glm53", "glm53-soak-relay", "glm53_soak_nginx_conf", "glm53-perception-check"})
        live = (ROOT / GPU04_LIVE).read_text()
        self.assertNotIn(f"http://{gen.REPLICAS[3]}:8000", block(live, "proxy-glm53"))
        self.assertIn(f"http://{gen.REPLICAS[3]}:8000", block(TARGET, "proxy-glm53"))
        self.assertEqual(block(TARGET, "proxy-glm53"), block(SOURCE, "proxy-glm53"))
        self.assertEqual(block(TARGET, "proxy-glm53"), block((ROOT / CANARY).read_text(), "proxy-glm53"))

    def test_v7_fleet_and_canary_files_differ_only_engine_side(self):
        for path in (gen.SOURCE, CANARY):
            only, differ = self.differing(path)
            self.assertEqual(only, set(), path)
            self.assertEqual(differ, ENGINE_SIDE, path)

    def test_runbook_names_the_live_files_and_the_proxy_check(self):
        text = RUNBOOK.read_text()
        for needle in (str(GPU03_LIVE), str(GPU04_LIVE), str(CANARY), str(gen.TARGET), str(gen.SOURCE), "VLLM_BACKEND_URLS",
                       "dry_run", "exp53-glm53-base-fp8-load-ladder", "C08", "84c6a6", "3f1971b"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
