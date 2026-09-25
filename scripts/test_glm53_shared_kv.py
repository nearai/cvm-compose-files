#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_glm53_shared_kv
"""The shared-KV-cache variant generator: only the sharing change differs from the live long-context file."""

import re
import unittest
from pathlib import Path

from scripts import prepare_glm53_shared_kv as generator

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "docker.io/nearaidev/sglang@sha256:" + "ab" * 32
SOURCE = (ROOT / generator.SOURCE).read_text()


def engine_block(text: str, replica: int) -> str:
    """The r<replica> service block (r1 inherits the common anchor, r2 overrides it)."""
    start = text.index(f"  model-sg-glm53-w4afp8-tp4-r{replica}:\n")
    end = text.index("\n\n", start)
    return text[start:end]


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out = generator.render(SOURCE, IMAGE)

    def test_both_engines_run_the_shared_kv_image(self) -> None:
        live = "\n".join(l for l in self.out.splitlines() if not l.lstrip().startswith("#"))
        for src in generator.SOURCE_IMAGES:
            self.assertNotIn(src, live)
        self.assertEqual(self.out.count(f"image: {IMAGE}\n"), 2)

    def test_both_engines_enable_the_file_store(self) -> None:
        self.assertEqual(self.out.count("--hicache-storage-backend file\n"), 2)
        self.assertEqual(self.out.count("--hicache-storage-prefetch-policy wait_complete\n"), 2)
        self.assertEqual(self.out.count(f"- SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR={generator.STORE_MOUNT}\n"), 2)
        self.assertEqual(self.out.count("- SGLANG_HICACHE_SHARED_STORE_BUDGET="), 2)
        self.assertEqual(self.out.count("- SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE="), 2)
        # The private 406 GiB host tier per replica is kept.
        self.assertEqual(self.out.count("SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-406GiB}"), 2)

    def test_one_shared_tmpfs_volume_mounted_by_the_engine_anchor(self) -> None:
        self.assertEqual(self.out.count(f"- shared_kv:{generator.STORE_MOUNT}\n"), 1)
        # r2 must not override volumes, or it would lose the shared mount.
        self.assertNotIn("volumes:", engine_block(self.out, 2))
        self.assertIn("  shared_kv:\n    driver: local\n    driver_opts:\n      type: tmpfs\n", self.out)

    def test_replica_cap_is_half_the_store_budget_by_default(self) -> None:
        budget = int(re.search(r":-(\d+)GiB", generator.STORE_BUDGET).group(1))
        cap = int(re.search(r":-(\d+)Gi}", generator.REPLICA_CAP).group(1))
        self.assertLessEqual(cap * 2, budget)
        tmpfs_g = int(re.search(r":-(\d+)g}", generator.TMPFS_SIZE).group(1))
        self.assertGreater(tmpfs_g * 10**9, budget * 1024**3, "tmpfs must be larger than the store budget")

    def test_replica_flags_are_otherwise_unchanged(self) -> None:
        # r1 stays the chunk-8192/pdi1 control, r2 the chunk-16384/pdi2 + indexer-split canary.
        self.assertIn("--chunked-prefill-size 8192", self.out)
        self.assertIn("--chunked-prefill-size 16384", engine_block(self.out, 2))
        self.assertIn("SGLANG_DSA_INDEXER_QSPLIT=1", engine_block(self.out, 2))

    def test_telemetry_names_the_new_config(self) -> None:
        self.assertNotIn(generator.SOURCE_VARIANT_PREFIX, self.out)
        variants = set(re.findall(re.escape(generator.VARIANT_PREFIX) + r"[a-z0-9-]+", self.out))
        self.assertEqual(len(variants), 2)
        for v in variants:
            self.assertIn("l3-shared-file", v)
        self.assertNotIn('engine_image: "fde25985aea3"', self.out)
        self.assertNotIn('engine_image: "8ff1a487b98a"', self.out)

    def test_nothing_else_changes(self) -> None:
        # Undo every intended edit; the remainder must be byte-identical to the source.
        body = self.out[len(generator.HEADER):]
        added = [  # anchored at line starts, so the 6-space lines never match inside the 8-space ones
            "\n        --hicache-storage-backend file\n", "\n        --hicache-storage-prefetch-policy wait_complete\n",
            "\n      --hicache-storage-backend file\n", "\n      --hicache-storage-prefetch-policy wait_complete\n",
            f"\n    - shared_kv:{generator.STORE_MOUNT}\n",
        ]
        for line in added:
            body = body.replace(line, "\n")
        body = re.sub(r"\n *# Shared KV store: one tmpfs volume in guest RAM, mounted by both replicas\.\n(?: *- SGLANG_HICACHE_(?:FILE_BACKEND_STORAGE_DIR|SHARED_STORE_BUDGET|FILE_BACKEND_MAX_SIZE)=.*\n){3}", "\n", body)
        body = re.sub(r"  shared_kv:\n    driver: local\n    driver_opts:\n      type: tmpfs\n      device: tmpfs\n      o: .*\n", "", body)
        body = body.replace(generator.VARIANT_PREFIX, generator.SOURCE_VARIANT_PREFIX).replace(generator.HICACHE_TAG, generator.SOURCE_HICACHE_TAG)
        new_short = IMAGE.rsplit(":", 1)[1][:12]
        shorts = list(generator.SOURCE_IMAGES.values())  # r1, r2 in file order
        for src in generator.SOURCE_IMAGES:
            body = body.replace(f"image: {IMAGE}\n", f"image: {src}\n", 1)
        for short in shorts:
            body = body.replace("engine_image:" + new_short, "engine_image:" + short, 1)
        # OTel labels (r1, r2) then collector attributes (r1, r2).
        for short in shorts + shorts:
            body = body.replace(f'engine_image: "{new_short}"', f'engine_image: "{short}"', 1)
        self.assertEqual(body, SOURCE)

    def test_rejects_unpinned_images(self) -> None:
        with self.assertRaises(SystemExit):
            generator.render(SOURCE, "docker.io/nearaidev/sglang:latest")


if __name__ == "__main__":
    unittest.main()
