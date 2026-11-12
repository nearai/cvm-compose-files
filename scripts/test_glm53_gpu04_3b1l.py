"""Contract tests for gpu04's mixed GLM-5.3 file (3 base replicas + 1 long-context replica).

Run: python3 -m scripts.test_glm53_gpu04_3b1l
"""

import re
import unittest
from pathlib import Path

from scripts import prepare_glm53_gpu04_3b1l as gen

ROOT = Path(__file__).resolve().parents[1]


def services(text: str) -> dict[str, str]:
    """Top-level service blocks by name (raw text)."""
    body = text.split("\nservices:\n", 1)[1].split("\nnetworks:\n", 1)[0]
    parts = re.split(r"(?m)^  ([a-z0-9-]+):\n", body)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


class Gpu04MixedTier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Base r1-r3 come from the V12Q8 file as rendered without gpu03's r4 v0.5.23 canary (v12).
        cls.source = gen.v12_base_text()
        # l4 copies gpu23's r2b as rendered without the r2b v0.5.23 canary (v12), not the committed long file.
        cls.long_source = gen.v12_long_text()
        cls.text = (ROOT / gen.TARGET).read_text()
        cls.svc = services(cls.text)
        cls.src_svc = services(cls.source)
        cls.long_svc = services(cls.long_source)

    def test_committed_file_is_generated(self) -> None:
        self.assertEqual(self.text, gen.generate(self.source, self.long_source))

    def test_l4_does_not_follow_the_r2b_v0523_canary(self) -> None:
        l4 = self.svc[gen.LONG_L4]
        self.assertNotIn(gen.long_generator.V0523_CANARY_IMAGE, l4)
        self.assertNotIn("--cuda-graph-backend-prefill", l4)
        self.assertNotIn("v0523", l4)
        self.assertIn(f"image: {gen.long_generator.TP2_IMAGE}\n", l4)

    def test_service_set(self) -> None:
        expected = (set(self.src_svc) - {gen.BASE_R4}) | {gen.LONG_L4, gen.LONG_PROXY}
        self.assertEqual(set(self.svc), expected)

    def test_base_replicas_unchanged(self) -> None:
        for r in ("r1", "r2", "r3"):
            name = f"{gen.PREFIX}{r}"
            strip = lambda block: [l for l in block.splitlines() if not l.startswith("  # ")]
            self.assertEqual(strip(self.svc[name]), strip(self.src_svc[name]), name)

    def test_l4_is_gpu23_r2b_with_queue_32(self) -> None:
        l4 = self.svc[gen.LONG_L4]
        r2b = self.long_svc[gen.LONG_R2B]
        argv = lambda block: block.split("command: >\n", 1)[1].split("    environment:\n", 1)[0].split()
        diff = [(a, b) for a, b in zip(argv(r2b), argv(l4)) if a != b]
        self.assertEqual(diff, [("8", "32")])
        self.assertIn('device_ids: ["6","7"]', l4)
        self.assertIn("SGLANG_GHOST_CACHE_REPLICA=l4", l4)
        self.assertNotIn("SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE", l4)
        # HiCache OFF on the long tier since 2026-11-06 (docs/glm53-long-hicache-off.md);
        # l4 inherits that from gpu23's r2b, which the argv diff above already pins.
        self.assertNotIn("--enable-hierarchical-cache", l4)
        self.assertNotIn("SGLANG_HICACHE_RAM_BUDGET", l4)

    def test_proxy_pools(self) -> None:
        base = re.search(r"VLLM_BACKEND_URLS=(\S+)", self.svc["proxy-glm53"]).group(1)
        long = re.search(r"VLLM_BACKEND_URLS=(\S+)", self.svc[gen.LONG_PROXY]).group(1)
        self.assertEqual(base, ",".join(f"http://{gen.PREFIX}r{i}:8000" for i in (1, 2, 3)))
        self.assertEqual(long, f"http://{gen.LONG_L4}:8000")
        self.assertIn("MODEL_NAME=z-ai/glm-5.3-flash\n", self.svc[gen.LONG_PROXY])

    def test_nginx_routes(self) -> None:
        nginx = self.svc["nginx"]
        self.assertIn('- "8001:8001"', nginx)
        self.assertIn(f"- {gen.LONG_PROXY}", nginx)
        conf = self.text.split("  nginx_conf:\n", 1)[1]
        self.assertEqual(conf.count("listen 443 ssl http2 default_server;"), 1)
        long_vhost = conf.split("glm-5-3-flash-long.completions.near.ai\n", 1)[1]
        self.assertIn("http://proxy-glm53-long:8000", long_vhost.split("}\n      }", 1)[0])
        plain = conf.split("        listen 8001;\n", 1)[1].split("\n      }\n", 1)[0]
        self.assertIn("auth_request /_glm53_long_health;", plain)
        self.assertIn('"id":"z-ai/glm-5.3-flash-long"', plain)
        self.assertNotIn("proxy-glm53:8000", plain)

    def test_registrar(self) -> None:
        script = self.text.split("  registrar_script:\n", 1)[1].split("  nginx_conf:\n", 1)[0]
        self.assertNotIn('unregister_endpoint "$${HOST_IP}:8001"', script)
        self.assertIn('register_model "z-ai/glm-5.3-flash" "glm-5-3-flash.completions.near.ai"', script)
        self.assertIn('register_model "z-ai/glm-5.3-flash-long" "glm-5-3-flash-long.completions.near.ai"', script)
        self.assertNotIn('register_model "z-ai/glm-5.3-flash" "glm-5-3-flash-long', script)
        self.assertIn("        long_tick\n        sleep 60\n", script)
        self.assertIn("LONG_HTTP_PORT=${LONG_HTTP_PORT:-8001}", self.svc["model-proxy-registrar"])

    def test_no_r4_left(self) -> None:
        body = self.text.split("\nservices:\n", 1)[1]
        self.assertNotIn(gen.BASE_R4, body)
        self.assertIn(f"job_name: sglang-{gen.LONG_L4}", body)
        self.assertIn(f"job_name: inference-proxy-{gen.LONG_PROXY}", body)


if __name__ == "__main__":
    unittest.main()
