"""Exercise the actual promotion and production validator in a temporary tree."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('promotion', ROOT / 'scripts/prepare_glm53_hicache_canary.py')
promotion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promotion)
FIXTURE_IMAGE = 'docker.io/nearaidev/sglang@sha256:' + '1' * 64


def parse_yaml(path):
    return json.loads(subprocess.check_output([
        'ruby', '-ryaml', '-rjson', '-e',
        'puts JSON.generate(YAML.load_file(ARGV[0], aliases: true))', str(path)], text=True))


class PromotionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in (promotion.COMPOSE, Path('scripts/validate_glm53_prod_config.rb'),
                     Path('scripts/prepare_glm53_hicache_canary.py')):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.root / name)
        (self.root / promotion.RELEASE).parent.mkdir(parents=True, exist_ok=True)
        self.path = self.root / promotion.COMPOSE
        self.original = self.path.read_text()
        # Continue exercising the transition after the activation PR is merged.
        # The generated overrides are bounded between identity and depends_on.
        if (ROOT / promotion.RELEASE).exists():
            start = self.original.index('    container_name: model-sg-glm53-fp8-tp4-r2\n')
            start += len('    container_name: model-sg-glm53-fp8-tp4-r2\n')
            end = self.original.index('    depends_on:\n', start)
            self.original = self.original[:start] + self.original[end:]
            self.original = self.original.replace(promotion.VARIANT,
                'official-upstream-fc91d24-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192')
            self.path.write_text(self.original)

    def validate(self, success=True):
        result = subprocess.run(['ruby', str(self.root / 'scripts/validate_glm53_prod_config.rb')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0 if success else 1, result.stdout + result.stderr)

    def activate(self):
        self.path.write_text(promotion.candidate(self.original, FIXTURE_IMAGE))
        (self.root / promotion.RELEASE).write_text(FIXTURE_IMAGE + '\n')

    def test_promotion_scope(self):
        self.validate()
        baseline = parse_yaml(self.path)
        self.activate()
        self.validate()
        updated = parse_yaml(self.path)
        changed = [name for name, service in baseline['services'].items()
                   if updated['services'][name] != service]
        self.assertEqual(changed, ['model-sg-glm53-fp8-tp4-r2'])
        for name, value in baseline.items():
            if name not in ('services', 'configs'):
                self.assertEqual(updated[name], value, name)
        for name, config in baseline['configs'].items():
            if name != 'otelcol_app_config':
                self.assertEqual(updated['configs'][name], config)
        before = baseline['configs']['otelcol_app_config']['content']
        after = updated['configs']['otelcol_app_config']['content']
        # Exactly one scrape label changes; r1 and other scrape targets are intact.
        self.assertEqual(after.count(promotion.VARIANT), 1)
        self.assertEqual(after.replace(promotion.VARIANT,
            'official-upstream-fc91d24-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192'), before)

    def test_preview_does_not_write(self):
        result = subprocess.run(['python3', str(self.root / 'scripts/prepare_glm53_hicache_canary.py'),
                                 '--image', FIXTURE_IMAGE], capture_output=True, text=True, check=True)
        self.assertIn('--enable-hierarchical-cache', result.stdout)
        self.assertEqual(self.path.read_text(), self.original)
        self.assertFalse((self.root / promotion.RELEASE).exists())

    def test_invalid_and_repeated_promotion(self):
        for image in ('nearaidev/sglang:latest', FIXTURE_IMAGE + '\n',
                      FIXTURE_IMAGE.replace('nearaidev', 'untrusted')):
            with self.assertRaises(ValueError):
                promotion.candidate(self.original, image)
        self.activate()
        with self.assertRaises(ValueError):
            promotion.candidate(self.path.read_text(), FIXTURE_IMAGE)

    def test_rejects_unreleased_image(self):
        self.path.write_text(promotion.candidate(self.original, FIXTURE_IMAGE))
        self.validate(False)

    def test_rejects_runtime_drift(self):
        self.activate()
        valid = self.path.read_text()
        # These mutations target only the candidate override, or all arguments
        # for the deliberate r1 contamination case. Each must fail the contract.
        for before, after in (
            ('--hicache-size 32', '--hicache-size 64'),
            ('--hicache-io-backend direct', '--hicache-io-backend kernel'),
            ('SGLANG_HICACHE_POOLED_TRANSFERS=1', 'SGLANG_HICACHE_POOLED_TRANSFERS=0'),
            ('SGLANG_HICACHE_STAGING_PAGES=64', 'SGLANG_HICACHE_STAGING_PAGES=128'),
            ('        --chunked-prefill-size 4096', '        --chunked-prefill-size 8192'),
            ('      --kv-cache-dtype bfloat16', '      --enable-hierarchical-cache\n      --kv-cache-dtype bfloat16'),
            (promotion.VARIANT, 'incorrect-variant'),
        ):
            with self.subTest(mutation=after):
                self.assertIn(before, valid)
                self.path.write_text(valid.replace(before, after))
                self.validate(False)


if __name__ == '__main__':
    unittest.main()
