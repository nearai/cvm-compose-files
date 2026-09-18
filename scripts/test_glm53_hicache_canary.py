"""Exercise the actual generator and production validator in a temporary tree."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('promotion', ROOT / 'scripts/prepare_glm53_hicache_canary.py')
promotion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promotion)
FIXTURE_IMAGE = 'docker.io/nearaidev/sglang@sha256:' + '1' * 64
OTHER_FIXTURE_IMAGE = 'docker.io/nearaidev/sglang@sha256:' + '2' * 64
OFFICIAL_VARIANT = 'fc91d24-admission-reserve-v10-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192'
LONG_CONTEXT = Path('prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml')
LONG_CONTEXT_CONTROL_VARIANT = 'fc91d24-long-context-admission-reserve-disabled-hicache-disabled-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192'
LONG_CONTEXT_HICACHE_VARIANT = 'fc91d24-long-context-admission-reserve-disabled-hicache-cuda-host-pooled-v1-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192'
# Substrings that would appear in a Ruby exception's default-formatted
# message or a backtrace line; none of these should ever reach stderr.
RUBY_CRASH_MARKERS = ('(NoMethodError)', '(TypeError)', '(Psych::SyntaxError)')
RUBY_BACKTRACE_LINE = re.compile(r'\.rb:\d+:in')


def parse_yaml(path):
    return json.loads(subprocess.check_output([
        'ruby', '-ryaml', '-rjson', '-e',
        'puts JSON.generate(YAML.load_file(ARGV[0], aliases: true))', str(path)], text=True))


def replace_nth(text, needle, index, replacement):
    """Replace only the `index`-th (0-based) occurrence of `needle` in `text`."""
    positions = []
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    target = positions[index]
    return text[:target] + replacement + text[target + len(needle):]


def assert_no_ruby_crash(testcase, output):
    for marker in RUBY_CRASH_MARKERS:
        testcase.assertNotIn(marker, output)
    testcase.assertIsNone(RUBY_BACKTRACE_LINE.search(output), output)
    testcase.assertNotIn('Traceback', output)


class SyncTest(unittest.TestCase):
    """The committed HiCache file must always be exactly what the generator produces
    from the committed canonical file and the committed RELEASED_IMAGE."""

    def test_committed_candidate_matches_generator(self):
        canonical = (ROOT / promotion.COMPOSE).read_text()
        released_image = (ROOT / promotion.RELEASE).read_text().strip()
        expected = promotion.candidate(canonical, released_image)
        actual = (ROOT / promotion.CANDIDATE).read_text()
        self.assertEqual(actual, expected)

    def test_check_flag_passes(self):
        result = subprocess.run(['python3', str(ROOT / 'scripts/prepare_glm53_hicache_canary.py'), '--check'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class GeneratorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in (promotion.COMPOSE, Path('scripts/validate_glm53_prod_config.rb'),
                     Path('scripts/validate_glm53_dcgm_metrics.rb'),
                     Path('scripts/prepare_glm53_hicache_canary.py')):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.root / name)
        (self.root / promotion.RELEASE).parent.mkdir(parents=True, exist_ok=True)
        self.canonical_path = self.root / promotion.COMPOSE
        self.candidate_path = self.root / promotion.CANDIDATE
        self.release_path = self.root / promotion.RELEASE
        self.canonical = self.canonical_path.read_text()

    def run_validator(self):
        if self.candidate_path.exists() and self.release_path.exists():
            long_context_path = self.root / LONG_CONTEXT
            long_context_path.parent.mkdir(parents=True, exist_ok=True)
            committed_image = (ROOT / promotion.RELEASE).read_text().strip()
            fixture_image = self.release_path.read_text().strip()
            long_context = (ROOT / LONG_CONTEXT).read_text().replace(
                committed_image, fixture_image, 1)
            long_context_path.write_text(long_context)
        return subprocess.run(['ruby', str(self.root / 'scripts/validate_glm53_prod_config.rb')],
                              capture_output=True, text=True)

    def run_dcgm_validator(self):
        return subprocess.run(['ruby', str(self.root / 'scripts/validate_glm53_dcgm_metrics.rb')],
                              capture_output=True, text=True)

    def run_generator(self, *extra_args):
        return subprocess.run(['python3', str(self.root / 'scripts/prepare_glm53_hicache_canary.py'), *extra_args],
                              capture_output=True, text=True)

    def validate(self, success=True):
        result = self.run_validator()
        self.assertEqual(result.returncode, 0 if success else 1, result.stdout + result.stderr)
        return result

    def activate(self, image=FIXTURE_IMAGE):
        self.candidate_path.write_text(promotion.candidate(self.canonical, image))
        self.release_path.write_text(image + '\n')

    def test_generation_scope(self):
        # Canonical alone (no HiCache file yet) must already pass.
        self.validate()

        result = self.run_generator('--image', FIXTURE_IMAGE, '--write')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # The canonical file is source-only; --write must never touch it.
        self.assertEqual(self.canonical_path.read_text(), self.canonical)
        self.validate()

        baseline = parse_yaml(self.canonical_path)
        updated = parse_yaml(self.candidate_path)
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
        # Exactly one scrape label changes; r1 and every other scrape target is intact.
        self.assertEqual(after.count(promotion.VARIANT), 1)
        self.assertEqual(after.replace(promotion.VARIANT, OFFICIAL_VARIANT), before)

    def test_preview_does_not_write(self):
        result = self.run_generator('--image', FIXTURE_IMAGE)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--enable-hierarchical-cache', result.stdout)
        self.assertEqual(self.canonical_path.read_text(), self.canonical)
        self.assertFalse(self.candidate_path.exists())
        self.assertFalse(self.release_path.exists())

    def test_invalid_and_repeated_generation(self):
        control_image = parse_yaml(self.canonical_path)['services']['model-sg-glm53-fp8-tp4-r1']['image']
        for image in ('nearaidev/sglang:latest', FIXTURE_IMAGE + '\n',
                      FIXTURE_IMAGE.replace('nearaidev', 'untrusted'), control_image):
            with self.assertRaises(ValueError):
                promotion.candidate(self.canonical, image)
        # A canonical text that already has HiCache in it is rejected.
        already = promotion.candidate(self.canonical, FIXTURE_IMAGE)
        with self.assertRaises(ValueError):
            promotion.candidate(already, FIXTURE_IMAGE)

    def test_write_guard_requires_force_to_replace_recorded_image(self):
        # First --write establishes a recorded RELEASED_IMAGE.
        result = self.run_generator('--image', FIXTURE_IMAGE, '--write')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        candidate_after_first = self.candidate_path.read_text()
        release_after_first = self.release_path.read_text()

        # A different image without --force refuses, leaving both files untouched.
        result = self.run_generator('--image', OTHER_FIXTURE_IMAGE, '--write')
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--force', result.stdout + result.stderr)
        self.assertEqual(self.candidate_path.read_text(), candidate_after_first)
        self.assertEqual(self.release_path.read_text(), release_after_first)

        # Re-running with the same digest still works without --force.
        result = self.run_generator('--image', FIXTURE_IMAGE, '--write')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.candidate_path.read_text(), candidate_after_first)
        self.assertEqual(self.release_path.read_text(), release_after_first)

        # --force replaces the recorded image.
        result = self.run_generator('--image', OTHER_FIXTURE_IMAGE, '--write', '--force')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.release_path.read_text().strip(), OTHER_FIXTURE_IMAGE)
        self.assertIn(OTHER_FIXTURE_IMAGE, self.candidate_path.read_text())

    def test_check_reports_malformed_released_image_cleanly(self):
        self.candidate_path.write_text(promotion.candidate(self.canonical, FIXTURE_IMAGE))
        self.release_path.write_text('nearaidev/sglang:latest\n')
        result = self.run_generator('--check')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('is not a valid signed image digest', result.stdout + result.stderr)
        self.assertNotIn('Traceback', result.stdout + result.stderr)

    def test_rejects_unreleased_image(self):
        self.activate()
        other_image = 'docker.io/nearaidev/sglang@sha256:' + '2' * 64
        self.release_path.write_text(other_image + '\n')
        self.validate(False)

    def test_rejects_missing_released_image(self):
        self.candidate_path.write_text(promotion.candidate(self.canonical, FIXTURE_IMAGE))
        self.validate(False)

    def test_rejects_hicache_added_to_canonical(self):
        for target in ('r1', 'r2'):
            with self.subTest(replica=target):
                marker = f'    container_name: model-sg-glm53-fp8-tp4-{target}\n'
                self.assertEqual(self.canonical.count(marker), 1)
                contaminated = self.canonical.replace(
                    marker, marker + '    environment:\n      - SGLANG_HICACHE_POOLED_TRANSFERS=1\n', 1)
                try:
                    self.canonical_path.write_text(contaminated)
                    self.validate(False)
                finally:
                    self.canonical_path.write_text(self.canonical)

    def test_rejects_runtime_drift_in_hicache_file(self):
        self.activate()
        valid = self.candidate_path.read_text()
        # Each mutation targets only the candidate override (or, for the
        # deliberate r1-contamination and proxy/nginx cases, a shared block
        # that reaches r1 or a non-replica service). Every one must fail.
        for before, after in (
            ('--enable-hierarchical-cache', '--enable-hierarchical-cache --hicache-size 64'),
            ('${GLM53_HICACHE_RAM_BUDGET:-80%}', '64GB'),
            ('--hicache-io-backend direct', '--hicache-io-backend kernel'),
            ('SGLANG_HICACHE_POOLED_TRANSFERS=1', 'SGLANG_HICACHE_POOLED_TRANSFERS=0'),
            ('SGLANG_HICACHE_STAGING_PAGES=64', 'SGLANG_HICACHE_STAGING_PAGES=128'),
            ('${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}', '${GLM53_HICACHE_CUDA_HOST_MEMORY:-0}'),
            # Hard-coding regression: a literal value instead of the overridable expression.
            ('SGLANG_HICACHE_CUDA_HOST_MEMORY=${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}', 'SGLANG_HICACHE_CUDA_HOST_MEMORY=1'),
            # A reintroduced SGLANG_HICACHE_CUDA_MANAGED_MEMORY key must fail too.
            ('      - SGLANG_HICACHE_CUDA_HOST_MEMORY=${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}\n',
             '      - SGLANG_HICACHE_CUDA_HOST_MEMORY=${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}\n'
             '      - SGLANG_HICACHE_CUDA_MANAGED_MEMORY=0\n'),
            ('        --chunked-prefill-size 4096', '        --chunked-prefill-size 8192'),
            ('      --kv-cache-dtype bfloat16', '      --enable-hierarchical-cache\n      --kv-cache-dtype bfloat16'),
            (promotion.VARIANT, 'incorrect-variant'),
        ):
            with self.subTest(mutation=after):
                self.assertIn(before, valid)
                try:
                    self.candidate_path.write_text(valid.replace(before, after))
                    self.validate(False)
                finally:
                    self.candidate_path.write_text(valid)

    def test_rejects_proxy_nginx_drift_between_files(self):
        # VLLM_BACKEND_CONVERSATION_AFFINITY is already covered by the common
        # proxy contract check, so it never reaches the cross-file equality
        # check. Mutate the HiCache file's embedded nginx_conf content instead
        # (a single line inside a config nothing else inspects) and confirm
        # the *cross-file* check is the one that fails, not some other rule,
        # and that its message points at the differing path.
        self.activate()
        valid = self.candidate_path.read_text()
        needle = '      client_body_buffer_size 1m;\n'
        self.assertEqual(valid.count(needle), 1)
        mutated = valid.replace(needle, '      client_body_buffer_size 2m;\n', 1)
        self.assertNotEqual(mutated, valid)
        self.candidate_path.write_text(mutated)
        result = self.run_validator()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        output = result.stdout + result.stderr
        self.assertIn('must match the canonical file outside r2', output)
        self.assertIn('first difference:', output)
        self.assertIn('nginx_conf', output)

    def test_rejects_hicache_variant_misplaced_in_canonical_telemetry(self):
        label_needle = f'nearai.otel.config_variant: "{OFFICIAL_VARIANT}"'
        log_needle = f'config_variant:{OFFICIAL_VARIANT}'
        scrape_needle = f'\n                      config_variant: "{OFFICIAL_VARIANT}"'
        self.assertEqual(self.canonical.count(label_needle), 2)
        self.assertEqual(self.canonical.count(log_needle), 2)
        self.assertEqual(self.canonical.count(scrape_needle), 2)

        cases = (
            ('canonical r2 metric label', label_needle, 1,
             f'nearai.otel.config_variant: "{promotion.VARIANT}"'),
            ('canonical r2 log tag', log_needle, 1, f'config_variant:{promotion.VARIANT}'),
            ('canonical r1 scrape label', scrape_needle, 0,
             f'\n                      config_variant: "{promotion.VARIANT}"'),
        )
        for description, needle, index, replacement in cases:
            with self.subTest(case=description):
                mutated = replace_nth(self.canonical, needle, index, replacement)
                self.assertNotEqual(mutated, self.canonical)
                try:
                    self.canonical_path.write_text(mutated)
                    self.validate(False)
                finally:
                    self.canonical_path.write_text(self.canonical)

    def test_rejects_wrong_variant_on_hicache_r1(self):
        self.activate()
        valid = self.candidate_path.read_text()
        needle = f'nearai.otel.config_variant: "{OFFICIAL_VARIANT}"'
        # r1 is the first (only remaining, after r2's own HICACHE_VARIANT) match.
        self.assertEqual(valid.count(needle), 1)
        mutated = valid.replace(needle, f'nearai.otel.config_variant: "{promotion.VARIANT}"', 1)
        try:
            self.candidate_path.write_text(mutated)
            self.validate(False)
        finally:
            self.candidate_path.write_text(valid)

    def test_rejects_missing_r2_scrape_job_in_hicache_file_cleanly(self):
        self.activate()
        valid = self.candidate_path.read_text()
        needle = 'job_name: sglang-model-sg-glm53-fp8-tp4-r2\n'
        self.assertEqual(valid.count(needle), 1)
        mutated = valid.replace(needle, 'job_name: sglang-model-sg-glm53-fp8-tp4-r2-renamed\n', 1)
        self.candidate_path.write_text(mutated)
        result = self.run_validator()
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertIn('missing sglang-model-sg-glm53-fp8-tp4-r2 scrape job', output)
        assert_no_ruby_crash(self, output)

    def test_rejects_malformed_yaml_in_hicache_file_cleanly(self):
        self.activate()
        self.candidate_path.write_text('services: [1, 2\n')
        result = self.run_validator()
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertIn('HiCache file', output)
        self.assertIn('not valid YAML', output)
        assert_no_ruby_crash(self, output)

    def test_dcgm_validator_reports_hicache_image_drift_and_absence(self):
        self.activate()
        # A drifted dcgm-glm53 image in the HiCache file must fail, naming that file.
        valid = self.candidate_path.read_text()
        needle = 'image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:613ab03c11d442fd960ff515f547e9921537454a712d08160bc8f677f89f1c35'
        self.assertEqual(valid.count(needle), 1)
        mutated = valid.replace(needle, 'image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:' + '0' * 64, 1)
        try:
            self.candidate_path.write_text(mutated)
            result = self.run_dcgm_validator()
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 1, output)
            self.assertIn('GLM-5.3-Flash-SGL-TP4-HiCache.yaml', output)
        finally:
            self.candidate_path.write_text(valid)

        # With the HiCache file absent, the DCGM validator skips it and still passes.
        self.candidate_path.unlink()
        self.release_path.unlink()
        result = self.run_dcgm_validator()
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn('GLM-5.3 DCGM telemetry contract skipped', output)
        self.assertIn('GLM-5.3-Flash-SGL-TP4-HiCache.yaml not present', output)

    def test_priority_switches_require_normalizing_proxy(self):
        # The priority switches are only allowed behind a proxy build that
        # overwrites every request's priority. Repoint the shared proxy image
        # at a different digest and confirm the gate names the proxy.
        self.activate()
        needle = 'nearaidev/vllm-proxy-rs@sha256:b3a8c6260834231271b4356c56a7aa2718608c8a537b35973916e0a56dc88fba'
        other = 'nearaidev/vllm-proxy-rs@sha256:' + 'a' * 64
        for path in (self.canonical_path, self.candidate_path):
            text = path.read_text()
            self.assertEqual(text.count(needle), 1, path)
            path.write_text(text.replace(needle, other))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        output = result.stdout + result.stderr
        assert_no_ruby_crash(self, output)
        self.assertIn('not a priority-normalizing inference-proxy build', output)

    def test_rejects_engine_default_priority(self):
        # The proxy assigns priority; an engine-side default must be refused.
        self.activate()
        needle = '      --disable-priority-preemption\n'
        text = self.canonical_path.read_text()
        self.assertEqual(text.count(needle), 1)
        self.canonical_path.write_text(text.replace(needle, needle + '      --default-priority-value 0\n'))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        output = result.stdout + result.stderr
        assert_no_ruby_crash(self, output)
        self.assertIn('must not set --default-priority-value', output)

    def test_rejects_missing_admission_reserve_env(self):
        # Admission-reserve v10 must be enabled on every replica; dropping the
        # opt-in env from the shared block must fail with the REQUIRED_ENV message.
        needle = '    - SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096\n'
        text = self.canonical_path.read_text()
        self.assertEqual(text.count(needle), 1)
        self.canonical_path.write_text(text.replace(needle, ''))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        output = result.stdout + result.stderr
        assert_no_ruby_crash(self, output)
        self.assertIn('must set SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096', output)

    def test_rejects_forbidden_admission_reserve_min_wait(self):
        # SGLANG_ADMISSION_RESERVE_MIN_WAIT_S desyncs the TP ranks and crashes
        # the engine; it must never be set anywhere in the file.
        needle = '    - SGLANG_ADMISSION_RESERVE_MAX_FRACTION=0.75\n'
        text = self.canonical_path.read_text()
        self.assertEqual(text.count(needle), 1)
        self.canonical_path.write_text(
            text.replace(needle, needle + '    - SGLANG_ADMISSION_RESERVE_MIN_WAIT_S=1\n'))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        output = result.stdout + result.stderr
        assert_no_ruby_crash(self, output)
        self.assertIn('must not set SGLANG_ADMISSION_RESERVE_MIN_WAIT_S', output)


class LongContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in (promotion.COMPOSE, promotion.CANDIDATE, promotion.RELEASE, LONG_CONTEXT,
                     Path('scripts/validate_glm53_prod_config.rb')):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.root / name)
        self.long_context_path = self.root / LONG_CONTEXT

    def run_validator(self):
        return subprocess.run(['ruby', str(self.root / 'scripts/validate_glm53_prod_config.rb')],
                              capture_output=True, text=True)

    def validate(self, success=True):
        result = self.run_validator()
        self.assertEqual(result.returncode, 0 if success else 1, result.stdout + result.stderr)
        return result

    def test_accepts_committed_long_context_contract(self):
        result = self.validate()
        self.assertIn('GLM-5.3-Flash-SGL-TP4-LongContext.yaml', result.stdout)

    def test_rejects_missing_long_context_file(self):
        self.long_context_path.unlink()
        result = self.validate(False)
        self.assertIn('long-context file not found', result.stdout + result.stderr)

    def test_rejects_hicache_on_long_context_r1(self):
        valid = self.long_context_path.read_text()
        marker = '    container_name: model-sg-glm53-fp8-tp4-r1\n'
        self.assertEqual(valid.count(marker), 1)
        mutated = valid.replace(
            marker, marker + '    environment:\n      - SGLANG_HICACHE_POOLED_TRANSFERS=1\n', 1)
        self.long_context_path.write_text(mutated)
        result = self.validate(False)
        self.assertIn('long-context r1 must remain the HiCache-disabled control',
                      result.stdout + result.stderr)

    def test_rejects_long_context_r2_contract_drift(self):
        valid = self.long_context_path.read_text()
        released_image = (ROOT / promotion.RELEASE).read_text().strip()
        cases = (
            (released_image, OTHER_FIXTURE_IMAGE, 'long-context r2 image must be RELEASED_IMAGE'),
            ('--hicache-io-backend direct', '--hicache-io-backend kernel',
             'long-context r2 must set --hicache-io-backend direct exactly once'),
            ('${GLM53_HICACHE_RAM_BUDGET:-256GiB}', '${GLM53_HICACHE_RAM_BUDGET:-80%}',
             'SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-256GiB}'),
            ('SGLANG_HICACHE_STAGING_PAGES=64', 'SGLANG_HICACHE_STAGING_PAGES=128',
             'SGLANG_HICACHE_STAGING_PAGES=64'),
        )
        for before, after, message in cases:
            with self.subTest(mutation=after):
                self.assertIn(before, valid)
                self.long_context_path.write_text(valid.replace(before, after, 1))
                result = self.validate(False)
                self.assertIn(message, result.stdout + result.stderr)
        self.long_context_path.write_text(valid)

    def test_rejects_long_context_non_hicache_runtime_drift(self):
        valid = self.long_context_path.read_text()
        needle = '        --chunked-prefill-size 4096'
        self.assertEqual(valid.count(needle), 1)
        mutated = valid.replace(needle, '        --chunked-prefill-size 2048', 1)
        self.long_context_path.write_text(mutated)
        result = self.validate(False)
        self.assertIn('long-context r2 must preserve all control serving arguments outside HiCache',
                      result.stdout + result.stderr)

    def test_rejects_admission_reserve_on_long_context_replicas(self):
        valid = self.long_context_path.read_text()
        marker = '    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n'
        self.assertEqual(valid.count(marker), 2)
        mutated = replace_nth(
            valid,
            marker,
            0,
            marker + '    - SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096\n'
                     '    - SGLANG_ADMISSION_RESERVE_MAX_FRACTION=0.75\n',
        )
        self.long_context_path.write_text(mutated)
        result = self.validate(False)
        self.assertIn('long-context replicas must not set admission-reserve environment',
                      result.stdout + result.stderr)

    def test_rejects_long_context_telemetry_variant_drift(self):
        valid = self.long_context_path.read_text()
        for expected in (LONG_CONTEXT_CONTROL_VARIANT, LONG_CONTEXT_HICACHE_VARIANT):
            with self.subTest(variant=expected):
                self.assertEqual(valid.count(expected), 3)
                self.long_context_path.write_text(valid.replace(expected, 'incorrect-variant'))
                result = self.validate(False)
                self.assertIn('config_variant must be', result.stdout + result.stderr)
        self.long_context_path.write_text(valid)

    def test_rejects_missing_long_context_collector_content(self):
        valid = self.long_context_path.read_text()
        marker = '  otelcol_app_config:\n    content: |\n'
        self.assertEqual(valid.count(marker), 1)
        self.long_context_path.write_text(
            valid.replace(marker, '  otelcol_app_config: {}\n  unused_app_config:\n    content: |\n', 1))
        result = self.validate(False)
        self.assertIn('long-context file otelcol_app_config is missing',
                      result.stdout + result.stderr)

    def test_rejects_renamed_or_conflicting_log_variant_tags(self):
        valid = self.long_context_path.read_text()
        exact = f'config_variant:{LONG_CONTEXT_CONTROL_VARIANT}'
        cases = (
            (exact, f'bogus_{exact}'),
            (exact, exact + '","config_variant:conflicting-variant'),
        )
        for before, after in cases:
            with self.subTest(mutation=after):
                self.long_context_path.write_text(valid.replace(before, after, 1))
                result = self.validate(False)
                self.assertIn('log metadata must carry exactly config_variant:',
                              result.stdout + result.stderr)
        self.long_context_path.write_text(valid)


if __name__ == '__main__':
    unittest.main()
