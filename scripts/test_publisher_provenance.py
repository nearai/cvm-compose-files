"""Exercise the publishers' actual jq policies, without builds or registry writes.

The synthetic predicates use the two documented BuildKit SLSA layouts:
https://github.com/moby/buildkit/blob/master/docs/attestations/slsa-provenance.md
No provenance-normalization code is duplicated here: the jq program and its
arguments are extracted from each committed workflow and executed unchanged.
Requires Python 3, Ruby (stdlib YAML/JSON), and jq.
"""
import json
from pathlib import Path
import re
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1" * 40
REPOSITORY = "https://github.com/nearai/cvm-compose-files"
LEGACY_TYPE = "https://mobyproject.org/buildkit@v1"
V1_TYPE = "https://github.com/moby/buildkit/blob/master/docs/attestations/slsa-definitions.md"
LEGACY_METADATA = LEGACY_TYPE + "#metadata"
WORKFLOWS = (
    "publish-dsv4-0731.yaml",
    "publish-glm53-admission-reserve.yaml",
    "publish-glm53-fc91d24.yaml",
    "publish-glm53-hicache.yaml",
    "publish-glm53-v0519.yaml",
    "publish-glm53-v0523.yaml",
    "publish-vllm-dsv41-lora.yaml",
    "publish-vllm-dsv41-v033.yaml",
)


def load_cases():
    """Use the same Ruby YAML parser already used by repository tests."""
    cases = []
    for filename in WORKFLOWS:
        path = ROOT / ".github/workflows" / filename
        workflow = json.loads(subprocess.check_output([
            "ruby", "-ryaml", "-rjson", "-e",
            "puts JSON.generate(YAML.load_file(ARGV[0], aliases: true))", str(path),
        ], text=True))
        steps = workflow["jobs"]["publish"]["steps"]
        script = next(step["run"] for step in steps if
                      step.get("name", "").startswith("Verify embedded BuildKit provenance"))
        tokens = shlex.split(script.replace("\\\n", " "), comments=True)
        start = tokens.index("jq")
        end = tokens.index("buildkit-provenance.json", start)
        command = tokens[start:end]
        # The final quoted argument is the production jq filter. Nothing from
        # the shell script is sourced or executed by this test.
        program = command[-1]
        context = workflow["env"].get("BUILD_CONTEXT")
        if context:
            contexts = [context]
        else:
            selection = "\n".join(step.get("run", "") for step in steps)
            contexts = re.findall(r'echo "BUILD_CONTEXT=([^"\n]+)"', selection)
        if not contexts:
            raise AssertionError(f"No build contexts found in {filename}")
        for context in contexts:
            dockerfile = (ROOT / context / "Dockerfile").read_text()
            base = re.search(r"(?m)^FROM (\S+)@sha256:([a-f0-9]{64})(?:\s|$)", dockerfile)
            if not base:
                raise AssertionError(f"No digest-pinned FROM in {context}")
            repository, digest = base.groups()
            repository = repository.removeprefix("docker.io/")
            substitutions = {
                "$SOURCE_REVISION": REVISION,
                "$BASE_DIGEST_HEX": digest,
                "$BUILD_CONTEXT": context,
            }
            args = []
            for token in command[1:]:
                if token.startswith("$"):
                    token = substitutions[token]
                args.append(token)
            cases.append({
                "name": filename + ":" + context,
                "context": context,
                "base_repository": repository,
                "base_digest": digest,
                "args": args,
                "program": program,
            })
    return cases


def predicate(case, version):
    """Minimal synthetic data in the documented layouts, with pinned policy inputs."""
    vcs = {
        "source": REPOSITORY,
        "revision": REVISION,
        "localdir:context": case["context"],
        "localdir:dockerfile": case["context"],
    }
    materials = [{
        "uri": "pkg:docker/" + case["base_repository"] + "?platform=linux%2Famd64",
        "digest": {"sha256": case["base_digest"]},
    }]
    if version == "v0.2":
        return {
            "buildType": LEGACY_TYPE,
            "metadata": {
                "buildStartedOn": "2026-11-01T00:00:00Z",
                "buildFinishedOn": "2026-11-01T00:00:01Z",
                LEGACY_METADATA: {"vcs": vcs},
            },
            "materials": materials,
        }
    return {
        "buildDefinition": {"buildType": V1_TYPE, "resolvedDependencies": materials},
        "runDetails": {"metadata": {
            "startedOn": "2026-11-01T00:00:00Z",
            "finishedOn": "2026-11-01T00:00:01Z",
            "buildkit_metadata": {"vcs": vcs},
        }},
    }


def policy_fields(data, version):
    if version == "v0.2":
        metadata = data["metadata"]
        return metadata[LEGACY_METADATA]["vcs"], data["materials"], metadata, \
            "buildStartedOn", "buildFinishedOn"
    metadata = data["runDetails"]["metadata"]
    return metadata["buildkit_metadata"]["vcs"], \
        data["buildDefinition"]["resolvedDependencies"], metadata, "startedOn", "finishedOn"


class PublisherProvenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def check(self, case, data, accepted):
        result = subprocess.run(["jq", *case["args"]], input=json.dumps(data),
                                capture_output=True, text=True)
        if accepted:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), "true")
        else:
            self.assertNotEqual(result.returncode, 0, "Invalid predicate was accepted")

    def test_both_supported_layouts_for_every_publisher_and_variant(self):
        for case in self.cases:
            for version in ("v0.2", "v1"):
                with self.subTest(publisher=case["name"], version=version):
                    self.check(case, predicate(case, version), True)

    def test_existing_policy_rejects_wrong_provenance(self):
        for case in self.cases:
            for version in ("v0.2", "v1"):
                for mutation in ("source", "revision", "context", "base_digest",
                                 "base_repository", "started", "finished", "empty_materials"):
                    data = predicate(case, version)
                    vcs, materials, metadata, started, finished = policy_fields(data, version)
                    if mutation == "source":
                        vcs["source"] = "https://github.com/example/untrusted"
                    elif mutation == "revision":
                        vcs["revision"] = "2" * 40
                    elif mutation == "context":
                        vcs["localdir:context"] = "docker/wrong-context"
                    elif mutation == "base_digest":
                        materials[0]["digest"]["sha256"] = "0" * 64
                    elif mutation == "base_repository":
                        materials[0]["uri"] = "pkg:docker/example/untrusted?platform=linux%2Famd64"
                    elif mutation == "started":
                        del metadata[started]
                    elif mutation == "finished":
                        del metadata[finished]
                    else:
                        materials.clear()
                    with self.subTest(publisher=case["name"], version=version, mutation=mutation):
                        self.check(case, data, False)

    def test_existing_dockerfile_and_platform_requirements_remain_enforced(self):
        for case in self.cases:
            for version in ("v0.2", "v1"):
                for field in ("localdir:dockerfile", "platform=linux%2Famd64"):
                    # These assertions exist only in particular publishers;
                    # the migration must preserve, rather than invent, policy.
                    if field not in case["program"]:
                        continue
                    data = predicate(case, version)
                    vcs, materials, *_ = policy_fields(data, version)
                    if field == "localdir:dockerfile":
                        vcs[field] = "docker/wrong-dockerfile"
                    else:
                        materials[0]["uri"] = materials[0]["uri"].replace("amd64", "arm64")
                    with self.subTest(publisher=case["name"], version=version, field=field):
                        self.check(case, data, False)

    def test_unknown_mixed_and_malformed_layouts_fail_closed(self):
        for case in self.cases:
            malformed = [None, [], "not provenance", {}, 1, True]
            old = predicate(case, "v0.2")
            old["buildType"] = "https://example.com/unknown-build-type"
            malformed.append(old)
            new = predicate(case, "v1")
            new["buildDefinition"]["buildType"] = "https://example.com/unknown-build-type"
            malformed.append(new)
            for legacy_key in ("buildType", "metadata", "materials"):
                new = predicate(case, "v1")
                new[legacy_key] = None
                malformed.append(new)
            for v1_key in ("buildDefinition", "runDetails"):
                old = predicate(case, "v0.2")
                old[v1_key] = None
                malformed.append(old)
            for value in (None, {}, "not an array"):
                new = predicate(case, "v1")
                new["buildDefinition"]["resolvedDependencies"] = value
                malformed.append(new)
            for value in (None, [], "not metadata"):
                new = predicate(case, "v1")
                new["runDetails"]["metadata"] = value
                malformed.append(new)
            for data in malformed:
                with self.subTest(publisher=case["name"], data=data):
                    self.check(case, data, False)


if __name__ == "__main__":
    unittest.main()
