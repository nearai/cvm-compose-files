---
paths:
  - "ci/**"
---

# ci/ guards and data

Python 3 standard library only. After every change here run `python3 ci/run.py --self-test` and
`python3 ci/run.py --base origin/main`.

## Guards
- One guard per file in `ci/checks/`, exporting `RULES`, `DIFF_RULES`, `CASES` and `check(ctx)`.
  Register it in `GUARDS` in `ci/run.py` and add every new `ci/` file to `FIXED` in `ci/checks/layout.py`.
- Rule ids are `<guard>.<name>`. Every rule needs a red case that fires only that rule, every guard a
  green case, and a diff rule a green case without base data. The self-test fails otherwise.
- A finding's `key` is the matched text an exception hashes. Messages never print it; the self-test
  fails if the fixture sentinel shows up in output.
- Fixtures that hold strings the public guard bans are assembled at run time, so the guard file never
  contains them.
- Read YAML with `parse_yaml` from `ci/lib/common.py`. It rejects anchors, tags and other constructs it
  cannot judge; turn that into a finding, as `layout.yaml` does, never a crash.
- A change that alters findings on the real tree comes with `--write-expected`; the diff of
  `ci/data/expected.json` is part of the review.

## Data files
- Paths in `ci/data/legacy.txt` and `ci/data/exceptions.json` are exact, no globs or comments.
  `legacy.txt` is sorted and unique.
- `ci/data/public.json` lists only hosts, owners and repos that are public already. An addition widens
  what every file may contain: list it under Gates in the PR and review it like a gate.
