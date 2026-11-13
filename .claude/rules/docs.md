---
paths:
  - "docs/**"
  - ".github/**"
  - "prod/**"
  - "docker/**"
  - "ci/data/**"
---

# Public text, docs and SECURITY.md

Applies to PR titles and bodies, commit messages, branch names, comments, docs, READMEs under
`docker/`, `ci/data/` reasons, and replies in reviews and issues.

## Wording
Describe the config as it is after the change. Invented examples:
- State, not delta. Pass: `prod: require setting X for service A`. Fail: a title that says what was
  missing before.
- No misuse or class names, no advisory identifiers. Name a patch by what it does. Pass: `patch 3: accept
  only values of kind K`. Fail: a title that names what the patch blocks.
- No discovery or history: who reported, how it was found, when it was seen. Pass: `Endpoint /foo: 1 s
  timeout.` Fail: a note on when or by whom a timeout was noticed.
- A why-comment names the property a setting gives, not the failure it prevents. Pass: `# Service A has its
  own queue, so its latency does not depend on load.` Fail: a comment about what happens without it.
- No impact rating, no list of hosts, replicas, files or users, no word on whether a weakness remains
  or when it is fixed. Pass: `docs: describe setting X`. Fail: `rollout of the fix in a later PR`.
- An upstream commit or public PR may be cited as the source of a patch, described by what it does. Pass:
  `Backport of upstream <sha>: validate input Y.` Fail: a note on its security status.
- A reply about a weakness cites `<agent>:<check>` and `path:line` only; never quote, restate or ask
  about it in public. Other review feedback on config may explain. Pass:
  `public-reviewer:7 | <path>:12 | state current behaviour`.
- Do not reword a sentence only to pass CI; remove what it says.

## Known gaps
- List limits by design and waiver classes only; a defect is fixed privately and is not listed.
- Each item says what a verifier cannot check and what to do instead.
- Name rule IDs and counts; no file, workflow, image, host or service lists.
- Order by rule ID or verification step; no impact order.
- Say what a guard checks; never list the cases it skips.
- No cause, trigger or history of a limit.

## Private fix path
Use it when the change fixes a weakness that `main` has, someone without control of the deploying host
could use it, and the diff or its description would point at it. Hardening with no known misuse, and
changes to files never deployed, use a normal public PR.
1. An author who is not a maintainer reports it through .github/SECURITY.md. A maintainer opens a draft
   GitHub security advisory, starts its temporary private fork and adds the author.
2. Every commit message, code comment and diff written in the fork follows the wording rules above from
   the first commit, because the diff and its comments become public.
3. Actions do not run in the fork: run `python3 ci/run.py --self-test` and `--base origin/main` locally
   and put the output in the private PR. Approvals in the fork are maintainer reviews; CI and the
   merge approvals run on the public PR. `main` changes only through the public PR; never use the
   advisory's own merge button, which bypasses branch protection and required checks.
4. When the fix is ready to deploy, a maintainer squashes it into one commit on a public branch with a
   neutral name, authors that commit, and opens a public PR described by what the config does now.
   The reporter is not named and the fork's commit messages are not published.
5. A fix that needs a new image follows the normal new-build-dir flow on the public PR; a merged build
   dir or publish workflow is not edited.
6. Publish the advisory only when people who copy these files have to act; the maintainers decide.
