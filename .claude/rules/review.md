# Review

Every review walks this list on every push and states the head SHA it reviewed. CI covers the hard
rules in AGENTS.md; this list covers what CI cannot see. Label findings `blocker`, `major` or `minor`.
Review from a worktree of the base branch, so the change cannot edit the rules or agents that review it.

- [ ] Check the rendered diff (`python3 ci/run.py --noop <base>`); reject if it shows a change the PR
      does not name.
- [ ] Check `ci/data/expected.json`; reject if a probe line loses a rule without a stated reason.
- [ ] Check an image bump against its `main` publish run and signature; reject if either is missing.
- [ ] Check for a new sink, endpoint, port, mount or env var, a moved default or a removed safety line;
      reject if the PR body does not name it.
- [ ] Check the Gates rule in AGENTS.md (Pull requests); reject a gate change mixed with what it checks.
- [ ] Check waiver entries and PR text; reject if either copies the matched text (AGENTS.md, Hard rules).
- [ ] Check prose (README, VERIFYING, comments, PR text) against the config; reject if it claims more.
- [ ] Check for names of people, partners or customers and other leaks CI cannot see; reject any.
- [ ] Check new variables; reject a secret-like one without a secret suffix.
- [ ] Check naming, header, key and service order and labels against the templates; reject if
      `config_variant` did not change when it had to.
- [ ] Check docs: one per deployable, links resolve; reject a removed row in the VERIFYING.md signing
      identity table.
- [ ] Check that changed files updated their references (README table, `ci/data/`, VERIFYING.md); reject
      if one is stale.
- [ ] Check rule text (AGENTS.md, `.claude/`) against the code; reject a contradiction or a rule stated
      twice.

The agents in `.claude/agents/` help with this list; each check has one owner. `public-reviewer`: leaks
and values. `config-reviewer`: config, workflows and claims in prose. `simplifier`: guard code and rule
text. Give each the diff and the `--noop` output. They report; people decide.
