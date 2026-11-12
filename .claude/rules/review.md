# Review

Every review walks this list on every push and states the head SHA it reviewed. CI covers the hard
rules in AGENTS.md; this list covers what CI cannot see. Label findings `blocker`, `major` or `minor`.

- [ ] Read the rendered diff (`python3 ci/run.py --noop <base>`), not only the YAML.
- [ ] `expected.json`: a probe line that loses a rule means a weaker guard; reject unless justified.
- [ ] Image bump: digest from a `main` publish run, signature verifies, no tag-only refs.
- [ ] New sink, endpoint, port, mount or env var; moved default; removed safety line.
- [ ] Gate edited together with the thing it gates; waiver text in the PR, not `ci/data/exceptions.json`.
- [ ] Prose (README, VERIFYING, comments, PR text) says the same as the config and claims no more.
- [ ] Names of people, partners or customers and other leaks CI cannot see.
- [ ] New secret-like variable without a secret suffix.
- [ ] Naming, header, key and service order, labels follow the templates; `config_variant` changed if due.
- [ ] Docs: one per deployable; links resolve; the signing-identity table in VERIFYING.md only gains rows.
- [ ] Changed files updated their references: README table, `ci/data/` (services, images, legacy),
      VERIFYING.md identities and known gaps.
- [ ] Rule text (AGENTS.md, `.claude/`) agrees with the code and the hard rules and states each rule once.

The agents in `.claude/agents/` help with this list: `public-reviewer` for leaks, `config-reviewer`
for config and workflows, `simplifier` for guard code and rule text. They report; people decide.
