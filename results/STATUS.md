# Status

- Phase: 1 (first environment) complete; Phase 2 (Modal baseline) not started
- Last gate: Phase 1 passed, see `gate` in `results/reacher.json`. Phase 0
  passed: CI green on `main`, zero and random Go1 rollouts ran on the Mac.
- Next step: Phase 2. Before any GPU job the owner must set the $130 workspace
  budget in the Modal dashboard, and the `modal` CLI must be on the PATH.
- Not done from Phase 0: Modal agent skill, `commit-commands` plugin, Context7
  authorisation, `modal-train` and `eval-policy` skills (written in their phases).
