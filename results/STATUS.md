# Status

- Phase: 2 (Modal baseline) run; its gate waits on the owner watching the video
- Last gate: Phase 1 passed, see `gate` in `results/reacher.json`.
- Phase 2 results: L4 has the most steps per dollar
  (`results/gpu_benchmark.json`). Go1 seed 0 trained 3,000 iterations on an L4
  in 49.5 minutes. Reward levels off at iteration 1,100, so later runs use
  1,320 iterations (`results/go1_baseline_curve.json`), about $0.45 each.
  Estimated Phase 2 spend $1.52 of the $10 cap (`results/spend.csv`); costs are
  list prices times job time, not read from the Modal invoice.
- Changes from the plan in Phase 2: the first run was 3,000 iterations, not
  mjlab's default 10,000; the plateau rule is applied to a 100-iteration
  trailing mean because single iterations are too noisy for a 2% band.
- Modal account: GPU jobs need a payment method on file. The owner set the
  usage limit to $130 and the spend limit to $0, so only credits are used.
  The workspace had $61.35 of other usage this billing period, which counts
  against the same $130 limit.
- Next step: Phase 3, after the owner confirms the robot walks and follows
  commands in the video and raises the usage limit or the period rolls over.
- Not done: Modal agent skill, `commit-commands` plugin, Context7
  authorisation, `eval-policy` skill (written in Phase 3).
