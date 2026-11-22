# Status

- Phase: 3 (fault ladder and evaluation) done on the branch `phase-3-ladder`;
  waits on the owner's review.
- Last gate: Phase 3 passed. G1, G2, G4 and G6 pass for the Phase 2 policy
  (`results/gate.md`), and the ladder tests run in CI.
- Phase 3 results (ladder v0, 4,096 episodes per cell, L4): default PPO scores
  0.650 (fault 0.665, shift 0.629, nominal 0.999); zero and random agents
  score 0.000; rescoring with another seed differs by 0.0003.
- CPU transfer check: 256 nominal episodes on the Mac CPU give 0.996 success
  (`results/cpu_transfer/ladder.json`) against 0.999 on the L4. Both use
  MuJoCo Warp, so this compares devices, not a second simulator.
- Changes from the plan in Phase 3:
  - Each episode holds one command, with a commanded speed of at least
    0.4 m/s. With the plan's resampled commands a standing robot met the
    tolerance on about 9% of episodes.
  - The eval task removes the base task's own pushes, friction, encoder-bias
    and centre-of-mass randomisation, so a cell changes one thing.
  - Calibration round 1 of 3: payload levels 0.1 to 0.5 became 0.12 to 0.6,
    because level 5 scored 0.73 (G4 needs at most 0.5).
- Known weak spots of ladder v0, left as they are since G4 passes: latency
  drops from 0.96 at level 1 to 0.12 at level 2, and push from 0.99 at level 2
  to 0.00 at level 3 (robots mostly stay up but miss the speed tolerance).
  Levels 3 to 5 of those two axes may not separate baselines; revisit with
  the Phase 4 policies.
- Spend: Phase 3 about $1.84 of the $25 cap; project total about $3.36
  (`results/spend.csv`, list prices times job time).
- Modal account: usage limit $130, spend limit $0, so only credits are used.
- Next step: Phase 4 baselines, after the owner reviews and merges Phase 3.
- Not done: Modal agent skill, `commit-commands` plugin, Context7
  authorisation.
