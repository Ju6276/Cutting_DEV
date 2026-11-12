# Plan

Updated 2026-11-12 (free-object task). The only current task list. Research scope:
[PAPER](PAPER.md); measurements and history: [RESULTS](RESULTS.md); implementation:
[METHOD](METHOD.md); task proposal: [TASK_FREE_CUT](TASK_FREE_CUT.md); assets: [DATA](DATA.md).
Earlier plans are in `archive/` (latest: [before the task decision](archive/PLAN_20261112_pre_review.md)).

## Current task (user decisions)

- Method paper for ICRA/IROS/RA-L; platform, objects and baselines are experiments. The
  paper's core must follow from the task, not from a failure seen in our own runs.
- **Task:** single cut of an object placed freely on the board. No fixture. Passive sliding
  is recorded, has no threshold and no penalty; active rolling and repositioning are out of
  scope. Sawing is allowed; neither learned sawing nor an advantage over pure pressing is presumed.
- **Success:** the slice is fully separated from the body (no remaining material connection)
  **and** no force-limit event occurred. Separation, force-limit events and task success are
  recorded separately. A remaining connection is incomplete; 98 % release is a diagnostic.
- **Failure endings:** force limit, explicit workspace bounds; a cut not completed within
  the time limit is incomplete (its handling in training is specified in TASK_FREE_CUT). "The object left the
  blade's reach" is not a termination rule.
- One shared WM/policy/critic across objects and materials; one force limit; no object
  identity, material label or per-object reward/success setting reaches the learner.
  The per-component six-axis sensor limit is kept as the interface in use; force norm and
  torque are recorded. It is a simulation convention, not a Mini45-derived safety limit.
- Simulation first, sim-to-real intended; real robot later (materials calibrated then).
- One change at a time; a written plan with a budget is approved before training.

## State

- Platform check of the free-object task on one scene: done (RESULTS). Severed-state check
  defined for it (mesh has two components joined only by cut springs).
- Reset from a settled object: implemented (`settle()` with each world's material; contact
  checked over every substep), verified on that scene.
- Net contact wrench feedback and friction transition velocity: implemented, repaired and
  re-measured (RESULTS); friction and ground parameters are uncalibrated.
- free_cut_v1 implementation and GPU/grouped-reset/gradient checks are complete.
  The first explicit reward weights FAILED the speed-preference check (fast press
  93.33 vs commanded press 72.77). This is not an approved training configuration.
- No training is running. Old S0 configs and held scenes are reproduction inputs only.

## Next steps, in order

1. Claude reviews [FREE_CUT_REVIEW](FREE_CUT_REVIEW.md), code and the four completed
   physical checks. Implementation authorized by the user; no long training launched.
2. Set explicit reward weights consistent with commanded-speed preference. The first
   candidate is rejected, so the remaining retract/slow trials were stopped rather
   than spending the full 1200 s budget. No physics or limits are tuned to help it pass.
   Recheck the same reference cases with revised weights before a training plan.
3. Once accepted, submit a bounded end-to-end engineering training plan. Newly fitted
   WM multi-step progress/completion and return ordering remain unverified until data
   and fitting exist. This is not a paper performance claim.
4. New exported objects get the same topology check during asset preparation.

## Known limits of the platform (not a repair queue)

- The cut plane is fixed in the object and the knife is constrained to one plane; this bounds
  what sliding across the cut can mean in this simulator. Its quantitative effect is not
  measured (an earlier "1 mm offset" explanation was withdrawn, RESULTS).
- Results depend on initialization: every world is settled with its own material before
  episodes (`settle()`); a nominal-material settle followed by a material change is not used.
- Free-cucumber forces with 2 ms sampling: peaks 2.7-23 N, utilisation 0.06-0.64 of the old
  S0 limits over six reference cases (RESULTS). No scene, free or held, is validated against
  this project's real cuts; parameters are not tuned to make the task harder. Whether the
  paper can centre on the force limit is decided with real measurements, not in simulation.
- Sawing changes force direction through friction only; the fracture-side effect is not in
  DiSECt's damage law. Long-horizon per-spring damage repeatability is an open item.

## Storage and execution

Use Remote SSH on GPU7, /dataHDD/dong/Projects/Cutting_DEV, main; the A6000 holds a second
checkout. Code, data, environments and outputs stay on servers, not the Mac. Check
`nvidia-smi` before using a GPU. Not to be launched without a new approved plan:
`train_gpu.py --conditions configs/multi_condition/s0_*.json` (with or without
`--risk-constraint`), `train_official_supermimic.py`, `train_sac_gpu.py` on S0.
