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
  scope. Sawing is part of the task (a pure press is not enough to show a method).
- **Success:** the slice is fully separated from the body (no remaining material connection)
  **and** no force-limit event occurred. Separation, force-limit events and task success are
  recorded separately. A remaining connection is incomplete; 98 % release is a diagnostic.
- **Failure endings:** force limit, timeout, explicit workspace bounds. "The object left the
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
- Reset from a settled object: implemented (`settle_seconds`), verified on that scene.
- Net contact wrench feedback and friction transition velocity: implemented, repaired and
  re-measured (RESULTS); friction and ground parameters are uncalibrated.
- No training is running. Old S0 configs and held scenes are reproduction inputs only.

## Next steps, in order

1. **Reward and its WM implementation** — proposal in [TASK_FREE_CUT](TASK_FREE_CUT.md),
   awaiting the user's approval. No task code changes before that.
2. After approval: implement the task version, run the fixed-trajectory reward preference
   check stated there, then submit a training plan with a budget.
3. New exported objects get the same topology check (two components joined only by cut
   springs) as part of asset preparation.

## Known limits of the platform (not a repair queue)

- The cut plane is fixed in the object and the knife is constrained to one plane. On the
  free cucumber a sideways offset of about 1 mm between them changed the peak force from
  about 2 N to 16 N and left the cut incomplete (RESULTS). Sliding across the cut plane is
  therefore not represented faithfully; sliding along the blade is.
- From a settled start the free cucumber is separated by a plain press at 1.5-2.5 N for
  Young x1 and x2: this scene alone poses no force difficulty.
- Sawing changes force direction through friction only; the fracture-side effect is not in
  DiSECt's damage law. Long-horizon per-spring damage repeatability is an open item.

## Storage and execution

Use Remote SSH on GPU7, /dataHDD/dong/Projects/Cutting_DEV, main; the A6000 holds a second
checkout. Code, data, environments and outputs stay on servers, not the Mac. Check
`nvidia-smi` before using a GPU. Not to be launched without a new approved plan:
`train_gpu.py --conditions configs/multi_condition/s0_*.json` (with or without
`--risk-constraint`), `train_official_supermimic.py`, `train_sac_gpu.py` on S0.
