# Current plan

The single current task list. [PAPER](PAPER.md) holds the research question and
algorithm; [METHOD](METHOD.md) maps implementation and sources;
[BENCHMARK](BENCHMARK.md) defines the task and metrics; [RESULTS](RESULTS.md)
holds measurements. Updated 2026-11-10 using saved evaluations; evidence is packaged in EVIDENCE.

## Current decision — use the trained SAC; no new training now

SAC has already been trained and evaluated in the unified environment. Keep the
step-85000 checkpoint as the existing model-free controller for the fixed-task
comparison. Retraining SAC, increasing its environment count or changing its
learning rate is not a prerequisite for interpreting these results or preparing
the paper's argument. The previous parallel-SAC / lr 3e-4 proposal is deferred;
it is not an approved next job.

Name this controller **SliceIt SAC adapted to the unified environment, 85k**.
It retains the source SAC implementation and listed training settings, but uses
our MuJoCo Warp/public-prism environment and integration patches. It is neither
the separate Gazebo run nor a full reproduction of the SliceIt paper's
calibrated-material and robot experiments.

The current comparison describes named, frozen controllers in one common task.
Their training data budgets and information interfaces differ. Report those
differences and checkpoint selection; do not infer algorithm-wide superiority
or sample efficiency. A single environment is not, by itself, a reason to
discard or retrain an already trained SAC.

## Research question and working algorithm

How can a learned robot/material interaction model support policy optimization
for force-limited compliant cutting? The connected method is:

physical interaction data → motion/force world model and physical-trajectory
value learning → short imagined rollouts with behavior regularization →
BPTT policy updates → executed control.

Keep the working algorithm: `real` critic, actor horizon 8, BC 0.25, continued
from policy62/WM562 with a fresh critic and 100 pre-fit rounds. Deployment uses
the policy and compliance controller. The specific values are implementation
settings; the algorithm is the connected learning procedure, not three numbers.

A (isolating terminal BCE from dynamics fitting) stays a completed mechanism
variant, not the selected main method. Its pre-stated final-success comparison
did not detect a clear benefit; its prediction, force and training-stability
observations remain in RESULTS. No extra A seed pair or B/C redesign is planned.

## Fixed scope and existing controllers

- Task: free-bottom `public-prism-ur5e-v1`, unchanged physics, rewards, bounds,
  force-limit and goal-pose predicates. Goal completion is not a test of full
  material separation. Definitions live only in BENCHMARK.
- WM + BPTT: `critic_abl_a250_20261106` final,
  `runs/sm_ckpts/abl_a250_last.pt`; policy62/WM562 is also shown as the starting
  controller. Report the later same-configuration run's temporary collapse,
  rather than describing the configuration as uniformly stable.
- SAC: `runs/sac_frozen_ckpt85000_20261107`, selected after observing later
  training collapse. Training used one environment, lr 0.01 and one 24-D
  observation (including its source force-history fields); the WM policy uses
  ten frames plus the goal. The 32 worlds used for frozen SAC evaluation do not
  turn its training into parallel training.
- Same-WM MPPI: the working baseline's own checkpoint, verified with
  `--paired-checkpoint`; H16, 512 samples, four refinements, temperature 0.1,
  noise 0.5, gamma 0.99. Critic-enabled and critic-disabled are two separate
  controllers. No further planner search is scheduled.
- Fixed controls: zero action through the common action mapper, and a prescribed
  reference trajectory through its separate reference branch.
- Multi-cut tasks, new simulators, visual policies and DiSECt direct trajectory
  optimization remain outside the current work.

## Completed comparison and allowable conclusions

Full numbers and output directories: RESULTS, 2026-11-09 main-comparison entries.

- Unperturbed: working policy completes in 19 steps; reference completes in 40.
  Constant-zero reached only 323 steps before an 840 s computation budget ended.
  It is an unfinished prefix, not a completed failure or proof it cannot finish.
- Paired working-baseline WM + MPPI: each terminal-value configuration completes
  one of three planning seeds and hits the force limit in two. Planning takes
  about 0.5 s per action, versus about 1 ms policy inference; physics time is
  separate. The critic's out-of-range candidate estimates remain disclosed.
  This is evidence about those fixed implementations, not MPC in general.
- Executed-action perturbation: 256 predefined cases per controller, all complete.
  Success is 74.6% for policy62, 83.2% for the working policy and 89.1% for SAC85k.
  Continuing the whole learning procedure improved the selected policy over its
  starting checkpoint; this does not isolate the WM gradient from extra data,
  critic changes or behavior regularization.
- Working policy vs SAC85k: success difference -5.9 percentage points
  [-10.5, -1.2]; mean material peak difference -15.7 N [-17.3, -14.0].
  Force-limit ending and total impulse differences are inconclusive. Lower
  material peak is not evidence of fewer force-limit failures.
- The logged `safety_margin` is a sensor limit-utilization ratio: larger means
  closer to/exceeding a limit. Material force is a different signal. Do not
  compare the material peak directly with the sensor's 50 N component limit.
- All-case metrics and the subset where both controllers succeed are reported
  separately. Case-paired intervals describe these checkpoints on this
  perturbation distribution, not training-seed variation.
- MPPI disturbance robustness is unmeasured. No new 256-case MPPI batch is
  scheduled. The old A comparison used raw-output perturbations and remains
  separate from the new executed-action rule.

Historical protocols are retained in Git at `3f65d23` (A pair) and `d5a876e`
(main evaluation plan); their completed results remain in RESULTS. They are not
pending tasks.

## Completed 2026-11-10 — evidence, argument and next-condition specification

1. **Evidence package:** CPU-only analysis of saved reports; four PNG/PDF
   figures, raw-case/paired CSV, controller provenance and SHA256 input/output
   manifest in `runs/paper_evidence_20261110`. It covers 768 recorded perturbed
   cases and 18 unperturbed records. No new simulation or optimizer updates.
2. **Paper argument:** [EVIDENCE](EVIDENCE.md) maps claims to measurements and
   limits; PAPER explains the connected method and inherited components.
   No manuscript skeleton, new architecture or mandatory SAC retraining.
3. **Next condition:** the frozen elastic-transfer pilot below is specified,
   including material provenance, controller identities, data use and budget.
   Its execution adapter is still to be implemented before running it.

The immediate next implementation is the bounded material-condition evaluation
option. There is no remaining A decision or broad regression/diagnostic stage
to resolve before that work.

## Next-stage protocol — frozen transfer under elastic-modulus changes

**Status:** protocol prepared; no new physical run, data collection or fitting
has started. This is one bounded single-cut pilot, not multi-material training.
Question: do the existing frozen controllers still reach the goal when prism
elasticity changes, and how do outcomes and forces change? A material shift may
hurt the learned model/controllers; either outcome is informative.

### Conditions and provenance

Use the existing free-bottom prism, robot, knife, target, captured initial
physical state, action mapper, reward and task definitions. Change only Young's
modulus at fixed Poisson ratio 0.45:

| Condition | E scale | E (Pa) | mu (Pa) | lambda (Pa) |
|---|---:|---:|---:|---:|
| Softer prism | 0.5 | 20,000 | 6,896.551724 | 62,068.965517 |
| Existing prism | 1 | 40,000 | 13,793.103448 | 124,137.931034 |
| Stiffer prism | 2 | 80,000 | 27,586.206897 | 248,275.862069 |

These scales come from the completed Nov 8 P0 material-scale check, not a new
search for favorable results. Use the existing `set_material_scales` operation
to scale both Lamé arrays relative to their source values. Keep density
830 kg/m³, damping, cutting-spring stiffness/softening, friction/contact and all
other settings unchanged. These are synthetic elasticity conditions, not
calibrated apple/potato/tomato materials.

Restore the full captured reset frame, including controller/filter and material
history, and keep the selected material arrays in effect across reset. Verify
the actual arrays and unchanged initial state in the run manifest. This tests a
common initial-state parameter shift, not material-specific equilibrium resets.
The existing P0 reference peaks were 9.68/9.97/10.33 N; elasticity alone may have
a modest effect. Do not widen the range or change spring strength after seeing
an uninteresting result.

### Controllers, data and budget

Use five frozen controllers: working WM + BPTT, SAC85k, prescribed reference,
and same-working-WM MPPI with/without terminal critic. Keep the existing MPPI
settings and planning seed 0 in both settings, including its known nominal
failure without a critic; do not select each setting's best seed. One physical
world and one episode per controller/condition; no action perturbation.
Do not add material identity to the model or policy inputs.

The nominal condition already has saved results in the evidence recipe.
Reuse the five nominal records (SAC's existing repeats remain repeats) when
checkpoint tensors, asset/reset identity and execution configuration match.
Only the softer and stiffer conditions need new execution: **10 new episodes**.
Run the softer condition then the stiffer one, in this fixed controller order:
working policy, SAC85k, reference, MPPI+critic, MPPI without critic.
If provenance cannot match, stop that cell for resolution rather than silently
substitute a new nominal experiment.

No training set is required in advance. New transitions are evaluation logs;
they are not used to update WM, critic or policy. The frozen nominal-material
WM deliberately tests transfer under model mismatch. If later learning across
materials is selected, that needs a separate data-coverage/training protocol.

Budget: one available GPU7, at most **2 GPU-hours total** including startup,
and at most **1,200 wall seconds per new episode**, in addition to the unchanged
500-step task limit. Finish each case at its own ending. Keep censored prefixes
and mark unstarted cells if the total budget expires; do not extend the budget
or count wall-budget interruptions as task failures. No extra seeds or
256-case disturbance evaluations in this pilot.

Before execution, add the material-condition option to the existing frozen
evaluator, preserve base-contract validation, and record a distinct condition
identity with the base contract hash, applied scale, effective material-array
hashes, reset hash and frozen checkpoint identity. This option/condition
registration is **not implemented by this document update**; do not label
changed material as an unchanged `public-prism-ur5e-v1` result. Use the existing
P0 scaling implementation, not a second material path.

### Measurement and stopping rule

Report each cell's ending, elapsed simulation time, goal error, material
peak/mean force and impulse, sensor utilization, released springs, and decision
latency. Keep failed and censored cells alongside successes. No confidence
interval or success probability is inferred from a single episode.

This pilot answers feasibility for the tested conditions only. If all methods
work similarly, report that the perturbation is mild. If a controller fails,
retain its trajectory and identify the observed ending without automatically
opening a repair, parameter sweep or retraining stage. A reproducible code
exception is fixed separately; physical failure is a result. The pilot closes
after the ten planned executions or the budget, and informs one explicit next
decision about whether multi-material training is useful.

## Conditional work, not current jobs

- A matched-data or sample-efficiency claim would need a new explicit comparison
  protocol: total physical transitions including policy62/WM pretraining, input
  access, checkpoint selection, gradient-update-to-data ratio, tuning and
  wall/GPU budgets. Decide then whether fresh SAC training is required; neither
  parallelism nor identical learning rates alone makes a comparison fair.
- A task outside existing training coverage may require collection/training for
  both methods. Existing single-prism policies are not assumed to generalize.
- More training seeds follow a fixed method and protocol, per the user's choice.
- Robot experiments follow [REAL2SIM](REAL2SIM.md) once hardware/sensing and
  the physical protocol are settled. No claim of real-world safety is made.

## Storage and scheduling

Use Remote SSH; keep source, data, weights and artifacts on the servers.
GPU7 canonical checkout: `/dataHDD/dong/Projects/Cutting_DEV`; machine routes
and protected dependencies: `runs/operations/MACHINES.md`. GPU9 is retired.
A-pair training/evaluation results are complete; previous running snapshots are
historical. Recheck actual GPU/process occupancy before any future job and leave
other users' work alone. This document update starts no training or physics
evaluation.
