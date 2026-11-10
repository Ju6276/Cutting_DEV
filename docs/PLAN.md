# Current plan

The single current task list. [PAPER](PAPER.md) holds the research question and
algorithm; [METHOD](METHOD.md) maps implementation and sources;
[BENCHMARK](BENCHMARK.md) defines the task and metrics; [RESULTS](RESULTS.md)
holds measurements. Updated 2026-11-10 against results at `d6f3865`.

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

## Next work — reuse recorded evidence, then scope the next task

1. **Conclusions and naming (this update).** Align PLAN, PAPER, README,
   BENCHMARK and RESULTS with the decision above. Retain historical measurements
   and failed runs. No manuscript skeleton is created.
2. **One evidence package from saved outputs.** Prepare the comparison figures
   and their captions from `runs/main_compare_20261109`,
   `runs/eval_fixed_p62_20261109`, `runs/eval_perturbed_main_20261109` and
   `runs/mpc_baseline_pair_20261109`: completion/failure reasons, material
   peaks, impulse, paired successful-case duration, and online computation.
   Mark budget-censored and unmeasured cells explicitly. Include selected
   checkpoints, source settings, input/history access and available training
   counts alongside the figures. Do not rerun physics to reproduce saved numbers.
   Deliverable: server-side figures and an evidence-to-claim summary.
3. **Set the paper argument using that package.** The working algorithm can be
   described now; A is analysis, not its central novelty claim. Identify which
   result supports each proposed contribution, distinguish inherited components,
   and keep unsupported safety/sample-efficiency/generalization claims out.
   This is a research-direction update, not restarting the withdrawn skeleton.
4. **Then scope one next single-cut condition.** Prefer a bounded material
   variation on the existing prism before introducing a new geometry. Specify
   whether the question is frozen-policy transfer or learning across conditions;
   define material values, task contract, data coverage, evaluation and budget
   before running it. Apple needs its own scene/data-coverage decision. No
   collection, fitting or multi-object training is launched by this document.

The immediate work is figures and interpretation of completed runs. It does not
wait for A6000 availability. Do not turn an inconvenient metric into another
unbounded repair stage.

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
