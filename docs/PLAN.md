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

## Current scope correction — nominal task first, no material pilot

The user questioned adding transfer experiments without a multi-material
training/data design. The proposed 20/40/80 kPa pilot is withdrawn from current
work; no adapter is to be implemented and no episodes are queued. Its proposed
ten new episodes meant five controllers at two new elasticities, once each.
That was a development budget, not a statistically justified sample size.
The unexecuted proposal remains in Git at `a097099`, not on the task list.

The completed action-noise evaluation is a supplementary, locally designed
stress test. It was introduced because the deterministic fixed-start task
provided little separation between policies. Its noise distribution is not
calibrated to a measured robot disturbance and is not a required reproduction
protocol for this project. Do not make its 256-case proportions the headline
success rate of the original task, or infer unseen-material generalization.

Keep the research scope at the trained, nominal prism task: the connected
WM + BPTT method, the existing SAC85k controller and the paired same-WM MPPI,
with their actual task outcomes, forces and online cost. Existing comparisons
and figures are available in EVIDENCE; retain their failures and limitations.
Clarify what the learning/control procedure contributes within this task before
deciding on any additional experiment. No new physics, training, material sweep,
action-noise sweep or mandatory SAC retraining is scheduled by this correction.

## Research review handoff — prepared 2026-11-10

[PAPER](PAPER.md) is now the review memo for the user and Claude: recommended
question, exact implemented data flow, inherited versus task-specific elements,
nominal-task results, claim-to-evidence gaps, and four review questions.
It is not a manuscript skeleton or a new experimental protocol.

Recommended scope: model-based refinement of an already functional compliant
cutting controller. The working continuation starts from successful policy62,
so learning from scratch and nominal task feasibility are different claims.

### Next action after review

1. Record one selected contribution claim and its scope. Review the existing
   BC-only evidence before asking for a new ablation; retain all working-run
   checkpoints, the later collapse and failed MPPI seeds.
2. If the scope is a task-level method/continuation study, assemble the method
   and nominal-result narrative from existing records. No new collection or
   retraining is required merely to do that. Any manuscript is a separate
   deliverable; the previously withdrawn skeleton is not restored here.
3. Only if attributing a benefit to physical-data critic training is central,
   design one matched recorded-target versus imagined-target continuation with
   H8 and BC held fixed. It is a conditional counterfactual, not a queued run.
   Initialization/pre-fit treatment, measured endpoints, computational budget
   and the interpretation of mixed/negative results must be fixed before it
   becomes a runnable proposal. No current implementation change is authorized
   by this review memo alone.
4. If the evidence and method distinction are insufficient for a convincing
   paper claim, report that decision directly. Do not replace the missing
   contribution with more disturbance cases, material sweeps or bug-fix plots.

The user will ask Claude to review. No external message, training or evaluation
job is sent/launched. A methods claim and an applied-system claim have different
evidence requirements; this review must end with a choice, not a longer repair
list.

## Completed evidence work

- Four figures and their CSV/provenance are on GPU7 in
  `runs/paper_evidence_20261110`; no measurements were deleted.
- EVIDENCE maps observations to appropriately limited claims. Action-noise
  results are supplementary; nominal execution is the current task evidence.
- PAPER describes the working method. A remains a completed mechanism analysis.
  Transfer is not a current contribution claim or next-step requirement.

## Conditional work, not current jobs

- A matched-data or sample-efficiency claim would need a new explicit comparison
  protocol: total physical transitions including policy62/WM pretraining, input
  access, checkpoint selection, gradient-update-to-data ratio, tuning and
  wall/GPU budgets. Decide then whether fresh SAC training is required; neither
  parallelism nor identical learning rates alone makes a comparison fair.
- Material or geometry generalization is deferred. If selected later, first
  define whether the claim concerns untrained transfer or learning across a
  specified training distribution, then design its data and held-out evaluation.
  Existing single-prism policies are not assumed to generalize.
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
