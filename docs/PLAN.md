# Current plan — 2026-11-17

## Active authorized runs

Both long runs are launched and online history is verified. The user authorized
long training on GPU7 and A6000 after documentation,
visualization, deployment and capacity checks. The previous no-launch status is
superseded by this instruction; it is preserved in the archived plan below.
See [shared training record](TRAINING_SHARED_20261116.md) for the launch state,
commands, verification and online run links.

- GPU7: current DiSECt material backend, 3072 worlds (1024/object). Full
  learner/reset/replay-restore capacity passed; follow-up runner fixes passed
  root569/skip5 and bundled backend41 (including the new offline wrench panel).
- A6000 GPU1: current BladeMaster material backend. 510 worlds
  (170/object) passed actual learner, masked reset, snapshot and replay restore
  checks with 7.90 GiB minimum measured free memory after updates. GPU0 is occupied and untouched.
- Both start fresh. Each ends at24 hours or1,280,000 new physical transitions,
  whichever comes first. No automatic extension or old-task checkpoint reuse.
- Every long run requires online W&B and a remotely verified first history.
  Checkpoints/replay and physical snapshots remain on the server.
- GPU9: user authorized current-task multi-environment SAC on 2026-11-16 after baseline validation. Capacity preflight and online launch are recorded in [TRAINING_SAC_20261116](TRAINING_SAC_20261116.md). Its historical BladeMaster SIGILL in libcuda is not declared resolved.

## Fixed task and algorithm

Source of scene truth: [SHARED_SCENE](SHARED_SCENE.md) and
[scene_v1.json](../configs/shared/scene_v1.json). Resource counts may differ;
task settings must not be edited independently for each backend.

- Freely placed cucumber003, flat-bottom half potato and prism; equal object
  allocation within a run. One shared policy per backend.
- Same UR5e, FDCC, mounted finite blade, sensor frame and action meaning.
  Six output channels, pitch off; five effective controls.
- Task pose_depth_v2: prescribed final pose gives+100 without higher-priority
  failure; board boundary, force/workspace/timeout rules retained. Maximum-depth
  progress replaces spring-release shaping. Material separation is diagnostic.
- User choice: final-position-only. Final EE XY radius<40mm; no particle-motion
  penalty, no extra path penalty. Optional path-and-final remains available but
  is not selected for these runs.
-180 control steps/9s, gamma1, componentwise simulated limits[50,50,50,5,5,5].
- Start: each settled object's top plus paired5–30mm clearance, fixed per world.
  Each episode samples E(.5–2), density(.8–1.25), cut resistance(.5–2), speed
  (1–5cm/s), then settles for1s with that material. Same scale distributions do
  not make the native fracture laws or resulting forces identical.
-34 observations/frame including six-axis wrench;10-frame history.
  Real-data critic32, actor imagination8, BC.25; no risk-constraint variant.
-640 physical transitions per learning round;30720 warmup transitions,
  then500 WM-only rounds. WM/policy effective batches2048/4096;
  microbatches512/critic8192. Pending learning rounds are checkpointed.

## Measurements and remaining limits

Track per-object reward, pose successes, all failure reasons, speed error and
wrench utilization. Capture actual first-step training geometry; use the shared
studio renderer and all-world overview. Final frozen evaluation uses predeclared
development conditions, records their identities and preserves terminal traces.
No outcome threshold is a license to change the task or extend the budget.

The board stop is an execution boundary, not compliant board-force simulation.
The support plane is infinite; visible board edges are not physical edges.
Native material parameters are uncalibrated. A measured adapter mesh is absent;
do not invent display hardware. Tilted contact remains unverified and disabled.

Current-task baseline implementation and source search are recorded in
[CURRENT_BASELINES](CURRENT_BASELINES.md). The original TensorFlow SAC runner is
historical single-world training; the inspected parallel SAC commit is batched
evaluation. No reusable cutting-connected PyTorch parallel trainer was found.
The user authorized SB3 as fallback. Its new vector bridge and learner use
physical-transition update credits, terminal replay, current observations and
task metadata. Frozen WM+MPPI and an explicitly approximate nonlearned FDCC+MPPI
model share the current pose-task evaluation/recording entry point.
GPU9 bounded three-object DiSECt-material validation completed; measured results
and remaining backend coverage are in CURRENT_BASELINES/RESULTS. Do not claim a formal trained comparator, a calibrated
analytic force model or a reproduction of a published cutting MPC.

## Vision route 1 (2026-11-17)

User corrected the goal: select the cut plane from vision, then try a physical
feedback-controlled episode. The optional visible-major-axis midpoint selector
does not read prescribed target XY. Seven scan shapes tested for perception;
the native BladeMaster single-world handoff uses an explicitly cross-backend,
frozen policy66, since the A6000 run has no trained actor yet. No ongoing job,
reward, material law or long-training budget is changed. This is one-shot
pre-cut vision followed by motion/force feedback, not visual servoing.
The bounded physical trial completed its180-step budget but timed out without
pose success; blade remained2.476 mm above the board. Do not broaden the
training task or claim visual generalization from this one transfer trial.
See [VISION_ROUTE1](VISION_ROUTE1.md) for commands, measurements and limits.

## Method 2: centered cut with fixed heading (user decision, 2026-11-17)

First implement the learned visual-goal pipeline on the existing object poses.
Do not randomize object headings or broaden the low-level task in this stage.
A plane passes through the object's bounding-box midpoint along the fixed cut
normal; orientation comes from the prescribed robot-compatible heading.
This is a geometric midpoint, not equal-volume partition or an optimal cut.

Use calibrated depth/point clouds as model input. Ground-truth mesh geometry
may generate supervised labels, but is not an inference input. Predict the
center coordinate along the normal; combine it with measured height,
board/tool calibration and the existing speed command to construct the same
start pose and 8D goal interface as route 1. Keep the low-level policy frozen.

At fully fixed placement a constant output can match every label. Report that
constant-center baseline and perception error; passing this pilot only validates
data -> prediction -> goal -> physical execution. Do not call it localization,
material adaptation or generalization. Position/heading and new-object
generalization are later work, not prerequisites for this pilot.

Implementation order:
1. Prepare centered labels and depth examples for the current fixed poses,
   with source identities and held-out views; do not split duplicate frames
   across train/test and call them unseen objects.
2. Add the learned center predictor and validate coordinate/goal conversion.
3. Execute a bounded frozen-policy trial, reporting pose success, failure,
   force and material separation independently.
4. Reuse that explicit visual-goal handoff for SAC / WM+MPPI / analytic MPC.
   Their independent current-task evaluation is already available; visual
   execution adapters are still missing.

No method-2 model has been trained yet. The current DiSECt assets bake x=0
in material coordinates, approximately the middle of all three objects.
A different inferred plane requires a separately prepared DiSECt asset or
the native BladeMaster backend; changing an RL goal cannot change cut topology.
No new long run or change to an active run is implied by this pilot plan.

## Historical evidence

Old GPU7 pose-v1 training completed its24h budget:policy1109,1,062,912
transitions; frozen0/48 target-pose completions. Old GPU9 stopped with SIGILL:
last logged policy998; evaluated saved policy866 had48/48 nominal-replica pose
completions. These differ in tools, tasks and conditions, and are not a matched
backend comparison. See RESULTS and DATA; preserve original checkpoints.

[Previous full plan](archive/PLAN_before_shared_long_20261116.md) retains older
capacity, cleanup, deployment and interpretation records. Historical source,
v1–v3, S3 and perturbation scripts remain only for identified reproduction.
The main task list is this file; dated evidence belongs in RESULTS.
