# Current plan — 2026-11-16

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
- GPU9 is not a launch target. Its historical SIGILL in libcuda is unresolved.

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

WM+MPPI is the retained frozen-model planning baseline, using the upstream
pytorch-mppi optimizer. Historical fixed-task/v3 experiments are separate from
the current shared pose_depth_v2 multi-object task. The optional Python
environment/import tests do not certify a new-task physical MPPI rollout.
The current pose-policy evaluator does not offer MPPI. The retained SAC runner
is historical single-world training and is not a current shared-task multi-object
baseline. A nonlearned-model MPC controller has not been integrated. A current-task planning
comparison needs a compatible checkpoint, runner/metadata adapter and matched
physical evaluation; do not label it complete.

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
