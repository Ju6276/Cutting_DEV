# Current plan — 2026-11-22

This page owns current decisions and remaining work. Measurements and historical
results belong in RESULTS, commands in RUNNING, hardware status in TAMSCAGE and
the Nano field record. Check progress/process before reporting a job as active.

## Fixed task and algorithm

Source of scene truth: [SHARED_SCENE](SHARED_SCENE.md), the UR10e
[five-food contract](../configs/shared/ur10e_five_food_v1.json), and retained UR5e
[scene_v1.json](../configs/shared/scene_v1.json); task rules: [TASK_FREE_CUT](TASK_FREE_CUT.md).
Resource counts may differ; task settings must not be edited independently per backend.

- UR10e five-food distribution: freely placed cucumber003, flat-bottom half
  potato, peeled banana, tofu and luncheon. Earlier two-food configs remain
  reproducible. Equal world allocation; one policy per backend (DiSECt or BladeMaster).
- Same robot, FDCC, mounted finite blade, sensor frame and action meaning.
  Six output channels, pitch off; five effective controls.
- Task pose_depth_v3 with `board.motion=project_continue_v2`: no permanent
  board freeze, geometric nonpenetration, cumulative arrival >10, 2 mm tolerance,
  analytic imagined survival. +100 for the prescribed final pose without
  higher-priority failure; -100 for force/workspace failure; timeout terminal.
  Maximum-depth progress shaping. Material separation is diagnostic.
  No board reaction-force model.
- User choice: final-position-only. Final EE XY radius <40 mm; no particle-motion
  penalty, no path penalty (optional path-and-final exists, not selected).
- 500 control steps / 25 s (0.05 s period), gamma 1, componentwise simulated
  limits [50,50,50,5,5,5].
- Start: each settled object's top plus paired 5–30 mm clearance, fixed per world.
  Each episode samples E (.5–2), density (.8–1.25), cut resistance (.5–2), speed
  (1–5 cm/s), then settles 1 s with that material. Same scale distributions do
  not make the native fracture laws or forces identical.
- 34 observations per frame including six-axis wrench; 10-frame history; 8D goal.
- WM+BPTT: real-data critic 32, actor imagination 8, BC 0.25; no risk variant.
  640 physical transitions per learning round; 30720 warmup transitions, then
  500 WM-only rounds. WM/policy effective batches 2048/4096; microbatches
  512/critic 8192. Pending learning rounds are checkpointed.
- Paper baselines: vector SB3 SAC and frozen WM+MPPI. Nonlearned FDCC+MPPI
  remains a controller-interface engineering tool
  ([CURRENT_BASELINES](methods/CURRENT_BASELINES.md)). No matched-budget
  superiority claim follows from integration tests.

## Current work

- UR10e/ATI/IKEA five-food integration is merged. The DiSECt reset-order and
  analytic MPC tare fixes are in main. Preserve active runs' source checkouts.
- DiSECt and BladeMaster have completed their authorized budgets; compare their
  actual data/update counts before interpreting policies. GPU9 SAC's run status
  must be read from its progress. See RESULTS for the dated long-run snapshot.
- Evaluate frozen checkpoints under identified conditions: per-food pose
  completion, failure/timeout, return, speed error, wrench utilization and
  independent separation diagnostics. A budget-complete run is not a mature
  policy-performance claim.
- Main paper comparisons are WM+BPTT, SAC and frozen WM+MPPI. Do not start a new
  physical cutting-model project just to add analytic MPC. Keep its corrected
  implementation for offline/controller-chain testing.
- User retained final-only position constraints and existing speed cost. No new
  region, force limit, reward, startup schedule or contact law is selected.
- Hardware work remains offline while the UR10e is powered off. The aligned ROS1
  FDCC candidate has passed fixed-input output parity but is not activated.
  ATI acquisition, online command/parameter timing, calibration and real closed
  loop remain to validate. Use the current FDCC guide, not a new ROS2 installation.
- Camera API/units/intrinsics are recorded on ju-ROG; empty-board/food captures,
  metric validation and eye-to-hand calibration remain. The 350 mm simulation
  profile is still provisional and must not be presented as the 550 mm lab model.

Every additional long run needs a concrete approved budget and online W&B.
Existing authorizations and settings are not reset by this documentation cleanup.
No unrelated native-differentiation or CPU-fixture investigation is a training gate.

## Vision

Vision is pre-cut goal selection followed by motion/wrench feedback, not
continuous visual servoing or a core RL contribution
([VISION_ROUTE1](methods/VISION_ROUTE1.md)). Official DP3 XYZ encoder plus our
regression head is the default (chosen on a reused development test set). Known-
shape pilots and DP3 controller videos are completed; SAC visual handoff is not
implemented. New shapes, rotations and real depth are deferred; no visual tuning
or training is scheduled.

## Boundaries that still matter

- Pose completion is not validated complete material separation.
- The board boundary is a geometric execution constraint, not compliant
  board-force simulation. The support plane is infinite; rendered board edges are
  not physical.
- Native material parameters and simulated wrench limits are uncalibrated.
  Shared settings do not equate the two fracture/contact laws.
- Pitch remains disabled; tilted contact is not verified.
- No fabricated connector, blue overlay or replacement physical blade in videos.
- DiSECt needs a prepared cut interface; a new visual plane cannot silently
  move that topology. Native BladeMaster handoff uses reset/IK placement,
  not a validated real-robot approach trajectory.
- Feature clusters or force/action correlations do not prove causal adaptation.
