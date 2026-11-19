# Current plan — 2026-11-19

## Authorized runs and next actions

Runtime status is a timestamped observation, not inferred from this plan.
Frozen launch protocol, budgets and W&B links:
[shared WM+BPTT runs](archive/TRAINING_SHARED_20261116.md),
[vector SAC](archive/TRAINING_SAC_20261116.md). Read each run's progress and process
before reporting that it is still active.

- GPU7: 3072-world WM+BPTT, DiSECt material backend.
- A6000 GPU1: 510-world WM+BPTT, native BladeMaster material backend.
- Both runs (pose_depth_v2, now read/evaluate-only) have ended; no restart or
  extension. A6000: budget_complete at the 24-hour bound, 482,460 transitions,
  705 WM / 205 policy updates. GPU7: FAILED at 22.4 h (2026-11-17 17:55) with a
  CUDA allocation failure inside a physics reset, 933,888 transitions, 1406 WM /
  906 policy updates; last named checkpoint 00000288 (see RESULTS). Same cap,
  different actual interaction; SAC used 1,281,024. Compare at equal transition
  counts only.
- Before any next long run: GPU-memory headroom for 3072 DiSECt worlds on a
  24 GB card must be fixed and verified (the GPU7 run grew from 16.7 to 23.8 GB
  and failed on an 8 MB Warp allocation). Status: trainer defaults now release
  Torch cache before physics/reset, use expandable segments and guard headroom
  (METHOD); verified only on a short 3072-world run without resets (RESULTS
  2026-11-17). Behaviour over hours with resets remains unverified.
- GPU9 SAC finished its transition budget (1,281,024 transitions, vector-step
  granularity). Preserve its checkpoint and evaluate under the recorded task.
- After completion, evaluate the frozen policies on identified development
  conditions: per-object pose completion, failures/timeouts, return, speed
  error, wrench utilization and independent material-separation diagnostics.
  Record checkpoint identities, conditions, timing and source versions.
- Keep the frozen training checkouts unchanged. Development cleanup and offline
  rendering changes do not replace their code. All long runs require online W&B.

## Current verification status — 2026-11-19

The a1852dc review confirmed the explicit-reference fix and its limited call
scope. The dbdddf6 training/evaluation/resume evidence remains applicable to
unchanged action-driven code. Its temporary verification scripts had two
false-pass holes (missing stages, omitted world_model tensors); maintained
checkers now live in scripts and the actual stored runs were rechecked.
See RESULTS and RUNNING for evidence and commands. Current development remains
on main; no new long training was launched during this review.

## Follow-up before the next pose_depth_v3 long run (2026-11-18)

Existing user decisions remain in effect: final-position-only, particle-slip and
optional path penalties disabled. This is not a request to decide those again.

- Check the existing final region against the recorded blade/object geometry.
  Historical v2 policies drag potato and prism ~20 mm along +x within the current
  40 mm XY radius. Any proposed change to this radius is a separate task change;
  no new radius is selected here.
- Speed cost already exists (weight 20, progress- and time-weighted components).
  Analyse its effect and the observed fast descent before proposing a change to
  that existing cost; do not describe speed cost as absent.
- GPU-memory behaviour over hours with repeated resets still needs a bounded
  check before setting the long-run capacity and budget. Short regression tests
  do not establish long-run memory stability.

Unified run entry: /dataHDD/dong/Projects/Cutting_DEV on main on every server,
environment from scripts/env (RUNNING).

## Fixed task and algorithm

Source of scene truth: [SHARED_SCENE](SHARED_SCENE.md) and
[scene_v1.json](../configs/shared/scene_v1.json). Resource counts may differ;
task settings must not be edited independently for each backend.

- Freely placed cucumber003, flat-bottom half potato and prism; equal object
  allocation within a run. One shared policy per backend.
- Same UR5e, FDCC, mounted finite blade, sensor frame and action meaning.
  Six output channels, pitch off; five effective controls.
- Task pose_depth_v3 now uses `board.motion=project_continue_v1`: no permanent
  cross-step freeze; cumulative arrival >10, 2 mm tolerance and analytic survival
  retained. Board reaction physics remains outside this task. v2 kept only for
  traceability of the 2026-11-16 runs. Prescribed final pose gives+100 without higher-priority
  failure; board boundary, force/workspace/timeout rules retained. Maximum-depth
  progress replaces spring-release shaping. Material separation is diagnostic.
- User choice: final-position-only. Final EE XY radius<40mm; no particle-motion
  penalty, no extra path penalty. Optional path-and-final remains available but
  is not selected for these runs.
- 500 control steps/25s (user-aligned to SliceIt on 2026-11-19), gamma1, componentwise simulated limits[50,50,50,5,5,5].
- Start: each settled object's top plus paired5–30mm clearance, fixed per world.
  Each episode samples E(.5–2), density(.8–1.25), cut resistance(.5–2), speed
  (1–5cm/s), then settles for1s with that material. Same scale distributions do
  not make the native fracture laws or resulting forces identical.
- 34 observations/frame including six-axis wrench;10-frame history.
  Real-data critic32, actor imagination8, BC.25; no risk-constraint variant.
- 640 physical transitions per learning round;30720 warmup transitions,
  then500 WM-only rounds. WM/policy effective batches2048/4096;
  microbatches512/critic8192. Pending learning rounds are checkpointed.

## Current baseline and vision status

[CURRENT_BASELINES](methods/CURRENT_BASELINES.md) is the implementation/coverage ledger.
Current entries are vector SB3 SAC, frozen WM+MPPI and approximate nonlearned
FDCC+MPPI. The original TensorFlow SliceIt SAC remains historical reproduction.
No claim of matched-budget superiority follows from integration tests.

Vision is pre-cut goal selection followed by motion/wrench feedback, not
continuous visual servoing or a core RL contribution.
[Vision protocol](methods/VISION_ROUTE1.md) documents geometric midpoint and learned
midpoint routes. Official DP3 XYZ encoder plus our regression head is the new
default; the local encoder remains a named comparison. This choice was made on
a reused development test set (RESULTS 2026-11-17), not a fresh final evaluation.

Completed: known-shape visual-data pilot, three-seed encoder comparison,
requested policy videos and nominal native-BladeMaster DP3 controller videos.
The frozen policy66 and paired WM566 were deliberate historical references,
not the latest A6000 actor. RL and WM+MPPI timed out; analytic MPC reached the
pose target in its one case. These are illustrations, not an algorithm ranking.
SAC visual handoff is not implemented. New shapes, rotations and real depth
remain deferred; no additional visual tuning or training is scheduled.

## Initial real-robot bring-up

The user selected UR10e + ATI Mini45-E (SI-580-20). First validate driver,
motion and FDCC wrench feedback on the control PC without a knife, RL, MPC
or vision. Follow [the Chinese handoff](hardware/UR10E_FDCC_SETUP_ZH.md). The Mech-Mind
camera and tool/mount CAD are later inputs. Hardware installation, calibration
and tests remain pending; this does not alter frozen simulations or authorize
a new training run. No additional near-board controller handover is selected.

## Boundaries that still matter

- Pose completion is not validated complete material separation.
- The board stop is an execution boundary, not compliant board-force simulation.
  The physical support plane is infinite; rendered board edges are not physical.
- Native material parameters and simulated wrench limits are uncalibrated.
  Shared settings do not equate the two fracture/contact laws.
- Pitch remains disabled; tilted contact is not verified.
- No fabricated connector, blue overlay or replacement physical blade in videos.
  Current export/render commands are in [RUNNING](RUNNING.md).
- DiSECt needs a prepared cut interface; a new visual plane cannot silently
  move that topology. Native BladeMaster handoff uses reset/IK placement,
  not a validated real-robot approach trajectory.
- Feature clusters or force/action correlations do not prove causal adaptation.

## History and completed budgets

[RESULTS](RESULTS.md) contains dated evidence; [DATA](DATA.md) locates artifacts.
The [pre-cleanup plan](https://github.com/Ju6276/Cutting_DEV/blob/e7b9e69/docs/PLAN.md)
preserves all completed vision budgets and their conditions.
The [previous shared-launch plan](archive/PLAN_before_shared_long_20261116.md)
retains older decisions. Archived S3, perturbation and old-task commands are
not current launch instructions.
