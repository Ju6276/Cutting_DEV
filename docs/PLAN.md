# Current plan — 2026-11-20

Runtime status is a timestamped observation, not inferred from this plan. Read a
run's progress and process before reporting that it is active. Every paid/long
run needs an approved plan (question, hypothesis, setting, metric, decision rule,
wall/GPU budget) and online W&B.

## Fixed task and algorithm

Source of scene truth: [SHARED_SCENE](SHARED_SCENE.md) and
[scene_v1.json](../configs/shared/scene_v1.json); task rules: [TASK_FREE_CUT](TASK_FREE_CUT.md).
Resource counts may differ; task settings must not be edited independently per backend.

- Freely placed cucumber003 and flat-bottom half potato; equal object
  allocation within a run. One shared policy per backend (DiSECt or BladeMaster).
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
- Baselines: vector SB3 SAC, frozen WM+MPPI, nonlearned FDCC+MPPI
  ([CURRENT_BASELINES](methods/CURRENT_BASELINES.md)). No matched-budget
  superiority claim follows from integration tests.

## Active work — UR10e / ATI / IKEA knife integration (2026-11-20)

Authorized: finish both material backends and the shared learning/evaluation/render
path for the user's UR10e, ATI Mini45 and CAD knife. No formal long run is started.
Keep the UR5e profile runnable and preserve V3 reward/arrival/force/horizon rules.

Implementation: immutable canonical knife mesh + curved edge; mesh-derived SDF
for DiSECt (old analytic wedge remains legacy); arbitrary-mesh contact/edge rules
for the BladeMaster forward solver; same fixed mounting and sensor wrench origin.
Re-export the three free objects with explicit knife provenance. Robot profile,
workspace and board placement are explicit; observations and action semantics stay
documented; checkpoint robot/geometry mismatches must fail.

Acceptance: geometry/SDF/contact tests and independent wrench transform checks;
food-object reference and actual-action episodes on both backends; all WM,
critic-32/actor-8 updates, SAC/planners, save/resume, partial resets; shared render
matching the physical knife. Per user scope, do not repeat hundreds-world learning
probes or native differentiation suites as acceptance gates. The two completed
9-world learning runs stand; finish natural terminal/reset checks independently
per backend. Publish a finite launch config and measured throughput.

Validation budget: each geometry/contact probe <= 10 min, each small full-pipeline
probe <= 1 h, capacity/reset probes <= 2 h per backend. These are bounded engineering
checks, not a substitute for a trained policy or a hardware contact validation.
Review candidate and evidence: [UR10e integration handoff](hardware/UR10E_INTEGRATION_REVIEW.md).

## Before the next pose_depth_v3 long run

Existing user decisions remain in effect: final-position-only, particle-slip and
optional path penalties disabled.

- GPU memory: the 2026-11-16 3072-world DiSECt run on GPU7 grew from 16.7 to
  23.8 GB and failed on an 8 MB Warp allocation in a reset. Trainer defaults now
  release Torch cache before physics/reset, use expandable segments and guard
  headroom ([METHOD](METHOD.md#torchwarp-gpu-memory-sharing)); verified only on a
  short run without resets. A bounded check over hours with repeated resets is
  needed before setting long-run capacity and budget.
- Final region: historical v2 policies drag potato and prism ~20 mm along +x
  within the 40 mm XY radius (RESULTS). Any change to this radius is a separate
  task change; none is selected.
- Speed cost already exists (weight 20, progress- and time-weighted). Analyse its
  effect and the observed fast descent before proposing a change.
- Evaluate frozen policies on identified conditions: per-object pose completion,
  failures/timeouts, return, speed error, wrench utilization and independent
  material-separation diagnostics, with checkpoint/source identities.

## Vision

Vision is pre-cut goal selection followed by motion/wrench feedback, not
continuous visual servoing or a core RL contribution
([VISION_ROUTE1](methods/VISION_ROUTE1.md)). Official DP3 XYZ encoder plus our
regression head is the default (chosen on a reused development test set). Known-
shape pilots and DP3 controller videos are completed; SAC visual handoff is not
implemented. New shapes, rotations and real depth are deferred; no visual tuning
or training is scheduled.

## Real-robot bring-up

UR10e + ATI Mini45-E (SI-580-20). First validate driver, motion and FDCC wrench
feedback on the control PC without a knife, RL, MPC or vision
([Chinese handoff](hardware/UR10E_FDCC_SETUP_ZH.md), [REAL2SIM](hardware/REAL2SIM.md)).
The Mech-Mind camera is a later input. Hardware installation, calibration and
tests are pending; this does not authorize a training run.

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

## Key decisions

- 2026-11-12: force-spike problem retired; contact coordinates corrected (`edge_coordinate_consistent_v2`).
- 2026-11-14: free-placed multi-object pose task; full six-axis wrench observation; transition-scaled schedule.
- 2026-11-15: every long run uses online W&B.
- 2026-11-16: shared scene, knife and paired material sampling for both backends; pose_depth_v2; final-only position mode.
- 2026-11-16/17: shared v2 long runs (GPU7 DiSECt 3072 worlds failed at 22.4 h on GPU memory; A6000 BladeMaster 510 worlds reached the 24 h cap); GPU9 SAC reached 1,281,024 transitions. All v2, evaluate-only.
- 2026-11-17: pose_depth_v3; DP3 encoder for visual goals; UR10e + ATI selected as hardware.
- 2026-11-19: horizon 500 steps; permanent board latch removed (`project_continue_v2`).
- 2026-11-20: UR10e/ATI/CAD-knife integration authorized; no formal long run.


## 2026-11-21 authorized consolidation
Merge the reviewed board-contact fixes, UR10e platform, corrected compliance axes,
fixed presentation entry and docs-prune branch after current regression checks.
Synchronize GPU7, GPU9, A6000 and tams61. No formal long training is launched by this
merge. Use the two-food configs (cucumber, half potato); new candidate foods require
their own scene preparation. Do not repeat unrelated native differentiation tests.
