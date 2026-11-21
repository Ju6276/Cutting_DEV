# Free-object single cut — task pose_depth_v3

The current task is **pose_depth_v3** with `task.board.motion = project_continue_v2`,
shared by the DiSECt and BladeMaster material backends
([scene_v1.json](../configs/shared/scene_v1.json), [SHARED_SCENE](SHARED_SCENE.md)).
Training entry points (`train_gpu.py`, `train_sac_multi.py`) refuse every other
version: pose_depth_v2, V3 metadata without `board.motion` (old permanent latch)
and `project_continue_v1` are read/evaluate-only, as are the older
free_cut_v1–v4, free_cut_pose_v1 and blademaster_pose_v1 tasks. Checkpoints,
replay and manifests record the version; metadata checks refuse mixing them.
Regenerate configs from the shared contract instead of editing a version field.

## Rule provenance

Upstream reference: official SliceIt superproject `3a6964862f1f135af9d98dd400efeb44ba394c15`
pinning ur_python_utilities `7cd7cb8ce491b7953fce61ca891382e893e2ec43`. The local
reproduction checkout `1e79aba` is not the unmodified upstream revision. Source
snapshots and the reproduction diff: GPU7 `runs/sliceit_task_rule_audit_20261119/evidence.json`.

| Rule | Official SliceIt source | Implemented pose_depth_v3 | Provenance / reason for difference |
|---|---|---|---|
| Arrival and +100 | Current pose error satisfies selected-axis thresholds and cumulative arrival counter exceeds threshold; no force/torque collision. No material-separation test. | Pose arrival plus cumulative counter >10; no higher-priority failure; +100 once. Also requires the configured final XY region. | Pose-based completion is inherited. Final XY region and explicit failure precedence are our task choices; completion does not certify cut-through. |
| Height tolerance and counter | Ordinary simulation: 0.5 mm, >10. **DiSECt simulation: 2 mm, >10.** Real: 2 mm, >1. | 2 mm, >10. Counter increments on qualifying samples; it is not a consecutive-hold requirement. | The current pair already exists in official DiSECt simulation. Do not remove the counter merely to match upstream. |
| Force/torque limits | Component limits; collision ends the episode and contributes -100. DiSECt simulation uses 50 N / 10 N m; real config uses 50 N / 5 N m. | Six-component limits 50 N / 5 N m, checked on fresh 2 ms filtered samples; -100 once. | Our shared sensor/limit contract uses the latter scale. Sample timing and peak bookkeeping are adapter-specific. |
| Workspace | Out-of-workspace ends the episode; no separate fixed -100. | Explicit robot-base workspace; exit ends the episode with -100. | Our failure settlement; a deviation, not an upstream rule. |
| Time and step cost | 500 steps. Step cost -1 is part of the normalized continuous reward (contribution -0.25 with equal weights). | 500 steps / 25 s; timeout is terminal with no bootstrap. No separate per-step -1. | Horizon aligned by user request (2026-11-19). Continuous rewards and timeout semantics are our design. |
| Continuous rewards | Normalized distance, force and jerk terms, plus normalized step cost. | Maximum geometric-depth increment, target-speed deviation, normalized six-axis load and jerk costs (below). | Our motion and speed objectives, not the full SliceIt reward. |
| Board geometry and reaction | Official scene contains a board collision mesh. No board latch or board-hit failure in task code. | Finite-blade geometric boundary; no calibrated blade-board reaction model. | Added here to prevent penetration; missing board physics is not an upstream requirement. |
| At the board | No permanent freeze in SliceIt task code. | Per-substep nonpenetration projection; material, controller and sensors continue; lateral/lift proposals execute; arrival counter accumulates. | Our geometric boundary, not a physical board reaction. |
| Material separation | Not checked by the pose-success code. | Separate diagnostic only; never a success gate or reward. | Do not relabel pose success as physical separation. |

Official sources:
[DiSECt simulation config](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_rl/config/simulation/slicing_3d_disect.yaml),
[ordinary simulation config](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_rl/config/simulation/slicing_3d.yaml),
[real config](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_rl/config/real/slicing_3d.yaml),
[termination](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_openai/src/ur3e_openai/task_envs/force_control/disect_slicing.py),
[reward](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_openai/src/ur3e_openai/cost_utils.py),
[board collision geometry](https://github.com/omron-sinicx/sliceit/blob/3a6964862f1f135af9d98dd400efeb44ba394c15/catkin_ws/src/o2ac_scene_description/urdf/cooking_environment.xacro).
[SliceIt III-C.4](https://arxiv.org/html/2404.02569v1#S3.SS3.SSS4) uses target-pose completion.

## Completion, failure and timeout

- Goal: the captured robot-base end-effector pose translated along the board
  normal until the lowest point of the actual finite knife is at the board stop
  height (blade-to-tool offset and orientation included). The policy does not
  choose it.
- Arrival sample: final EE distance in robot-base XY strictly below 40 mm
  (`completion_position.radius_m = 0.04`), z error < 2 mm, pitch-error
  component < 0.04 (quaternion-vector component, about 0.08 rad; not radians).
  The counter increases only when all hold; success needs the current sample
  inside and counter > 10 (cumulative, not consecutive).
- Success earns +100 once, only without force or workspace failure. Failure
  takes precedence and costs -100 once, including on the same step as arrival.
  The environment requires reset after an ending.
- Timeout at 500 control steps (25 s at 0.05 s) is an unsuccessful terminal
  without completion bonus or value bootstrap. Gamma is 1. Initialization
  settling is outside this clock.
- Board contact is neither success nor failure. Penetration below
  -2 micrometres aborts the simulator transition (an implementation error, not an
  RL failure).
- Material separation, residual springs, coverage and sliding are reported
  separately. Pose-success rate must not be described as a separation rate.

## Board boundary (`project_continue_v2`)

| `task.board.motion` | Code | `board_contact` | Joint projection | Training |
|---|---|---|---|---|
| absent | b367fcd–6ec9f34 | absorbing latch | shortened swept step, then frozen | refused; evaluate only |
| `project_continue_v1` | f12ca04 | projected AND clearance <= 10 um | minimum-norm clearance-gradient correction | refused; evaluate only |
| `project_continue_v2` | current | clearance <= 10 um (state property) | pure normal EE translation | current |

- Both backends correct an inward proposed joint endpoint so that the removed
  end-effector displacement is a pure translation along the board normal: the
  corrected pose is the proposal shifted by exactly the clearance deficit plus a
  0.2 um float32 margin, with zero rotation and zero tangential displacement
  (`tool_mount.project_joint_endpoint`: Newton on the 6-D pose residual with the
  full EE Jacobian; minimum joint norm among exact solutions for redundant
  chains). Joint limits are re-applied afterwards; if a limit binds, the clamped
  pose is kept only if it still clears the stop, otherwise projection raises. A
  Jacobian too singular to realise the normal translation (more than 1e3 rad
  joint correction per metre of lift, or an unreachable normal direction) raises.
  The inward normal EE velocity is removed the same way. The configured endpoint
  gap is 1 micrometre. This is a geometric execution constraint, **not** a
  knife-board force or friction model. Material-ground and blade-material
  contacts remain active.
- After projection, the conservative swept-curvature check can shorten a
  substep whose joint increment is large relative to the clearance; shortening
  scales the whole substep, so it reduces (never redirects) lateral motion.
- Every 2 ms material interval and sensor/filter update continues at the
  boundary. Force violations and evaluation pause still stop their own worlds.
  No board-contact flag prevents the next action from executing.
- `board_contact` is 1 iff the finite-blade clearance of the end-of-step pose is
  <= 10 um (`tool_mount.board_contact`), whether or not a projection acted. The
  physical adapters evaluate it on the rigid-blade FK of the recorded joints
  (also at reset); the WM, analytic MPC and MPPI on the post-projection imagined
  joints, with the same function.
- WM, analytic MPC and MPPI call the same endpoint/velocity projection
  (autograd Jacobian, differentiable for BPTT; the physical adapters use the
  chain's analytic Jacobian, same residual and tolerance). The WM's discrete
  transition does not resolve the physical substep swept trajectory; that is a
  model approximation.
- There is no simulated board reaction in the sensor, no first-arrival terminal,
  no board-hit penalty and no V4 task.

Probe: `scripts/check_board_continue.py`; measurements in [RESULTS](RESULTS.md).

## Rewards

Let g_t be the lowest point of the recorded finite-blade rigid proxy above the
board, from robot joint state and blade mounting; s the positive board stopping
clearance; D = g_reset - s > 0 the per-world travel recorded at reset:

    rho_t = max(rho_(t-1), clamp(1 - (g_t - s)/D, 0, 1)),   rho_reset = 0

- Progress reward: 10 times the increase in rho. Retraction/re-entry cannot earn
  progress again. Descending through free space does earn progress: this is
  motion shaping, not a cutting or contact detector.
- Speed penalty, with positive commanded feed v* and m = ((v-v*)/(|v|+v*))^2:
  20*m*(positive delta-rho + 1/horizon).
- Load/jerk cost at the 50 ms control period: 0.05*(min(u,1)+0.1*min(j,1)), with
  u the maximum absolute normalized six-axis sensor wrench and j the normalized
  jerk measure.
- Success +100, failure -100 (above). Success does not require rho = 1.
- With final-only defaults the running-penalty bound is
  2*20 + 500*0.05*(1+0.1) = 67.5; adding maximum progress 10 gives 77.5 < 100,
  so terminal settlement dominates. With the optional tool-motion cost the bound
  plus progress is 97.5 < 100.

The backends and the WM call the same depth helper at control-step endpoints.
Depth span and accumulated progress are known history fields; the WM derives
progress analytically from predicted joints. The per-episode span supports
different objects and start heights without per-object weights. Physical
separation is independent: DiSECt reports residual springs, BladeMaster its
native topology diagnostics. Do not report rho = 1 as material separation.

## Force sampling and WM state

- Within one control step, every 2 ms filtered reading produced in that step
  enters the component-wise peak, the force-limit check and the safety_margin
  metric exactly once, right after it is produced; the carried reading from the
  previous step is not counted again. A violation stops the world at that sample.
  Code: `CuttingBatch.fresh_sample_peak` (set by GPUEnvironment for v3) and
  `BladeMasterBatch.step`. `peak_force` (material force magnitude) is not a
  filtered reading and is not affected.
- WM state adds `last_wrench` (6; newest filtered sensor sample, predicted,
  scaled by the force limits) and `board_contact` (derived, not predicted).
  Contact-start transitions are fitted normally. The dynamics loss is averaged
  over valid transitions (`wm_loss_normalization=valid_transitions_v1`); the
  latch mask (`separate_fit_valid_v1`) applies only to old latch metadata.
- Imagined survival: success, failure and timeout are settled analytically,
  survival = prod(1 - absorbed) in actor imagination, GAE targets and WM+MPPI.
  The terminal head is still fitted with BCE and logged, not used as survival.
- Reset (full or partial) re-initialises `last_wrench` and recomputes
  `board_contact` only for reset worlds. Generic `restore_frame` is unsupported
  for board tasks.
- The v3 WM prediction diagnostic is `scripts/diagnose_pose_predictions.py`
  ([RUNNING](RUNNING.md#pose_depth_v3-wm-prediction-diagnostic)).

## Actions and scene boundaries

The six outputs are x stiffness, pitch stiffness, vertical P gain, pitch P gain,
x target increment and pitch target increment. With max_twist=[1,0] and pitch
disabled, the last channel does not change the target: five effective mapped
controls, not six independent Cartesian commands. Pitch can still respond
mechanically through the compliant controller. The x mapper accumulates up to
2 mm of target offset per 50 ms step; there is no separate accumulated-offset clamp.

Failure checks constrain the robot-base EE position (x=[0.038,0.238],
y=[0.405,0.605], z=[0.096,0.269] m) and the six-axis wrench. Both backends
support the material on an infinite plane; the rendered wood board is display
geometry, not an object-fall detector or workspace boundary.

## Robot-position modes

Both measure the EE relative to the prescribed goal in robot-base XY; the knife
is rigidly mounted. There is no particle-motion penalty in the shared task.

- `tool_motion.enabled = false` (selected): final only. Off-target intermediate
  movement carries no position cost or failure.
- `tool_motion.enabled = true`: path plus final. Distance <= 25 mm is free;
  25–60 mm subtracts `20/500 * clamp((d-.025)/(.060-.025),0,1)^2` per step;
  > 60 mm is `tool_motion` failure (-100 once, no +100). At 25/35/42.5/50/60 mm
  the per-step costs are 0/.0032653/.0100000/.0204082/.0400000.

`completion_xy_error` is derived from FK and the goal (a state/history field, not
a prediction head); policy inputs stay 34 per frame and outputs six. Boundary
comparisons allow eight machine epsilons at metre scale (about 1 um in float32);
this is not a physical tolerance. Path checks use 50 ms endpoints.
`check_pose_depth.py --tool-reward-audit` compares both modes on measured
transitions with counterfactual goal offsets.

## Optional pitch

`bounded_pitch_v1` is disabled by default. When enabled, pitch target updates
use the 50 ms agent period, a configured degrees/s rate and a total-angle bound;
the WM uses the same differentiable mapping. Tilted contact is not validated.

## BladeMaster cut-topology diagnostics (logging only)

At each BladeMaster episode end, episode rows (training `episodes.jsonl` and
`evaluate_pose_policy` cases) carry `bm_separation`
(`cutting_world_model/blademaster_separation.py`). It is never reward, success,
policy or WM input. Native BladeMaster has no "prescribed slice detached" field,
so `separated` is always null with `separation_note`; `material_components` > 1
is not separation of the prescribed cut. Fields:

- `component_sizes`: sorted particle counts of native topology.components.
- `side_coded_fraction`, `side_1_particles`, `side_2_particles`: digit of cut 0.
  With the external tool `free_blade` is 0, so assign labels any particle the
  edge has passed within 6 grid spacings of the blade plane (or pinched on the
  board), independent of damage (BladeMaster batch_contact.py:444-469).
- `bridging_code0_particles`: unlabelled particles whose stencil nodes are shared
  with both side-1 and side-2 particles (one-hop bridges only).
- `band_damage_saturated_fraction`: damage>=1 among particles within
  cut_band_cells x grid spacing of the prescribed plane (reference coordinates).
- `side_boundary_offset_m`: best 1-D split of side-1/side-2 reference positions
  along the prescribed plane normal.
- `board_pinched_particles`: null; the pinch test is internal to assign.
- `separated_bm_candidate`: candidate only (two largest components >= 10% each,
  reference centroids on opposite sides of the plane); used nowhere.

## Key decisions

- 2026-11-14: free_cut_pose_v1 — +100 for prescribed pose arrival instead of full spring release; finite-blade board stop added.
- 2026-11-16: pose_depth_v2 — geometric depth progress replaces spring-release shaping in both backends; final-only XY radius selected; particle-slip penalty retired.
- 2026-11-17: pose_depth_v3 — board-state/last-wrench WM fields, per-step fresh-sample force rule, analytic imagined survival.
- 2026-11-19: horizon aligned with SliceIt, 180 -> 500 steps; old manifests keep 180.
- 2026-11-19: permanent board latch removed (`project_continue_v1`, f12ca04), then replaced by pure-normal projection and state-property contact (`project_continue_v2`).
- 2026-11-21: prism removed from the current scene by user decision; objects are cucumber_003 and half potato (results with three objects are not directly comparable).
