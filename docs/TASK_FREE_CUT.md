# Free-object single cut — shared geometric-progress pose task

Updated 2026-11-19. The current task is **pose_depth_v3**, with explicit
`task.board.motion = project_continue_v2`. Permanent cross-step freezing is
removed; the cumulative arrival rule is retained. Historical V3 metadata without
this field retains its old latch interpretation for evaluation, and
`project_continue_v1` (f12ca04: projection-only contact flag) is evaluated with
its recorded rules. Do not mix their replay/checkpoints with the current task;
metadata compatibility checks apply. pose_depth_v2 remains read/evaluate-only.
Training entry points reject V3 configurations whose `board.motion` is not
`project_continue_v2`; regenerate from the shared contract.
Origin (2026-11-16): the depth-progress pose task is shared by
DiSECt and BladeMaster. The user requested removing spring-release shaping and
using BladeMaster-style downward progress. Previously trained checkpoints retain
free_cut_pose_v1 / blademaster_pose_v1; their results have not been relabelled.

## Rule provenance and implemented continuation (2026-11-19)

This table separates upstream rules, **implemented behaviour**, and proposed changes.
It is not authorization to silently change a task while documenting it.
Reference: official SliceIt superproject `3a6964862f1f135af9d98dd400efeb44ba394c15`
pins ur_python_utilities `7cd7cb8ce491b7953fce61ca891382e893e2ec43`.
The local reproduction checkout `1e79aba` is not the unmodified upstream revision.
Source snapshots and the reproduction diff are recorded on GPU7 in
`runs/sliceit_task_rule_audit_20261119/evidence.json`.

| Rule | Official SliceIt source | Implemented shared pose_depth_v3 | Provenance / reason for difference |
|---|---|---|---|
| Arrival and +100 | Current pose error satisfies selected-axis thresholds and cumulative arrival counter exceeds threshold; no force/torque collision. No material-separation test. | Pose arrival plus cumulative counter >10; no higher-priority failure; +100 once. Also requires the configured final XY region. | Pose-based completion is inherited. Final XY region and explicit failure precedence are our task choices; completion does not certify cut-through. |
| Height tolerance and counter | Ordinary simulation: 0.5 mm, >10. **DiSECt simulation: 2 mm, >10.** Real: 2 mm, >1. | 2 mm, >10. Counter increments on qualifying samples; it is not a consecutive-hold requirement. | The current pair already exists in official DiSECt simulation. It is **not** a mixture of real tolerance and simulation counter. Do not remove the counter merely to match upstream. |
| Force/torque limits | Component limits; collision ends the episode and contributes -100. DiSECt simulation uses 50 N / 10 N m; real config uses 50 N / 5 N m. | Six-component limits 50 N / 5 N m, checked on fresh 2 ms filtered samples; -100 once. | Our shared sensor/limit contract uses the latter scale. Current sample timing and peak bookkeeping are adapter-specific, not an upstream reproduction claim. |
| Workspace | Out-of-workspace ends the episode. The slicing reward has no separate fixed -100 for this event. | Explicit robot-base workspace; exit ends the episode with -100. | Our failure settlement makes workspace violations costly; it is a deviation, not an upstream -100 rule. |
| Time and step cost | Configured 500 steps, passed to the environment loader. Configured step cost -1 is part of the normalized continuous reward; with the listed equal weights its contribution is -0.25. | 500 steps / 25 s (aligned 2026-11-19); timeout is terminal with no bootstrap. No separate fixed per-step -1. | Horizon aligned by user request. Continuous rewards and timeout/bootstrap semantics remain our task design. |
| Continuous rewards | Normalized distance, force and jerk terms, plus normalized step cost. | Historical maximum geometric-depth increment, target-speed deviation, normalized six-axis load and jerk costs; formulas below. | Our motion and speed objectives, not the full SliceIt reward. Spring-release shaping has been removed from the current task. |
| Board geometry and reaction | Official scene contains a board collision mesh. Task code has no permanent board latch or special board-hit failure. Geometry alone does not validate the executed contact response. | Finite-blade software boundary; no calibrated blade-board reaction model in the formal task. | Boundary was added here to prevent penetration after migration; do not describe missing board physics as an upstream requirement. |
| After software board stop | No corresponding permanent freeze in SliceIt task code. | Per-substep nonpenetration projection; material, controller and sensors continue. Lateral/lift proposals can execute. Arrival counter still accumulates. | Our geometric boundary, not a physical board reaction. The permanent latch introduced in b367fcd is historical only. |
| Material separation | Not checked by the cited pose-success code. | Separate diagnostic only; never a success gate or reward. BladeMaster separated is unknown where no validated diagnostic exists. | Preserve pose success as the evaluation standard; do not relabel it as physical separation. |

Official sources:
[DiSECt simulation config](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_rl/config/simulation/slicing_3d_disect.yaml),
[ordinary simulation config](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_rl/config/simulation/slicing_3d.yaml),
[real config](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_rl/config/real/slicing_3d.yaml),
[termination](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_openai/src/ur3e_openai/task_envs/force_control/disect_slicing.py),
[reward](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_openai/src/ur3e_openai/cost_utils.py),
[board collision geometry](https://github.com/omron-sinicx/sliceit/blob/3a6964862f1f135af9d98dd400efeb44ba394c15/catkin_ws/src/o2ac_scene_description/urdf/cooking_environment.xacro).

## Current V3 continuation (`project_continue_v2`)

Revisions of `task.board.motion` (machine-checked; the task name stays v3):

| Value | Code | `board_contact` | Joint projection | Training |
|---|---|---|---|---|
| absent | b367fcd–6ec9f34 | absorbing latch | shortened swept step, then frozen | refused; evaluate only |
| `project_continue_v1` | f12ca04 | projected AND clearance <= 10 um | minimum-norm clearance-gradient correction | refused; evaluate only |
| `project_continue_v2` | current | clearance <= 10 um (state property) | see below | current |


- Success is unchanged: cumulative arrival count >10, 2 mm height tolerance,
  final XY region, no higher-priority failure, +100 once. Board contact itself
  is neither success nor a new failure. Horizon remains 500 steps / 25 s.
- Both backends correct an inward proposed joint endpoint using the finite
  blade clearance Jacobian (minimum joint correction). A conservative swept
  curvature check can further shorten the motion to prevent crossing the
  physical plane. The configured positive endpoint gap is 1 micrometre.
  This is a geometric execution constraint, **not** a knife-board force model.
  Projection can change joint direction; it is not an exact Cartesian friction
  or sliding model. Material-ground and blade-material contacts remain active.
- Every 2 ms material interval and sensor/filter update continues at the
  boundary. Force violations and evaluation pause still stop their own worlds.
  No board-contact flag prevents the next action from executing.
- `board_contact` is a property of the end-of-step pose: 1 iff the finite-blade
  clearance above the board plane is <= 10 um (`tool_mount.board_contact`),
  whether or not a projection acted in that step. DiSECt and BladeMaster evaluate
  it on the rigid-blade FK clearance of the recorded joints; the WM, analytic
  MPC and MPPI on the post-projection imagined joints, with the same function.
  A recorded state resting on the board therefore has contact 1 in both, and a
  blade hovering more than 10 um above has 0. It is not an absorbing latch.
  `last_wrench` remains the newest filtered sensor sample, never a held old reading.
- WM uses the same finite-blade endpoint projection and fresh field prediction.
  Its discrete transition does not resolve the physical substep swept trajectory;
  that remains a model approximation. Peak is at least the predicted absolute
  last reading. Contact-start transitions contribute to dynamics loss normally;
  the historical latch mask/independent denominators only apply to old metadata.
- Analytic survival, force sampling fixes, cumulative counting and the reward
  formula/weights are retained. WM+MPPI and analytic MPC use this state spec;
  SAC uses the same physical environment. Diagnostics accept either V3
  interpretation and record which one was evaluated.
- The independent blade-board force/FDCC prototype is not merged. There is
  still no simulated board reaction in the sensor. No first-arrival terminal,
  board-hit failure penalty or new V4 task has been introduced.

Validation and limitations: see `RESULTS` entry for this change and
`scripts/check_board_continue.py`. Old latch-specific probe
`check_wm_latch_learning.py` explicitly refuses the current continuation mode.

## Historical V3 latch semantics (2026-11-17 to 2026-11-19)

The following description and measurements apply only to V3 records whose
`task.board` has **no** `motion` field. They are retained for audit, not current
training instructions.

## pose_depth_v3 — software board-stop consistency (2026-11-17)

Historical task semantics since 2026-11-17 (both
backends; v2 copy in configs/archive/shared_scene_v1_pose_depth_v2_20261116).
Users do not choose v2/v3: checkpoints, replay and manifests record the version
and code checks compatibility. scripts/train_gpu.py and scripts/train_sac_multi.py
refuse pose_depth_v2 (fresh or resumed); legacy evaluation of v2 checkpoints
is retained. diagnose_pose_predictions.py is v3-only. Operational training
status belongs in PLAN and each run's progress/manifest, not this task definition.
All four controllers coexist on main; see the
[controller/version ledger](methods/CURRENT_BASELINES.md).

pose_depth_v3 = pose_depth_v2 task + software board-stop boundary consistency
fix + analytic imagined survival. The board stop is the execution interlock
introduced in this repository (commit b367fcd). It is not a board contact-force
model and does not reproduce DiSECt or SliceIt behaviour; neither has a latch.

Changed (v3 only):
- Frame/WM state: `board_contact` (latch flag 0/1; history field, derived, not
  predicted) and `last_wrench` (6; force-limit signal at the last 2 ms sample,
  net model = filtered sensor wrench; physical field predicted by the dynamics,
  scaled by the force limits).
- Latch: contact' = contact OR (step shortened by the stop AND end clearance
  <= 10 um). Same criterion in DiSECt, BladeMaster and the WM board projection.
- Latched state: q, wrench history, last_wrench, depth progress and object slip
  held; qd = 0; peak = |last_wrench|. Step count, arrival counter, remaining
  time, motion command and observation continue; observed EE velocity is zero,
  as in both physical adapters.
- Peak rule, both backends: the latch interval keeps its true in-interval peak;
  every later interval reports |frozen reading| (never zero, never carried).
  DiSECt already did this; BladeMaster v2 reported zero after the latch.
  The force-limit check and load cost read the same signal.
- Per-step sampling rule, both backends (v3 only; 2026-11-17): within one control
  step, every 2 ms filtered reading produced in that step (each substep's new
  reading, including the last) enters the component-wise peak, the force-limit
  check and the safety_margin metric exactly once, right after it is produced.
  The reading carried over from the previous step (or the initialised reading
  after reset) is not counted again. A violation stops the world at that sample:
  the next substep no longer moves it (same stop sample as before). Worlds
  stopped without a latch (collided, or scripts' evaluation_paused) produce no
  reading and add nothing; board-latched worlds follow the frozen-reading rule
  above. Code: CuttingBatch.fresh_sample_peak (set by GPUEnvironment for v3) and
  BladeMasterBatch.step; post-latch rule shared in
  cutting_gpu_backend/knife_board.latched_interval_peak.
  v2 difference (kept for traceability, unchanged): the DiSECt adapter reads the
  previous substep's reading at the START of each substep, so its peak, check and
  safety_margin include the previous step's last reading (counted in two steps);
  with the net force model it also checks the step's last reading after the
  loop. It also sampled stopped-but-not-collided worlds every substep. BladeMaster
  v2/v3 already counted only new readings; its v3 safety_margin now also includes
  the frozen reading of latched worlds, as its peak does.
  peak_force (material force magnitude) is not a filtered reading and is not
  changed by this rule.
  Checked (GPU9, 2026-11-17): stub tests on the real step code of both adapters
  (tests/test_peak_sampling_v3.py; identical reading sequences give identical
  peak, collision and safety_margin); real simulation, cucumber_003, 2 worlds per
  backend, 24 reference steps: peak equals the component-wise max of that step's
  25 readings in every step; a v2-style carried reading would have raised some
  component in 11/24 (DiSECt) and 6/24 (BladeMaster) steps. No latch or force
  violation occurred in that short run; those cases are covered by the stub tests.
- Reset (full or partial): latch and last_wrench reset only for reset worlds;
  last_wrench = the initialised sensor reading. Generic restore_frame remains
  unsupported for board tasks.
- WM fit (normalization corrected 2026-11-19): transitions whose recorded start
  state is latched are zero-weighted in the dynamics loss; the first-contact
  transition is kept. Dynamics error is divided by the number of valid unlatched
  starts, while terminal BCE is divided by all valid transitions. Microbatches
  use these two full-batch denominators independently. A batch without unlatched
  starts skips the dynamics optimizer update (including Adam momentum/decay);
  terminal-head fitting continues. Logged: dynamics_fit_count,
  dynamics_masked_count, dynamics_masked_fraction, dynamics_skipped_batches.
  Checkpoints record wm_loss_normalization=separate_fit_valid_v1; resuming an older
  v3 checkpoint warns that its training loss changes. Network weights and
  evaluation semantics remain load-compatible. v2/free_cut loss rules are unchanged.
- Imagined survival: success (+100), failure (−100) and timeout are all settled
  analytically, so survival = prod(1 − absorbed) in actor imagination, GAE
  targets and WM+MPPI. The terminal head is still fitted with the same BCE and
  gradient route (with terminal_gradient=dynamics its gradient still reaches the
  dynamics); its probabilities are logged, not used as survival.
  scripts/diagnose_free_cut_predictions.py is the historical free_cut_v3
  separation diagnostic; it refuses every pose task (including v3) at load, so its
  analytic-survival line is never reached. The v3 WM prediction diagnostic is
  scripts/diagnose_pose_predictions.py (docs/RUNNING.md).
- Reading historical WM losses: before the 2026-11-19 normalization correction,
  v3 divided masked dynamics error by all valid transitions, so adding latched
  rows diluted that term. New losses average only the transitions being predicted;
  report the fit/masked counts and do not interpret the resulting increase in
  state_loss relative to old logs as worse prediction. Adding latched rows can
  still change BCE and its gradient; its population has not been restricted.
- v3 refuses v2 checkpoints, critics and replay (state shape/version checks).

Unchanged: reward weights, +100 arrival rule, −100 failures, thresholds,
horizon, workspace, force limits, board stop gap. Final-pose region still
permissive (user decision pending). Pose success is still not cut-through;
separation remains a diagnostic. pose_depth_v2, free_cut_v3, free_cut_pose_v1
and blademaster_pose_v1 behave as before.

Measured vs not: v3 guarantees post-latch rule consistency between the physical
adapters and the WM (tests/test_pose_depth_v3.py; real-simulation latch check below). First-contact timing in imagination still
depends on dynamics prediction error and the rigid finite-blade projection
proxy; that is not measured.

Real-simulation latch check, GPU9, 2026-11-17 (temporary v3 copy of the shared
scene, prism and cucumber_003, 2 worlds each per backend; reference controller at
.05 m/s with its end pushed 20 mm past the board goal so the stop latches; task
target unchanged). In every world: latch interval peak = true in-interval max of
the 2 ms samples, last_wrench = last sample; every later step peak = |last_wrench|,
q, wrench history and last_wrench unchanged, qd = 0, step/arrival counters kept
running until pose success (counter 11); packed v3 WM state matched the frame;
partial reset cleared the latch with last_wrench = sensor reading (0 after settle);
other worlds unchanged; no NaN.
The 20 mm push past the goal is a test-only device to trigger the stop; it is not a
training setting. Under the task rules, reaching the target pose within the 2 mm
height tolerance without touching the board is a valid success (the unmodified
reference did so at steps 38–45).

| Backend | Object, world | Latch step | Gap at latch (um) | Latch peak (N) | Later peak (N) | End |
|---|---|---|---|---|---|---|
| DiSECt | prism 0 / 1 | 29 / 23 | .97 / 1.03 | 9.79 / 9.97 | 1.44 / 5.72 | success 38 / 33 |
| DiSECt | cucumber 0 / 1 | 31 / 29 | 1.02 / .99 | 2.41 / 3.57 | 2.19 / 2.87 | success 40 / 38 |
| BladeMaster | cucumber 0 / 1 | 28 / 23 | 1.02 / .99 | 1.12 / .027 | .031 / .027 | success 38 / 32 |
| BladeMaster | prism 0 / 1 | 30 / 29 | 1.06 / 1.02 | .53 / .18 | .29 / .038 | success 39 / 38 |

Forces: largest force component. Not a training or success-rate result.

### BladeMaster cut-topology diagnostics (v3, logging only; 2026-11-17)

At each pose_depth_v3 BladeMaster episode end, episode rows (training
episodes.jsonl and evaluate_pose_policy cases) carry `bm_separation`
(cutting_world_model/blademaster_separation.py). It is never reward, success,
policy or WM input. Native BladeMaster has no "prescribed slice detached" field,
so `separated` is always null with `separation_note`; `material_components` > 1 is
not separation of the prescribed cut. Fields:

- `component_sizes`: sorted particle counts of native topology.components.
- `side_coded_fraction`, `side_1_particles`, `side_2_particles`: digit of cut 0
  (the batched coupling always uses cut index 0). With the external tool
  `free_blade` is 0, so assign labels any particle the edge has passed within
  6 grid spacings of the blade plane (or pinched on the board), independent of
  damage (BladeMaster batch_contact.py:444-469).
- `bridging_code0_particles`: unlabelled particles whose own stencil nodes are
  shared with both side-1 and side-2 particles (one-hop bridges only).
- `band_damage_saturated_fraction`: damage>=1 among particles within
  cut_band_cells x grid spacing of the prescribed plane in reference coordinates.
- `side_boundary_offset_m`: best 1-D split of side-1/side-2 reference positions,
  signed along the prescribed plane normal (blade local x at the task target).
- `board_pinched_particles`: null; the pinch test is internal to assign.
- `separated_bm_candidate`: candidate only (two largest components >=10% each,
  reference centroids on opposite sides of the plane); used nowhere.

## Horizon alignment — 2026-11-19

The user requested aligning the episode duration with SliceIt: the current
shared contract uses **500 control steps x 0.05 s = 25 s** for both backends.
Initialization settling is outside this clock. Earlier success/failure still
ends the episode; timeout remains terminal without bootstrap. No arrival rule,
latch behaviour, reward formula or weight is changed here.

The speed time term and optional tool-motion term use 1/horizon, so their
per-step contributions now use 1/500 rather than 1/180. This follows the existing
formula, not a retuned weight. Load and jerk remain integrated over physical
time, so their maximum full-episode cost increases with the horizon.
With final-only defaults, the finite running-penalty bound is
2*20 + 500*0.05*(1+0.1) = 67.5; adding maximum progress 10 gives 77.5 < 100,
so the existing terminal-settlement validation still passes. If optional
tool-motion cost is enabled (weight 20), the bound plus progress is 97.5 < 100.
These are algebraic bounds, not a new physical reward-preference experiment.

Regenerate both backend input files from the shared contract. Its fingerprint
changes; do not relabel old 180-step manifests/checkpoints/replay or bypass
contract checks to continue them as 500-step runs. Frozen evaluations retain
their recorded horizons.

Validation on GPU7: regenerated both backend configs and checked their identical
500-step task contracts; actual WM analytic termination with synthetic states
does not time out at steps 180 or 499 and does at 500 for both backend identities.
Root GPU suite: 713 passed / 5 skipped (legacy Gym/TF and opt-in bridge);
backend suite: 43 passed. Evidence: `runs/horizon_500_check_20261119/`.
No new 500-step physical reward-preference audit or training was run.

## Current reward and state semantics

Let g_t be the lowest point of the recorded finite-blade rigid proxy above
the board, obtained from robot joint state and blade mounting. Let s be the
positive board stopping clearance and D = g_reset - s > 0 the per-world travel
recorded at episode reset. Define

    rho_t = max(rho_(t-1), clamp(1 - (g_t - s)/D, 0, 1))

Reset rho is zero.

- Progress reward is 10 times the increase in rho.
- Both parts of the speed penalty remain. With positive commanded feed v*,
  m = ((v-v*)/(abs(v)+v*))^2, the penalty is
  20*m*(positive delta-rho + 1/horizon). Its progress factor is now **depth**,
  never released springs.
- At the 50 ms control period, load/jerk cost is
  0.05*(min(u,1)+0.1*min(j,1)), where u is the maximum absolute normalized
  six-axis sensor wrench and j is the existing normalized jerk measure.
- Success remains +100 for prescribed pose arrival without force/workspace
  failure, with the existing arrival counter and board boundary. It does not
  require rho = 1; reaching the depth alone does not imply success.
- Force/workspace failure is -100 once; timeout ends without bootstrapping.
  Gamma remains 1. The current shared contract uses 500 steps / 25 s;
  archived check configurations and historical runs retain their recorded 180 steps.
- Retraction/re-entry into a depth already visited cannot earn progress again.
  Descending through free space does earn geometric progress: this is motion
  shaping, not a material-cutting or contact detector.
- Physical separation is independent: DiSECt reports actual residual springs;
  BladeMaster reports its native topology diagnostics. Do not report geometric
  rho = 1 as material separation.

The two backends and WM call the same depth helper at control-step endpoints.
Depth span and accumulated progress are known history fields, not learned
physical outputs; the WM derives progress analytically from predicted joints.
The dynamics inputs no longer contain spring progress in this version.
The actor observation/action interfaces are unchanged. The per-episode span
supports different objects and start heights without per-object reward weights.

This changes WM state/input/output dimensions relative to the old spring-progress
model. Old replay/WM/critic cannot be silently resumed as this task. New metadata
records task identity, backend identity and geometric progress semantics. No
long training is started by the verification configurations:
configs/archive/scene_checks_20261116/free_cut/depth_three_object_check.json and
configs/archive/scene_checks_20261116/blademaster/depth_three_object_check.json.

The optional tool revision `disect_knife_rigid_mount_v1` now unifies physical
blade geometry and the rigid UR5e mount; see [shared knife](SHARED_SCENE.md#shared-physical-knife-and-mounting).
In that revision the DiSECt physical knife follows robot FK through substep
interpolation, and the two backends share the finite-tool board interlock.
Material laws, settling and sampling distributions remain separate. Existing
checkpoints retain their original tools and rewards. Residual material
connections remain a separately recorded outcome.

## Historical pose-v1 definition (previous completed runs)

Scope note (2026-11-17): this section and its subsections through "Entry points
and retained history" describe `free_cut_pose_v1` only. For the current
`pose_depth_v3` (and v2) task, release-progress shaping is replaced by geometric depth
progress (above), and `configs/free_cut/pose_*.json` are historical fixtures,
not current development configurations.

Updated 2026-11-14. The implemented task identity is `free_cut_pose_v1`.
The user authorized changing the +100 success event to prescribed pose arrival and
adding a nonpenetrating cutting-board boundary. Existing v3 running rewards are
retained in this change; it is not a replacement of the entire reward with SliceIt's.

### Completion and failure (pose-v1)

- The prescribed goal is the captured robot-base end-effector pose, translated
  along the board normal until the lowest point of the actual finite knife is at
  the board stop height. The blade-to-tool offset and orientation are included.
  It is not a target chosen by the policy, and not the old below-board goal.
- The existing SliceIt interface checks configured axes `[0,2,4]` (x, z, pitch),
  with thresholds `[0.04 m, 0.002 m, 0.04]`; the last is the selected quaternion-vector pose-error component, not radians. Its accumulated arrival counter
  must exceed 10 while the current observation is inside those thresholds.
  This is not a new consecutive-hold requirement.
- Pose completion earns +100 only without force or workspace failure.
  Force/workspace failure takes precedence and costs -100 once, including when
  arrival happens on the same step. The environment requires reset after ending.
- Timeout is an unsuccessful finite-horizon terminal, with no completion bonus
  and no value bootstrap. Historical pose-v1 configuration: 180 steps, 9 s (not the current shared horizon).
- All springs need NOT be released to earn pose completion. Material separation,
  residual springs, geometric coverage and sliding remain separately reported.
  Pose-success rate must not be described as a physical-separation rate.

### Board requirement (pose-v1)

The board is the DiSECt plane y=0. The existing finite-blade swept-motion stop
prevents downward and rotational penetration; it is a software interlock, not a
calibrated compliant board-contact force model. Ordinary board arrival is not a
failure and has no added penalty. Geometric coverage is not a success gate.
Invalid penetration below -2 micrometres aborts the simulator transition; it is
not scored as an RL failure. The stop aims for +1 micrometre clearance.

The WM uses the recorded finite-blade geometry with forward kinematics to project
predicted endpoints to the board boundary. This is a rigid kinematic proxy;
physical execution checks the integrated DiSECt knife, including the swept arc.
The boundary projection runs eagerly (including its implicit BPTT derivative); the rest of the WM can use the existing compilation path. The physical/proxy gap discrepancy is measured by `scripts/check_pose_cut.py`.
This proxy must not be described as a learned board contact-force model.

### Running rewards and world model (pose-v1)

Unchanged from v3: progress weight 10; speed weight 20 (progress and finite-time
components); load weight 1; jerk weight 0.1; gamma=1. Six-axis sensor limits remain
explicit configuration values, currently [50,50,50,5,5,5] for simulated development,
not calibrated Mini45 safety limits. See [historical equations](archive/TASK_FREE_CUT_V1_V3.md).

Existing release-progress prediction remains for running reward shaping, not
completion. No new coverage head is added. The +100 settlement is computed from
predicted pose/counter and failure conditions; the learned terminal head cannot
issue a probabilistic +100 completion reward. Its labels now mean task termination,
not separation. Hard pose/force/workspace/deadline termination is enforced in
imagination so terminal rewards and bootstrap are not repeated.

This changes the task identity and terminal semantics. Old v3 replay, critic and
completion-head weights must not silently resume as the new task. Old checkpoints
remain usable under their own version.

### Entry points and retained history (pose-v1)

- Historical pose-v1 configuration: `configs/free_cut/pose_cucumber.json`.
- Physical/update/checkpoint check: `scripts/check_pose_cut.py`.
- Training entry: `scripts/train_gpu.py --conditions configs/free_cut/pose_cucumber.json`;
  this file has four worlds and is not an approved long-training budget.
- Source SliceIt and v1-v3 remain for old checkpoints.
- v4 is the retired coverage-task evaluation, not a training entry.
- [PLAN](PLAN.md) lists remaining work and actual object readiness.
- [RESULTS](RESULTS.md) records physical measurements and their limits.

[SliceIt III-C.4](https://arxiv.org/html/2404.02569v1#S3.SS3.SSS4) uses target-pose
completion. The complete-spring-release criterion was our later task choice.
The new software board stop is our integration safeguard, not a claim to
reproduce SliceIt's compliant board-contact dynamics.


## Current action and scene boundaries (2026-11-16)

The six outputs are x stiffness, pitch stiffness, vertical P gain, pitch P gain,
x target increment and pitch target increment. With inherited max_twist=[1,0],
the final channel does not change the target. This is five effective mapped
controls, not six independent Cartesian position/rotation commands. Pitch can
still respond mechanically through the compliant controller.

The x mapper accumulates up to 2 mm of target offset per 50 ms control step.
There is no separate accumulated-offset clamp. Historical/default configurations
have no object-motion penalty. The current shared contract instead offers the
robot-position modes below, without changing historical checkpoints.
Failure checks constrain the robot-base EE position and six-axis wrench.
Verification boxes use x=[0.038,0.238], y=[0.405,0.605],
z=[0.096,0.269] m. Pose completion uses the source x/z/pitch-error tolerances
and arrival counter; the current shared contract additionally checks final XY
radius <40 mm. Material displacement, lifting and separation are recorded
independently; reaching depth alone does not complete the task.

Both backends support the material on a plane; the wood rectangle in the studio
image is display geometry. Its drawn edge is not an object-fall detector or the
EE workspace boundary. The finite-blade board interlock prevents downward and
rotational penetration and is separate from a compliant physical board-force law.

The superseded shared-knife checks used different material/start/speed settings.
They are archived. Current checks generate both configurations from the same
shared contract, including paired episode sampling; see SHARED_SCENE. Material
integration and constitutive laws still differ.

## Historical particle-motion constraint — 2026-11-16

Superseded in the shared contract by the tool-position modes below. This section
documents the earlier optional interface; fragment opening is not penalized in
the current scene.

User choice: small movement is free, excess movement has a soft cost, larger
movement terminates unsuccessfully. This is `task.object_slip` on pose_depth_v2,
revision `planar_mass_rms_v1`. Absence retains the prior task exactly.

The metric is mass-weighted RMS horizontal material-point displacement from
the **settled reset state of this episode**:

`d = sqrt(sum_i (m_i / sum_j m_j) * ||P_horizontal(x_i - x_i_reset)||^2)`.

DiSECt uses its particle masses and horizontal x/z; BladeMaster's equal-mass
particles use horizontal x/y. Opposite fragment motions do not cancel.
Rotation, deformation and fragment opening also contribute: this is a material
motion envelope, not a pure rigid-body translation estimate. Partial reset
replaces only reset worlds' reference clouds. DiSECt physical-frame restoration
requires the saved reference cloud. BladeMaster full-state restore remains
unsupported; it must not invent a reference from a mid-episode frame.

Development settings, common to both backends:
- d <= 25 mm: no extra cost.
- 25 < d <= 60 mm: subtract
  `(20 / horizon_steps) * clamp((d - .025) / (.060 - .025), 0, 1)^2` each step.
- d > 60 mm: `object_motion` failure; no success bonus, failure -100 once even
  with another simultaneous failure. Exactly 60 mm is still inside.
- Total additional running cost is bounded by 20 for any accepted finite
  episode; task validation includes it in the existing settlement bound.
- Measurement is at 50 ms control endpoints, not a continuous safety interlock.
  It is not a finite-board edge test or a lifting constraint.

25/60 mm are configurable engineering choices informed by the existing recordings,
not measured tolerances, optimized hyperparameters or Mini45 hardware limits.
Old-policy cases used different tools/material distributions and are not a fair
backend comparison. New matched experiments must freeze the chosen thresholds.

Physical execution and WM imagination share the penalty and hard-event formula.
The compact WM adds one supervised residual output/state, `object_slip` in
metres, normalized by failure_m (output delta scale failure_m/5), nonnegative.
Its ten-frame history and critic receive that field; policy observation remains
34 values / ten frames and receives no particle ground truth. Hard failure stops
imagined survival/bootstrap. Pose success is masked by the same failure.
This is privileged simulation supervision; a scalar is not a complete description
of fragment geometry. Learned motion accuracy and real-robot observability are
not established by the interface checks. No new camera pipeline is claimed.

WM/critic state layouts change; old task checkpoints/replay cannot silently
resume. Policy architecture stays the same, but old policies were not trained
with this historical cost. Archived reproduction inputs are under
configs/archive/scene_checks_20261116; --motion-reward-audit requires their
object_slip field. Current shared configurations use --tool-reward-audit instead.
Both audits compare task equations on recorded physical transitions and
counterfactual distances, not trained-policy responses.

## Shared scene initialization and optional pitch — 2026-11-16

Current scene pairing is defined in [SHARED_SCENE](SHARED_SCENE.md). The
earlier shared_motion_check configurations above remain motion-only historical
checks; they are not the current initialization scheme.

Scene initialization uses settled_surface_gap_v2, explicit paired per-world
clearances and a 5–30 mm range. It aligns the finite physical blade and the
robot tool together after settling, including masked resets, and recomputes
geometric progress span from the actual start. See SHARED_SCENE for the full table.

bounded_pitch_v1 is optional and disabled by default. When enabled, pitch target
updates use the 50 ms agent period, a configured degrees/s rate and total-angle
bound. The WM uses the same differentiable mapping; disabled mode preserves the
existing five effective channels. This option alone does not establish validity
of tilted contact. The pose goal and blade/board boundary remain unchanged.


## Current robot-position modes — 2026-11-16

The user selected two options. Both measure the robot EE relative to the
prescribed cut-completion target in the robot-base horizontal XY plane. The knife
is rigidly mounted; final height and pitch conditions keep their existing roles.
There is no new particle-motion field or particle-based penalty in the shared task.

- completion_position.radius_m = .04: final planar distance must be strictly
  below 40 mm. Original z error <2 mm, pitch-error component <.04 (quaternion-vector
  component, about 0.08 rad; not radians), and arrival counter
  >10 also apply. The counter increases only when all these conditions hold.
- tool_motion.enabled = false: final only. Off-target intermediate movement
  carries no additional position cost or position failure. Force/workspace/board
  rules still apply; failure to finish within the configured 500 steps is timeout.
- tool_motion.enabled = true: path plus final. Distance <=25 mm is free;
  25–60 mm has quadratic running cost, weight20 normalized by the current horizon500;
  >60 mm is tool_motion failure with -100 once. No +100 on a failure.

At 25/35/42.5/50/60 mm the extra per-step costs are respectively
0 / .0032653 / .0100000 / .0204082 / .0400000 (500-step horizon). At 61 mm the active path mode
adds the failure settlement. These distances are engineering defaults, not
measured real-robot tolerances. Settings apply to both backends identically.

Physical adapters and WM derive completion_xy_error from FK and the target.
It is a deterministic state/history field, not an independent prediction head;
policy inputs stay 34 per frame and outputs stay six. The optional hard event
is included in imagined termination, pose-success masking and episode metrics.
The changed task/state identity requires matched new checkpoints/replay.
Tests include both modes, both horizontal directions, exact boundary values,
failure precedence, physical/WM equation agreement, fit/BPTT and save/restore.

Floating-point boundary comparisons allow eight machine epsilons at metre scale
(about 1 micrometre for float32). This prevents a computed 60 mm endpoint from
spuriously failing after world-frame subtraction; it is not a physical tolerance
calibration. Final 40 mm remains excluded. Path checks use 50 ms endpoints.

The current path weight is a mild running preference, not a 25 mm hard limit:
ten 50 ms steps at 50 mm cost about .567, whereas exceeding 60 mm charges the
failure settlement once. The bounded physical rule demonstration is recorded
in RESULTS (2026-11-16). It validates settlement and rendering, not post-training
drift reduction. No weights were tuned to improve the demonstration.
