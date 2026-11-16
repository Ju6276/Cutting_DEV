# Free-object single cut — shared geometric-progress pose task

Updated 2026-11-16. New opt-in task **pose_depth_v2** is shared by
DiSECt and BladeMaster. The user requested removing spring-release shaping and
using BladeMaster-style downward progress. Previously trained checkpoints retain
free_cut_pose_v1 / blademaster_pose_v1; their results have not been relabelled.

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
  Gamma remains 1; these check configs retain 180 steps.
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
configs/free_cut/depth_three_object_check.json and
configs/blademaster/depth_three_object_check.json.

The optional tool revision `disect_knife_rigid_mount_v1` now unifies physical
blade geometry and the rigid UR5e mount; see [SHARED_KNIFE](SHARED_KNIFE.md).
In that revision the DiSECt physical knife follows robot FK through substep
interpolation, and the two backends share the finite-tool board interlock.
Material laws, settling and sampling distributions remain separate. Existing
checkpoints retain their original tools and rewards. Residual material
connections remain a separately recorded outcome.

## Historical pose-v1 definition (previous completed runs)

Updated 2026-11-14. The implemented task identity is `free_cut_pose_v1`.
The user authorized changing the +100 success event to prescribed pose arrival and
adding a nonpenetrating cutting-board boundary. Existing v3 running rewards are
retained in this change; it is not a replacement of the entire reward with SliceIt's.

## Completion and failure

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
  and no value bootstrap. Current development configuration: 180 steps, 9 s.
- All springs need NOT be released to earn pose completion. Material separation,
  residual springs, geometric coverage and sliding remain separately reported.
  Pose-success rate must not be described as a physical-separation rate.

## Board requirement

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

## Running rewards and world model

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

## Entry points and retained history

- Development configuration: `configs/free_cut/pose_cucumber.json`.
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
  below 40 mm. Original z error <2 mm, pitch-error <.04 rad, and arrival counter
  >10 also apply. The counter increases only when all these conditions hold.
- tool_motion.enabled = false: final only. Off-target intermediate movement
  carries no additional position cost or position failure. Force/workspace/board
  rules still apply; failure to finish before 180 steps is timeout.
- tool_motion.enabled = true: path plus final. Distance <=25 mm is free;
  25–60 mm has quadratic running cost, weight20 normalized by horizon180;
  >60 mm is tool_motion failure with -100 once. No +100 on a failure.

At 25/35/42.5/50/60 mm the extra per-step costs are respectively
0 / .0090703 / .0277778 / .0566893 / .1111111. At 61 mm the active path mode
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
