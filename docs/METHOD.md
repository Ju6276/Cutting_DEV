# Implementation reference

Current implementation of the learner, physics interfaces and tooling for task
pose_depth_v3 (`board.motion=project_continue_v2`). Task equations:
[TASK_FREE_CUT](TASK_FREE_CUT.md); scene: [SHARED_SCENE](SHARED_SCENE.md);
decisions: [PLAN](PLAN.md); upstream revisions and patches: [SOURCES](reference/SOURCES.md).

## Active configuration

- Task pose_depth_v3 in both material backends: shared analytic maximum
  blade-depth progress in the backends and the WM, pose +100 with failure
  precedence, analytic imagined survival, geometric board boundary without
  permanent freezing. `train_gpu.py` and `train_sac_multi.py` refuse retired
  task versions (TASK_FREE_CUT).
- WM state: adds `last_wrench` (6, predicted, scaled by the force limits) and
  `board_contact` (end-of-step clearance <= 10 um, derived, not predicted).
  Contact-start transitions are fitted normally; dynamics loss and terminal BCE
  are averaged over valid transitions (`valid_transitions_v1`). The latch mask
  `separate_fit_valid_v1` applies only to old latch metadata.
- Imagined survival: success, failure and timeout are settled analytically in
  actor imagination, GAE targets and WM+MPPI; the learned terminal head is
  fitted and logged, not used as survival.
- Force sampling: every new 2 ms filtered reading of a control step enters the
  component-wise peak, force-limit check and safety margin exactly once.
- Policy input: 34 observations per frame (full six-axis sensor wrench) x10
  frames plus an 8D goal; six outputs, pitch disabled.
- Learner settings (selected, passed explicitly): real-data critic with 32-step
  windows ([`critic_target=real`](#variant-critic_targetreal)), 8-step actor
  imagination ([`actor_horizon`](#variant-critic_targetreal_wm-with-actor_horizon)),
  BC 0.25 ([`actor_bc_alpha`](#variant-actor_bc_alpha)); gamma 1.
- Schedule: transition-scaled (`--schedule transitions`), 30720 warmup
  transitions, 640 physical transitions per learning round, then 500 WM-only
  rounds; effective batches 2048 (WM) / 4096 (actor), microbatch 512, critic
  microbatch 8192. Pending learning rounds are checkpointed
  ([lifecycle](#training-lifecycle)).
- GPU memory: Torch cache release before every physics step/reset, expandable
  segments and a headroom guard are trainer defaults
  ([Torch/Warp GPU memory sharing](#torchwarp-gpu-memory-sharing)).
- `train_gpu.py` defaults remain the original SuperMimic `source` variant; the
  settings above must be passed explicitly ([RUNNING](RUNNING.md)).

## External implementation evidence

[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym/tree/0171f5eadee50492d6ff30fa04b55e16cc11b61e)
provides full-update equivalence checks, independent physical evaluation and
500 world-model-only iterations; all three are applied here. Its RunningNorm
critic and analytic soft termination are separate method choices, not adopted.
The implementation keeps static scales and still fits the Cuboi learned terminal
head, whose probabilities are only logged (imagined survival is analytic).
ndm-gym's soft termination is not used. SliceIt physical thresholds stay fixed. Analytic termination
alone cannot restore a force-limit event that the dynamics model failed to
predict.

## Control-history state

One frame has 107 coordinates: 42 predicted physical ones (joint position/velocity
12, three-sample wrench history 18, interval peak wrench 6, `last_wrench` 6) and
65 derived ones (motion command, previous pose/velocity/acceleration, the
34-wide observation, success counter, step count, depth progress, depth span,
completion XY error and `board_contact`). Ten frames give 1070 state coordinates
(`prediction_layout=control_history_v2`). The dynamics input takes 54
coordinates per frame, 540 in total: joints 12, wrench history/peak/last 30,
motion command 6, previous action 6. The model predicts a 42-wide delta. The
policy sees 34 observations x 10 frames plus an 8D goal. The WM fits 8-step
windows, the actor imagines 8 steps (`--actor-horizon 8`) and the real-data
critic uses 32-step recorded windows.

FK, observation, motion-command, derivative-history, counter, reward and
terminal formulas advance the derived coordinates analytically; derived pose,
velocity, acceleration, observation copies and counter feed the analytic heads
and critic, not the dynamics MLP. This follows the pinned source's separation
of physical dynamics state from analytic observation/reward features. History
length 10 matches the policy history; reset repeats the actual initial frame,
and each transition shifts only past frames. Particles, spring damage and native
controller/filter packets remain in physical recordings but are absent from
model inputs and targets. The SuperMimic residual MLP widths, normalized
squared-error form, terminal BCE, optimizer and effective batches are kept.
No learned latent state or recurrent network is added.

## World-model-first training

`SequenceReplay` retains the upstream storage and uniform sampling among eligible
start rows. It pads completed short windows to the maximum horizon with their
final recorded state and a false valid-step mask. Actions/rewards/done labels
in padding are zero and contribute no supervision. A short unfinished buffer
tail is excluded. A truncation ends a sequence but retains `done=0`; a natural
termination retains `done=1` on its entering transition. Same-episode and write
pointer checks prevent reset/overwrite crossings.

Masked training uses a recorded initial state as a finite inactive input and
freezes the predicted state after each window ends. Each microbatch contributes
in proportion to its valid transition count, preserving the effective-batch
gradient. State/terminal losses are means over real transitions; the first/last
prediction-error metrics average windows, with `nse_kstep` evaluated at each
window's last valid step. The sampler and masking are local data-handling changes
to the [pinned source replay](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/src/supermimic/data/replay_buffer.py).
The maximum horizon, model, policy/critic equations, optimizers, batch sizes and
physical task remain unchanged. Per-update counts report sampled real
transitions, terminal labels and short windows.

Task-terminal handling follows the episodic bootstrap rule in
[Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/):
true termination removes bootstrap; an external time truncation does not.
The current pose task's500-step horizon is part of the task and is a true
termination, not an external truncation; it does not bootstrap.
The correction adds one task-head evaluation per imagined step, changes neither
physical trajectories nor task rewards/thresholds, and does not retrain the world
model's terminal classifier using predicted labels.

The user-selected startup stage follows ndm-gym `0171f5e`:
[train.py](https://github.com/Frank-ZY-Dou/ndm-gym/blob/0171f5eadee50492d6ff30fa04b55e16cc11b61e/ndm_gym/train.py)
uses 500 world-model-only iterations before policy/critic updates. Cutting uses
`--wm-only-updates 500`: each completed iteration fits 10 batches of 2048
eight-step windows. Updates 1–500 train the world model, including its learned
termination head; update 501 also starts policy and critic training. The frozen
initial policy plus the existing 0.1 action noise continues collecting data.
The change addresses policy optimization through an insufficiently fitted
predictor; its effect is measured by force/terminal errors and physical episodes.


Schedule: `--schedule transitions` (`physical_transitions_v2`): warmup and update
cadence count physical transitions; LR and the schedule endpoint are indexed by
physical transitions / 256 (`lr_reference_worlds`). Checkpoints store completed
WM rounds, policy rounds, actual Adam step counts, stage length and training
configuration. `--wm-only-updates 0` selects immediate joint updates. Frozen
evaluation loads model weights independently of training-stage metadata.
RunningNorm is not adopted; static input scales are retained.

## Learner mapping

The connected learner keeps the source MLP, eight-step autoregressive loss
(sum over state coordinates, mean over batch and time), learned terminal head
(logged), actor objective and GAE critic updates. The actor objective spans 8
steps with `--actor-horizon 8` (32 in the source). Microbatch gradient
accumulation reproduces the effective batch sizes (2048 world model, 4096
actor); optimizer steps and gradient clipping occur after accumulation.

Robot end-effector pose is derived by differentiable URDF forward kinematics.
Controller motion accumulation, finite-difference velocity, acceleration, jerk,
observation, success counter and target follow the existing SliceIt equations;
targets and derived observations are not learned outputs. Wrench uses the
force/torque limits as scale; history is decoded component-major. No damage
projection is applied. Observation noise follows the SliceIt interface; per-episode material and
target-speed sampling is the project distribution described in SHARED_SCENE.

URDF FK returns the same quaternion sign convention as
[KDL Rotation::GetQuaternion](https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp).
SliceIt's angular finite difference uses the signed quaternion components, so
matching the rotation alone is insufficient for matching its velocity history.

Training quality is assessed with learning curves and held-out physical episodes.
Short-term loss/reward oscillation does not trigger architecture or threshold
changes. Non-finite computations, broken data alignment and incorrect gradients
are implementation failures to repair before interpreting training quality.

## Batched GPU collection

| Item | Existing source behavior | GPU collection |
|---|---|---|
| Physical backend | Gazebo/ROS robot and single DiSECt instance | MuJoCo Warp selected UR5e/UR10e profile plus packed DiSECt worlds; equations and source clocks retained, details in CuttingGpuBackend METHOD |
| Replay and updates | 4096 worlds × 1000 rows; one source step collects 4096 transitions; 40-step warmup, UTD 0.4 | 1000 rows per world; asynchronous terminal rows and resets; schedule historically advanced per completed vector step, now counts physical transitions (below) |
| Reset observation | Corrected ROS task returns an observation sampled after physical reset | GPU reset already uses the restored pose and zero derivative histories; the shared corrected reward head applies to physical steps and model imagination |
| Wrench samples | Oldest three entries of a 25-entry sensor queue, appended at filter callbacks | Same queue ordering at the deterministic 500 Hz exchange; initialize queue from the reset sensor baseline |
| Initial condition | ROS moves the arm to its start pose before resetting DiSECt | Restore the exported start and initialized controller/filter snapshot per world; shared public material configuration and prescribed cut topology |


`train_gpu.py` reuses OnlineLearner, policy/critic networks, objectives,
learning rates and effective batches. Checkpoints record the schedule state and
environment count; a resume with a different count is rejected.
`--max-hours`, `--max-transitions` and `--max-iterations` bound a run.

Resume: the replay paired with the checkpoint is restored (`last.pt` with
`replay.pt`, a named checkpoint with `<stem>.replay.pt`); name and iteration must
match. Open episodes are closed and every world restarts. The saved warmup
origin, transition credit and pending learning rounds are kept, so no new warmup
runs. Only `--resume-without-replay` starts from an empty replay, and then warmup
restarts at the current transition count.

The GPU physical snapshot layout `coupled_recurrent_v1` has 8944 coordinates:
FDCC integration velocity/error/position-command history, cached tip pose, the
200-sample filter window and IIR state, 25-sample sensor queue, MuJoCo warm start
and previous coupling pose; filter tare and controller compensation are
constant. These values affect the next transition even when particle and joint
states match. Tensor batch packing shifts each world's history independently; a
reset repeats that world's new initial frame and preserves every other world's
history. Full-physical-state checkpoints have different metadata and are rejected.

At a force-limit terminal boundary, inactive worlds retain their robot, material,
controller and filter state while other worlds finish the 50 ms action. Masked
writeback discards their candidate integration before finite-state checks.

The pinned source logs with TensorBoard. The GPU entry additionally uses W&B
online logging for phase timing, physical episode returns/success/collision,
world-model losses, BPTT/critic metrics and checkpoint artifacts.

SliceIt Section III-B3 and Figure 2 specify the bridge: robot knife position and
velocity go to CutSim, and CutSim contact force goes to the compliance controller
([source](https://arxiv.org/html/2404.02569v1#S3.SS2.SSS3)). Task metadata
includes `task_interface_revision=sliceit-177cf21`; checkpoint metadata equality
rejects the old reward interpretation.

## Trajectory baselines

Reference execution uses CuttingBatch.step(reference=True) and the same physical
FDCC/sensor/task path, bypassing the six-action mapper. In shared scenes the
default reference follows the episode target speed and stops its commanded
target at the prescribed endpoint. An explicitly installed KnifeTrajectory
takes priority; piecewise velocities are converted from the material frame
before integration. Actual motion still depends on compliance and contact.

The historical fixed-prism velocity/start-pose recipe is not the current
five-food reference. Reference audits use explicit trajectories and verify the
executed commands, including stall and sawing, rather than only checking exit
status. Measurements: [RESULTS](RESULTS.md).

## Variant `critic_target=real`

`source` (`--critic-target imagined`, default) fits the critic on GAE targets
computed along the 32-step WM-imagined rollouts of each policy update. Measured
(fixed-prism development, 2026-11-05): full-episode WM position error 5–7 mm exceeds the 2 mm
goal-z threshold, imagined terminal events are wrong, and the policy62 critic
has held-out return MAE ~153. Run C1 (`runs/critic_real_c1_20261105`, outside
Git) fitted a fresh source critic on recorded windows of the same data: held-out
MAE 10.9, max |target| 103.5.

`--critic-target real` changes only the critic's training data. The actor
objective, its BPTT through the WM and its `gamma^H * survival_H * V(s_H)`
bootstrap are unchanged. Each critic update draws episode-bounded H=32 windows
from the real `SequenceReplay` (`sample_windows`: recorded reward, terminal
label and valid-step mask) and applies the source GAE recursion
(source `gamma 0.99, lambda 0.95`; current pose task `gamma 1.0`, `lambda 0.95`) per window: a window with L recorded steps uses
`R_L = V(s_L)`, as the source uses `R_H = V_H`. A natural terminal (`done=1`)
on the entering transition keeps that reward once and removes all later value
(absorbing); a time truncation bootstraps from the recorded truncation state.
Padding steps after the window end are excluded from the loss. With all steps
valid the recursion is bit-identical to `gae_targets` (`masked_gae_targets`).

Data size: the source fits each update on `batch_size x horizon = 4096 x 32 =
131072` imagined states. Recorded windows stop at episode ends (a successful
cut lasts 20 steps), so batches of 4096 windows are drawn until their valid steps
reach 131072 (the 20-step figure is the fixed-prism task), and a uniform subset of exactly 131072 (state, goal, target)
triples is kept. Critic network, inputs, Adam (lr 2e-4), 5 epochs x 4 shuffled
minibatches and grad clip 1.0 are the source settings. `--critic-microbatch`
only chunks the gradient accumulation of a minibatch (default `--microbatch`).
Values for the targets come from the critic before the update, as in the source.

Starting from an existing checkpoint, `--critic-reset --critic-prefit-rounds N`
keeps the freshly initialized critic and Adam state instead of the saved critic,
and runs N real-target critic updates on the refilled replay before the next
policy update (each round: one data draw plus 5 x 4 Adam steps). This pre-fit is
part of the variant: a critic trained on imagined targets is biased by ~−150
and would otherwise bootstrap the actor during the first updates.

Checkpoints store `learner.critic_variant = {target, reset_update,
prefit_rounds, prefit_pending}`. Resuming with a different `--critic-target`
is rejected; checkpoints without the field are treated as `imagined`. Critic
optimizer counts are validated against `updates - reset_update + prefit_rounds`
critic rounds, and a pending pre-fit blocks policy updates until it is run.
In this variant, `policy/critic_target_*`, `critic_value_mean` and
`critic_value_target_mae` describe the real targets; imagined-target means are
logged as `policy/imag_critic_*` diagnostics only.

Every GPU checkpoint also writes the complete replay ring atomically (`last.pt`
with `replay.pt`, named checkpoints with `<stem>.replay.pt`) (states, actions, rewards, labels, episode ids, pointers; unwritten
rows are omitted while no world has wrapped), so online data survive the run.
`BatchedOnlineReplay.load_state_dict` restores it exactly.

Related methods train value functions from real transitions while using a model
for policy improvement or target expansion (MVE, STEVE, MAAC); this variant is
an adaptation of the SuperMimic source to the cutting task, not a novelty claim.

## Variant `critic_target=real_wm` with `actor_horizon`

The real_wm critic is a retained experimental option, not the selected current
real-data critic. The observations below are historical fixed-prism evidence,
not a current UR10e policy result.

Measured on the held-out episodes of `wm_frozen_fit_20261105` with frozen
policy62/WM562 (`runs/critic_real_c3_20261105/{horizon_probe,drift_probe,field_probe}.py`):
450 of 648 imagined H32 paths end in a workspace failure although every physical
episode succeeds; at least 365 of these occur at or after the step where the physical cut ended,
because the 2 mm goal-z success is missed. All paths terminate, so the H32
bootstrap weight is zero and the source actor objective is biased by −139 relative
to physical returns. Shortening the horizon alone does not help with a critic
fitted only on recorded states: one WM step changes the C1 critic by −638 on
average, almost entirely through the finite-difference `previous_acceleration`
field (prediction error 14 data standard deviations).

`--critic-target real_wm` keeps the `real` targets and adds, for every sampled
window, the current WM's open-loop predictions of the recorded states s_{t+j}
(j ≥ 1, recorded executed actions from s_t), each paired with the recorded
target of s_{t+j}. The pool of recorded and predicted samples is subsampled to
the same 131072 samples per update. `--actor-horizon h` sets the number of
imagined steps of the actor objective and its `gamma^h c_h V(s_h)` bootstrap
(default: the source 32); the critic windows keep H=32. Checkpoints record
`actor_horizon`; a resume with a different horizon is rejected unless it starts
the variant with `--critic-reset`.

## Variant `actor_bc_alpha`

The option is current (selected value0.25); the motivation measurement below
predates UR10e and is not evidence of current-policy failure.

Measured in run `critic_real_v2_20261106` (real_wm, actor horizon 8, from
policy62/WM562): the H8 bootstrap carries most of the actor objective (survival
0.4–0.8, bootstrap ≈ 55 of J ≈ 60), the policy gradient norm rises to 40–90
(C3: 2–18), and the mean |action| grows 0.95 → 1.37 within ten updates. The
physical policy then fails by force-limit collision in 1–2 steps; for these
recorded transitions the WM predicts the collision in 22% of cases (predicted
peak/limit 0.67–0.83, recorded 1.04) and its learned terminal probability is ~0.
The actor moved to actions whose force consequences the WM does not represent.

`--actor-bc-alpha a` adds the TD3+BC behavior term
([Fujimoto & Gu 2021](https://arxiv.org/abs/2106.06860)) to the actor loss:
`-J/h + (mean|J| / (a h)) * MSE(pi(o_t), a_t)` on a separate batch of
`batch_size` recorded transitions (clean recorded actions, ten-frame observation
history read from the stored control-history state). This is `1/lambda` times
`-lambda J + MSE` with `lambda = a / mean|J|`; `a = 2.5` is the source value.
It keeps the policy near actions present in replay, where the WM is fitted.
Default: off (source). Checkpoints record the value; a resume with a different
value is rejected unless it starts the variant with `--critic-reset`.


## Variants not adopted

`actor_objective=bc`, `force_penalty=linear` and `wm_terminal_gradient=head` were
tested and not adopted.

## Frozen world-model MPPI controller

`cutting_world_model/mpc.py` connects the unmodified `pytorch_mppi.MPPI`
optimizer (revision in SOURCES) to `CuttingWorldModel.predict`. The optimizer
samples normalized six-dimensional actions in [-1, 1], mapped affinely to the
task's original bounds. Candidate rollouts use the checkpoint's motion/force
history, goal, analytic reward and task predicates. No particle or damage fields
enter the planner. The WM dynamics receives ten frames of 54 control features:
joints, wrench samples, interval peak wrench, `last_wrench`, motion command and
previous action. The rollout additionally carries derived pose/velocity/
acceleration, observation, success-counter and board-contact history. The
policy instead receives ten frames of 34 observations and the 8D goal; these are
different information interfaces.

The accumulated task cost is the negative discounted survival-weighted reward.
Hard task endings absorb the rollout, keeping the terminal transition's reward
once. Survival weighting is the same as in `OnlineLearner.imagine` (analytic for
pose_depth_v3; learned terminal probabilities only for older checkpoints). `--mpc-terminal-value none` uses the finite-horizon
reward alone. `critic` adds `gamma**H * survival_H * V(state_H, goal)`, with
the checkpoint's frozen critic, canonical state scaling and pose-error inputs
used by `OnlineLearner.critic_values`. Absorbed candidates do not query the
critic and have zero terminal value. There is no BC cost.
The optimizer adds its upstream importance-sampling correction
`lambda * sum(U * Sigma_inverse * noise)` to sampled trajectory costs. This
correction is separate from the benchmark's executed reward. The returned
weighted-mean action sequence is also evaluated in the WM and logged, because
its score can differ from those of the samples.

| Component | Upstream / working policy | MPPI adapter |
|---|---|---|
| Prediction | Supplied dynamics callback | Frozen existing WM and exact task decoding |
| Control | Upstream sampling, bounded perturbations and importance weights | Unchanged; reset to the action-bound midpoint, shift the previous plan thereafter |
| Horizon | Working actor H8 | Development-selected H16 (0.8 s) |
| Objective | Actor includes critic bootstrap and BC | Finite-horizon task return, optional frozen critic, and upstream MPPI sampling correction |
| Execution | Six-dimensional compliance/motion action | Same action mapper and physical benchmark |

Solver settings are explicit CLI arguments: horizon, number of samples,
refinements, temperature, normalized noise standard deviation and discount.
They do not change the physical task. The frozen single-world adapter replans
after each measured transition and clears both history and plan on reset.
`planning.json` contains every call's duration, predicted reward and terminal
value separately, effective sample count, predicted sensor limit ratio, survival
and task ending. The physical evaluator supplies executed metrics. Temperature and repeated
MPPI refinements use the upstream optimizer controls; no task parameters change.
The finite-horizon running/terminal-cost formulation follows
[Williams et al. (2017)](https://arxiv.org/abs/1707.02342); solver controls and
the terminal callback are implemented by
[pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi#parameter-tuning-and-hints).

## Recorded geometry and offline video

Physical recordings retain joint/particle states, forces, actual blade vertices
and faces, body poses and platform identity. DiSECt reconstructs its virtual
cut-surface vertices from the recorded interpolation; BladeMaster uses its own
material-surface representation. These representations are not interchangeable.

The presentation entry is scripts/render_scene.py, using
configs/render/studio_v1.json. The recorded UR5e or UR10e platform selects its
actual robot/tool assets; UR10e includes metal ATI/IKEA and pure-white PLA.
No fabricated connector or overlay blade is added. Pinned food identity,
appearance, board, lighting and fixed camera apply across videos and stills;
missing required assets fail instead of substituting a palette or robot.

Half potato stays half potato; the current homogeneous banana has peeled-flesh
appearance. Tofu/luncheon have explicit synthetic materials. Mesh reconstruction
and display smoothing are presentation approximations, not physical fracture
evidence. Cut-face highlighting is diagnostic only. Sensor RGB uses the same
appearance path with its own optical camera; depth is geometric rasterization.
Recording/rendering never changes physical states, rewards or learned weights.
Commands: [RUNNING](RUNNING.md); camera model: [MECH_EYE_NANO](methods/MECH_EYE_NANO.md).

## Material feedback and diagnostic labels

`SliceItCoupling.feedback` sums contact magnitudes into a single-axis material
proxy with zero material torque, then adds native MuJoCo wrench before filtering.
It is not net material contact wrench at the Mini45 mounting. Zero material
torque does not imply zero total sensor torque. `source_wrench_transform`
preserves upstream block ordering, not a newly validated general transform.
A revised physical interface must be derived, validated and versioned separately.

`measure_sawing.py` reports with schema `simultaneous_coupling_samples_v2` sample at 500 Hz: every component and
vector norm uses the same contact-to-release window. Means are means of those
samples, not means of 50 ms peaks. Contact onset uses the explicitly recorded
sum-of-magnitudes threshold (default 0.5 N); 98% spring release remains a diagnostic
fraction, not a validated task-success definition.

## Material force model option (`force_model`, `friction_velocity`)

Default `force_model='scalar'` is the source bridge described above (sum of contact-force
magnitudes on one axis, no material torque). This preserves the signal convention;
the material discretization revision below separately records physical corrections.
`force_model='net'` (per scene: `CuttingBatch`, `GPUEnvironment`, object-group config
keys `force_model`, `friction_velocity`) selects a different, explicitly versioned sensor-feedback convention:

- `BatchedDiSECt.knife_wrench()` returns the vector sum of the per-contact forces on the
  knife and their moment about the DiSECt origin; contact points are the interpolated
  points on the cut edges that `eval_spring_sdf_contacts` uses. The particle positions
  are cached immediately before the final material substep's force evaluation,
  because the in-place integrator subsequently overwrites them. The cache participates
  in per-world pause and reset; this does not change material integration.
- `coupling.rigid_wrench_transform` carries that wrench to the base and sensor frames
  (force rotated, moment += p x force), replacing the source `convert_wrench` ordering
  for this term. Filtering is unchanged. Net observations and force-limit checks use
  sensor axes and moments about the sensor origin, matching the controller input and
  wrench history. They do not use the historical scalar observation conversion,
  which mixes torque into force. The final coupling sample is checked before task
  heads may declare success. Scalar reproduction retains its original convention.
- `friction_velocity` overwrites the scene's `sdf_kf`. The DiSECt contact kernel divides
  the tangential speed by this value (Brown's three-parameter friction law), so it is a
  transition velocity: with the exported 50 (or 1000 in other public configs) friction is
  negligible at cutting speeds; DiSECt's paper default is 0.01.

The feedback option itself changes no material equation; the separate contact-coordinate
correction below does change material contact. The damage law and the one-way
coupling (DiSECt forces are not applied to the MuJoCo robot; the robot reacts through the
compliance controller reading the sensor wrench) remain unchanged. Not validated: friction coefficient and
transition velocity against real cuts, the moment arm against a real mount, and whether
friction redirection alone accounts for the real sawing benefit (the fracture-side effect
of slicing is not in DiSECt's damage law).


The net metadata identifier is `net_contact_sensor_v1`; nondefault force/friction
settings also record `material_contact` with the revision, override and effective
friction values. Object groups record these values per group and must share the
same observation/limit convention. Scalar retains its feedback tag; every mode now
also records the separate contact discretization revision below. Older unversioned net runs cannot be treated
as measurements from this corrected convention.

The optional `sample_callback` on single-group steps is a read-only diagnostic
hook at each 2 ms coupling boundary. It does not alter actions, rewards or material
updates; the measurement script copies the tensors immediately. These samples
resolve coupling-rate forces, not every microsecond material-substep peak.


Net-mode work statistics use `-(F · v + tau_knife · omega)`, with the moment
shifted from the DiSECt origin to the knife origin. This includes lateral sawing
and rotation; gross work integrates the absolute net power. The historical scalar
downward-only proxy is retained in scalar mode. These are 2 ms coupling quadrature
statistics, not an exact material energy balance.



## Contact coordinate consistency correction

User-authorized physical bug fix, revision `edge_coordinate_consistent_v2`, in the
bundled `dflex/sim.py`. The earlier interface audit identified inconsistent weights
but had not traced the coordinate producer; swapping only the nodal weights would
have been incomplete.

The SDF edge search evaluates `p=(1-u)*x1+u*x2` (and its normal/distance), but used
to store `c=1-u`. Its caller then evaluated position and velocity as
`(1-c)*x1+c*x2`, while distributing force with `(c,1-c)`. Thus the search geometry
and nodal distribution agreed with one another, but the force's velocity/moment arm
and the reported knife moment used the opposite point.

The correction stores `c=u`. Position, velocity and force now use the same weights:
`p=(1-c)*x1+c*x2`, `v=(1-c)*v1+c*v2`,
`f1=(1-c)*F`, `f2=c*F`. Apply the same distribution correction to the optional
triangle-contact branch, whose producer already uses the second-endpoint coordinate.
The right side's endpoint reversal is retained consistently. Search iterations,
contact/friction coefficients, material law, damage law, timesteps, thresholds and
rewards are unchanged.

For a fixed contact, summing the nodal forces, their moments and their powers now
recovers `F`, `p cross F` and `F dot v`. This is a local force-transfer consistency
property; it is not a proof of whole-simulation energy conservation or real-material
accuracy. The production CUDA kernels are checked directly, including a forced
endpoint case, an independent SDF query at the reconstructed point, both cut sides,
four independent worlds, and the inactive triangle branch.

`GPUEnvironment.metadata.contact_discretization_revision` identifies the correction
for both scalar and net signals; it propagates to grouped environments/checkpoints.
Old strict checkpoint/environment comparisons must detect the mismatch. Do not
strip the key to resume old training as an identical-physics run. Reproduce the
old material implementation from commit `38b85a5` or earlier in an isolated checkout.
This correction changes future material motion and therefore can change cutting
forces, despite leaving all numerical parameters unchanged.


## DiSECt execution changes

Reset execution: cached material scratch batches compute only resetting worlds, with exact
material draws and the original duration; a stationary 32-substep CUDA graph removes Python
dispatch overhead. These are execution changes, not new contact/damage equations.

Third-party execution changes (introduced in 1b287d5) affect all DiSECt paths:

- `backends/mujoco/third_party/disect/disect/dflex/adjoint.py`: generated forward AND backward
  kernels launch on PyTorch's current CUDA stream. Necessary for dependent tensor operations
  and CUDA graph capture to execute in order. Compilation cache identity includes generated
  CPU and CUDA sources, so stale CUDA launch code cannot be loaded as an unchanged module.
- `.../dflex/sim.py`: scalar zero initialization uses device `torch.zeros` (same int64 value),
  avoiding a host-to-device scalar copy inside graph capture.

These change scheduling/cache/initialization, not the contact/fracture equations. Numerical
bitwise reproducibility is not promised. Original third-party licenses remain. New v3 work
adds no further third-party kernel changes. GPU regression includes packed materials,
controller/coupling, graph/eager comparisons and scratch eviction; passing these does not

## Tool mounting

The retained UR5e profile in both backends uses `disect_knife_rigid_mount_v1`
with a canonical rigid EE mount ([SHARED_SCENE](SHARED_SCENE.md#retained-ur5e-physical-knife-and-mounting));
the UR10e profile uses the mounted CAD knife (below). This is a kinematic
coupling, not a frictional gripper grasp or tool slippage model. The inherited
`sr305` camera meshes are not policy image inputs: `GPUEnvironment._observe`
builds observations from target-pose error, motion derivatives, actions and
wrench history, plus task time.

## Native BladeMaster adapter

The BladeMaster adapter runs the external forward material solver with the
existing robot, six-action FDCC interface and 500 Hz sensor-wrench feedback, and
the same analytical board-normal depth progress. It has no DiSECt cut-spring
input or separation label. Material reaction reaches the robot through
controller feedback, not an additional two-way MuJoCo contact solve. The batched
generated kernels load per-world particle mass, Lamé coefficients and fracture
threshold from persistent GPU arrays. Per-object settling settings belong to
`group_reset_settings` when object groups differ; reward, force limits and
robot/task fields must match.

## Observation and cadence options

The transition schedule earns WM updates from newly collected physical
transitions (integer credit, LR clock and budget in transitions) rather than a
world-count-independent vector-step counter. The `sensor_wrench6_v1` observation
adds all six sensor components while preserving pose/velocity axes, action
mapping and reward. Both are engineering settings, not methodological
contributions; the `source` mode remains available.


## Torch/Warp GPU memory sharing

Torch (learning, DiSECt state) and Warp (MuJoCo Warp collision, `wp.empty` from the
CUDA stream-ordered pool) share one device. Warp does not change the pool's release
threshold, so its blocks return to the device at synchronization and every physics
step or reset needs its in-use high-water as free device memory. Torch keeps idle
blocks after a learning round that Warp cannot use.

Defaults in scripts/train_gpu.py (recorded in the manifest as `gpu_memory_policy`):

- `--cuda-cache-release always`: before every physical step and every reset, release
  unoccupied Torch allocator cache with empty_cache. Live model/replay/optimizer
  tensors stay allocated. `reserve` (with `--cuda-cache-reserve-gib`, the previous
  trigger) and `off` remain available.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set by the trainer entry point
  unless the variable is already set, so idle pages of a segment can be returned and
  not only wholly free segments.
- Headroom guard (`--gpu-headroom-margin-gib`, default 1): after the release, free
  device memory must be at least the measured Warp pool high-water (since process
  start, which includes the full-batch initial reset) plus the margin. The guard is
  armed after the first joint learning round and raises a RuntimeError naming the
  numbers; it never reduces worlds or batches. Before arming it only records.

Per-phase Torch reserved/allocated peaks and device free memory are logged as
`gpu_mem/<phase>/*` and in steps.jsonl (`gpu_memory`, `gpu_memory_window`,
`gpu_headroom`, `cuda_cache`). None of this changes reward, task, effective batch
sizes, gradient accumulation or optimizer schedule. Measured numbers: RESULTS
(GPU memory headroom).

## Shared scene, reset and episode sampling

`shared_scene.py` generates paired backend configurations from the selected
contract (current UR10e: `configs/shared/ur10e_five_food_v1.json`; retained UR5e:
`configs/shared/scene_v1.json`); `scene_alignment.py`
measures settled particle height, solves the robot start pose, updates the
physical rigid tool and validates the resulting gap. The once-per-world clearance
is recorded but is not an actor feature. The DiSECt reference path follows the
goal's target speed. A packed DiSECt loader rejects CPU-resident scene tensors.

`episode_conditions.py` provides per-object/world/episode random streams shared
by both backends. E and density use the same log-uniform multipliers; the third
multiplier maps to DiSECt cutting-spring stiffness and BladeMaster cutting
energy (distinct quantities). Target speed is sampled uniformly and passed to
the goal and reference controller. BladeMaster selects dt using the upper E and
lower density bounds, updates only resetting worlds, restores their raw states
and settles with the actual parameters. Material draws are logged, not appended
to actor/WM state. Checkpoints retain the sampler; resume starts new physical
episodes at the next stream indices.

`completion_xy_error` is computed analytically from joint FK and the goal and
stored in state/history; final-only versus path-and-final is selected by
`common.task.tool_motion.enabled` (TASK_FREE_CUT). The retired particle-RMS
`object_slip` option remains only for old checkpoints.

## Training lifecycle

The GPU runner persists unstarted learning rounds (`update_queue`) separately
from the collection schedule's fractional credits; restored pending rounds run
before further collection. Saved optimizer state from an interrupted round is
retained, but the round is not replayed wholesale (RUNNING). Environment
shutdown is attempted before W&B finalization, and shutdown errors enter failed
status. Checkpoint payload hashes reject mixed model/replay generations.

## Baselines and vision

[CURRENT_BASELINES](methods/CURRENT_BASELINES.md) owns controller interfaces,
comparison scope and planner approximations. Physical execution uses one
action mapper/FDCC per platform for WM policy, SAC and both planners. A planner's
internal surrogate is not the physical execution controller.

[VISION_ROUTE1](methods/VISION_ROUTE1.md) owns geometric/learned goal selection,
DP3 encoder details and historical visual results. Vision selects the goal
before cutting; it is not a policy image input. The current native BladeMaster
visual handoff supports wm_policy, wm_mppi and analytic_mpc, not SAC.

WM transfer between material backends is explicit (--allow-backend-transfer);
task, controller, action/observation/model and robot metadata must still match.
Evaluation checks frozen weights and marks premature budget exhaustion as
incomplete. Diagnostic scripts reject non-finite predicted/replayed values;
unavailable separation measurements are null.

## UR10e CAD knife integration

Both material backends load the mounted IKEA CAD geometry: DiSECt uses its
signed grid distance and prepared cut topology; BladeMaster uses mesh contact
and curved-edge forward kernels without a preset cut plane. Robot/controller/
sensor identities are checked when loading checkpoints. Five-food assets and
mounts: [SHARED_SCENE](SHARED_SCENE.md); measured checks: [RESULTS](RESULTS.md).
Real-controller alignment and pending hardware work:
[FDCC setup](hardware/UR10E_FDCC_SETUP_ZH.md).

### Reset preparation correction (2026-11-21)

The DiSECt adapter restores the selected raw export geometry before aligning the
stationary knife for material settling, then aligns again to the settled surface
for the episode start. Previously the first alignment used the previous settled
snapshot, so restoring a taller raw soft object could introduce knife contact and
abort reset. This is an implementation-order correction, not a change to the
5–30 mm episode clearance, material distribution, reward or termination rules.
Non-reset worlds remain untouched. Contact checking is retained; failures now
identify the scene and original world indices. BladeMaster restores the robot's
high preparation pose before raw material reset/settling and uses the shared
post-settling alignment; its separate material reset does not execute the faulty
DiSECt ordering. Existing running checkouts are not hot-patched.

### Analytic MPC sensor tare correction (2026-11-21)

The nonlearned FDCC surrogate previously called controller.reset at every
predicted step, thereby replacing the real sensor's initial gravity tare with
the predicted orientation's gravity. The input wrench retained the real
measurement baseline, so nonzero payloads could acquire a fictitious contact
force during rotation. This affected analytic MPC, not SB3 SAC or the learned
WM transition.

Both physical backends now export controller_tare_compensation as read-only
execution context (DiSECt already did so). It is not appended to WM state,
actor/critic observations or checkpoint state_shapes. Each analytic MPPI
rollout has a distinct cloned sensor-frame baseline for its physical world;
candidate batches and horizon steps reuse that baseline. Observe and explicit
reset refresh context from the actual frame; masked reset updates only selected
worlds. Missing context with nonzero gravity/payload is an error, not an inferred
retare. Zero-load historical models retain their zero baseline without requiring
this field. Standalone AnalyticFDCCModel callers must set the physical context
with set_tare_compensation before predicting a nonzero-load platform. The
SharedMPPIPolicy.model prototype is not a per-world bound model; use the relevant
rollout model for direct diagnostic prediction.

This correction does not replace the held last-supplied-history-wrench
approximation, its existing delay, coarse FDCC recurrence or ideal joint
tracking. It does not establish accurate material-force forecasting, real-time
planning or improved task success. Reward, task events and SAC are unchanged.

Validation and old-code regression evidence are recorded once in
[RESULTS](RESULTS.md#ur10e-baseline-verification). This fix does not activate
or certify the real-robot command bridge.
