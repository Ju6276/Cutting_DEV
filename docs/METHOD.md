# Implementation reference

## Active configuration — pose_depth_v3 (updated 2026-11-18)

Current task version: **pose_depth_v3** in both material backends
([scene_v1.json](../configs/shared/scene_v1.json)): the pose_depth_v2 task rules
(shared analytic maximum blade-depth progress in both backends and the WM, no
spring-release shaping or learned release-progress head, pose +100 and failure
precedence) plus the software board-stop consistency fix and analytic imagined
survival. Exact equations: [TASK_FREE_CUT](TASK_FREE_CUT.md#pose_depth_v3--software-board-stop-consistency-2026-11-17).
pose_depth_v2 is read/evaluate-only for the 2026-11-16 runs; `train_gpu.py` and
`train_sac_multi.py` refuse it. No pose_depth_v3 long training run has been launched.

- WM state (v3): adds `last_wrench` (6, predicted, scaled by the force limits)
  and `board_contact` (latch flag, derived, not predicted). Transitions whose
  recorded start state is latched are zero-weighted in the dynamics loss.
- Imagined survival (v3): success, failure and timeout are settled analytically
  in actor imagination, GAE targets and WM+MPPI; the learned terminal head is
  still fitted and logged, not used as survival.
- Force sampling (v3): every new 2 ms filtered reading of a control step enters
  the component-wise peak, force-limit check and safety margin exactly once;
  latched worlds report the frozen reading.
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
  ([lifecycle](#gpu-training-lifecycle-correction-2026-11-16)).
- GPU memory: Torch cache release before every physics step/reset,
  expandable segments and a headroom guard are trainer defaults
  ([Torch/Warp GPU memory sharing](#torchwarp-gpu-memory-sharing-default-since-2026-11-17)).
- `train_gpu.py` defaults remain the historical `source` variant; the current
  settings above must be passed explicitly ([RUNNING](RUNNING.md)).

Statements below about the 2026-11-16 shared runs refer to pose_depth_v2; earlier
pose-v1 and free_cut_v3 results retain their recorded definitions.
[SHARED_SCENE](SHARED_SCENE.md) is the scene/configuration authority;
[PLAN](PLAN.md) gives decisions and next work. Online W&B is required for long runs.

Sections describing the fixed-prism source adapter mapping, the native
ROS/Gazebo state path, the one-off pre-policy checkpoint migration, the retired
S0 multi-condition task, the force-risk penalty and the free_cut_v1–v4 task
interfaces moved verbatim to
[archive/METHOD_historical_20261118](archive/METHOD_historical_20261118.md).
Source revisions and patches: [SOURCES](reference/SOURCES.md).

## External implementation evidence

[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym/tree/0171f5eadee50492d6ff30fa04b55e16cc11b61e)
provides full-update equivalence checks, independent physical evaluation and
500 world-model-only iterations; all three are applied here. Its RunningNorm
critic and analytic soft termination are separate method choices, not adopted.
The implementation keeps static scales and still fits the Cuboi learned terminal
head; since pose_depth_v3 imagined survival is analytic and the head is only logged.
ndm-gym's soft termination is not used. SliceIt physical thresholds stay fixed. Analytic termination
alone cannot restore a force-limit event that the dynamics model failed to
predict.

## Control-history candidate

**Current dimensions (pose_depth_v3; from the spec and the 2026-11-18 v3
integration run's task.json/manifest.json).** One frame has 107 coordinates:
42 predicted physical ones (joint position/velocity 12, three-sample wrench
history 18, interval peak wrench 6, `last_wrench` 6) and 65 derived ones (motion
command, previous pose/velocity/acceleration, the 34-wide observation, success
counter, step count, depth progress, depth span, completion XY error and
`board_contact`). Ten frames give 1070 state coordinates. The dynamics input takes
54 coordinates per frame, 540 in total: joints 12, wrench history/peak/last 30,
motion command 6, previous action 6. The model predicts a 42-wide delta. The
policy sees 34 observations x 10 frames plus an 8D goal. The WM fits 8-step
windows, the actor imagines 8 steps (`--actor-horizon 8`) and the real-data critic
uses 32-step recorded windows. Imagined survival is analytic.

*Historical layout.* The rest of this section gives the original numbers of the
fixed-prism SliceIt task. The field roles and history mechanism still apply; the
numbers do not.

The user-approved control predictor (`prediction_layout=control_history_v2`)
learns 36 next-state coordinates: robot joint position/velocity (12), the exact
three-sample observed wrench history (18), and interval peak wrench (6).
The original FK, observation, motion-command, derivative-history, success-counter,
reward and terminal formulas advance the other 50 current-state coordinates.
Nine previous 86-coordinate frames accompany the current frame, giving 860
rollout-state coordinates. Dynamics input selects 48 coordinates per frame:
joints (12), observed/peak wrench (24), accumulated motion command (6) and
previous executed action (6), for 480 total. Derived pose, velocity, acceleration,
jerk, observation copies and success counter remain available to analytic heads
and critic, but do not feed the dynamics MLP. This follows the pinned source's
separation of physical dynamics state from analytic observation/reward features;
the v1 frozen diagnostic identifies derivative-input feedback as an error amplifier.
History length 10 matches the pinned policy history;
reset repeats the actual initial frame, and each transition shifts only past
frames. Particles, spring damage and native controller/filter packets remain in
physical recordings but are absent from model inputs and prediction targets.

This changes the state representation and supervised targets of the pinned
SuperMimic residual MLP. Its widths, normalized squared-error form, terminal BCE,
optimizer, effective batches, eight-step fitting and (in that version) 32-step BPTT remained intact.
Physical/task equations and policy observations are unchanged. The finite history
approximates unobserved material and controller memory. Evaluation separates
reset-first transitions, held-out full episodes, force-limit termination and
8/32-step predictions; model gradients are checked independently of physical
policy success. No learned latent state or recurrent network is added.

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

*Historical vector-step schedule:* Cuboi's vector-step collection schedule used
warmup 40 and UTD 0.4, replay 1000 rows per world, with the same loss, effective batches, learning rates,
networks, reward and physical thresholds. NDM uses 64 initial collection steps
and indexes its LR by learner iteration; those changes are not adopted. The
500-update stage adds 5000 world-model optimizer steps before policy learning.
At that cadence, WM-only finished at vector step 1289 and the first joint update
occurred at 1292. Current runs use `--schedule transitions`
(`physical_transitions_v2`): 30720 warmup transitions, one learning round per 640
physical transitions. LR and the schedule endpoint are indexed by physical
transitions / 256 (`lr_reference_worlds`). The vector-step figures above do not
apply.

Checkpoints store completed WM rounds, policy rounds, actual Adam step counts,
stage length and training configuration. Resuming refills replay while retaining
these counters. `--wm-only-updates 0` selects the original immediate joint-update
behavior. Frozen evaluation loads model weights independently of training-stage
metadata. RunningNorm is not adopted; static input scales are retained. The learned
head is still fitted. Since pose_depth_v3, imagined survival is analytic and the
head's probabilities are only logged ([Active configuration](#active-configuration--pose_depth_v3-updated-2026-11-18)).

## Connected trainer mapping

This section maps the original fixed-prism SliceIt task onto the trainer. Current
task equations are in [TASK_FREE_CUT](TASK_FREE_CUT.md); current dimensions and
horizons are in [Control-history candidate](#control-history-candidate).

The connected candidate keeps the source MLP, eight-step autoregressive loss
(sum over state coordinates, mean over batch and time), learned terminal head,
survival-weighted actor objective and GAE critic updates. The actor objective
spans 32 steps in the source and 8 with `--actor-horizon 8`. Its survival weight
is learned up to pose_depth_v2 and analytic from pose_depth_v3. A 4090 uses
microbatch gradient accumulation for the original effective batch sizes (2048
world model, 4096 actor); optimizer steps and gradient clipping occur after
accumulation. This changes storage/scheduling, not the loss or update ratio.

Physical recordings contain DiSECt vertex positions/velocities and
spring stiffness, robot joints/velocities, bridge knife pose/twist, the three
wrench samples used by SliceIt, and the componentwise peak wrench checked during
the control interval. Robot end-effector pose is derived by differentiable URDF
forward kinematics. Controller motion accumulation, finite-difference velocity,
acceleration, jerk, observation, success counter and fixed target follow the
existing SliceIt equations. Targets and derived observations are not learned
outputs. Spring damping is static in this DiSECt integrator.

Metric/velocity/joint error and output scales reuse source StateSpec values;
spring stiffness uses its initial material stiffness, wrench uses the existing
force/torque limits. No damage projection is applied. The terminal head uses
normalized force-limit, workspace, goal and counter features in place of G1
height/tilt. Force history is decoded as component-major in both the corrected NumPy source
and the Torch reward. Existing component scales, averaging, reward weights and
thresholds are retained. Reward retains SliceIt's discrete collision/success bonuses and
continuous distance/force/jerk terms; gradients through the discrete bonus tests
are piecewise, while learned termination supplied the source survival gradient
(pose_depth_v3 uses analytic survival instead).

Only SliceIt's existing observation noise/randomization is used; humanoid sensor
noise magnitudes do not apply to the cutting observations. The fixed target
replaces reference-motion frames. Gazebo/ROS collection remains one environment.

Training quality is assessed with learning curves and held-out physical episodes.
Short-term loss/reward oscillation does not trigger architecture or threshold
changes. Non-finite computations, broken data alignment and incorrect gradients
are implementation failures to repair before interpreting training quality.

URDF FK returns the same quaternion sign convention as
[KDL Rotation::GetQuaternion](https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp).
SliceIt's angular finite difference uses the signed quaternion components, so
matching the rotation alone is insufficient for matching its velocity history.

## Batched GPU collection

| Item | Existing source behavior | GPU collection |
|---|---|---|
| Physical backend | Gazebo/ROS robot and single DiSECt instance | MuJoCo Warp UR5e/FDCC plus packed DiSECt worlds; equations and source clocks retained, details in CuttingGpuBackend METHOD |
| Replay and updates | 4096 worlds × 1000 rows; one source step collects 4096 transitions; 40-step warmup, UTD 0.4 | 1000 rows per world; asynchronous terminal rows and resets; schedule historically advanced per completed vector step, now counts physical transitions (below) |
| Reset observation | Corrected ROS task returns an observation sampled after physical reset | GPU reset already uses the restored pose and zero derivative histories; the shared corrected reward head applies to physical steps and model imagination |
| Wrench samples | Oldest three entries of a 25-entry sensor queue, appended at filter callbacks | Same queue ordering at the deterministic 500 Hz exchange; initialize queue from the reset sensor baseline |
| Initial condition | ROS moves the arm to its start pose before resetting DiSECt | Restore the exported start and initialized controller/filter snapshot per world; shared public material configuration and prescribed cut topology |

`train_gpu.py` uses the same `control_history_v2` layout as the ROS entry and
reuses OnlineLearner, policy/critic networks, objectives, learning rates and
effective batches.

Schedule (current): `--schedule transitions` (`physical_transitions_v2`), where
warmup and update cadence count physical transitions. Checkpoints record the
schedule state and environment count, and a resume with a different count is
rejected. `--max-hours`, `--max-transitions` and `--max-iterations` bound a run.
Environment count does not change the eight-step WM loss, the actor horizon (8
with the current flags) or the 32-step critic windows.

Resume (current): the replay paired with the checkpoint is restored: `last.pt`
with `replay.pt`, a named checkpoint with `<stem>.replay.pt`. Name and iteration
must match. Open episodes are closed and every world restarts. The saved warmup
origin, transition credit and pending learning rounds are kept, so no new warmup
runs. Only `--resume-without-replay` starts from an empty replay, and then warmup
restarts at the current transition count.

*Historical:* the vector-step schedule (`vector_steps_v1`, 320000-step endpoint)
used one overwritten `replay.pt`, and a resume repeated the source warmup.

The GPU physical snapshot layout `coupled_recurrent_v1` has 8944 coordinates. It includes
FDCC integration velocity/error/position-command history, cached tip pose, the
200-sample filter window and IIR state, 25-sample sensor queue, MuJoCo warm start
and previous coupling pose. Filter tare and controller compensation remain
constant snapshot coordinates. These values affect the next transition even when
particle and joint states match. The predictor uses the control history described
above: 1070 coordinates with 42 predicted for pose_depth_v3, 860 with 36 in the
historical source layout. Tensor batch
packing shifts each world's history independently; a reset repeats that world's
new initial frame and preserves every other world's history. GPU checkpoints
record `prediction_layout=control_history_v2`; full-physical-state checkpoints
have different metadata and are rejected on resume.

At a force-limit terminal boundary, inactive worlds retain their robot, material,
controller and filter state while other worlds finish the 50 ms action. Masked
writeback discards their candidate integration before finite-state checks.

The pinned source logs with TensorBoard. The GPU entry additionally uses W&B
online logging for phase timing, physical episode returns/success/collision,
world-model losses, BPTT/critic metrics and checkpoint artifacts.

SliceIt Section III-B3 and Figure 2 specify the bridge: robot knife position and
velocity go to CutSim, and CutSim contact force goes to the compliance controller.
The disconnected topics observed in our old running launch establish an
integration failure for that run. They do not establish that every published
launch combination or the authors' experiment had that failure.
Source: https://arxiv.org/html/2404.02569v1#S3.SS2.SSS3

ROS and GPU task metadata include `task_interface_revision=sliceit-177cf21`.
Existing checkpoint metadata equality rejects resuming with the old reward
interpretation. Earlier recorded datasets retain their original rewards; new
interface parity checks use episodes collected with the corrected source.
SAC's PER and TCN changes do not apply to the SuperMimic replay or policy.

## Trajectory baselines

Sources: backend reference branch (`CuttingBatch.step(reference=True)`);
DiSECt `SlicingMotion` and NVlabs
[`optimize_slicing.py`](https://github.com/NVlabs/DiSECt/blob/main/examples/optimize_slicing.py),
run through the DiSECt-Optimization wrapper (`372e8ac`).

| Item | Source | Cutting implementation |
|---|---|---|
| Reference target | Constant DiSECt-frame knife velocity `v` for the scene duration `T`: target = reset tip + `R v min(t + dt, T)` | `KnifeTrajectory`: piecewise-constant DiSECt-frame velocities; target = reset tip + `sum_i R v_i (clamp(t + dt, t_i, t_i+1) - t_i)`. The default is the one-interval scene trajectory (`v = (0, -0.05, 0)` m/s, `T = 1.3` s); velocities are rotated before integration, so its targets are bit-identical to the previous formula |
| Frame | `R` = rotation of the fixed URDF transform `b_bot_base_link -> cutting_board_disect` | Same `coupling.base_from_disect`; DiSECt -y maps to base -z |
| Execution | Same compliance controller, coupling and task heads | Unchanged; the 6-D action mapper is bypassed and the recorded action is zero |
| DiSECt motion | Velocity `v_y = sum_k w_k(t) p_k`, `v_z = sin(f t) sum_k w_k(t) a_k`, `w_k = exp(-(t - t_k)^2 / 0.02)`, five waypoints over 0.6 s, evaluated at each 1e-5 s substep | Exported as 600 intervals of 1 ms, each the mean of its substep velocities (float64; DiSECt evaluates in float32). Against the recorded free-knife positions the exported displacement differs by at most 1.1e-5 m |
| Start pose | Free knife at (0, 0.0895, 0) m, identity rotation | The trajectory is applied as a displacement from the robot's reset knife: (0, 0.08822, 4.9e-5) m in the DiSECt frame, rotation about x by 6.7e-4 rad |
| Time | DiSECt time from 0 to 0.6 s, knife free of the robot | Episode simulated time; after 0.6 s the target holds, and the episode continues until SliceIt success, a force/workspace terminal or 500 steps |

The trajectory is open loop: tracking error and force feedback in the compliance
controller decide the knife's actual path. Determinism: two runs of the identical
default reference at `71be9ec` differ in the last digits (peak 9.9746 vs 9.9763 N,
return 80.2903 vs 80.2899; both 40 steps, 288 springs) because the material uses
FP32 atomic force sums, so bit identity is tested on controller targets, not on
whole rollouts.

## Variant `critic_target=real`

`source` (`--critic-target imagined`, default) fits the critic on GAE targets
computed along the 32-step WM-imagined rollouts of each policy update. Measured
(RESULTS 2026-11-05): full-episode WM position error 5–7 mm exceeds the 2 mm
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

`actor_objective=bc`, `force_penalty=linear` and `wm_terminal_gradient=head` were tested and not
adopted; their descriptions are in [archive/METHOD_variants_20261112](archive/METHOD_variants_20261112.md).

## Frozen world-model MPPI controller

`cutting_world_model/mpc.py` connects the unmodified `pytorch_mppi.MPPI`
optimizer (revision in SOURCES) to `CuttingWorldModel.predict`. The optimizer
samples normalized six-dimensional actions in [-1, 1], mapped affinely to the
task's original bounds. Candidate rollouts use the checkpoint's motion/force
history, goal, analytic reward and task predicates. No particle or damage fields
enter the planner. For a pose_depth_v3 checkpoint the WM dynamics receives ten
frames of 54 control features: joints, wrench samples, interval peak wrench,
`last_wrench`, motion command and previous action. Fixed-prism checkpoints used
48, without `last_wrench`. The rollout additionally carries derived
pose/velocity/acceleration, observation, success-counter and board-latch history.
The policy receives ten frames of 34 observations and the 8D goal; fixed-prism
checkpoints used 24 observations and the target pose. These are different
information interfaces.

The accumulated task cost is the negative discounted survival-weighted reward.
Hard task endings absorb the rollout, keeping the terminal transition's reward
once. Survival weighting is the same as in `OnlineLearner.imagine`. It is analytic
for pose_depth_v3 checkpoints, where learned terminal probabilities are only
logged, and comes from the learned terminal probabilities for earlier checkpoints. `--mpc-terminal-value none` uses the finite-horizon
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
and task ending. BENCHMARK supplies executed metrics. Temperature and repeated
MPPI refinements use the upstream optimizer controls; no task parameters change.
The finite-horizon running/terminal-cost formulation follows
[Williams et al. (2017)](https://arxiv.org/abs/1707.02342); solver controls and
the terminal callback are implemented by
[pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi#parameter-tuning-and-hints).

## Recorded geometry and offline video

The recorder observes the benchmark's reset and completed physical transitions.
It stores raw joint/particle states and forces, plus world-frame material, knife,
robot-body and collision-geometry poses. URDF joint names select the MuJoCo qpos
addresses. Display frames interpolate joint angles and positions; recorded force
peaks and spring-release counts are held until their measured boundary.

DiSECt `cutting/visualizer.py` and `dflex/render.py` evaluate virtual cut vertices
as `(1-t)*particle[i] + t*particle[j]`, then render both `tri_indices` and
`cut_tri_indices`. The offline renderer uses the same construction. The prism
export contains 608 virtual vertices and 1338 cut triangles.

| Source component | Video implementation |
|---|---|
| MuJoCo physics import discards URDF visuals and keeps collision meshes | Offline video loads the original URDF visual meshes, local origins, scales and COLLADA material colors onto recorded body poses. `--robot-geometry collision` selects the physics meshes for inspection. |
| Robot `knife_and_adapter` includes blade, handle and mounting adapter; DiSECt uses a separate rectangular contact blade | Historical source-format renderer supports the URDF appearance with an explicit debug contact-proxy overlay. Current frozen-evaluation videos use only the actual backend blade; see the 2026-11-15 correction below. |
| DiSECt particle and virtual-vertex surface | Same surface topology and edge interpolation, with smooth normals and optional triangle edges. |
| Material contact plane | Add a board thickness and support below the plane for display. |

One EGL window owns both views. Force traces use measured control-step peaks;
playback speed uses recorded simulation timestamps, independently of solver wall
time. Render dependencies are isolated in `requirements-render.txt`.

The URDF tool is fixed to the gripper frame. The DiSECt contact blade follows
SliceIt's separate velocity coupling, with motion projected onto the cutting
plane. At reset, `b_bot_knife_sim` receives the source's vertical shift
`spine_height / 2 - 0.0043` (25.2 mm for this blade); during stepping, transformed
tool velocities drive DiSECt. DiSECt contact-force magnitudes are summed and
returned along its cutting axis through the sensor filter to FDCC. This is a
motion/force signal connection between simulators; it is not a MuJoCo fixed
joint connecting the DiSECt blade to the robot. Source implementations are
SliceIt `disect_slicing.py:reset`, `fzi_cartesian_compliance_controller.py`
knife publishing, and `cutting/ros_visualizer.py:update_knife_pose` /
`step_simulation`; the port is `cutting_gpu_backend/coupling.py`.

## Material feedback and diagnostic labels

`SliceItCoupling.feedback` sums contact magnitudes into a single-axis material
proxy with zero material torque, then adds native MuJoCo wrench before filtering.
It is not net material contact wrench at the Mini45 mounting. Zero material
torque does not imply zero total sensor torque. `source_wrench_transform`
preserves upstream block ordering, not a newly validated general transform.
A revised physical interface must be derived, validated and versioned separately.

Historical reports from `measure_sawing.py` used temporal component maxima;
their L2 norm is an envelope, not a simultaneous force norm. Reports with schema
`simultaneous_coupling_samples_v2` instead sample at 500 Hz: every component and
vector norm uses the same contact-to-release window. Means are means of those
samples, not means of 50 ms peaks. Contact onset uses the explicitly recorded
sum-of-magnitudes threshold (default 0.5 N); 98% spring release remains a diagnostic
fraction, not a validated task-success definition.

## Material force model option (`force_model`, `friction_velocity`) — 2026-11-12

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
of slicing is not in DiSECt's damage law). Measurements: RESULTS 2026-11-12.


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



## Contact coordinate consistency correction — 2026-11-12

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


## DiSECt execution changes from the free-cut reviews (2026-11-12/13)

Kept from the free-cut v2/v3 review sections (task-specific parts are in
[archive/METHOD_historical_20261118](archive/METHOD_historical_20261118.md)); these
execution changes apply to all DiSECt paths, including the current shared task.

Reset execution: cached material scratch batches compute only resetting worlds, with exact
material draws and the original duration; a stationary 32-substep CUDA graph removes Python
dispatch overhead. The vendored DFlex wrappers now launch on PyTorch's current CUDA stream,
the extension cache checks both CPU and CUDA generated source, and a per-step scalar zero
allocation is graph-compatible. These are execution changes, not new contact/damage equations.

Third-party execution changes introduced in 1b287d5 affect ALL DiSECt paths, including
legacy tasks; they are not a free-cut-only physical patch:
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
establish real-world physical calibration. Full provenance lives in FREE_CUT_REPAIR.


## Recorded tool mounting and unused camera meshes (2026-11-14)

The retained URDF mounts `knife_and_adapter.stl` as the visual geometry of
`b_bot_right_inner_knuckle`. Its joint and the intervening gripper-base joints
are fixed to the UR5e end-effector chain. The knife and adapter share one mesh;
this model does not simulate a frictional grasp or tool slippage. The DiSECt
contact blade is the coupled material-side proxy, driven by robot tool motion
and returning a wrench through the existing coupling.

`b_bot_cam_mount` uses `sr305_mount.stl`; inside/outside camera links use
`sr305_simplified_body.stl`. These inherited model parts are not policy image
inputs. `GPUEnvironment._observe` builds observations from target-pose error,
motion derivatives, actions and wrench history, plus task time where applicable.

Rendering defaults to no contrasting prepared-cut-face highlight. Original
texture is extended from nearby exterior display triangles without moving
vertices, deleting faces, filling openings or changing physical connectivity.
`--highlight-cut-faces` is an explicit diagnostic display option, recorded in
render provenance. Thus removing the pale band does not assert that a material
is intact or separated.


In the historical source-format render_episode.py path, the visible knife defaults to the mounted URDF knife/adapter only. The blue
DiSECt contact proxy remains active in physics but appears only with
`--show-contact-blade`. This corrects the confusing double outline, not any
geometric mismatch between the two representations. Still renders support
close and whole-arm views. Rendering settings do not alter tool transforms,
material contact, reaction forces, task termination or learning inputs.

`capture_environment_batch.py` records all worlds after actual randomized
initialization and settling; it performs no control actions or optimizer updates.
`render_environment_batch.py` uses each world's captured robot transforms and
particles, arranging isolated worlds in a display-only grid. Such images are
initialization previews, not screenshots of an ongoing training run.

## Additional material implementations (2026-11-14)

The [BladeMaster-derived DiSECt candidates](archive/BLADEMASTER_DISECT_ASSETS.md) keep the current prescribed-interface pose task. Only geometry and compatible elastic quantities (E, nu, rho) transfer; cutting-spring and contact values remain explicitly inherited from the DiSECt template. They are reproduction shapes, not scanned foods or real-food calibrations.

The [BladeMaster UR5e adapter](archive/BLADEMASTER_BACKEND_REVIEW.md) instead runs the external forward material solver with the existing robot, six-action FDCC interface and 500 Hz sensor-wrench feedback. It has the independent `blademaster_pose_v1` identity and analytical board-normal depth progress, no DiSECt cut-spring input or separation label. Real mapped actions feed the normal replay/WM/BPTT update path. Material reaction reaches the robot through controller feedback, not an additional full two-way MuJoCo contact solve. The initial adapter used serial independent material solvers. The later batched implementation was validated at256 worlds and used for the GPU9 engineering run; see RESULTS for its scope and subsequent SIGILL, rather than treating the initial serial limitation as current.

Per-object settling settings now belong to `group_reset_settings` when object groups differ. Shared reward, force limits and robot/task fields must still match. Homogeneous groups retain the former metadata shape for compatibility.


## Collection-scaled cadence and full-wrench policy observation (2026-11-14)

The source mode is preserved. The opt-in transition schedule earns WM updates
from newly collected physical transitions rather than a world-count-independent
vector-step counter; its warmup, integer credit, LR clock and budget are specified
in [TRAINING_SCALE_20261114](archive/TRAINING_SCALE_20261114.md).
The opt-in sensor_wrench6_v1 observation adds all six sensor components while
preserving pose/velocity axes, action mapping and reward. WM already predicted
six-dimensional wrench. See [WRENCH_OBSERVATION_REVIEW](archive/WRENCH_OBSERVATION_REVIEW.md)
for layout, normalization and checkpoint compatibility. These are explicit
engineering variants, not claimed methodological contributions.

### Torch/Warp GPU memory sharing (default since 2026-11-17)

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
2026-11-17 (GPU memory headroom).


## Frozen-evaluation physical-tool rendering (2026-11-15; historical geometry)

Current shared geometry/reset is defined in SHARED_SCENE (including its shared-knife section). The
120 mm wedge and proxy variation below describe the old frozen runs, not the
new 150 mm shared knife. The studio entry points remain supported.

Use evaluate_pose_policy.py --record-geometry, export_evaluation_scene.py and
render_episode_still.py --video for current frozen evaluations. The shared studio
renderer preserves the original UR5e visual meshes, scan appearance and one
actual physical blade, with no additional crack highlighting. It replaces the old blade portion of the combined URDF knife/adapter appearance
mesh, which does not match the BladeMaster wedge, while retaining the original
gripped handle (see the correction below).
The initial whole-mesh omission was a visualization regression: it also removed
the original handle. The current renderer clips only the old blade beyond the
mesh shoulder (source x=-0.04786805 m), retaining the gripped handle/adapter.
The original export added a metal shank and collar to bridge the handle and
blade visually. These were invented display geometry without CAD provenance or
user authorization and are now removed (FRAMEWORK_AUDIT_20261116). The original
handle and physical blade are retained without visual repositioning. A
full-trajectory rigidity check remains and rejects relative blade motion over
10 micrometres; it does not establish a real manufactured mounting. Main proxy/handle relative variation was measured at
0.488 mm in the selected cucumber recording and is not rewritten.

BladeMaster's configured wedge is120 mm along the edge,80 mm high,1 mm thick,
with5 mm bevel. Its adapter retains the initial source knife-center position
but sets the blade's initial quaternion to identity in robot-base coordinates.
The resulting fixed EE transform drives the material tool and wrench mapping.
This is a kinematic coupling, not a frictional gripper grasp, and the existing
URDF appearance must not be used as proof of physical blade placement.

BladeMaster videos transform the same vertices_in_ee used by board checks
through the saved EE pose and MuJoCo base. Main videos use the recorded DiSECt
contact blade vertices. Particle-derived BladeMaster display surfaces are
cosmetic and never used for collision or task labels. Recording and rendering
do not change task physics, parameters, rewards, or checkpoint weights.

### Shared studio appearance

The November 14 studio appearance is reused for frozen-evaluation video instead
of maintaining a second VTK presentation style. Evaluation geometry is adapted
by `export_evaluation_scene.py`; `render_episode_still.py` handles both stills
and animated output. The previous `render_pose_evaluation.py` is removed.
The canonical entry and commands are in RUNNING.

Knife vertices, body poses and material coordinates are recorded physics data.
The original handle remains visible; the historical mismatched visual blade is
removed. No shank, collar or other connector is drawn for either backend. The
earlier display shank and collar were invented geometry and were removed (see
above); do not restore them. The URDF cable
envelope (30 x 30 x 55 mm) is hidden in studio views, without modifying physics.

Cucumber FEM scan appearance is reused only after topology equality and rigid
geometry correspondence checks. Half potato and particle surfaces can use
explicitly approximate PCA-aligned scan-color transfer. Half-potato display
triangles subdivide the existing FEM surface; smooth shading normals are updated
from that surface each frame, without moving the surface. Particle colors follow
particle identity; changing display topology is reconstructed from recorded
points. None of these operations recover exact UV coordinates or validate
separation. No artificial crack bands are drawn; actual geometric gaps are
not filled. Board grain, lights and shading are decorative.

## Shared physical knife (2026-11-16)

The new opt-in `disect_knife_rigid_mount_v1` uses the same finite physical blade
and canonical rigid EE mount in both backends. See [SHARED_SCENE](SHARED_SCENE.md#shared-physical-knife-and-mounting)
for geometry, coupling differences and verification commands. Historical policies
retain their recorded tools. [Frozen policy analysis](archive/POLICY_MECHANICS_20261116.md)
separates measured force/history sensitivity from unproven material identification.

## Optional material-motion envelope — 2026-11-16

Implementation: `object_slip.py` reads settled-reference planar particle RMS;
`free_cut.py` shares soft cost and failure semantics; `pose_depth.py` adds the
supervised WM field and hard imagined termination. `pose_cut.py` excludes this
failure from analytical +100 settlement. Both environment adapters emit
`object_slip_m` and `object_slip_failure`; episode logs report object_motion.
Group concatenation excludes full reference clouds from learner state.
Policy input is unchanged. WM/critic/replay layouts and task metadata change.
See TASK_FREE_CUT for equations, limits and reproduction commands. This is our
task integration, not a claimed original BladeMaster reward or fracture model.

## Shared scene/reset implementation — 2026-11-16

shared_scene.py generates both configurations and paired initial-clearance cases
from configs/shared/scene_v1.json; scene_alignment.py measures settled particle
height, solves the robot start pose, updates the physical rigid tool and validates
the resulting gap. The once-per-world clearance is exposed in frame/condition
records but is not an extra actor feature. Actual realized heights may differ
across material models.

The shared DiSECt reference path now explicitly follows the goal's target speed
when this scene protocol is selected; historical exports retain their original
reference trajectory. The new scene preparer changes knife friction from .5 to
.2 to match the nominal BladeMaster coefficient; old assets are untouched. These
are new-configuration choices, not calibration or old-run-equivalence claims.

A packed DiSECt loader now rejects CPU-resident scene tensors before CUDA kernel
launch. No native damage law was changed in this work. Optional bounded pitch
has matching physical-action and WM equations (default off). See SHARED_SCENE for
the scene definition and the episode-sampling implementation below.

## Paired episode material distribution — 2026-11-16

episode_conditions.py provides per-object/world/episode random streams shared
by DiSECt and BladeMaster. E and density use the same log-uniform multipliers.
The third log-uniform multiplier maps to DiSECt cutting-spring stiffness and
BladeMaster cutting energy; these remain distinct physical quantities.
Target speed is sampled uniformly and passed to the goal and reference controller.

The BladeMaster generated batch kernels load per-world particle mass, Lamé
coefficients and fracture threshold from persistent GPU arrays. They preserve
the scalar solver equations; the native solver's heterogeneous contact tests
compare against independent scalar instances. The robot adapter selects dt
using the upper E and lower density bounds, updates only resetting worlds,
restores their raw states and settles with those actual parameters.

The learner still receives motion and wrench history. Material draws are logged,
not appended to the actor/WM state. Training checkpoints now retain the episode
sampler; resume starts new physical episodes at the next stream indices.


## Robot-position constraint options (2026-11-16)

Shared pose_depth_v2 (unchanged in pose_depth_v3) selects final-only versus path-and-final using
common.task.tool_motion.enabled. Final EE target error is evaluated in robot-base
XY (radius40 mm), extending the historical single-horizontal-axis goal check.
The optional path constraint uses a bounded quadratic running cost and a hard
60 mm failure, independently of particle motion. Neither mode penalizes fragment
opening. See TASK_FREE_CUT for exact equations and boundary conventions.

completion_xy_error is computed analytically from joint FK and the goal, stored
in state/history for critic and goal-free termination queries; the WM does not
predict its delta. Actor observation/action dimensions are unchanged. The physical
and imagined arrival counter includes the same final-position mask. Historical
particle-RMS checkpoints retain their old state specification and task metadata.


## Per-world physical-blade overview correction (2026-11-16)

The studio movie exporter already displayed the physical blade, but the separate
all-world overview still displayed the full old URDF knife. Corrected by recording
each world's authoritative physical tool vertices in world coordinates and
sharing the same robot/handle clipping with the studio exporter. No fake mount
is added. Old snapshots lacking these data fail explicitly. DiSECt episode files
now also include the base/world transform that BladeMaster already recorded.
This is a recording/rendering correction; task physics and learning are unchanged.
The overview retains explicitly illustrative material colors and distant mesh LOD;
it is not a scan-textured studio movie. Actual geometry validation is in RESULTS.


### GPU training lifecycle correction (2026-11-16)

The active GPU runner persists unstarted learning rounds separately from the
collection schedule's fractional credits. This fixes lost work when a checkpoint
is written between rounds earned by the same vector step. Restored pending rounds
run before further collection. No reward, actor/critic objective, network, batch
size or uninterrupted update cadence changes. Saved optimizer state from an
interrupted round is retained but the round is not replayed wholesale; this
limitation and legacy checkpoints lacking a queue are explicit in RUNNING.
The actual runner tests exercise milestone checkpoint/replay restoration with
CPU doubles for physics/learning, plus shutdown failure reporting. They test
control flow, not new policy performance. Physical learner/capacity evidence
remains separately recorded in RESULTS.

## Current-task baseline adapters (2026-11-16)

See [CURRENT_BASELINES](methods/CURRENT_BASELINES.md) for source provenance, input and
update semantics, the SB3 fallback decision, and model approximations. These
entry points share the current pose_depth_v3 physics/reward/recording (pose_depth_v2
when introduced); the main WM+BPTT algorithm and the frozen 2026-11-16 long-training
checkouts were unchanged. Native backend
physics remain distinct. Physical test coverage is explicitly recorded there.

## Pre-cut geometric vision (2026-11-17)

Route 1 is a nonlearned depth-to-goal preprocessing module. It preserves the
prescribed cut plane, uses calibrated board/tool geometry, estimates the visible
top and supplies base-frame EE pose plus commanded speed. It does not change
the policy observation or learning algorithm. This describes the original
prescribed-plane mode; the visible-midpoint selector and tested physical
handoffs below supersede its initial untested-execution status.
See [VISION_ROUTE1](methods/VISION_ROUTE1.md).


### Visual goal selector and physical handoff (2026-11-17)

visual_cut.visual_midpoint_plan chooses a visible geometric mid-section from
depth, independently of preset cut XY; ambiguous axes reject. Calibration then
maps the physical blade edge to an8D base-frame policy goal. visual_execution
places the robot only before an episode (native BM, single world); it verifies
IK, board clearance, goal validity and unchanged material positions. This reset
adapter is not a real-robot approach planner. The native BM handoff permits a
new cut plane without rebuilding DiSECt topology. The frozen policy sees its
existing motion/force observations and goal, not the input image or segmentation.
See VISION_ROUTE1 and RESULTS for the cross-backend nature and bounded evidence.

## Fixed-heading visual-goal predictor — 2026-11-17

The optional visual_center module predicts a one-dimensional midpoint offset
from observed local point clouds. It does not consume hidden material state,
object labels or full mesh geometry at inference, and does not change the
WM/BPTT/controller. Training supervision uses full surface bounds offline.
Both learned and geometric fixed-heading selectors call the same board/tool
goal converter, then the existing frozen-policy handoff. Camera and data
protocol, current test errors and failed physical trials are in
[VISION_ROUTE1](methods/VISION_ROUTE1.md). This is a modular task-input demonstration,
not an adopted algorithmic novelty claim.

## Visual midpoint DP3 encoder adapter (2026-11-17)

The optional visual selector now defaults, when newly trained, to the official
DP3 PointNetEncoderXYZ at commit47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97.
The class is vendored verbatim under cutting_world_model/vendor/dp3 with MIT
license and SOURCE.json; only colored terminal printing is suppressed. Its AST
was checked against the pinned upstream class. Official dp3.yaml settings:
XYZ3-64-128-256, ReLU, LayerNorm, max pooling,256-to64 projection + LayerNorm.
Our64-64-1 SiLU/tanh midpoint head is additional, not part of DP3. No diffusion
policy, action model or pretrained weights are imported.

The original local_v1 encoder stays available solely for paired comparison and
existing checkpoint loading. Versioned checkpoints dispatch explicitly; legacy
checkpoints without architecture load local_v1, unknown architectures fail.
Both consume the same512 observed-local XYZ points, fixed0.1m scaling, fixed
base-frame axes, and output the same +-50mm residual in base Y. The board/goal
conversion and frozen RL interfaces are unchanged. There is no arbitrary cut
orientation prediction. New runs use a dedicated shuffle RNG independent of
model parameter initialization so unequal architecture sizes do not alter data
order. Same epochs/optimizer, not same capacity or compute time, is the comparison.

## DP3 visual goals for planning controllers (2026-11-17)

evaluate_visual_cut now dispatches wm_policy,wm_mppi,analytic_mpc after
the same DP3 depth-based plane selection and pre-episode IK placement.
The planner is constructed after the visual target/start is installed, ensuring
its packed history begins at the actual visual start. Each physics step updates
the planner state. Inputs are robot/controller/sensor histories and geometric
depth; no material particles or spring progress are consumed by the planners.

WM transfer is explicitly opt-in for this native-BladeMaster visual prototype.
Default SharedMPPIPolicy validation stays strict. The explicit transfer admits
only the DiSECt/BladeMaster backend/contact solver distinction and equivalent
blade/board descriptions: unordered blade vertex sets agree within1e-8m,
normal within1e-10 and plane offset within1e-8m, identical stop gap. Task,
controller, action/observation/prediction layouts and robot metadata must still
match exactly. The frozen source WM is not rewritten or trained on the target
backend. Full checkpoint actor tensors must exactly match the reference frozen
policy; filenames/update counters alone are insufficient.

Nonlearned MPC uses the existing FDCC surrogate with held measured wrench,
not a fitted cutting/material model; MPPI is the optimizer in both planners.
The --policy file supplies quaternion/goal convention metadata in analytic mode;
its actor/WM/critic weights are not used to compute actions. No terminal critic
is enabled for this video comparison. Camera perception is pre-cut only.

A native-backend interface bug was found by actual execution: target_speed is
N in BladeMaster but N-by1 in the other interface. SharedMPPIPolicy now reshapes
one scalar per world before goal concatenation. Both layouts have integration
tests. The first failed zero-step attempts remain in the run directory.

## Evaluation persistence cleanup — 2026-11-17

Visual evaluation now closes the physical environment even if planner JSON,
trajectory serialization or summary writing fails. Persistence errors mark
the progress state failed when the filesystem permits writing; a failed error
log does not hide the original persistence exception. The pose evaluator
already closed unconditionally and now also marks persistence failure.
Task rules, checkpoints and controller/model outputs are unchanged.

## Pose prediction diagnostic validity (2026-11-18)

scripts/diagnose_pose_predictions.py reuses the training rollout without
optimization. It now rejects non-finite replayed/imagined states, rewards,
survival, logits, critic values and objective values before reporting them as
normal predictions. Statistics reject NaN/Inf instead of silently dropping
samples; JSON export also rejects unexpected non-finite values. Only unavailable
physical-separation measurements convert their collection NaN sentinel to
explicit null. An empty selected cohort is still valid and has count zero.

This changes diagnostic failure reporting only. It does not change the WM,
policy, replay, reward, task version or optimization schedule.
