# Source mapping

Source revision: SuperMimic-Reproduced `ba610534`; SliceIt `1e79abad`;
robot utilities `ddc85eca`; DiSECt `db7de0f`. Complete dependency pins and
source patches are listed in [SOURCES](SOURCES.md). Method and baseline names used
in results are defined in [BENCHMARK](BENCHMARK.md) ("Method names").

| Module | Source design | Cutting implementation |
|---|---|---|
| Repository layout | Learning and physics maintained in separate local repositories | Both histories merged; physics resides in `backends/mujoco`, and SliceIt is restored from pinned revisions and patches. Entry points default to the bundled backend; cache and logger paths are configurable. Physics, learning objectives and task parameters are unchanged by this integration. |
| Fixed comparison environment | SliceIt SAC uses Gazebo/ROS; run3 uses MuJoCo Warp | New comparisons use the existing UR5e + MuJoCo Warp + FDCC + DiSECt export for every method. The original tf2rl SAC/Trainer executes in Python 3.8 and exchanges observations/actions with a Python 3.12 physical worker. A YAML-backed parameter reader supplies the source SAC settings without starting ROS. Evaluation freezes an exact checkpoint, disables action noise and uses the same reset, reward, termination and six-action mapper. |
| Fixed scene | Captured SliceIt task metadata includes Gazebo randomization flags | The benchmark restores one exported prism/material/robot state every episode. No object or initial-pose sampling occurs. Its configuration is checked against the prepared assets before evaluation; the three asset-content identifiers pin non-Git physical inputs, while Git records code revisions. |
| SAC time-limit bootstrap | The pinned `cb_trainer.py` sets replay `done=False` at the maximum episode length even if that step also succeeds or collides | The new process installs a replay adapter that takes `done` from the physical `terminated` flag. A pure time truncation keeps bootstrap. The original buffer, PER, source training loop, saved format and batched replay restoration remain in place. The old Gazebo checkout is unchanged. |
| Replay | Episode-aware ring, 4096 environments × 1000 rows, clean/executed actions, terminal row | Original ring with 1000 rows per environment. Physical-transition counts exclude extra terminal rows. |
| Schedule | Warmup 40 vector steps with 4096 environments; UTD 0.4 per vector step | Original vector-step cadence for every environment count: warmup 40, UTD 0.4, first update at step 42. LR and step endpoint use vector steps; transition counts are telemetry. The collection regime differs when fewer environments are active. |
| GPU checkpoint timing | Save every 4000 vector steps | User-requested SIGUSR1 saves at a completed iteration, plus stage boundaries and shutdown; periodic saves are disabled at the user's request. |
| Terminal-label storage | Pinned recorder writes termination on both the last transition row and final state row; sampler reads the next state row | Cutting recorder labels only the reached terminal state, so a window marks the actual terminating transition once |
| WM sequence coverage | The pinned sampler requires eight consecutive transitions within one episode | The local `SequenceReplay` adapter includes complete H8 windows and shorter recorded prefixes ending at an episode boundary. The loss sums state error and terminal BCE over valid transitions and divides by their total count across the effective batch. Full H8 windows retain the original loss and gradient. |
| Objective | Survival-weighted 32-step return, terminal critic, GAE lambda 0.95 | Original objective and GAE functions; only their type-only import is changed |
| Dynamics MLP | Three 2048-unit SiLU layers, zero final layer | Original source module with cutting StateSpec and connected trainer |
| Policy | Goal encoder, 32-D latent, decoder; widths 2048/1024/512/256; decoder gain 0.01 | Same architecture with a 7-D target pose, ten frames of SliceIt's 24-D observation, and six task actions |
| Policy input scaling | G1 observations and reference motions use robot-specific scales | SliceIt observations retain their existing normalization; target pose uses its SI/quaternion representation |
| Actions | Gaussian execution noise sigma 0.1 | Same noise; execution respects SliceIt's existing action-space bounds; both clean and executed actions are recorded. SliceIt's last-action observation contains the executed action |
| Observation/reward | Analytic task functions, including robot FK | Torch translation of corrected SliceIt normalization, reward and termination; decode component-major force history before averaging each component |
| Termination | Learned probability head trained from simulator terminal labels | Connected cutting probability head with force, workspace, goal and counter features |
| Absorbing task terminals | Imagined survival uses the learned termination probability | Predicted SliceIt force/workspace/goal terminals also end imagination. Subsequent state is absorbing, reward is zero, and bootstrap is zero, using the existing task predicates. This removes repeated terminal penalties when the learned classifier misses an already-known task terminal. The hard event mask has zero derivative across its switch; nonterminal model/action gradients remain. |
| Success priority | SliceIt checks collision, then `termination_on_negative_reward`, then goal success | Shared task heads apply the same priority; a failure branch suppresses the success bonus while retaining counter advancement |
| Physical state | Simulator state read at the transition boundary | ROS `native_ros_recurrent_v1` exports FDCC solver/PD/force/target state, hardware command, filter and sensor queues alongside joints/material. GPU `coupled_recurrent_v1` includes its controller/filter state. Observation snapshots precede terminal shutdown |
| Force history | Observation reads the sensor history | Record that exact sensor read, rather than rereading a moving queue after env.step |
| Material-force wiring | DiSECt reaction feeds the robot force sensor and compliance controller | Initial Gazebo runs have disconnected material-force and controller-input topics; new runs check the complete relay/filter/controller graph before collection |
| Reset boundary | The direct DiSECt environment inherited a false step-control flag; pause/unpause wrappers published asynchronous world-control requests | Load the existing step-control setting, use synchronous Gazebo pause/unpause services, then sample the post-reset observation. The original control timestep and task parameters are retained |
| Process shutdown | ROS service discovery and the controller-switch shutdown hook can wait indefinitely after Gazebo exits | Shutdown-only unpause and controller switching each have a five-second wall-clock bound. Checkpoint/replay, environment close, atomic status and W&B completion run independently; failures produce a failed status. Ordinary stepping and controller activation are unchanged |
| Robot asset | SliceIt UR5e robot; legacy task class UR3eDisectSlicingEnv-v1 | Keep the active XML (upper-arm mass 8.058 kg, COM x=-0.2125 m) for both SAC and the world-model task |
| Geometry/material | Source humanoid task | SliceIt's public prism configuration and aligned initial knife pose; physical equations and task thresholds unchanged |
| Reference library | Motion-tracking goals, reward targets and reset frames | SliceIt's fixed cutting target and original task reward replace the humanoid motion library |

The preceding offline full-packet predictor, learned reward head and added damage
projection have been removed. Their implementations and experiment revisions
remain in Git history. New architecture changes require a source mapping before
implementation; a short fit is a data/interface check, not a model-selection result.

Sources: [world model](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/src/supermimic/world_model/world_model.py),
[training loop](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/src/supermimic/training/loop.py),
[SliceIt](https://github.com/omron-sinicx/sliceit).

## External implementation evidence

[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym/tree/0171f5eadee50492d6ff30fa04b55e16cc11b61e)
provides full-update equivalence checks, independent physical evaluation and
500 world-model-only iterations; all three are applied here. Its RunningNorm
critic and analytic soft termination are separate method choices, not adopted.
The current implementation keeps the Cuboi learned terminal head and static
scales; SliceIt physical thresholds and reward stay fixed. Analytic termination
alone cannot restore a force-limit event that the dynamics model failed to
predict.

## Control-history candidate

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
optimizer, effective batches, eight-step fitting and 32-step BPTT remain intact.
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

Cuboi's vector-step collection schedule remains warmup 40 and UTD 0.4, replay
1000 rows per world, with the same loss, effective batches, learning rates,
networks, reward and physical thresholds. NDM uses 64 initial collection steps
and indexes its LR by learner iteration; those changes are not adopted. The
500-update stage adds 5000 world-model optimizer steps before policy learning.
At this cadence, WM-only finishes at vector step 1289 and the first joint update
occurs at 1292. Environment count does not rescale either counter.

Checkpoints store completed WM rounds, policy rounds, actual Adam step counts,
stage length and training configuration. Resuming refills replay while retaining
these counters. `--wm-only-updates 0` selects the original immediate joint-update
behavior. Frozen evaluation loads model weights independently of training-stage
metadata. Analytic soft termination and RunningNorm remain separate method
choices; the candidate retains the Cuboi learned head and static input scales.

## Connected trainer mapping

The connected candidate keeps the source MLP, eight-step autoregressive loss
(sum over state coordinates, mean over batch and time), learned terminal head,
32-step survival-weighted actor objective and GAE critic updates. A 4090 uses
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
are piecewise, while learned termination supplies the source survival gradient.

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
| Replay and updates | 4096 worlds × 1000 rows; one source step collects 4096 transitions; 40-step warmup, UTD 0.4 | 1000 rows per world; schedule advances once per completed vector step; asynchronous terminal rows and resets |
| Reset observation | Corrected ROS task returns an observation sampled after physical reset | GPU reset already uses the restored pose and zero derivative histories; the shared corrected reward head applies to physical steps and model imagination |
| Wrench samples | Oldest three entries of a 25-entry sensor queue, appended at filter callbacks | Same queue ordering at the deterministic 500 Hz exchange; initialize queue from the reset sensor baseline |
| Initial condition | ROS moves the arm to its start pose before resetting DiSECt | Restore the exported start and initialized controller/filter snapshot per world; shared public material configuration and prescribed cut topology |

`train_gpu.py` uses the same `control_history_v2` candidate as the ROS entry and
reuses OnlineLearner, policy/critic networks, objectives,
learning rates, effective batches and the 320000-vector-step schedule. Environment count
changes the number of physical transitions collected per vector step. It does
not change the eight-step world-model loss or 32-step actor BPTT horizon.
`--max-hours` and `--max-iterations` bound wall time and physical vector steps.
LR and the step endpoint use vector steps. Checkpoints record
`vector_steps_v1` and the environment count. Resume refills replay with the source
warmup; checkpoints using transition-rescaled schedule units require conversion.

The GPU physical snapshot layout `coupled_recurrent_v1` has 8944 coordinates. It includes
FDCC integration velocity/error/position-command history, cached tip pose, the
200-sample filter window and IIR state, 25-sample sensor queue, MuJoCo warm start
and previous coupling pose. Filter tare and controller compensation remain
constant snapshot coordinates. These values affect the next transition even when
particle and joint states match. The predictor uses the 860-coordinate control
history described above and predicts its 36 physical coordinates. Tensor batch
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

## Native ROS state and lifecycle

The native ROS state has 8990 coordinates (8924 predicted increments). Read-only
snapshot services export completed FDCC updates, the hardware-saturated command,
200 filter samples, IIR memory, added-force freshness, the 25-sample wrench queue,
and the knife-coupling previous pose. Internal solver velocity and measured
joint velocity are separate fields. Joint and wrench observations retain the
exact atomic source samples used by the original observation call. The current
model scales reuse joint/force scales, queue capacities and the existing 50 ms
control period for timestamp ages.

State accounting: particle position/velocity 7122, spring stiffness 304,
robot/tool/force fields 49, native controller/filter/coupling packet 1453,
derived history and episode constants 62. The complete dynamics/termination
model has 45,124,125 parameters versus 8,815,111 with the pinned humanoid state.
Flat particle ordering fixes the model input/output size to this material mesh.

Source identities distinguish recorded state from independent prediction targets:
controller tip pose follows FK of internal controller joints; force-history and
current-force readouts follow the recorded wrench queue. Sensor/controller
callbacks have separate sampling phases, so measured and internal joint states
remain separate. The filter reprocesses its 200-sample window with carried IIR
memory; replacing it with a two-state streaming filter changes the source update.

The arm's coupling pose cache survives environment reset. Reset computes a new
knife pose through the environment helper without updating that arm cache.
The next arm coupling query refreshes it; the source's existing 1 m/s translation
guard returns zero twist for a sufficiently large displacement. Consequently
cache reconstruction must preserve reset/query timing and use `b_bot_knife_sim`
FK, rather than replace every initialized cache with current robot FK.

With one ROS environment, the unchanged 10 batches of 2048 eight-step windows
and UTD 0.4 request 65,536 training transition slots per collected transition;
at 4096 worlds the corresponding ratio is 16. Overlapping/repeated samples do
not supply independent trajectories. This is a collection-regime difference
despite matching the source batch and optimizer settings.

The base environment and arm must agree with the configured step-control flag.
Reset and cleanup use `/gazebo/pause_physics` and `/gazebo/unpause_physics`
services in call order. Native snapshots are read after the observation while
physics is paused, before terminal controller shutdown. ROS message delivery retains its source scheduling; snapshot records contain
source timestamps and ages. In stepped mode, dynamic-reconfigure writes finish
before advancing physics. The source launched concurrent writes, and recorded
action-dependent gains were requested only after the first 2 ms had advanced.
Continuous control retains asynchronous updates. Gain values, action mapping
and control timesteps are unchanged.

The source shutdown hook switches back to the trajectory controller on a Gazebo
tick. Workers therefore unpause before closing, including setup failures and
interruption. The launcher waits for the worker before stopping bridge/Gazebo.
`--max-iterations` bounds executed environment steps; it does not rescale the
learning schedule or alter batch sizes.

Critic diagnostics split the existing objective into discounted rewards and
terminal bootstrap, retain the source value/target means, and compare learned
termination with the analytic task predicates along imagined trajectories.
These measurements do not change the learned head, objective, gradients or
optimizer updates. The old reset-history error and uncontrolled reset-time
advancement corrupt the state-transition contract; they are repaired at the
source environment boundary.

The SliceIt Gazebo StepControl extension previously returned after publishing a
WorldControl request. Five measured calls supplied the pre-step joint sample to
DiSECt. Positive step requests now use Gazebo's native blocking `World::Step(n)`;
zero steps retain unpause semantics. This changes request completion ordering,
not the requested 2 ms interval or Gazebo/DiSECt equations.
Source: [Gazebo 11.15.1 World::Step](https://github.com/gazebosim/gazebo/blob/gazebo11_11.15.1/gazebo/physics/World.cc#L807).

StepControl responses report authoritative start/end simulation times and the
requested joints' position, velocity and effort after integration. The client
installs this ordered snapshot atomically before computing knife FK, retaining
the ROS hardware continuous angular branch and rejecting older/equal-time topic
updates. The source joint-state publisher can skip a busy publication opportunity;
with physics paused, waiting for that topic cannot guarantee a new sample.
ROS hardware reads on WorldUpdateBegin, so the response shifts the sample phase
by up to one physics tick. Requested tick counts and physical equations stay
unchanged. Episode resets retain the source NO_RESET_SIM monotonic clock;
recreating a world requires recreating its joint-state consumer.
Reset also requests an explicit `snapshot_only` pause/read, with zero elapsed
simulation time, before constructing its first observation. This establishes
the same authoritative joint-sample boundary before the first action.
Metadata `step_interface_revision=synchronous_paused_joint_snapshot_v1`
separates these recordings from earlier topic-sampled data. Dataset/checkpoint
metadata equality rejects mixing the two sampling contracts.
Source: [ROS joint-state controller](https://github.com/ros-controls/ros_controllers/blob/noetic-devel/joint_state_controller/src/joint_state_controller.cpp).

Action duration and trajectory segment timing use accumulated server intervals
in integer nanoseconds in stepped mode. The previous ROS-clock/float loop
produced 52 ms actions for the configured 50 ms interval. Continuous control
retains its original clock path and all trajectory/force predicates.

## Pre-policy checkpoint compatibility

The explicit terminal-rule migration uses a separate copy of the completed
WM-only checkpoint. Conditions: WM rounds equal the configured 500-round
startup stage; completed policy rounds are zero; policy and critic Adam states
are empty with recorded step counts zero; WM Adam step counts equal 5000;
trainer configurations and the 1024-environment schedule match the current
entry. All model tensors must be finite. The copy adds only
`learner.imagination_terminal_rule=task_absorbing_v1`; every original field is
checked for exact equality after saving and loading. The unchanged strict
loader then validates the copy. Checkpoints containing policy or critic updates
under the previous terminal rule retain the existing rejection.

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
(`gamma 0.99, lambda 0.95`) per window: a window with L recorded steps uses
`R_L = V(s_L)`, as the source uses `R_H = V_H`. A natural terminal (`done=1`)
on the entering transition keeps that reward once and removes all later value
(absorbing); a time truncation bootstraps from the recorded truncation state.
Padding steps after the window end are excluded from the loss. With all steps
valid the recursion is bit-identical to `gae_targets` (`masked_gae_targets`).

Data size: the source fits each update on `batch_size x horizon = 4096 x 32 =
131072` imagined states. Recorded windows stop at episode ends (a successful
cut lasts 20 steps), so batches of 4096 windows are drawn until their valid steps
reach 131072, and a uniform subset of exactly 131072 (state, goal, target)
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

Every GPU checkpoint also rewrites `replay.pt` atomically with the complete
replay ring (states, actions, rewards, labels, episode ids, pointers; unwritten
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

## Ablation `actor_objective=bc`

`--actor-objective bc` (requires `--actor-bc-alpha`) keeps the actor's TD3+BC
term at the same scale `mean|J| / (a h)` but drops the gradient of the imagined
objective: the H-step imagination still runs, without autograd, to supply
`mean|J|` and the imagination diagnostics. WM and critic training are
unchanged. It measures how much of a `model` run's behavior comes from the
model-based gradient rather than from behavior cloning toward replay actions.
Default: `model`. Checkpoints record the objective; checkpoints without the
field were `model`. A resume with a different objective needs `--critic-reset`.

## Variant `force_penalty=linear`

SliceIt's force term `-sigmoid(s/2 - 3)` (`s` = norm of the history-mean
`[Fx, Fz, Ty]`, mixed N and N·m units) saturates near 13–20 N: 6 N → −0.125,
10 N → −0.220, 20 N → −0.250. Above about 15 N the reward barely distinguishes
forces up to the 50 N collision limit. `--force-penalty linear
--force-penalty-scale 50` replaces only this term by `-s/50`, keeping the same
three readings, components, weights and every other reward and termination
term. DiSECt penalizes mean knife force directly; normalizing by the existing
50 N limit is our choice, not a literature value.

The setting is stored in the task metadata's `environment`
(`force_penalty`, `force_penalty_scale`); absent keys mean the source reward,
so existing checkpoints and metadata are unchanged. The real-environment reward
and the WM's analytic imagined reward read the same setting. A resume may change
only these reward keys, and only with `--critic-reset` (the critic estimates
returns of the reward it was fitted on). Frozen evaluation accepts such
checkpoints, records `policy_training_reward`, and scores every method with the
benchmark's SliceIt reward and the force/work/safety metrics.
`scripts/value_eval.py` compares a critic with returns of its own reward,
recomputing the force term of the recorded SliceIt rewards from the recorded
next observations.

Not adopted (RESULTS 2026-11-08); kept as a named variant for the record.

## Variant `wm_terminal_gradient=head` (variant A, completed analysis)

Status updated 2026-11-10: implementation retained as a named variant; its
paired control experiment is complete (RESULTS, 2026-11-09). The pre-stated
final-success criterion did not detect a clear benefit. A is not the selected
main method; the description below records its implementation.

Source route (`dynamics`): the terminal head reads `terminal_features` of the
predicted next state, so the terminal BCE of the WM fitting loss also trains the
dynamics MLP through that state (as in SuperMimic-Reproduced). With `head`, the
fitting loss (`window_loss`, `masked_window_loss`, through `fit_predict`) feeds the
terminal head the detached predicted state: the BCE trains only the head and the
state loss alone trains the dynamics. Forward values are unchanged. Policy
imagination uses `predict`/`step`, which is identical for both routes, so the
actor keeps the path action → predicted state → reward and terminal probability
(`tests/test_wm_terminal_gradient.py`). `wm_diagnostics.window_loss_components`
follows the configured route; set it to `dynamics` to measure the source BCE
gradient.

`--wm-terminal-gradient {dynamics,head}`; default `dynamics`. Checkpoints record
the route; checkpoints without the field used `dynamics`. `train_gpu.py` resumes
only with the same route; `OnlineLearner.restore(...,
change_wm_terminal_gradient=True)` starts the other route from a saved WM, used by
the fixed-data comparison in PLAN.

## Frozen world-model MPPI controller

`cutting_world_model/mpc.py` connects the unmodified `pytorch_mppi.MPPI`
optimizer (revision in SOURCES) to `CuttingWorldModel.predict`. The optimizer
samples normalized six-dimensional actions in [-1, 1], mapped affinely to the
task's original bounds. Candidate rollouts use the checkpoint's motion/force
history, goal, analytic reward and task predicates. No particle or damage fields
enter the planner. The WM dynamics receives ten frames of 48 control features
(joints, wrench samples, interval peak wrench, motion command and previous action).
The rollout additionally carries derived pose/velocity/acceleration, observation
and success-counter history. The policy receives ten frames of 24 observations
and the target pose. These are different information interfaces.

The accumulated task cost is the negative discounted survival-weighted reward.
Hard task endings absorb the rollout, keeping the terminal transition's reward
once; learned terminal probabilities supply the same survival weighting as
`OnlineLearner.imagine`. `--mpc-terminal-value none` uses the finite-horizon
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
| Robot `knife_and_adapter` includes blade, handle and mounting adapter; DiSECt uses a separate rectangular contact blade | Render the complete gray URDF tool and blue DiSECt contact blade at their recorded poses. Frame the close-up to include the mount. |
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

## Multi-condition task (`--conditions`)

`cutting_world_model/multi_env.py`. A JSON config lists object groups (exported
scene, worlds, start-clearance range, material-scale ranges, target-speed range),
`w_speed` and optionally one `force_limits` vector for every object. Each group is a
homogeneous `GPUEnvironment`; groups are concatenated along the world dimension.
Per world: knife start height fixed at construction by inverse kinematics
(`CuttingBatch(initial_lift=...)`, knife edge `clearance` above the object top).
Per episode at reset: Young's modulus, density and cut-spring stiffness scales
(log-uniform) and the target cutting speed (uniform). The task goal is
`[target_pose(7), target_speed]` (`goal_dim` 8): policy and critic receive it, the
WM dynamics receives the pose only, the reward adds
`-tanh(|v_down - v*| / v*)` while the cut depth is not reached (`SliceItHeads.speed_reward`).
No object identity or per-object limit is given to the learner. Groups are stepped
sequentially (PLAN step 2 parallelizes them). Objects: scanned surfaces
(`scripts/prepare_food_mesh.py`, fTetWild with a ~1 mm envelope) and DiSECt official
materials (`scripts/make_food_configs.py`), exported with `export_disect_scene.py`.

## Force-risk constraint (`--risk-constraint`)

`OnlineLearner(risk_constraint=...)`, `cutting_world_model/force_risk.py`.
R(s_t, a_t) = max_i |peak_wrench_i(t+1)| / limit_i. A `QuantileHead` on the last
hidden layer of the WM dynamics MLP, the WM's predicted R and the measured R_t is
fitted with the 0.95 pinball loss on recorded transitions (`head_steps` per WM
update, WM features frozen). In `imagine`, the upper estimate of every imagined
interval is computed before it is executed; `improve_policy` adds
`lambda * sum_t survival_t * relu(R_up,t - threshold) / H` to the actor loss, so the
penalty reaches the policy through the action. After each update, lambda <-
max(0, lambda + lr * (force-limit episode rate - target rate)) over the episodes
that ended since the previous update. Defaults: threshold 0.9, target rate 0.02,
lambda 10, lr 50. Head, optimizer and lambda are checkpointed; a checkpoint with
and one without the constraint cannot be mixed.

