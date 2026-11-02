# Source mapping

Source revision: SuperMimic-Reproduced `ba610534`; SliceIt `4c65cf7`;
robot utilities `177cf21`; DiSECt `db7de0f`.

| Module | Source design | Cutting implementation |
|---|---|---|
| Replay | Episode-aware ring, 1000 rows/environment, clean/executed actions, terminal row | Original replay code with a recorded-episode importer; existing task configuration is fixed for this scene |
| Schedule | Warmup 40 environment steps; UTD 0.4; one actor update per world-model update | Original defaults and update cadence; ROS sampler has one environment; GPU sampler has configurable worlds; source default has 4096/GPU |
| Terminal-label storage | Pinned recorder writes termination on both the last transition row and final state row; sampler reads the next state row | Cutting recorder labels only the reached terminal state, so a window marks the actual terminating transition once |
| Objective | Survival-weighted 32-step return, terminal critic, GAE lambda 0.95 | Original objective and GAE functions; only their type-only import is changed |
| Dynamics MLP | Three 2048-unit SiLU layers, zero final layer | Original source module with cutting StateSpec and connected trainer |
| Policy | Goal encoder, 32-D latent, decoder; widths 2048/1024/512/256; decoder gain 0.01 | Same architecture with a 7-D target pose, ten frames of SliceIt's 24-D observation, and six task actions |
| Policy input scaling | G1 observations and reference motions use robot-specific scales | SliceIt observations retain their existing normalization; target pose uses its SI/quaternion representation |
| Actions | Gaussian execution noise sigma 0.1 | Same noise; execution respects SliceIt's existing action-space bounds; both clean and executed actions are recorded. SliceIt's last-action observation contains the executed action |
| Observation/reward | Analytic task functions, including robot FK | Torch translation of corrected SliceIt normalization, reward and termination; decode component-major force history before averaging each component |
| Termination | Learned probability head trained from simulator terminal labels | Connected cutting probability head with force, workspace, goal and counter features |
| Physical state | Simulator state read at the transition boundary | ROS `native_ros_recurrent_v1` exports FDCC solver/PD/force/target state, hardware command, filter and sensor queues alongside joints/material. GPU `coupled_recurrent_v1` includes its controller/filter state. Observation snapshots precede terminal shutdown |
| Force history | Observation reads the sensor history | Record that exact sensor read, rather than rereading a moving queue after env.step |
| Material-force wiring | DiSECt reaction feeds the robot force sensor and compliance controller | Initial Gazebo runs have disconnected material-force and controller-input topics; new runs check the complete relay/filter/controller graph before collection |
| Reset boundary | The direct DiSECt environment inherited a false step-control flag; pause/unpause wrappers published asynchronous world-control requests | Load the existing step-control setting, use synchronous Gazebo pause/unpause services, then sample the post-reset observation. The original control timestep and task parameters are retained |
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

## Control-history candidate

The user-approved control predictor (`prediction_layout=control_history_v1`)
learns 36 next-state coordinates: robot joint position/velocity (12), the exact
three-sample observed wrench history (18), and interval peak wrench (6).
The original FK, observation, motion-command, derivative-history, success-counter,
reward and terminal formulas advance the other 50 current-state coordinates.
Nine previous 86-coordinate frames accompany the current frame, giving 860
input-state coordinates. History length 10 matches the pinned policy history;
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

## Connected trainer mapping

The connected candidate keeps the source MLP, eight-step autoregressive loss
(sum over state coordinates, mean over batch and time), learned terminal head,
32-step survival-weighted actor objective and GAE critic updates. A 4090 uses
microbatch gradient accumulation for the original effective batch sizes (2048
world model, 4096 actor); optimizer steps and gradient clipping occur after
accumulation. This changes storage/scheduling, not the loss or update ratio.

The physical prediction state contains DiSECt vertex positions/velocities and
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
| Replay and updates | 1000 rows/environment, 40 vector-step warmup, UTD 0.4 | Same capacity per world and cadence per vector step; asynchronous terminal rows and resets |
| Reset observation | Corrected ROS task returns an observation sampled after physical reset | GPU reset already uses the restored pose and zero derivative histories; the shared corrected reward head applies to physical steps and model imagination |
| Wrench samples | Oldest three entries of a 25-entry sensor queue, appended at filter callbacks | Same queue ordering at the deterministic 500 Hz exchange; initialize queue from the reset sensor baseline |
| Initial condition | ROS moves the arm to its start pose before resetting DiSECt | Restore the exported start and initialized controller/filter snapshot per world; shared public material configuration and prescribed cut topology |

`train_gpu.py` reuses OnlineLearner, StateSpec, policy/critic networks, objectives,
learning rates, effective batches and the 320000-step schedule. Environment count
changes the number of physical transitions collected per vector step. It does
not change the eight-step world-model loss or 32-step actor BPTT horizon.
`--max-hours` and `--max-iterations` bound execution without rescaling the schedule.

The GPU state layout `coupled_recurrent_v1` has 8944 coordinates. It includes
FDCC integration velocity/error/position-command history, cached tip pose, the
200-sample filter window and IIR state, 25-sample sensor queue, MuJoCo warm start
and previous coupling pose. Filter tare and controller compensation remain
constant state coordinates. These values affect the next transition even when
particle and joint states match. The existing residual MLP learns their dynamic
updates; network hidden widths, loss and optimizer settings are unchanged.
Checkpoints with the previous partial state layout have different dimensions.

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

StepControl responses also report authoritative start/end simulation times.
Before computing the knife pose, the source joint subscriber waits for a sample
strictly newer than the server start time. Its existing connection timeout
bounds callback delivery in wall time; physics does not advance while waiting.
Action duration and trajectory segment timing use accumulated server intervals
in integer nanoseconds in stepped mode. The previous ROS-clock/float loop
produced 52 ms actions for the configured 50 ms interval. Continuous control
retains its original clock path and all trajectory/force predicates.
