> Historical — superseded by [METHOD](../METHOD.md), kept for traceability.
> Split from docs/METHOD.md on 2026-11-18 without rewriting. These sections describe the fixed-prism
> source adapter, the native ROS/Gazebo path, retired variants (S0 multi-condition, force-risk
> penalty) and the free_cut_v1–v4 task interfaces. Relative links are adjusted to this folder.

# Implementation reference — historical sections

## Historical source-baseline mapping

The following source mapping describes the retained original fixed-prism/source
adapter. Its24D observation,7D goal, vector-step cadence and historical
checkpoint choices are not the current shared training configuration.
Later dated implementation sections document migrations; old run results keep
their original version.



> 2026-11-12: `RESULTS <date>` references up to 2026-11-10 point to
> [archive/RESULTS_20261102_20261110](archive/RESULTS_20261102_20261110.md). The risk penalty
> and the official SuperMimic adaptation failed their first runs and are retained historical variants, not the active route.

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


## Multi-condition task (`--conditions`)

Implemented, retired S0, not a validated new task. Keep numerical settings for
reproduction; see [configuration status](../configs/multi_condition/README.md).

`cutting_world_model/multi_env.py`. A JSON config lists object groups (exported
scene, worlds, start-clearance range, material-scale ranges, target-speed range),
`w_speed` and a required `force_limits` vector of six finite positive values shared by all objects. Each group is a
homogeneous `GPUEnvironment`; groups are concatenated along the world dimension.
Per world (drawn once when the environment is built, not at resets): knife start
height, set by inverse kinematics
(`CuttingBatch(initial_lift=...)`, knife edge `clearance` above the object top).
Per episode at reset: Young's modulus, density and cut-spring stiffness scales
(log-uniform) and the target cutting speed (uniform). The task goal is
`[target_pose(7), target_speed]` (`goal_dim` 8): policy and critic receive it, the
WM dynamics receives the pose only, the reward adds
`-tanh(|v_down - v*| / v*)` while the cut depth is not reached (`SliceItHeads.speed_reward`).
No object identity or per-object limit is given to the learner. With `parallel: true`,
one persistent worker process steps each group concurrently; the sequential path
remains available. Missing/invalid limits fail before scene loading. The smoke config
explicitly records historical 50 N / 10 N·m values for interface checks only; these
are not the selected S0 limits. Legacy single-scene callers retain their original
task defaults. Objects: scanned surfaces
(`scripts/prepare_food_mesh.py`, fTetWild with a ~1 mm envelope) and DiSECt official
materials (`scripts/make_food_configs.py`), exported with `export_disect_scene.py`.


## Historical force-risk penalty (`--risk-constraint`, not selected)

Retained variant, not the selected paper method. Old S3 cannot establish control
benefit. B1 is a historical screen with an already-used test split.

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
and one without the constraint cannot be mixed. The penalized quantity (predicted
excess) and the quantity adapting lambda (observed episode rate) differ, so this is an
adaptive penalty rather than an exact Lagrangian; the pinball loss targets the 0.95
quantile without guaranteeing coverage.

Ensemble-risk comparator (`--risk-estimator ensemble`, precedent approach in the style of
LAMBDA/CAP): identical penalty, threshold and lambda adaptation, but the upper estimate is
mean + 2 std of R predicted by the main WM and four further independently initialized
WMs, each fitted every update with the WM procedure on its own minibatches (the
setting selected on B1 development data). All ensemble WM optimizers follow the
main WM learning-rate schedule, including after checkpoint restore. The quantile
head retains its separately configured fitting learning rate. Only the risk
estimator is exchanged in the actor objective; the ensemble adds four WM fits
and their computation cost must be reported.

Multi-object frozen evaluation: `scripts/evaluate_multi.py --checkpoint ... --conditions
<eval json> --episodes N` runs the deterministic policy until every world finished N
episodes and writes one row per episode (object, sampled conditions, ending, steps,
peak material force, peak limit ratio, impulse, released-spring fraction, target-speed
error while cutting) plus per-object summaries.

Worker lifecycle: the parent closes its copy of the child pipe, reports an
unexpected worker exit instead of waiting forever for a reply, and closes every
started group on initialization failure. Training closes groups even when learner
setup/checkpoint loading or finalization fails. Normal close waits up to 30 s,
then warns and terminates a stuck worker (with kill fallback). This is process
cleanup, not a physics timeout or a changed task termination rule.


## Historical opt-in free-object task (2026-11-12)

The historical pose task above is unchanged. The user-authorized implementation
free_cut_v1 is described in [TASK_FREE_CUT](TASK_FREE_CUT.md); source and verification
entry points are in [FREE_CUT_REVIEW](archive/FREE_CUT_REVIEW_20261112.md). Its compact WM adds
cut_progress and deterministic step_count; completion supervision uses physical
separation, not pose success or failure labels. Force/workspace/deadline endings
absorb imagined trajectories. The policy inputs remain unchanged. This is a new
task/head interface, not evidence of successful training or a new method claim.


### Free-cut review revision v2 (2026-11-12)

See [TASK_FREE_CUT](TASK_FREE_CUT.md) for the exact versioned contract and
[FREE_CUT_REPAIR](archive/FREE_CUT_REPAIR_20261113.md) for bounded evidence. Changes are gamma=1 across
actor/critic/planning, policy-observable remaining time (25 observations, goal dimension 8),
an explicit EE XYZ workspace, a .001 initial fraction-of-remaining-progress output,
and physical-label completion diagnostics. The progress initialization is not a separation
threshold. Source/v1 network shapes and defaults remain supported.

Kinematic observation/controller-state derivation is shared with the parent state class.
The free-cut task has its own reward/termination; old pose-goal diagnostic labels are no
longer used for its imagined event metrics. Completion BCE diagnostics use physical
separation, matching WM fitting.


## Free-cut v3 review and DiSECt execution changes

The opt-in v3 task uses the progress-weighted, finite-budget speed cost and bounded
completion probability in TASK_FREE_CUT section 9. Source/v1/v2 formulas are preserved.
This task-specific reward/parameterization is not evidence of a new algorithm or success.

(The DiSECt execution-change paragraphs of this section remain in docs/METHOD.md.)


## 2026-11-13 — Opt-in board-path physical task

The free_cut_v4 physical environment adds a finite-knife swept stop at the DiSECt ground plane and projected interface coverage. Physical separation is separately logged; path completion does not assert all material connections vanish. This is an execution interlock, not a board contact-force model. Source v1-v3 dynamics and task conditions remain the defaults. Full definition, tolerance provenance, frozen-policy transfer and training incompatibility are in [BOARD_CUT_REVIEW](archive/BOARD_CUT_REVIEW.md).
