# Source mapping

Source revision: SuperMimic-Reproduced `ba610534`; SliceIt `4efdf37`;
robot utilities `a14c856`; DiSECt `db7de0f`.

| Module | Source design | Cutting implementation |
|---|---|---|
| Replay | Episode-aware ring, 1000 rows/environment, clean/executed actions, terminal row | Original replay code with a recorded-episode importer; existing task configuration is fixed for this scene |
| Schedule | Warmup 40 environment steps; UTD 0.4; one actor update per world-model update | Original defaults and update cadence; ROS sampler has one environment; GPU sampler has configurable worlds; source default has 4096/GPU |
| Objective | Survival-weighted 32-step return, terminal critic, GAE lambda 0.95 | Original objective and GAE functions; only their type-only import is changed |
| Dynamics MLP | Three 2048-unit SiLU layers, zero final layer | Original source module with cutting StateSpec and connected trainer |
| Policy | Goal encoder, 32-D latent, decoder; widths 2048/1024/512/256; decoder gain 0.01 | Same architecture with a 7-D target pose, ten frames of SliceIt's 24-D observation, and six task actions |
| Policy input scaling | G1 observations and reference motions use robot-specific scales | SliceIt observations retain their existing normalization; target pose uses its SI/quaternion representation |
| Actions | Gaussian execution noise sigma 0.1 | Same noise; execution respects SliceIt's existing action-space bounds; both clean and executed actions are recorded. SliceIt's last-action observation contains the executed action |
| Observation/reward | Analytic task functions, including robot FK | Torch translation of original SliceIt normalization, reward and termination formulas. No learned reward network |
| Termination | Learned probability head trained from simulator terminal labels | Connected cutting probability head with force, workspace, goal and counter features |
| Physical state | Simulator state read at the transition boundary | Gazebo joint/controller state plus DiSECt position, velocity and spring history, recorded during observation generation, before terminal controller shutdown |
| Force history | Observation reads the sensor history | Record that exact sensor read, rather than rereading a moving queue after env.step |
| Material-force wiring | DiSECt reaction feeds the robot force sensor and compliance controller | Initial Gazebo runs have disconnected material-force and controller-input topics; new runs check the complete relay/filter/controller graph before collection |
| Reset boundary | SliceIt reset reads its return observation before physically resetting the scene | Preserve that returned observation and its sensor sample; capture the initial physical state after reset. Subsequent snapshots occur during observation generation |
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
height/tilt. Reward retains SliceIt's discrete collision/success bonuses and
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
| Reset observation | ROS task returns an observation sampled before physical reset | GPU task computes observation after restoring robot/material/controller state; derivative histories start from the restored pose and zero velocity/acceleration |
| Wrench samples | Oldest three entries of a 25-entry sensor queue, appended at filter callbacks | Same queue ordering at the deterministic 500 Hz exchange; initialize queue from the reset sensor baseline |
| Initial condition | ROS moves the arm to its start pose before resetting DiSECt | Restore the exported start and initialized controller/filter snapshot per world; shared public material configuration and prescribed cut topology |

`train_gpu.py` reuses OnlineLearner, StateSpec, policy/critic networks, objectives,
learning rates, effective batches and the 320000-step schedule. Environment count
changes the number of physical transitions collected per vector step. It does
not change the eight-step world-model loss or 32-step actor BPTT horizon.
`--max-hours` and `--max-iterations` bound execution without rescaling the schedule.
