# Fixed cutting benchmark

Every method executes in **UR5e + MuJoCo Warp + the SliceIt compliance controller
+ DiSECt**, using `GPUEnvironment`. The current case is
`public-prism-ur5e-v1`: one prism, one material configuration, one initial state
and one cutting target. The existing Gazebo SAC training continues separately.

## Fixed inputs

`configs/public_prism/benchmark.json` records the existing exported configuration.
Evaluation checks the material settings, controller parameters, task settings,
captured reset, 1187 vertices and 304 cutting springs. It also checks the content
identities of the material export, compiled robot and robot XML. The 4090 and
A6000 prepared assets match. Each episode restores the entire initialized
physical frame, including controller/filter memory and material state.

The policy period is 0.05 s, robot step 0.001 s, coupling period 0.002 s, and
material step 0.00001 s. The original action bounds, reward weights, workspace,
force limits and success counter remain in the shared task heads.

## Methods

| Method | Action source | Status |
|---|---|---|
| Constant zero | Six zero actions through the same compliance action mapper | Fixed-control baseline |
| SliceIt SAC | Original tf2rl Wave actor; Python 3.8 inference process | Frozen checkpoint evaluation and training adapter |
| SuperMimic-style policy | Current learned policy; ten observation frames and fixed target pose | Frozen checkpoint evaluation |
| Prescribed reference | Scene knife velocity (0, -0.05, 0) m/s in the DiSECt frame for 1.3 s, then hold | `--method reference` |
| DiSECt trajectory optimization | Knife velocity trajectory from `optimize_slicing.py` parameters | `--method reference --trajectory FILE`; the gradient optimizer currently produces no update (below) |

The constant baseline uses `reference=False`, as do both learned policies.
The two trajectory methods use the **reference control branch** (`reference=True`),
not the 6-D policy action interface: the six-action mapper is bypassed and the
compliance controller's target pose is the reset tip pose translated by the
knife trajectory's displacement, rotated from the DiSECt frame into the robot
base by the fixed `cutting_board_disect` transform. At each 0.05 s policy step the
target is the displacement at the end of that step; orientation stays at the reset
tip orientation. The robot, compliance controller, coupling, material, reset,
reward, termination and spring/force metrics are the same as for the policies.
Recorded actions are zero; the observation's last-action field is therefore zero.
`benchmark_gpu_env.py` remains the multi-world scaling measurement of the default
reference.

### Trajectory file

`cutting-knife-velocity-v1` JSON (`backends/mujoco/cutting_gpu_backend/reference.py`):

```json
{"format": "cutting-knife-velocity-v1", "frame": "disect",
 "times": [0.0, 0.001, 0.002, ...],
 "linear_velocity": [[vx, vy, vz], ...],
 "source": {"kind": "...", ...}}
```

`times` (s) start at 0 and increase; interval `[times[i], times[i+1])` moves with
`linear_velocity[i]` (m/s, DiSECt frame: y up/pressing, z along the blade).
After the last time the displacement is held. Time is the episode's simulated
time. A one-interval file with the scene velocity and 1.3 s reproduces the default
targets bit for bit (tested). `scripts/export_disect_trajectory.py` converts a
DiSECt-Optimization run (checkpoint or initial parameters) into this format at
1 ms intervals; each interval is the mean of its 100 DiSECt substep velocities,
so boundary displacements equal DiSECt's prescribed free-knife displacement.

SAC receives the source 24-D observation, including three force-history samples;
the SuperMimic policy receives ten such frames plus the target pose. These source
architectures are retained. The common task, physical reset and six-action
interface are fixed. A Gazebo-trained SAC checkpoint evaluated here measures
transfer; the algorithm training comparison uses SAC trained in this same GPU
environment, with its original training settings.

## Train SAC in the fixed environment

The source endpoint is 100100 environment steps. The existing configuration
sets batch 1024, learning rate 0.01, automatic entropy tuning and seed 123456.
The source warmup is 1000 random-action steps; its Trainer already updates
SAC during this period. Periodic source checkpoints are saved every 5000 steps.
The wrapper also saves checkpoint/replay when its wall budget or SIGTERM ends
the run. W&B runs through the existing modern-Python logger.

```bash
export CUTTING_LOGGER_PYTHON="$PWD/.venv/bin/python"
"$SLICEIT_ROOT/.venv/bin/python" scripts/train_sac_gpu.py \
  --sliceit-root "$SLICEIT_ROOT" \
  --config "$SLICEIT_ROOT/configs/public_prism_ros.yaml" \
  --gpu-python "$PWD/.venv/bin/python" \
  --backend-root "$PWD/backends/mujoco" \
  --wall-seconds 259200 --output runs/sac_gpu_training
```

This command starts a fresh SAC. Frozen evaluation of an existing checkpoint
uses `evaluate_benchmark.py` below.

To continue a stopped or interrupted run, repeat the command with a new
`--output` and `--resume runs/sac_gpu_training`. Every 5000-step save and the
final save write `resume_state.json` (checkpoint, replay file, total steps,
updates and episode records) when replay rows equal steps and every optimizer
iteration count equals the update count. Resume restores the TensorFlow
checkpoint (networks, targets, Adam moments and counters, log-alpha and its
optimizer), loads the replay through the source Trainer, and continues the
total step count: the remaining warmup is `1000 - steps`, the 100100 endpoint
and 5000-step saves count all steps, and episode/TensorBoard indices continue.
The child `run_config.json` records the parent. Not restored: an unfinished
episode restarts from reset (its transitions stay in replay and the count is
recorded as `discarded_episode_steps`); restored replay rows start at cpprb's
default maximum priority (the source pickle stores transitions only); random streams restart from seed 123456, as in the
source Trainer's own `--model-dir` restore. Outputs written before resume
support resume from the final save in `result.json`.

## Evaluate

Set the CUDA/cache variables in [RUNNING](RUNNING.md) first. Each command writes
to a new output directory. Start with one complete episode; increasing `--episodes`
checks repeatability of the same fixed task. Exploration noise and optimizer updates are zero.

```bash
timeout --signal=TERM --kill-after=15s 660s .venv/bin/python \
  scripts/evaluate_benchmark.py --method constant-zero \
  --episodes 1 --wall-seconds 600 --output runs/benchmark_constant

timeout --signal=TERM --kill-after=15s 660s .venv/bin/python \
  scripts/evaluate_benchmark.py --method supermimic \
  --checkpoint runs/training/last.pt \
  --episodes 1 --wall-seconds 600 --output runs/benchmark_policy

timeout --signal=TERM --kill-after=15s 660s .venv/bin/python \
  scripts/evaluate_benchmark.py --method sac \
  --checkpoint "$SAC_CHECKPOINT_DIR" \
  --sac-python "$SLICEIT_ROOT/.venv/bin/python" \
  --sliceit-root "$SLICEIT_ROOT" \
  --sac-config "$SLICEIT_ROOT/configs/public_prism_ros.yaml" \
  --episodes 1 --wall-seconds 600 --output runs/benchmark_sac

timeout --signal=TERM --kill-after=15s 660s .venv/bin/python \
  scripts/evaluate_benchmark.py --method reference \
  --episodes 1 --wall-seconds 600 --output runs/benchmark_reference

.venv/bin/python scripts/export_disect_trajectory.py \
  --run "$DISECT_OPT_RUN" --parameters checkpoint --output runs/knife_trajectory.json
timeout --signal=TERM --kill-after=15s 5460s .venv/bin/python \
  scripts/evaluate_benchmark.py --method reference --trajectory runs/knife_trajectory.json \
  --episodes 1 --wall-seconds 5400 --output runs/benchmark_disect
```

A trajectory that stops before the goal runs to the 500-step time limit, so give
it a larger wall budget. `--parameters initial` exports the optimizer's starting
motion when no update has completed.

`report.json` records completed episodes, success, force/workspace failure,
time truncation, return, actual simulated seconds, peak material reaction force,
released springs and new spring releases. Inference, physics and total wall times
are separate. Physics time depends on other work sharing the GPU; repeat timed
comparisons on an idle card.

`episode_*.npz` contains T+1 complete physical frames and T actions/rewards/terminal
labels. Terminal frames are saved before reset. A budget expiry or exception
retains the valid prefix with `completed=false`. Task success follows SliceIt's
existing predicate; spring releases are reported separately as material evidence.
The material force metric is the maximum, over coupling intervals, of DiSECt's
sum of contact-force magnitudes. It is separate from the filtered robot wrench.

### Episode metrics

`CuttingBatch.step` integrates over every 2 ms coupling substep of the world while
it is active (a world stops integrating at its collision check). The step's
return values are unchanged; the integrals of the last step are
`CuttingBatch.step_metrics` and `GPUEnvironment.step` adds them to `info`
(`material_impulse`, `knife_work`, `knife_gross_work`, `safety_margin`, plus
`jerkiness`). `evaluate()` sums them per episode. `F` is DiSECt's summed contact-force
magnitude after the substep's material advance (`feedback['force_magnitude']`,
the quantity whose maximum is `material_peak_force_n`); `v` is the knife linear
velocity sent to DiSECt for that substep (DiSECt frame, after the coupling's
first-step and jump zeroing).

| Field | Definition |
|---|---|
| `material_impulse_ns` | ∫ F dt (N s) |
| `mean_material_force_n` | `material_impulse_ns / simulation_seconds` (time average over simulated time) |
| `knife_work_j` | ∫ F (-v_y) dt (J): net mechanical work by the knife on the material |
| `knife_gross_work_j` | ∫ F \|v_y\| dt (J); equals net work unless the knife rises while in contact |
| `safety_margin` | max over checks and components of \|observation_wrench_k\| / max_force_torque_k, the collision check's quantity; >= 1 means a limit was exceeded |
| `total_cut_springs`, `cut_fraction` | number of cut springs (304); `released_springs / total_cut_springs` (includes springs released at reset) |
| `mean_jerk` | mean over control steps of SliceIt's reward jerkiness, the norm of the observed jerk over `target_dims` divided by 500 (`SliceItHeads.jerkiness`; dimensionless, the reward clamps it at 1) |

**Work sign and frame.** DiSECt y is up; the scene reference presses at
v_y = -0.05 m/s. The coupling applies the material force to the knife purely
along +y of the DiSECt frame (`wrench_disect[..., 1] = magnitude`), opposing the
press, so the power delivered to the material is `-F_knife · v = F (-v_y)`:
positive when pressing down, negative when the knife is lifted against
contact force. The lateral (z, along-blade) sawing motion does no work in this
coupling because the applied force has no z component.

The safety margin uses the checks at the start of each substep, so the
wrench after the final substep of an episode is not included, as in the
collision check itself.

`report.json` also contains `summary`: over completed episodes, the mean and
population std (`ddof=0`) of every numeric row field, plus `success_rate` and
`collision_rate`.

## Trajectory baseline results (2026-11-05, 4090 shared with other jobs)

Measured, one deterministic-input episode each (the material's FP32 atomics make
repeated runs differ in the last digits; see METHOD):

| Run | Success | Steps / sim s | Return | Peak material force | Released springs | Physics wall |
|---|---|---|---|---|---|---|
| Prescribed reference (`runs/benchmark_reference_20261105`) | yes | 40 / 2.00 | 80.29 | 9.975 N | 288/304 | 310 s |
| DiSECt initial motion, robot-executed (`runs/benchmark_disect_initial_20261105`) | no, time limit | 500 / 25.0 | -250.27 | 13.22 N | 218/304 | 3580 s |

Physics wall time was measured while another user's training job held the GPU at
100%; it is not a speed result.

DiSECt optimization (`optimize.py` at `372e8ac`, DiSECt `db7de0f`; run
`/dataHDD/dong/Projects/DiSECt-Optimization/runs/prism_benchmark_20261105`; 100
iterations and 14400 s requested): the iteration-0 forward pass took 80 s, then
the first backward pass returned nonfinite parameter gradients and the run stopped
after about 3 minutes with **no optimizer update**. This reproduces the failure
recorded in that repository's README. There is therefore no optimized trajectory.
Iteration-0 free-knife result (0.6 s, no robot): mean force 7.533 N, peak 14.755 N,
end blade height residual +0.0119 m (blade bottom above the board), lateral
coverage residual 0 m, 192/304 springs released.

The iteration-0 motion (frequency 5 rad/s, amplitudes 0.02 m/s, pressing
-0.05 m/s at five waypoints) was exported with `--parameters initial` and executed
through the robot. Its knife displacement is (0, -0.0484, +0.0131) m in the DiSECt
frame. The robot knife followed it with a maximum lag of 13.2 mm during the
0.6 s motion and settled 0.5 mm above the held command (knife y 0.0403 m); it
released 218 springs by 1.3 s and none afterwards. The goal tip pose (base
z 0.126 m) corresponds to knife y about 0.025 m, reached by the prescribed
reference (tip z 0.1245 m); the DiSECt motion ends with tip z 0.1411 m and x offset 13 mm,
so SliceIt's goal predicate never triggers and the episode ends at 500 steps.
Inference: even a converged DiSECt solution targets blade bottom at the board
(knife y about 0.030 m) rather than the SliceIt goal pose; comparing it on task
success needs that difference stated.

## Next measurements

1. Frozen WM: compare train and held-out full physical episodes at H1, H8 and
   the complete action sequence, with policy/critic and their optimizers frozen.
2. Fit the critic against new complete physical returns after resolving the
   demonstrated model/target error; then check joint updates.
3. Train the original SAC in this fixed GPU environment, then evaluate SAC,
   constant control and the WM policy with the commands above. Compare success,
   force, duration, real transitions and wall time at saved checkpoints.
4. DiSECt trajectory optimization: resolve the nonfinite prism adjoint (or agree
   on a different optimizer) before an optimized trajectory can be compared.
