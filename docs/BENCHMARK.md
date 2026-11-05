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
| DiSECt trajectory optimization | Optimized knife trajectory | Robot/controller evaluation adapter remains to be connected |

The constant baseline uses `reference=False`, as do both learned policies.
The separate `benchmark_gpu_env.py` command measures the backend's prescribed
knife-reference task and scaling. Its `reference=True` control is a different
baseline and has its own results.

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
```

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

## Next measurements

1. Frozen WM: compare train and held-out full physical episodes at H1, H8 and
   the complete action sequence, with policy/critic and their optimizers frozen.
2. Fit the critic against new complete physical returns after resolving the
   demonstrated model/target error; then check joint updates.
3. Train the original SAC in this fixed GPU environment, then evaluate SAC,
   constant control and the WM policy with the commands above. Compare success,
   force, duration, real transitions and wall time at saved checkpoints.
4. Connect the model-based knife optimizer to the same robot/controller task.
   Record the action/trajectory interface and optimization budget before comparing.
