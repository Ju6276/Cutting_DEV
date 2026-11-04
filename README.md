# Cutting World Model

Adapt [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
(`ba610534`) to SliceIt's Gazebo UR5e robot and DiSECt public prism. Keep SliceIt's
six actions, reward, force limits and termination rules; compare with its SAC.

The ROS task uses corrected SliceIt `1477b0a` / robot utilities `c5f28a2`:
post-reset observations, component-major force history and synchronized steps.
ROS collection checks the material-force relay/filter/controller graph at startup.
The GPU collector includes controller/filter recurrence for world-model training.

The trainer collects physical transitions, fits an eight-step world model and
updates the actor through 32-step imagined trajectories with a GAE critic.
The first 500 completed world-model rounds train dynamics and termination only;
policy and critic updates begin with round 501. This startup stage follows
ndm-gym; `--wm-only-updates 0` selects the original immediate joint updates.
It uses the source effective batches (2048/4096), learning rates, update ratio,
network widths and gradient clipping. Microbatches control memory use.
The Gazebo entry collects one physical environment. The GPU entry batches
complete cutting worlds; its environment count is separate from the 4096-sample
actor learning batch.

## Run on the A6000

The ROS-free GPU entry uses `/dataHDD/dong/Projects/CuttingGpuBackend` and the
same learning modules as the Gazebo entry. Install `requirements-gpu.txt` in its
Python 3.12 environment. Prepare exported assets as described
in that backend's README. From `/dataHDD/dong/Projects/CuttingWorldModel`:

```bash
export CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST=8.6
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=/dataHDD/dong/.cache/torch_extensions_cutting_gpu
export XDG_CACHE_HOME=/dataHDD/dong/.cache TMPDIR=/dataHDD/dong/tmp MAX_JOBS=2
export TORCHINDUCTOR_CACHE_DIR=/dataHDD/dong/.cache/torchinductor_cutting
export TRITON_CACHE_DIR=/dataHDD/dong/.cache/triton_cutting
export NETRC=/dataHDD/dong/.config/wandb/netrc
export WANDB_CONFIG_DIR=/dataHDD/dong/.config/wandb
export WANDB_CACHE_DIR=/dataHDD/dong/.cache/wandb
export WANDB_DATA_DIR=/dataHDD/dong/.cache/wandb-data
.venv/bin/python scripts/train_gpu.py --backend-root ../CuttingGpuBackend \
  --worlds 1024 --wm-only-updates 500 --max-hours 24 --output runs/gpu_training
```

Online W&B is enabled by default (`--wandb-project sliceit-supermimic`);
`wandb.json` records the run URL. The existing authenticated account supplies the
entity, or set `--wandb-entity`. Final checkpoints are uploaded as model artifacts. W&B records
`learning/wm_updates`, `learning/policy_updates`, the training stage and actual
optimizer steps separately; policy updates remain zero during model preparation.

`--max-iterations` bounds a smoke run without changing the learning-rate
schedule. `--resume RUN/last.pt` restores learner/optimizer state and refills
replay with the source warmup. Stage configuration and completed WM rounds are
restored; changing the stage length on resume is rejected. Each environment has independent episode rows,
controller/filter history, material damage and selective reset. Logs are
`progress.json`, `steps.jsonl`, `episodes.jsonl`, `updates.jsonl` and `last.pt`.
`wm_only_complete.pt` stores the 500-round boundary; `first_joint_update.pt`
stores the first completed policy/critic update. At the measured 1024-world
throughput, collection to this boundary alone takes about 7.5 hours; model fitting
adds to that time.

`kill -USR1 PID` requests a GPU checkpoint at the next completed iteration while
training continues. Phase boundaries and normal shutdown also save checkpoints;
periodic wall-time and step-based saves are disabled. Checkpoints are written atomically.
Evaluate a frozen policy with `scripts/evaluate_gpu.py --backend-root
../CuttingGpuBackend --checkpoint RUN/last.pt --output runs/gpu_evaluation`.

For complete-task throughput, run `scripts/benchmark_gpu_env.py
--backend-root ../CuttingGpuBackend --worlds 16 --output runs/gpu_cut_16`.
It follows the public reference trajectory, evaluates the original task heads,
and records each world's goal/force termination, spring release and runtime.

## Collect interface data on the 4090

From `/dataHDD/dong/Projects/CuttingWorldModel`:

```bash
.venv/bin/python scripts/run_coupled_collection.py \
  --sliceit-root /dataHDD/dong/Projects/SliceIt \
  --output runs/interface_collection --mode collect --episodes 1 --wall-seconds 600
```

This starts headless Gazebo/DiSECt on ROS 11322/Gazebo 11356. Complete episodes
are written to `data/episode_*.npz`; metadata records the task interface revision.
ROS collection exports native FDCC, filter and complete wrench-queue memory
alongside joints, material and observation history (8990 state coordinates).
All run data stays outside Git.

## Control-history prediction on the 4090

`--model-state control-history` selects the 36-output robot/force candidate in
`run_coupled_collection.py --mode train`. It uses 10 causal frames (860 state
coordinates; 480 selected dynamics inputs), while collection still records the full physical state. Existing
native/GPU checkpoints use their original layout.

`scripts/check_control_model.py --train EPISODE... --held-out EPISODE...
--output runs/control_check` fits the source world-model update on complete
recorded episodes and evaluates separate episodes at 1/8/32 steps. Each episode
must have its source `task.json` in the same directory. Default budget: 50 updates,
600 seconds, unchanged effective batch/optimizer; reports include task-equation
parity, reset-start errors, force/torque errors, terminal counts and a checkpoint.
The command trains the world model; connected policy evaluation follows separately.

## Interface checks

Tracked SAC uses `scripts/train_sac.py --sliceit-root SLICEIT --output RUN`
followed by the original SAC arguments (`-e 3 --config CONFIG --logdir LOGDIR`).
The run launcher supplies `RUN/run_config.json` and the SliceIt ROS environment.
`scripts/run_sac_training.py --sliceit-root SLICEIT --output RUN` launches and
supervises its Gazebo, DiSECt and tracked trainer, returning nonzero on failure.
SAC's source `--model-dir` restores weights, optimizer state and replay; its
session counters and warmup restart. Copy checkpoint/replay into the new run's
restore directory because the source trainer writes replay back there.
The entry saves checkpoint/replay, closes ROS with bounded remote calls, writes
final status and finishes W&B independently. Coupled `--mode train` uses
`scripts/train_coupled_tracked.py` for the same W&B finalization. `scripts/guard_training_exit.py
--config RUN/guard_config.json` cleans up run-owned processes if the supervisor
exits; process identity and the unique run environment are checked before signaling.

Run the supervisor with `--mode collect --episodes 1 --wall-seconds 600`.
Then run `scripts/check_task_interface.py --dataset RUN/data` and
`scripts/check_state_interface.py --dataset RUN/data` to compare recorded
observations, rewards, terminal labels, URDF kinematics and controller history.
`PYTHONPATH=. .venv/bin/python scripts/check_native_state.py RUN/data` checks
native state packing, sample times and queue alignment. The learning schedule
counts completed vector steps, with source warmup 40 and UTD 0.4: first updates
occur at steps 42, 44, 47 and 49 for any environment count. Replay holds 1000 rows
per environment. Changing the world count changes the amount of newly collected
data per update. Runs 2 and 3 are stopped while the critic divergence is investigated;
current work and experiment results are in PLAN and RESULTS. The next candidate
uses model preparation with fresh weights and the same physical task.

Tests: `PYTHONPATH=. .venv/bin/python -m pytest tests -q`. GPU integration
checks additionally use `CUTTING_BACKEND_ROOT=/dataHDD/dong/Projects/CuttingGpuBackend`
and the Python 3.12 GPU environment above.
Independent object recording: `scripts/collect_disect.py`;
restart comparison: `scripts/check_state_restart.py`.

[Methods and source mapping](docs/METHOD.md) · [Results](docs/RESULTS.md) ·
[Plan](docs/PLAN.md) · [Source provenance](third_party/supermimic/README.md)
