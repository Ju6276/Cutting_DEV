# Cutting World Model

Adapt [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
(`ba610534`) to SliceIt's Gazebo UR5e robot and DiSECt public prism. Keep SliceIt's
six actions, reward, force limits and termination rules; compare with its SAC.

The initial `online_source_20261102` run and concurrent SAC run have a
disconnected DiSECt-to-controller force path in their Gazebo launch. Updated
collection checks the relay, filter and controller subscriptions at startup;
use the repaired SliceIt launch for subsequent runs (see METHOD).

The trainer collects physical transitions, fits an eight-step world model and
updates the actor through 32-step imagined trajectories with a GAE critic.
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
  --worlds 16 --max-hours 6 --output runs/gpu_training
```

Online W&B is enabled by default (`--wandb-project sliceit-supermimic`);
`wandb.json` records the run URL. The existing authenticated account supplies the
entity, or set `--wandb-entity`. Final checkpoints are uploaded as model artifacts.

`--max-iterations` bounds a smoke run without changing the learning-rate
schedule. `--resume RUN/last.pt` restores learner/optimizer state and refills
replay with the source warmup. Each environment has independent episode rows,
controller/filter history, material damage and selective reset. Logs are
`progress.json`, `steps.jsonl`, `episodes.jsonl`, `updates.jsonl` and `last.pt`.

For complete-task throughput, run `scripts/benchmark_gpu_env.py
--backend-root ../CuttingGpuBackend --worlds 16 --output runs/gpu_cut_16`.
It follows the public reference trajectory, evaluates the original task heads,
and records each world's goal/force termination, spring release and runtime.

## Run on the 4090

From `/dataHDD/dong/Projects/CuttingWorldModel`:

```bash
.venv/bin/python scripts/run_coupled_collection.py \
  --sliceit-root /dataHDD/dong/Projects/SliceIt \
  --output runs/online_source --mode train --max-hours 6 --wall-seconds 22500
```

This starts headless Gazebo/DiSECt on ROS 11322/Gazebo 11356; SAC keeps its own
ports. The six-hour budget saves `data/last.pt`. The source learning-rate
schedule remains 320000 training steps; the budget does not rescale it.
Resume with `--resume /absolute/path/to/data/last.pt` and a new output directory.
As in the source, resume refills replay during a 40-step warmup.

`data/progress.json` reports the current physical step or learning phase.
`data/steps.jsonl`, `episodes.jsonl` and `updates.jsonl` record physical rewards,
success/collision, model losses, actor/critic metrics, timing and peak GPU memory.
Completed physical episodes are stored as NPZ. `manifest.json` records versions
and configuration. All runs stay outside Git.

Evaluate a saved policy without action exploration:

```bash
.venv/bin/python scripts/run_coupled_collection.py \
  --sliceit-root /dataHDD/dong/Projects/SliceIt \
  --output runs/evaluation --mode evaluate --episodes 5 \
  --resume /absolute/path/to/data/last.pt --max-hours 1 --wall-seconds 3900
```

## Interface checks

Run the supervisor with `--mode collect --episodes 1 --wall-seconds 600`.
Then run `scripts/check_task_interface.py --dataset RUN/data` and
`scripts/check_state_interface.py --dataset RUN/data` to compare recorded
observations, rewards, terminal labels, URDF kinematics and controller history.

Tests: `PYTHONPATH=. .venv/bin/python -m pytest tests -q`. GPU integration
checks additionally use `CUTTING_BACKEND_ROOT=/dataHDD/dong/Projects/CuttingGpuBackend`
and the Python 3.12 GPU environment above.
Independent object recording: `scripts/collect_disect.py`;
restart comparison: `scripts/check_state_restart.py`.

[Methods and source mapping](docs/METHOD.md) · [Results](docs/RESULTS.md) ·
[Plan](docs/PLAN.md) · [Source provenance](third_party/supermimic/README.md)

For automatic evaluation after an already-running training job,
`scripts/finish_online_run.py --training-run RUN --sliceit-root SLICEIT
--output EVAL_RUN` waits for supervisor cleanup, then evaluates `data/last.pt`
for five episodes with a one-hour budget. Failures are recorded in
`EVAL_RUN.followup.json`.
