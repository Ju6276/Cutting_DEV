# Cutting World Model

Adapt [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
(`ba610534`) to SliceIt's Gazebo robot and DiSECt public prism. Keep SliceIt's
six actions, reward, force limits and termination rules; compare with its SAC.

The trainer collects physical transitions, fits an eight-step world model and
updates the actor through 32-step imagined trajectories with a GAE critic.
It uses the source effective batches (2048/4096), learning rates, update ratio,
network widths and gradient clipping. Microbatches control memory use.
Physical collection currently uses one Gazebo/DiSECt environment; 4096 is the
actor learning batch size, not the number of parallel physics environments.

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

Tests: `PYTHONPATH=. .venv/bin/python -m pytest tests -q`.
Independent object recording: `scripts/collect_disect.py`;
restart comparison: `scripts/check_state_restart.py`.

[Methods and source mapping](docs/METHOD.md) · [Results](docs/RESULTS.md) ·
[Plan](docs/PLAN.md) · [Source provenance](third_party/supermimic/README.md)
