# Cutting World Model

Migrate [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
(`ba610534`) to SliceIt's Gazebo robot and DiSECt public prism. Keep the six
SliceIt actions, task rewards and termination conditions.

Implemented: synchronized state recording, analytic SliceIt observation/reward
functions, upstream replay and return/GAE equations, source update schedule,
and the goal-encoder/decoder policy with cutting dimensions.
Next: cutting StateSpec and the connected online world-model/policy trainer.

## Check a complete robot episode

On the 4090, from `/dataHDD/dong/Projects/CuttingWorldModel`:

```bash
.venv/bin/python scripts/run_coupled_collection.py \
  --sliceit-root /dataHDD/dong/Projects/SliceIt \
  --output runs/interface_source_policy --episodes 1 --wall-seconds 600
.venv/bin/python scripts/check_task_interface.py \
  --dataset runs/interface_source_policy/data
```

The first command starts an isolated headless Gazebo/DiSECt instance on ROS
11322/Gazebo 11356 and records one episode from the initialized policy with
source action noise. It stops those child processes when complete or timed out.
`status.json` and `data/progress.json` report progress. The second command checks
analytic observations, rewards, termination and upstream replay-window alignment.
The existing SAC uses its original ports, process and environment.

Independent object-dynamics recording: `scripts/collect_disect.py`.
State restart comparison: `scripts/check_state_restart.py`.

Tests: `PYTHONPATH=. .venv/bin/python -m pytest tests -q`.
Methods: [METHOD](docs/METHOD.md). Numbers: [RESULTS](docs/RESULTS.md).
Next work: [PLAN](docs/PLAN.md). Source provenance: [third_party](third_party/supermimic/README.md).
