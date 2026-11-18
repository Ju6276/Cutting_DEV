> Historical — superseded by [RUNNING](../RUNNING.md), kept for traceability.
> Split from docs/RUNNING.md on 2026-11-18 without rewriting: independent Gazebo/SliceIt
> reproduction (fixed prism, original tf2rl SAC and Gazebo WM collection).

# Running Cutting DEV — historical reproduction commands

Original intro of docs/RUNNING.md (2026-11-17):

> Historical fixed-prism/source commands below are reproduction entries, not
> instructions to restart S3. Current long-run settings and frozen launch records
> are in [TRAINING_SHARED_20261116](TRAINING_SHARED_20261116.md). The versioned opt-in free-cut contract is in [TASK_FREE_CUT](../TASK_FREE_CUT.md); follow [PLAN](../PLAN.md).
> Commands run from the repository root. The GPU backend is bundled at
> `backends/mujoco`; `--backend-root` can select a separate prepared checkout.

## Independent Gazebo reproduction

These commands require a built SliceIt ROS Noetic checkout and its Python 3.8
environment. Export `CUTTING_LOGGER_PYTHON` to the modern environment above.
Use an absolute project path when sourcing `public_env.sh`, which changes cwd.

```bash
export CUTTING_DEV_ROOT="$PWD"
export CUTTING_LOGGER_PYTHON="$PWD/.venv/bin/python"
export SLICEIT_ROOT=/dataHDD/dong/Projects/SliceIt
source "$SLICEIT_ROOT/scripts/public_env.sh"
cd "$CUTTING_DEV_ROOT"
python scripts/run_coupled_collection.py --sliceit-root "$SLICEIT_ROOT" \
  --mode collect --episodes 1 --wall-seconds 600 --output runs/capture
python scripts/run_coupled_collection.py --sliceit-root "$SLICEIT_ROOT" \
  --mode train --model-state control-history --wm-only-updates 500 \
  --max-hours 6 --wall-seconds 22200 --output runs/gazebo_wm
```

SAC uses its source settings. Create `runs/sac/run_config.json` as `{}`
for a fresh run, or include a `resume` path for existing weights/replay.
Its original endpoint is 100100 environment steps.

```bash
mkdir -p runs/sac
printf '{}\n' > runs/sac/run_config.json
python scripts/run_sac_training.py --sliceit-root "$SLICEIT_ROOT" \
  --output runs/sac
```

SAC uses ROS/Gazebo ports 11321/11355; Gazebo WM uses 11322/11356.
Keep current running SliceIt processes on their original checkouts.
