# Cutting World Model

Learn cutting dynamics from DiSECt transitions using the dynamics MLP from
[SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced),
commit `ba6105349069bf040f1fdad60873d13138d505a2`.

The first experiment records the public prism, full particle position/velocity,
cutting-spring stiffness, knife pose/twist and knife force. The action is the
six-dimensional prescribed knife twist. The current SliceIt SAC run continues
in its original process and environment.

## Run on the 4090

DiSECt data collection uses the installed SliceIt physics environment:

```bash
source /dataHDD/dong/Projects/SliceIt/scripts/public_env.sh
python /dataHDD/dong/Projects/CuttingWorldModel/scripts/collect_disect.py \
  --config ../configs/public_prism.json --disect-root . \
  --output /dataHDD/dong/Projects/CuttingWorldModel/runs/prism_straight
```

Run the collector under `xvfb-run -a` if the installed Qt backend requires a display.
`--actions` accepts an NPY array of knife twists in m/s and rad/s. The dataset
stores T actions and T+1 states, plus unchanged physical configuration and mesh
connectivity. `progress.json` records every control step.

`scripts/check_state_restart.py --dataset <directory> --frame <index>` restarts
one control interval from a recorded, partially damaged state and reports
position, velocity and spring-stiffness differences.

The learned model has a separate `.venv` with PyTorch 2.4.1/CUDA 12.1 and NumPy
1.23.5. Its source import reads only the upstream dynamics module; it does not
install the G1 simulator or change SliceIt's dependencies.

For the selected robot-and-compliance task, collect the original SliceIt
actions, observations, rewards and terminal labels in an isolated instance:

```bash
.venv/bin/python scripts/run_coupled_collection.py \
  --sliceit-root /dataHDD/dong/Projects/SliceIt \
  --output runs/coupled_initial --episodes 3 --wall-seconds 900
```

This starts headless Gazebo and DiSECt on ROS 11322/Gazebo 11356, records full
episodes under `data/`, and stops those processes at the end. `status.json`
and `data/progress.json` show lifecycle and control-step progress.

Fit eight-step predictions, with a whole episode held out:

```bash
.venv/bin/python scripts/train_world_model.py \
  --train runs/coupled_initial/data/episode_000.npz runs/coupled_initial/data/episode_001.npz \
  --validation runs/coupled_initial/data/episode_002.npz \
  --upstream-root /dataHDD/dong/Projects/SuperMimic-Reproduced \
  --output runs/coupled_fit --updates 10 --wall-seconds 600
```

`report.json` compares physical-unit prediction errors with state persistence
and reports reward error, termination probability error and invalid material
predictions. `checkpoint.pt` contains network and optimizer state.

Methods and scope: [METHOD](docs/METHOD.md). Results: [RESULTS](docs/RESULTS.md).
Next experiments: [PLAN](docs/PLAN.md).
