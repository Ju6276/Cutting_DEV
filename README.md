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

The learned model has a separate `.venv` with PyTorch 2.4.1/CUDA 12.1 and NumPy
1.23.5. Its source import reads only the upstream dynamics module; it does not
install the G1 simulator or change SliceIt's dependencies.

Methods and scope: [METHOD](docs/METHOD.md). Results: [RESULTS](docs/RESULTS.md).
Next experiments: [PLAN](docs/PLAN.md).
