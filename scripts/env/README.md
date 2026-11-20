# Runtime dependencies and environments

Use [../../requirements.txt](../../requirements.txt) for the main training environment.
UR5e and UR10e share these dependencies. DiSECt and BladeMaster also share the
training environment; selecting a robot or material backend does not create a
new Python environment.

| Entry | Purpose | Existing environment |
|---|---|---|
| `requirements.txt` | WM+BPTT, W&B, and both material backends | `.venv` |
| `requirements/physics.txt` | Torch, Warp, MuJoCo, build and test dependencies; included above | `.venv` |
| `requirements/blademaster.txt` | SciPy runtime extra; included above | Legacy base installations use `make_bm_extra.sh` |
| `requirements/baselines.txt` | SAC, Gymnasium and pinned MPPI in one manifest | `.venv-sac`; `make_sac_env.sh` |
| `requirements/render.txt` | Complete pinned offline-renderer environment | `.venv-render`; `make_render_env.sh` |
| `requirements/cad.txt` | CAD/USDZ preparation; unnecessary for prepared-scene training | Separate CAD environment |

Only one renderer manifest is retained. The former direct-dependency list was
redundant with the complete pinned list. The former MPPI manifest is merged into
the baselines manifest because SAC and planner entry points use the same
environment. Package versions and the MPPI commit are unchanged.

The renderer uses MuJoCo 4.0, while training uses MuJoCo 3.16; install these into
separate environments. Existing running environments are not reinstalled by
this repository reorganization. CAD uses version ranges; record the installed
environment and generated asset identity when exporting a new tool.

From the repository root:

```bash
# New Linux training environment only:
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Existing workstation environments:
source scripts/env/cutting_env.sh
CUTTING_ENV=sac source scripts/env/cutting_env.sh
CUTTING_ENV=planner source scripts/env/cutting_env.sh
CUTTING_ENV=render source scripts/env/cutting_env.sh
```

Use a fresh shell or explicitly select `CUTTING_ENV=base` when returning to the
base environment. GPU selection must persist into the launched process:

```bash
export CUDA_VISIBLE_DEVICES=1
source scripts/env/cutting_env.sh
python scripts/train_gpu.py --help
```

Host GPU defaults are not reservations; inspect occupancy before launch.
All long runs use online W&B as specified in [RUNNING](../../docs/RUNNING.md).
The separate native BladeMaster repository has development dependencies for
its own experiments, including differentiable-physics tests. Those are not
required for this project's WM+BPTT training pipeline.
