# MuJoCo cutting backend

SliceIt's UR5e, tensor FDCC compliance controller and DiSECt material dynamics
run together in independent GPU worlds. The learning code and entry points are
at the [repository root](../../README.md). Current bundled contact uses
`edge_coordinate_consistent_v2`; optional net feedback and historical scalar
feedback are distinct conventions. See the root [METHOD](../../docs/METHOD.md).

Use the root Python 3.12 environment and follow
[RUNNING](../../docs/RUNNING.md) to prepare robot meshes, the captured reset,
controller parameters and exported material scene. Runtime assets stay under
this directory's ignored `runs/` directory.

From the repository root:

```bash
.venv/bin/python scripts/benchmark_gpu_env.py --worlds 1 \
  --wall-seconds 600 --output runs/reference
```

This executes the reference motion through the complete robot/controller/material
loop and records task termination, force and timing. `CuttingBatch` accepts a
time-varying DiSECt-frame knife trajectory (`cutting_gpu_backend/reference.py`);
the default is the scene's constant-velocity reference. A robot-only throughput
measurement is available through `scripts/benchmark_ur5e.py` in this backend;
it excludes controller and material integration.

The source pins are MuJoCo/MuJoCo Warp 3.16.0 and Warp 1.18.0, matching the
SuperMimic-Reproduced backend revision recorded in
[SOURCES](../../docs/SOURCES.md).

[Implementation](docs/METHOD.md) · [Historical measurements](docs/RESULTS.md) ·
[Third-party licenses](../../NOTICE.md)
