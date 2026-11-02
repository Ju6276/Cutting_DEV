# Results

2026-11-02, commit `23d2501774cbe1857dc07b29d07b39685cb70592`: current UR5e imported with 6 joints, 46 bodies and 15 collision geometries. `scripts/validate_robot.py --episode runs/source/episode_000.npz --output runs/fk.json`: 42 recorded poses; maximum position error **5.66e-16 m**, rotation-matrix error **1.44e-15**. Source/imported total mass both **23.384 kg**. `PYTHONPATH=. .venv/bin/python -m pytest tests -q`: **3 passed**, including fixed-body kinetic-energy conservation.

2026-11-02, same commit, RTX 4090 on `tamsgpu9`: `.venv/bin/python -u scripts/benchmark_ur5e.py --worlds N --steps 100 --output runs/ur5e_N/report.json`; 20 warmup steps, 2 ms timestep, timeout 180 s, at most 60 s measured stepping budget. The two measurements used 0.3954 s of measured stepping in total.

| Worlds | 100 batch steps | World-steps/s | Batch-steps/s | Process memory after stepping | Setup/JIT/warmup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.18335 s | 545 | 545 | 518 MiB | 48.425 s |
| 32 | 0.21206 s | 15,090 | 471.6 | 518 MiB | 0.429 s |

Both runs: all states finite; clock advance 0.20000014 s; world-0 CPU MuJoCo error **1.03e-6 rad** in joint position and **1.23e-6 rad/s** in velocity after 121 total steps. The 32 worlds have independent control phases and a maximum terminal joint spread of **0.003619 rad**. World 0 differs between 1/32 batches by at most **1.40e-9 rad**. Terminal constraint count is 16 per world. Allocator/process memory above is a post-step sample.

Concurrent GPU users stayed active: old SAC's DiSECt PID 99620 (450 MiB), online DiSECt PID 851622 (450 MiB), online learned-model/policy PID 851964 (4314 MiB). Old SAC TensorFlow PID 99901 was also left running. Throughput is measured under this concurrent load.

The completed experiment is robot-only batching at 1 and 32 worlds. Full cutting world count is **0** in this backend. The next integration stage is the original Cartesian compliance controller and batched DiSECt material/knife coupling described in METHOD. Thousands of full cutting environments require that stage's measured resource curve.
