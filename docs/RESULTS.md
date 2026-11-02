# Results

2026-11-02, commit `23d2501774cbe1857dc07b29d07b39685cb70592`: current UR5e imported with 6 joints, 46 bodies and 15 collision geometries. `scripts/validate_robot.py --episode runs/source/episode_000.npz --output runs/fk.json`: 42 recorded poses; maximum position error **5.66e-16 m**, rotation-matrix error **1.44e-15**. Source/imported total mass both **23.384 kg**. `PYTHONPATH=. .venv/bin/python -m pytest tests -q`: **3 passed**, including fixed-body kinetic-energy conservation.

2026-11-02, same commit, RTX 4090 on `tamsgpu9`: `.venv/bin/python -u scripts/benchmark_ur5e.py --worlds N --steps 100 --output runs/ur5e_N/report.json`; 20 warmup steps, 2 ms timestep, timeout 180 s, at most 60 s measured stepping budget. The 1/32-world measurements use commit `23d2501`; 256/1024/4096 use `7250318` (documentation-only change). All five measurements use **1.4410 s** of measured stepping in total. Before each larger batch, free GPU memory was 18,812 MiB; the process budget was half of that (9406 MiB).

| Worlds | 100 batch steps | World-steps/s | Batch-steps/s | Process memory after stepping | Setup/JIT/warmup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.18335 s | 545 | 545 | 518 MiB | 48.425 s |
| 32 | 0.21206 s | 15,090 | 471.6 | 518 MiB | 0.429 s |
| 256 | 0.14000 s | 182,857 | 714.3 | 678 MiB | 0.376 s |
| 1024 | 0.45486 s | 225,125 | 219.8 | 1286 MiB | 0.531 s |
| 4096 | 0.45078 s | **908,648** | 221.8 | **3558 MiB** | 0.577 s |

All five runs: all states finite; clock advance 0.20000014 s; world-0 CPU MuJoCo error **1.03e-6 rad** in joint position and **1.23e-6 rad/s** in velocity after 121 total steps. The 32 worlds have independent control phases and a maximum terminal joint spread of **0.003619 rad**. World 0 differs between 1/32 batches by at most **1.40e-9 rad**. Terminal constraint count is 16 per world. Allocator/process memory above is a post-step sample.

Concurrent GPU users stayed active: old SAC's DiSECt PID 99620 (450 MiB), online DiSECt PID 851622 (450 MiB), online learned-model/policy PID 851964 (4314 MiB). Old SAC TensorFlow PID 99901 was also left running. Throughput is measured under this concurrent load.

Robot-only batching completed at 1, 32, 256, 1024 and **4096 worlds**. At 4096 worlds, independent control phases produce a maximum terminal joint spread of 0.003754 rad. Complete cutting throughput is measured separately below.

2026-11-02, controller `3c2ac45`, coupling `87390c0`, adjacent-body import `6b7a360`:
42-frame CUDA/KDL controller comparison: command maximum error 1.69e-12 rad,
internal velocity 3.51e-11 rad/s. Coupled importer excludes 14 adjacent-body
pairs; initial robot contacts 0, mass 23.384 kg, FK error 5.66e-16 m.

2026-11-02, remote parent `861abdf` (dirty headless-core/controller changes), source DiSECt `db7de0f`:
`validate_disect.py --scene runs/disect_contact --output runs/disect_contact_check.json`:
continue 200 steps at 10 µs from 0.65 s inside a cut (113 released springs).
Position maximum/RMS error 7.45e-8/6.42e-9 m; velocity maximum/RMS
5.13e-4/3.25e-5 m/s; spring-stiffness maximum error 1.22e-4;
per-contact force maximum error 1.46e-5 N; knife pose/twist error 0.
Headless continuation took 0.309 s with other training active.

2026-11-02, live ROS wiring audit on masters 11321 and 11322:
`/disect/wrench` has a publisher and zero subscribers; controller
`ft_sensor_wrench` has a subscriber and zero publishers. The active filter
has no second-input option, and the controller remap follows the gzserver
launch. These runs collect robot-driven cuts with the material force path
disconnected. The tensor coupling connects that path directly.

2026-11-02, remote parent `861abdf` (dirty coupled-loop changes):
`run_coupled.py --wall-seconds 600 --output runs/coupled_sensor_startup`:
41 recorded actions, 2.05 s simulation, 165.64 s wall time, 160/304 springs
released, peak material force 7.6461 N. Stopped at the recorded sequence end;
vertical target error 24.77 mm. Position-motor holding/yielding checks and
the complete test suite: 18 passed in 1.32 s.
