# Historical backend measurements — 2026-11-02

These measurements use the recorded revisions below. They are not throughput or
validation results for current free_cut_v3. [Current results](../../../docs/RESULTS.md).

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

2026-11-02, `a0165ab` (start manifest):
`run_coupled.py --hold-last-action --wall-seconds 900 --output runs/coupled_endpoint`:
untrained recorded actions followed by final-action hold. Budget ended after
227 actions, 11.35 s simulation / 900.41 s wall time; goal failed at 2.91 mm
vertical error, 280/304 springs released, peak force 7.6459 N.

2026-11-02, `33449f8`:
`run_coupled.py --actions-file runs/sac_latest_actions.npz --hold-last-action
--wall-seconds 600 --output runs/coupled_sac_endpoint`:
latest complete SAC replay episode (indices 4955–4979, recorded return 88.2375).
Force-limit termination after 0.134 s simulation / 17.29 s wall time;
peak material force 64.689 N, filtered force 53.240 N, 0 springs released.

2026-11-02, `860b37a`:
`run_coupled.py --scene-motion --wall-seconds 300 --output runs/coupled_reference`:
public DiSECt reference velocity 0.05 m/s for 1.3 s, then hold; captured source
FDCC settings. Original task goal reached after 40 actions / 2.0 s simulation,
162.70 s wall time; final vertical error 1.496 mm, peak material force 9.9741 N,
288/304 springs released. Removing the filtered material contribution at the
same controller state changes internal joint velocity by up to 1.526e-4 rad/s.
Other SAC/world-model training stayed active during the measurement.

Gazebo wiring repair is committed in SliceIt `31852dc` (5 launch/relay checks
passed). Training startup verifies the three-topic relay/filter/controller
chain in CuttingWorldModel `7d34e27` (24 tests passed). Existing processes keep
their startup wiring; new runs load the repair.

2026-11-02, `1069081`, tamsgpu5 GPU 1 (RTX A6000 48 GB, driver 610.43.02):
Python 3.12.15, Torch 2.4.1+cu121, MuJoCo/MuJoCo Warp 3.16.0, Warp 1.18.0;
CUDA 12.9 compilation for sm_86. `PYTHONPATH=. .venv/bin/python -m pytest tests -q`:
18 passed in 2.48 s. `validate_disect.py --scene runs/disect_contact
--output runs/a6000_setup/disect_check.json`: 200 material steps in 0.0917 s,
position maximum/RMS error 7.08e-8/6.43e-9 m, velocity 4.41e-4/3.09e-5 m/s,
spring-stiffness maximum error 2.44e-4, contact-force maximum error 1.27e-5 N,
knife pose/twist error 0, 113 released springs.

Same commit and GPU: `run_coupled.py --scene-motion --wall-seconds 600
--output runs/a6000_reference`: goal reached after 40 actions / 2.0 s simulation;
115.36 s wall time including first-use Warp compilation and initialization.
Final vertical error 1.497 mm, peak material force 9.97446 N, 288/304 springs
released, force-feedback velocity response 1.526e-4 rad/s. Sampled process
memory during cutting: 451 MiB. Both GPUs otherwise idle during this run.
Source material, controller settings and task termination thresholds unchanged.


2026-11-02, backend `d81579e`, task/trainer `f656efc`, RTX A6000 GPU 1:
`CuttingWorldModel/scripts/benchmark_gpu_env.py --backend-root ../CuttingGpuBackend
--worlds 32 --wall-seconds 600 --output runs/gpu_cut_32`: 32/32 complete reference
cuts reached the task goal, each with 288/304 springs released. 1280 world-actions
in 88.06 s stepping (14.536 world-actions/s), 2.71 s setup, sampled process
memory 453 MiB. Peak forces range 9.97376–9.97683 N. One-world stepping took
66.91 s for 40 actions; aggregate 32-world throughput increased 24.3 times.
Backend tests: 19 passed on both A6000 and 4090. Complete scaling table and
learning runs are retained in the [root results archive](../../../docs/archive/RESULTS_20261102_20261110.md).
