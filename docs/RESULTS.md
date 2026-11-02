# Results

## 2026-11-02

ROS wiring audit, running `online_source_20261102` and SAC masters 11322/11321:
DiSECt force-topic subscribers 0; controller force-input publishers 0.
Recorded observations/rewards match that running environment, while its
material-to-controller force feedback is disconnected. New collection verifies
the complete relay/filter/controller path before recording transitions.

Collector `9e7273d`, SuperMimic-Reproduced `ba610534`, DiSECt `db7de0f`.
`collect_disect.py --config ../configs/public_prism.json --disect-root .
--output /dataHDD/dong/Projects/CuttingWorldModel/runs/prism_straight`:
1.3 s, 26 control intervals, 1187 vertices, 304 springs, 281 released at the
endpoint; peak force 10.3782 N, wall time 87.10 s while SAC was also running.
Full positions, velocities, remaining spring stiffness, knife pose/twist and
force are stored in `runs/prism_straight/trajectory.npz`.

`check_state_restart.py --dataset runs/prism_straight --frame 13`:
110 springs released at restart. One-step position RMS/max difference
3.78e-8/3.80e-7 m; velocity RMS/max 4.39e-5/6.17e-4 m/s; spring stiffness
RMS/max 0.00572/0.07373. Record: `restart_check.json`.

During additional physics collection, GPU utilization reached 98% and one
SAC episode took 96.22 s for 22 steps, versus about 1.76 s/step beforehand.
The collector exited after one trajectory. Further large physics collection
must be scheduled rather than continuously competing with the SAC baseline.

Task interface `ec8c49f`, source `ba610534`:
`run_coupled_collection.py --sliceit-root /dataHDD/dong/Projects/SliceIt
--output runs/interface_source_policy_aligned --episodes 1 --wall-seconds 600`:
41 transitions, 233.39 s total wall time (197.08 s collector); episode ended at
the original force limit, measured x force -52.318 N. Policy updates: 0.

`check_task_interface.py --dataset runs/interface_source_policy_aligned/data`:
observation maximum absolute error 0; reward maximum absolute error 5.55e-17;
termination and success mismatches 0; success-counter error 0. Upstream replay:
42 rows; eight-step window batch shape [9, 2048, 7535].
`python -m pytest tests -q`: 12 passed in 0.74 s.

Cutting state interface, parent `5a75ca5` + connected-trainer changes:
`run_coupled_collection.py --sliceit-root /dataHDD/dong/Projects/SliceIt
--output runs/state_interface --episodes 1 --wall-seconds 600`:
41 transitions, 191.89 s collector time; original force-limit termination.
`check_state_interface.py --dataset runs/state_interface/data`:
URDF FK position/rotation maxima 6.94e-16 m / 8.95e-16 rad;
analytic observation/reward error 2.78e-14 / 6.94e-15;
controller accumulation error 2.17e-19; collision/termination mismatches 0.

Connected update check on the same 41 recorded transitions:
`OnlineLearner.fit_world_model` (10 batches × 2048, horizon 8), then
`OnlineLearner.improve_policy` (4096 starts, horizon 32, five critic epochs):
completed; model/actor gradient norms 43.9297 / 0.003292; peak allocated GPU
memory 2577.35 MiB. First model/actor+critic updates took 61.88 / 54.71 s,
including compilation. These are execution measurements before online learning.
`python -m pytest tests -q`: 20 passed in 1.83 s.


2026-11-02, world-model `f656efc`, backend `d81579e`, RTX A6000 GPU 1:
`benchmark_gpu_env.py --backend-root ../CuttingGpuBackend --worlds N
--wall-seconds 600 --output runs/gpu_cut_N`. Same public material and reference
trajectory; every episode reaches the original goal after 40 actions / 2 s,
with 288/304 springs released. Full task includes robot, material, compliance,
force filtering, observation, reward and termination.

| Worlds | Successful cuts | Stepping time (s) | World-steps/s | Setup (s) |
|---:|---:|---:|---:|---:|
| 1 | 1/1 | 66.91 | 0.598 | 2.64 |
| 2 | 2/2 | 67.48 | 1.186 | 2.69 |
| 16 | 16/16 | 80.26 | 7.974 | 2.70 |
| 32 | 32/32 | 88.06 | 14.536 | 2.71 |

16/32 manifests record clean commits above. The 1-world manifest records
parents `7d34e27`/`403a143` with dirty changes; the 2-world manifest records
`f656efc`/`d81579e` with dirty index state. The 32-world peak force range is
9.97376–9.97683 N; sampled process memory 453 MiB. Aggregate throughput is
24.3 times the one-world run. Torch allocation peak alone is 30.04 MiB.
These are prescribed-reference episodes; learned-policy episodes are logged
by the training run below.

GPU test suites on A6000: backend 19 passed (2.35 s), world model 26 passed
(10.58 s). On 4090 with the Python 3.12 backend environment: backend 19 passed
(26.82 s including kernel rebuild), world model 26 passed (22.66 s). Checks
cover independent knife trajectories, material/reset isolation, per-world
force termination, replay boundaries and existing source learning formulas.


2026-11-02, `f656efc` / backend `d81579e`, A6000 GPU 1:
`train_gpu.py --backend-root ../CuttingGpuBackend --worlds 32 --max-iterations 42
--max-hours 1 --output runs/gpu_train_check`: 1344 physical transitions and one
complete learning update in 205.20 s. Physics 92.84 s, world model 57.86 s,
actor/critic 49.55 s; Torch peak allocation 3490.60 MiB. Source settings retained:
40 vector-step warmup, ten 2048-sample eight-step model batches, 4096-sample
32-step BPTT, five critic epochs. Model loss 19.7608, actor gradient norm
6.686e-4; finite gradients throughout. `last.pt` is 541 MiB; completed physical
training episodes at this endpoint: 0. Reference-task success counts above
are separate from the learned policy.


2026-11-02, world model `93f7efa`, backend `5279f77`: GPU recurrence/terminal
review fixes. `CUTTING_BACKEND_ROOT=../CuttingGpuBackend PYTHONPATH=. python -m
pytest tests -q`: A6000 backend 19 passed / 2.43 s and model 27 passed / 35.78 s;
4090 backend 19 passed / 2.39 s and model 27 passed / 81.42 s. Checks include
live stopped-world state, independent reset, controller-memory observability
and exact next-command reconstruction from a restored contact state.
State width 8944; original 1000-row replay costs 35,887,024 bytes per world
(1280 worlds: 42.781 GiB; 4096 worlds: 136.898 GiB).

W&B authenticated entity: `ju-dong6276-technical-university-of-munich`, project
`sliceit-supermimic`. Reference scaling upload `3e6fvj66` contains the completed
1/2/16/32/64/128-world results; 64/128 cuts passed in 105.59/147.72 s.
