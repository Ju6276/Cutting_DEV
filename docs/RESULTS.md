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


2026-11-02, `4723b34` / backend `5279f77`, A6000 GPU 1:
`benchmark_gpu_env.py --backend-root ../CuttingGpuBackend --worlds 1280
--wall-seconds 1800 --output runs/gpu_cut_final_1280`: 1280/1280 reference
cuts succeeded in 40 actions, zero force-limit failures, 288/304 released springs
per world; stepping 1027.92 s.

`train_gpu.py --worlds 1280 --max-hours 6 --resume
runs/gpu_capacity_1280/last.pt --backend-root ../CuttingGpuBackend
--output runs/gpu_supermimic_1280_20261102`: failed at cumulative vector step 84
(two cumulative learning updates). MuJoCo Warp convex collision allocation
failed for 59,310,080 bytes on the first physics step after a learning update;
Torch peak 47,811.64 MiB. The 42-step capacity test ended at the learning update
and omitted the return to physics. Capacity validation must cross that boundary.
Physical per-step mean reward ranged from -0.53849 to -0.51134; completed
episodes 0. Final learner/optimizer checkpoint saved (611 MiB).

User-selected restart: 1024 worlds, same learning settings, `--resume
runs/gpu_supermimic_1280_20261102/last.pt --max-hours 6`; output
`runs/gpu_supermimic_1024_20261102`, W&B `i02x34rz`. First two resumed physical
steps completed; sampled process memory 37,295 MiB.


2026-11-02, user-requested stop and weight removal: SAC, single-environment and
batched model training, simulators and launch monitors have exited on both hosts.
The 1024-world run stopped at cumulative iteration 115 with two cumulative
learning updates. Deleted experimental SAC/model checkpoints and W&B cache
copies: 17 files, 6,024,901,743 bytes by file size. Deleted three W&B model
artifacts (1,921,590,938 bytes); scalar histories, configurations and physical
scene assets remain. No training or evaluation is queued.


2026-11-02, parent `f8b372c` + dirty interface changes, task interface
`sliceit-177cf21`; corrected SliceIt `4c65cf7`. Earlier parity records use the
previous force-history decoding. Full suite on RTX 4090 with the Python 3.12
backend environment and `CUTTING_BACKEND_ROOT=../CuttingGpuBackend`:
`PYTHONPATH=. ../CuttingGpuBackend/.venv/bin/python -m pytest tests -q`:
29 passed in 52.03 s. Two independent reviewers verified reset sampling,
non-square history decoding, policy history, BPTT and GAE. Independent CPU
reward/gradient errors: 5.55e-17 / 3.47e-18; actor finite-difference error 1.14e-10.

`run_coupled_collection.py --sliceit-root ../SliceIt --mode collect --episodes 1
--wall-seconds 600 --output runs/interface_sync_ros_20261102`, followed by
`check_task_interface.py` and `check_state_interface.py` on `RUN/data`:
44 physical transitions, force-limit termination, 109.20 s including cleanup.
Reward parity error 1.11e-16, observation reconstruction error 0, terminal/success
mismatches 0. URDF/state reconstruction: position error 8.05e-16 m,
observation error 3.22e-14, reward error 1.42e-14. All ROS children exited.

A6000 GPU 1, `train_gpu.py --backend-root ../CuttingGpuBackend --worlds 2
--max-iterations 43 --max-hours 0.15 --output runs/interface_sync_20261102
--wandb-mode online`: 43 vector steps, 86 transitions, one full learning update
and a subsequent physics step; 173.88 s, Torch peak 3052.58 MiB, zero completed
physical episodes. WM loss 20.15477; actor gradient norm .0014653;
critic loss 85833.37. W&B run `1cxhqutb`. The bounded run exited normally.


2026-11-02, parent `9a0b677` + dirty native-state/lifecycle changes, RTX 4090:
`CUTTING_BACKEND_ROOT=../CuttingGpuBackend PYTHONPATH=.
../CuttingGpuBackend/.venv/bin/python -m pytest tests -q --disable-warnings`:
43 passed in 28.14 s. Production FDCC regression
`SliceIt/scripts/check_native_fdcc_state.py`: equal measured joints, internal
velocity difference 0.5 produces command difference 0.01 and next velocity
difference 0.45; equation error 8.67e-18, restored-state next-step error 0.

2026-11-02, parent `9a0b677` + dirty diagnostics, A6000 GPU 1,
`runs/critic_fresh_replay_20261102/run.py`: 30 source-configuration updates on
44 recorded transitions in 923.84 s. Final objective -411.259 = discounted
reward -189.055 + terminal bootstrap -222.205; terminal value mean -772.972,
maximum absolute terminal value 1007.20. W&B `mtsr5tll`. This dataset predates
native-state export. Historical positive value explosion remains unresolved.
Frozen-weight probe: 18.18% of 32-step predicted states violate the workspace;
mean learned termination probability on those states is 5.38e-5. No network,
objective, training threshold or reward change was made from this probe.

Counterfactual on 45 old physical trajectories changes only initial derivative
history: maximum velocity 28.1167 -> 0.72597; acceleration 562.333 -> 14.5194.
The old reset also advanced physics outside recorded control intervals; those
unrecorded transitions cannot be reconstructed from the saved episodes.


2026-11-02, native state candidate based on `9a0b677`, source controller
`6d6fea2`, utilities `c5f28a2`, Gazebo `e2ac2c9`, RTX 4090:
final read-only reset probe (three resets, immediate/0.25 s/1 s reads) shows
zero clock, solver/filter, queue and particle-state changes while paused.
Native source sensitivity and interval checks were reviewed independently.
Live ordering checks: first-step pending parameter updates 45/45 -> 0/45;
pre-step FK samples 5/5 -> 0/5. Final requested substeps all execute 2 ticks;
four consecutive action intervals are exactly 50 ticks. The 500 Hz joint
sample falls within each requested (start,end] interval.

2026-11-02, WM parent `9a0b677` + native-state/lifecycle changes,
SliceIt `3d0801d`, RTX 4090: final suite in the ROS-compatible environment
43 passed / 2 optional GPU checks skipped (1.94 s); complete suite using
`CUTTING_BACKEND_ROOT=../CuttingGpuBackend PYTHONPATH=.
../CuttingGpuBackend/.venv/bin/python -m pytest tests -q`: 45 passed (29.12 s).
The ROS environment has no Warp package; the GPU environment runs both backend
checks. Constructor-failure cleanup is included in these checks.

`runs/native_learning_capture.py` via the supervised training entry,
`--max-iterations 45 --max-hours 0.3 --wall-seconds 1200`:
45 physical steps, 2 source learning updates, return to physics/reset,
224.69 s learner / 272.56 s supervisor, exit 0, peak Torch allocation 2909.25 MiB.
World-model, policy and critic weights all changed and remained finite.
Final objective -3.867958 = discounted reward -3.867126 + bootstrap -0.0008327;
critic loss 20236.38 -> 73917.29. Physical episode: 44 steps, return -122.97679,
force-limit termination, 278 springs released; the next episode executed one step.
Online metrics: [W&B ugsu01gt](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/ugsu01gt).
Independent review verified replay fields, actual parameter updates and exited
process groups. Long training remains stopped for critic diagnosis.

`run_coupled_collection.py --mode collect --episodes 1 --wall-seconds 600`
in `runs/native_independent_final_20261102`: 45 transitions / 46 frames,
114.30 s, exit 0. `check_native_state.py`, `check_task_interface.py` and
`check_state_interface.py`: native FK error 8.88e-16 m, exact observation/queue
reconstruction error 0; state-derived observation error 2.80e-14, reward error
1.42e-14, termination/success/collision mismatches 0. Native solver, filter and
material snapshot ages are 0; source joint/wrench observation ages reach 2 ms. The 2 ms joint age occurs
only at reset; all 45 post-action joint samples match the native packet exactly.
44 normal actions advance 50 ticks; the terminal action stops at 34 ticks on
the source force limit.

Frozen final weights on the independent episode (`runs/native_terminal_probe.py`):
45 measured transitions have zero terminal-label mismatches. Teacher-forced
one-step prediction misses the single force-limit event. In 32-step policy
predictions, 33.75% of states are outside the workspace; their mean predicted
termination probability is 0.10044. Terminal value mean -10.8955,
maximum absolute value 11.5832, terminal bootstrap -0.0034576. Both source updates
change 14 world-model, 20 policy and 6 critic tensors; all weights and gradients
are finite. Next experiment compares growing physical replay against fixed-replay
updates while recording the first value/target departure and rollout state.
