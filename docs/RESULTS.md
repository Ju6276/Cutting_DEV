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

2026-11-02, WM `23db4c2`, run 2 `native_critic_long_20261102` on RTX 4090:
first positive-bound event at update 16, objective 1.60520059e8 = discounted
reward -284.757 + terminal bootstrap 1.60520349e8; maximum terminal value
3.56389806e10. Retained before/after learner, replay, optimizer and RNG snapshots.
Run stopped at a physical-step boundary after 176 steps / 54 updates (3401.51 s),
with final checkpoint and automatic cleanup. Three completed physical episodes:
returns -122.97779, -122.87392, 74.91947; one success, two force-limit terminations.
Frozen replay audit: 80 valid input states, all finite; largest normalized
feature 2.00997, maximum acceleration 1.88324, both reset derivative histories 0.

Run 2 W&B repair: 176 physical `reward` rows and 3 `episode_return` rows
backfilled after the training logger exited; server/local maximum error 0.
All 54 world-model and 54 policy metric rows remain present. Run 1
`064z1a85` logs `episode_return`; its first 27 server/local values match exactly.

Frozen phase substitution on the same worst recorded update-16 start/history/goal
(`runs/critic16_causal_20261102/phase_probe.json`, CPU, 1.75 s): pre-fit world
model gives final-step maximum normalized state 2.29, terminal value -408,
objective -174.91. Changing only the world model to post-fit weights gives
4.23476e11, 3.56390e10 and 1.39892e10 respectively. Post-fit zero-action rollout
still reaches 4.2757e11; pre-fit zero-action rollout remains at 2.29. The actor
change affects the magnitude by under 1%; subsequent critic fitting amplifies
the terminal value to 9.63e10. The initiating departure is world-model update 16.

2026-11-02, captured run 2 `23db4c2`, exact optimizer replay
`runs/critic16_samebatch_trace_20261102/trace.json`: 66.93 s on RTX 4090,
final model maximum weight difference 0 against the original post-update-16
snapshot. The tenth Adam step, evaluated on the identical sampled 2048 windows
including multiplicities and terminal BCE, changes loss 8.97483247 ->
11097.31519559. Along the same parameter displacement, quarter/half steps yield
8.45624199 / 9.31428881. Clipped gradient dot displacement -0.03820362.
The error is finite-step overshoot of the actual objective; training settings
and optimizer implementation remain unchanged. Independent review confirmed the
same-batch evaluation and exact replay.

Run 1 reward logging: attach a producer to existing W&B service/core and tail
`metrics.jsonl`, with no learner restart or parameter change. Server checks:
1751 `reward/step` rows, 63 `reward/episode` rows, maximum local/server error 0;
all 77 pre-attachment history rows preserved and primary logging continues.
`reward/episode_mean_20` reports the last 20 completed episodes. Original task
terminal reward is +100 for success / -100 for force-limit failure; at 63
completed episodes success return mean 85.047, failure mean -113.851.

2026-11-02, WM `4809e8b`, frozen state reassessment on RTX 4090/CPU:
`PYTHONPATH=. .venv/bin/python runs/state8990_reassessment.py`;
report `runs/state8990_reassessment_20261102/state_reassessment.json`.
80 replay rows supply 64 overlapping eight-step windows. Pre-update state loss
8.63448: spring stiffness 50.71%, coupling pose 36.61%, filter history 5.66%,
particle position/velocity 4.95%. Two reset-start windows contribute 99.29% of
the coupling-pose loss. Particle coordinates occupy 79.22% of the state.

Independent source/record audit, four episodes, report
`runs/state_reassessment_20261102/coupling_cache_reset_correction.json`:
nonreset coupling poses match explicit `b_bot_knife_sim` FK within 8e-16.
The second and third long-run reset caches equal the preceding final cache
bitwise, with reset-FK position differences 0.0622816 / 0.0628629 m. Reset uses
a separate helper that leaves arm memory unchanged; the next arm query applies
its existing velocity guard. First micro-query motion is absent from these
records. Retain reset cache memory; unconditional FK replacement fails here.

2026-11-02, `aeacdd7`, RTX 4090 control-history candidate:
`PYTHONPATH=. .venv/bin/python scripts/check_control_model.py --train
runs/native_critic_long_20261102/data/episode_00000.npz
runs/native_critic_long_20261102/data/episode_00001.npz
runs/native_critic_long_20261102/data/episode_00002.npz --held-out
runs/native_independent_final_20261102/data/episode_000.npz --output
runs/control_history_fit_20261102 --updates 50 --max-seconds 600`.
36 predicted coordinates, 860 state/history inputs, 10,262,373 parameters.
137 training transitions / 45 held-out; physical-task equation check across all
182 transitions: observation max error 3.11e-14, reward 1.60e-14, terminal mismatches 0.
47 tests passed, 2 optional GPU integration tests skipped.

Budget ended at 601 s: 32 complete update groups / 328 Adam steps, checkpoint saved.
Held-out 8-step normalized RMSE 0.06461 -> 0.03493; FK translation RMSE 2.92 mm,
peak-force RMSE 3.95 N; sole force-limit event missed. At 32 steps: normalized
RMSE 14.68, FK translation 225.9 mm, peak-force RMSE 1854.9 N, 13 collision false
positives. Training episodes also fail at 32 steps (RMSE 5.40 / 26.65 / 24.04).
The 8-step fitting result does not transfer to the 32-step prediction horizon.
[W&B curves and final measurements](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/84na9ay5).

Independent frozen-checkpoint BPTT, `runs/control_history_fit_20261102/check_trained_bptt.py`:
8/32-step survival-weighted discounted reward, float64, three central-difference
steps (1e-3 / 1e-4 / 1e-5), action bounds and branch fingerprints unchanged.
Directional gradient relative errors 1.75e-8 to 8.46e-7. Gradient calculation
passes; 32-step dynamics prediction fails. No physical policy training started.

Run 1 SAC stopped at 2026-11-02 19:53:24 CET: 3257 steps, 3124 updates,
133 episodes. A completed two-tick step was followed by a 1 s wall-time timeout
waiting for joint-state stamp > 747.073. Final checkpoint and replay saved;
supervisor cleanup complete. This precedes the compact fit start by 34 min 33 s.

Frozen v1 derivative-input intervention, same checkpoint and 14 held-out 32-step
windows (`runs/control_history_fit_20261102/check_derivative_feedback.py`):
replace only 180 derivative-history dynamics-input coordinates with aligned
recorded values; all remaining inputs and propagated physical outputs unchanged.
State RMSE 14.6728 -> 0.08983, FK 225.79 -> 11.55 mm, peak-force RMSE
1854.35 -> 8.924 N. This isolates a derivative-feedback amplification route.
Recorded future measurements enter this intervention; the deployable v2 instead
excludes derived observations from dynamics input while retaining analytic heads.

2026-11-02, `ed91239`, same episode split, source hyperparameters and 600 s budget:
repeat `scripts/check_control_model.py` above with output
`runs/control_history_v2_fit_20261102`. Dynamics input now selects 480 physical/
command-history coordinates; rollout state remains 860, outputs 36, parameters
9,484,133. 50 tests passed, 2 optional GPU integration tests skipped.
32 complete groups / 324 Adam steps in 600.603 s; checkpoint saved, no policy updates.

Held-out 1/8/32-step normalized RMSE 0.03059 / 0.04076 / 0.12302.
At 32 steps, v1 -> v2: FK translation 225.90 -> 15.69 mm, peak-force RMSE
1854.89 -> 8.968 N, return RMSE 1692.30 -> 25.705, collision false positives
13 -> 0. The sole collision remains missed. Training-episode 32-step RMSE is
0.0780 / 0.0612 / 0.0840. At matched update 20, held-out RMSE is 69.012 ->
0.13744. Explosive feedback is removed on this dataset. The initial constant-
velocity/force-persistence prior has 32-step FK 9.91 mm and peak-force 7.83 N;
v2 task-relevant errors still exceed those prior values despite lower composite
state error (prior 0.21873). Held-out 8-step state RMSE worsens from v1 0.03493
to v2 0.04076. More whole-episode data and physical evaluation remain next.

Independent final-model BPTT, `runs/control_history_v2_fit_20261102/check_trained_bptt.py`:
central FD at 1e-3 / 1e-4 / 1e-5 gives H8 relative errors
6.64e-5 / 6.60e-7 / 6.69e-9 and H32 3.14e-7 / 3.09e-9 / 2.92e-9;
all branch fingerprints unchanged. H32 action-gradient norm 4465.86 -> 7.70.
[W&B v2](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/8j3k5uuo)
contains all 32 updates and final measurements; server status finished.

2026-11-02, SAC run `sac_long_20261102`: deleted checkpoint data/index and
checkpoint pointer from `quarantined_learner` (3 files, 3,728,288 bytes).
`checkpoint_status.json` records `weights_deleted`; logs and replay remain.
The failing `env.step` raises before replay insertion (tf2rl `cb_trainer.py`
lines 211/228); the final checkpoint is saved by the worker's exception cleanup.
The timeout itself does not establish corruption of all earlier transitions.

2026-11-02, `9d48c7b`, SliceIt `3bc5024`, utilities `7ca3977`, Gazebo `c93e50a`:
`run_coupled_collection.py --sliceit-root /dataHDD/dong/Projects/SliceIt
--output runs/poststep_joint_collection_20261102 --mode collect --episodes 3
--wall-seconds 900`: exit 0 in 262.41 s including startup/cleanup. Initial
SuperMimic policy plus source noise produced 45/44/43-step episodes, all ending
at the source force limit; 132 transitions, 135 recorded frames.
`check_state_interface.py`, `check_task_interface.py`, `check_native_state.py`:
FK position error <=6.39e-16 m, observation error <=2.56e-14, reward error
<=1.43e-14, termination/collision mismatches 0, wrench-queue read error 0.
Post-integration joint positions differ from the preceding native hardware read
by up to 5.42e-4 rad; METHOD records their one-tick sampling-phase difference.
Full tests on 4090: 50 passed, 2 GPU integration checks skipped (2.64 s).

2026-11-02, `9e763c6`, SliceIt `1420ade`, utilities `97ba329`, Gazebo `ef2face`:
`python runs/sac_paused_snapshot_20261102/run.py` in SliceIt, 600 s budget,
original SAC entry/configuration: 31 transitions, 30 actor/Q1/Q2/alpha updates,
return 84.00210, task success=true, collision=false. First and subsequent reset
pose/history errors 0; both snapshot intervals have start=end, with preceding
topic caches 2/1 ms old. The earlier `3bc5024` reset check differed by 0.519 mm.
Force observation/accumulator errors 0, TD arrays (1024,), weights/TD finite.
DiSECt advanced 1.552 s in 776 calls, released 288/304 springs; next reset 0/304.
Exit 0 in 112.75 s, no remaining processes or checkpoint files. Actions remain
in the original random warmup (1000); its source trainer updates during warmup.
Shared source tests: SliceIt 59 passed; CWM 50 passed, 2 skipped (2.67 s).

2026-11-03, source `7359378`, SliceIt `11fc054`, utilities `97ba329`:
run 1 starts fresh with `python runs/sac_long_20261103/supervisor.py --output
runs/sac_long_20261103` in SliceIt: source endpoint 100100 steps / existing
259200-second execution budget. Startup: 62 physical steps, 60 optimizer updates.
[W&B xqmkhjbc](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/xqmkhjbc)
receives physical step/episode rewards and success rate.

Same revisions, run 2: `python runs/control_history_online_20261103_bootstrap/supervisor.py
--sliceit-root /dataHDD/dong/Projects/SliceIt --output runs/control_history_online_20261103
--mode train --model-state control-history --max-hours 6 --wall-seconds 21900
--microbatch 64`. Fresh connected learning, no resumed checkpoint; source
320000-iteration endpoint, 40-step warmup, UTD 0.4, batches 2048/4096 and horizons
8/32 unchanged. Six hours is the execution budget, with 300 s for startup/cleanup.
At iteration 49, three complete world-model/BPTT/critic update groups have run.
First group at iteration 42: world-model loss 0.707695, policy gradient norm
8.20e-5, critic loss 27820.3, finite updates. First physical episode: 44 steps,
return -122.90303, source force-limit termination, 278 released springs.
[W&B 9lptr72s](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/9lptr72s)
server history contains real rewards, episode return and model/policy/critic
updates. JSON-to-W&B run wrappers change logging only. Both runs were interrupted
for the shutdown repair below; their training checkpoints were retained.

2026-11-03, `702b90c`, SliceIt `5b7d208`, utilities `ddc85ec`:
`PYTHONPATH=. .venv/bin/python -m pytest tests -q`: CWM 74 passed, 2 skipped;
SliceIt 68 passed. Two source SAC updates followed by SIGINT saved checkpoint
and two replay rows, exited 0, and finished W&B in 55.151 s. Killing the owned
Gazebo server after two updates saved both artifacts, retained the injected
error, bounded unpause at 5 s, attempted close, and reported controller shutdown
failure separately; worker/supervisor exited 1 in 50.055 s. W&B server states:
[normal finished](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/m4tctkup),
[disconnect failed](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/spvsmw6r).
Both cases left zero processes. Six injected failures through the tracked coupled
entry preserved checkpoint and final failed status and attempted W&B finish;
longest case 5.002 s. Four diagnostic runs' checkpoint/replay files were removed
(14,916,211 bytes); logs and reports remain in `runs/shutdown_fix_20261103`.

Same source revisions: run 1 resumes through `scripts/run_sac_training.py
--sliceit-root /dataHDD/dong/Projects/SliceIt --output
/dataHDD/dong/Projects/SliceIt/runs/sac_resume_20261103`, with the saved TensorFlow
checkpoint and replay. Original 100100-step / 72-hour endpoint retained;
W&B server at 66 new physical steps / 64 updates: first complete episode
return 84.18421 and success=true.
[W&B xofe4ps8](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/xofe4ps8).

Run 2 resumes with `python scripts/run_coupled_collection.py --sliceit-root
/dataHDD/dong/Projects/SliceIt --output runs/control_history_resume_20261103
--mode train --model-state control-history --resume
runs/control_history_online_20261103/data/last.pt --max-hours 6
--wall-seconds 21900 --microbatch 64`. Model/optimizer/schedule/RNG restored at
iteration 133 / update 37; source replay starts empty and collects 40 new steps
before updates. Both runs use `scripts/guard_training_exit.py` for owned-process
cleanup after supervisor exit.
At iteration 175 the first resumed world-model/BPTT/critic group completed,
update 38, in 236.93 s from worker loop start: model loss 0.0112034, policy
gradient norm 20.9793, critic loss 759.112, finite values. W&B servers report both
runs running; run 2 has reached iteration 177 and receives physical reward,
model and policy/critic metrics.
[W&B rzv5jrqm](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/rzv5jrqm).

2026-11-03, `4e13589`: `PYTHONPATH=. .venv/bin/python -m pytest tests -q`:
90 passed, 2 skipped on both RTX 4090 (3.12 s) and A6000 (3.07 s).
`check_value_learning.py --checkpoint runs/control_history_resume_20261103/data/last.pt
--episodes runs/control_history_resume_20261103/data --output
runs/critic316_mc_probe_20261103/report.json`: 18 natural episodes / 735 transitions;
observed discounted return mean -70.9657, critic mean -69359.2151, RMSE 70107.0458.
The critic checkpoint remains stopped at update 316.

2026-11-03, `4e13589`, A6000 GPU 1: `train_gpu.py --backend-root
../CuttingGpuBackend --worlds 1024 --max-iterations 176 --max-hours 2
--output runs/gpu_source_budget_check_20261103`: started fresh control_history_v2,
source-equivalent 4096-transition schedule and 4096000-row replay. Endpoint is
176 vector steps; timeout stops at a learner callback and saves last.pt.
W&B `1t0hkx98`; first physical vector step completed in 20.23 s, GPU process
allocation about 15 GiB. First learning update is due at vector step 168.

2026-11-03, `4e13589`: `run_sac_training.py --sliceit-root ../SliceIt --output
../SliceIt/runs/sac_frozen_eval_20261103_full --evaluate --ros-port 11323
--gazebo-port 11357 --wall-seconds 7200`: frozen copy of SAC session-step-20000
checkpoint ckpt-5, original ten-episode deterministic evaluation; training remains
on ROS 11321. Evaluation rejects optimizer calls and checks unchanged actor weights.

The initial 1800-second-budget evaluation was stopped before completion and
replaced by the same frozen policy with a 7200-second budget because another
user’s GPU job reduced available throughput. No completed evaluation result
was produced by that initial run.


## 2026-11-03 — Source schedule and critic update equivalence

Commits `1a2a88c`, `05d145a`, `1f71016`. Command on the 4090:
`PYTHONPATH=. .venv/bin/python -m pytest tests -q`: **94 passed, 2 skipped** (3.18 s).
The two skips require the separate GPU backend integration setup.
Source schedule comparison through 320000 vector steps gives 127984 updates;
1/1024/4096 worlds share first updates 42, 44, 47, 49 and 1000 replay rows/world.
A batch of 8 versus microbatches 3/3/2 with four critic minibatches gives maximum
critic parameter difference **4.604533e-4 before**, **7.45058e-9 after** the sample-order
repair; actor difference is 2.32831e-10. This comparison checks update equivalence.

## 2026-11-03 — Frozen terminal prediction audit

Base commit `0663cc5`, learner repair `05d145a`. CPU command on the 4090:
`PYTHONPATH=. .venv/bin/python runs/terminal_frozen_audit_20261103/probe.py`.
The 18 saved episodes contain 735 transitions: task-equation terminal mismatches
**0**, maximum reward difference **1.42e-14**. Checkpoint update 316 gives mean
terminal probability **0.97769** on recorded terminal states, **0.30872** on one-step
predicted states and **0.78210** on eight-step predicted states. At probability 0.5,
missed recorded terminal labels are **0/18**, **14/18** and **3/18**, respectively.
The frozen audit updates no network. Next: separate physical-state prediction
error from terminal-head error on the same rollouts, then test their effect on
critic targets before changing the learning method.

## 2026-11-03 — Frozen SAC task evaluation

Execution commit `4e13589`; command `scripts/run_sac_training.py --evaluate
--ros-port 11323 --gazebo-port 11357 --wall-seconds 7200` with the frozen session-step
20000 checkpoint in `SliceIt/runs/sac_frozen_eval_20261103_full/restore`.
Ten deterministic physical episodes: **10/10 successes**, mean return **91.02107**,
actor weights unchanged, optimizer updates **0**. Process exit **1**: the one-shot
evaluator returned into the source training loop; the update guard stopped it.
Commit `1f71016` dispatches evaluation directly and fixes the fallthrough;
four dispatch/lifecycle tests pass. The original process record retains its failure.

## 2026-11-03 — A6000 collection stopped

Execution commit `4e13589`, `scripts/train_gpu.py --worlds 1024
--max-iterations 176 --max-hours 2`, run `gpu_source_budget_check_20261103`.
Interrupted at **122 vector steps**, **124928 transitions**, **0 learner updates**,
**0 complete episodes**, **2579.09 s**. No GPU process remains from this run.


## 2026-11-03 — Terminal error components and failure precedence

Frozen diagnostic commit `29858b4`, checkpoint update 316; command:
`PYTHONPATH=. .venv/bin/python runs/terminal_components_20261103/probe.py`.
Of 18 recorded terminal steps, **16** exceed the force limit; one-step predicted
force exceeds it on **2** of those terminal steps. Replacing only the terminal
head's force features with recorded values reduces missed terminal labels
**14 → 3** and increases false positives **8 → 12** across 735 transitions.
On the same predicted states, the analytic rule marks **8** terminals; the learned
head misses **6** and adds **10** false positives. Both force prediction and head
consistency contribute to terminal error.

Commit `2493259` restores SliceIt's failure-before-success priority. With
`termination_on_negative_reward=True`, the old model added **100** success reward;
the repaired shared head matches the physical branch. The failed training run has
this flag **False**. Command `PYTHONPATH=. .venv/bin/python -m pytest tests -q` on
4090: **98 passed, 2 skipped**, 3.16 s. Independent replay of 735 transitions gives
motion error **6.94e-18**, observation error **3.33e-14**, reward error **1.42e-14**,
terminal mismatches **0**, and exact control history.


## 2026-11-03 — World-model-first startup and execution

Commit `9188c59`, 4090 command `PYTHONPATH=. .venv/bin/python -m pytest tests -q`:
**125 passed, 2 skipped**, 5.13 s. Stage tests verify unchanged actor/critic weights
and empty optimizer states during model preparation, the joint-stage boundary,
partial updates and checkpoint restoration. Main/run cleanup tests preserve the
original failure while recording save/close failures.

A6000 command `CUTTING_BACKEND_ROOT=/dataHDD/dong/Projects/CuttingGpuBackend
PYTHONPATH=. .venv/bin/python -m pytest tests/test_gpu_env.py -q`, commit `33f3e22`:
**2 passed**, 34.69 s. Physical environment files are unchanged in `9188c59`.

A6000 execution at `9188c59`:
`PYTHONPATH=. .venv/bin/python runs/wm_stage_execution_20261103/probe.py`.
735 recorded transitions; source configuration (10 batches × 2048 windows × 8
steps), microbatch 64, CUDA compilation. **1 complete WM round / 10 WM Adam steps**,
**0 policy / 0 critic steps**; actor/critic weights exactly unchanged; checkpoint
restore passed. Elapsed **40.28 s**, peak PyTorch memory **451.10 MiB**. This is the
source-size update and phase execution check.

## 2026-11-03 — A6000 world-model-first run

Execution commit `9188c59`, backend `5279f77`; command:
`scripts/train_gpu.py --backend-root ../CuttingGpuBackend --worlds 1024
--wm-only-updates 500 --max-hours 24 --output runs/gpu_wm_first_20261103`.
Fresh weights, A6000 GPU 1; W&B `dm6e5jan`. Endpoint is the 24-hour wall budget or
320000 vector steps. Internal stopping saves the checkpoint and finishes W&B;
external timeout sends SIGINT at 24 h 15 min and SIGKILL after another 300 s.
WM-only finishes at vector step 1289; first joint update is at 1292 for an
uninterrupted run. Estimated collection to that boundary is about 7.5 h from
measured physical throughput, plus model fitting. The run records separate WM,
policy and optimizer counts and saves both phase-boundary checkpoints.

## 2026-11-04 — GPU checkpoint and frozen evaluation entry

Commit `f77b548`; 4090 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`:
**125 passed, 2 skipped**, 5.09 s. Normal SIGTERM saved the previous run at
**1806 vector steps, 706 WM rounds, 206 policy updates**. A6000 resume command:
`scripts/train_gpu.py --backend-root ../CuttingGpuBackend --worlds 1024
--wm-only-updates 500 --resume runs/gpu_wm_first_20261103/last.pt
--checkpoint-minutes 30 --max-hours 11.340668234560225
--output runs/gpu_wm_resume_20261104`. W&B `t21ma46i`; original wall deadline
retained. Model/optimizer counts restored; replay and physical episodes restart.
`kill -USR1 1249025` wrote **checkpoint_00001807.pt (182 MiB)** while training
continued. Frozen physical evaluations use `scripts/evaluate_gpu.py`, one fixed
scene episode per checkpoint with zero execution noise; outputs are
`runs/gpu_frozen_eval_20261104_{initial,policy206}`.

## 2026-11-04 — Frozen policy result and critic drift

Commit `f77b548`, `scripts/evaluate_gpu.py --backend-root ../CuttingGpuBackend
--checkpoint runs/gpu_wm_first_20261103/last.pt
--output runs/gpu_frozen_eval_20261104_policy206`: one deterministic physical
episode, **success, 144 steps / 7.2 simulated seconds, return 44.7112**, peak
force 11.9093 N, 282 released springs, no force-limit termination. Weights
unchanged, zero optimizer updates; wall 534.24 s while another evaluation ran.

Training `9188c59` update log: critic means at policy updates 137/180/200/206
are **-106.22/-383.86/-976.64/-1402.69**; update 206 critic loss **119971.91**.
Its imagined discounted reward is **-6.307**, terminal bootstrap **-978.421**.
Three-number trend shows renewed critic drift; task completion and value learning
are evaluated separately. Resumed training was stopped before additional
optimizer updates and saved at vector step **1836**, policy update **206**.
Next check: compare frozen critic values and bootstrap targets against physical
discounted returns and separate terminal-prediction error from value-target feedback.
