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
The recorded trend shows renewed critic drift; task completion and value learning
are evaluated separately. Resumed training was stopped before additional
optimizer updates and saved at vector step **1836**, policy update **206**.
Next check: compare frozen critic values and bootstrap targets against physical
discounted returns and separate terminal-prediction error from value-target feedback.

Frozen initial-policy control (`scripts/evaluate_gpu.py`, checkpoint
`gpu_wm_first_20261103/wm_only_complete.pt`, output
`gpu_frozen_eval_20261104_initial`): **500 steps / 25 s, timeout, return -193.7770**,
no force-limit termination, 281 released springs. Zero execution noise and
zero optimizer updates, same physical task as policy 206.

Policy 206 video repeat at code `344e604`, output
`runs/policy206_robot_video_20261104`: **144 steps / 7.2 s, success, return 44.7387**,
283 released springs. Control-boundary material-force samples at 20 Hz:
peak **10.8548 N**, trapezoidal time mean **2.4119 N**. SAC frozen 20k video
`runs/sac_robot_video_20261104`: **19 steps / 0.95 s, success, return 91.0862**,
286 released springs, sampled force peak **31.9968 N**, time mean **7.7572 N**.
Both videos replay recorded robot/material states at 0.5x; robot FK position
error for policy 206 is **7.22e-16 m**. Robot backends differ (Gazebo vs MuJoCo Warp).

## 2026-11-04 — Known task terminals in imagination

Frozen value probes use `/dataHDD/dong/tmp/probe_cutting_values.py CHECKPOINT
EPISODE OUTPUT`, 48 evenly spaced physical starting states and 32 imagined steps.
Gazebo run `control_history_resume_20261103`, update 316, episode 17:
physical-state critic mean **-73684.30**, observed discounted-return mean
**-94.38**. Original imagined step 32 has mean reward **-77.69**, minimum
**-200.75**, even after predicted task failures. Absorbing the existing task
terminals removes repeated terminal penalties: step-32 mean **-0.1468**, minimum
**-0.6349**; terminated paths have zero future reward/bootstrap. Frozen weights
are unchanged, so their physical-state value error remains.

GPU update 206 on a new on-policy trace `policy206_value_trace_20261104`:
**138 steps, success, return 46.6444**, physical-state critic mean **-1276.00**
versus observed discounted-return mean **38.4822**. Initial critic estimate
**-434.80** versus actual discounted return **-6.5264**. This separates the
critic error from physical task success. Next validation is fresh value learning
with the corrected terminal rule before restarting long training.

4090 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`: **130 passed,
2 skipped**, 5.23 s. Force/workspace/success terminal examples retain the entering
reward once, make subsequent rewards and bootstrap zero, and preserve bootstrap
at a nonterminal rollout horizon. Old-terminal-rule training resumes are rejected.

## 2026-11-04 — Frozen-model critic target feedback

Code `7675283`; command `.venv/bin/python /dataHDD/dong/tmp/critic_fixed_test_20261104.py`
on RTX 4090; output `runs/critic_fixed_comparison_20261104/report.json`.
Frozen Gazebo WM/policy, episode 17, 48 fixed starts, horizon 32, seed 41;
each fresh critic runs 200 rounds × 5 epochs × 4 minibatches, source Adam LR
2e-4 and gradient norm limit 1. Total experiment time **28.30 s**.
Previous termination reaches target absolute maximum **146381.23**; corrected
termination still grows from **100.59 to 21810.18**. Holding the initial corrected
targets fixed gives final imagined value mean **-16.25** versus target **-17.87**.
Updating bootstrap targets reproduces instability with WM and policy frozen.
The fixed-target control isolates this feedback; it is not installed in training.
Next: audit horizon-state coverage and bootstrap evaluation against the source
implementation before selecting a value-learning correction. These fixed-start
critic-only rounds do not run online collection or policy updates.

Follow-up commands on the same frozen data/code:
`python /dataHDD/dong/tmp/critic_normalized_probe_20261104.py` and
`python /dataHDD/dong/tmp/critic_shac_target_probe_20261104.py`.
NDM `0171f5e` input normalization alone reaches target absolute maximum
**1212410.50**; SHAC's target update (0.4 old + 0.6 online, once per round)
alone reaches **18874.46**, both at round 200. Neither candidate is merged.
`python /dataHDD/dong/tmp/critic_bootstrap_trace_20261104.py` traces the corrected
baseline: 12/48 imagined paths remain nonterminal at H=32; all imagined rewards
are in **[-100.5893, 0]**, but endpoint value reaches **22031.08**.
Terminal target minus entering reward is exactly zero. Recomputing GAE with
zero value inputs gives target absolute maximum **100.5893**, versus **21810.18**
with learned values. This isolates the large targets to value bootstrap on
nonterminal paths. The source GAE recurrence matches the local implementation;
finite-window endpoint coverage and repeated fixed-data fitting remain to audit.

## 2026-11-04 — Critic data coverage and physical-transition comparison

Code `7675283`; frozen WM/policy, fresh critic, original GAE/Adam/batch/epoch
settings. `/dataHDD/dong/tmp/critic_resampled_probe.py --rounds 1000 --seconds 180
--seed SEED` samples 4096 starts each round from Gazebo episodes 0–15; episodes
16–17 are held out. Seeds 41/42/43 finish with target absolute maxima
**1059.05 / 647.12 / 200.76** and held-out value MAE
**97.44 / 131.04 / 75.60**. Replacing predicted future states/rewards with
recorded transitions (`--physical`, seed 41) gives maximum **102.84**, MAE
**15.42** after 1000 rounds. The observed future returns include the collection
policy's execution noise. These runs update only the critic on fixed recordings.
Outputs: `runs/critic_paired_20261104_seed*`.

A6000 `/dataHDD/dong/tmp/collect_critic_gpu_validation.py` collects eight new
physical episodes with frozen GPU policy 206 and source action noise 0.1:
**8/8 successes, 128–234 steps, return 16.09–49.74**, 282–285 released springs.
All task-terminal labels match recomputation; observation error is at most
**3.00e-14**, reward error **3.52e-6** from float32 reward storage.
Output: `runs/critic_gpu_validation_20261104`.

The frozen GPU WM evaluated on every valid recorded-action window has
position RMSE **0.42 / 3.20 / 19.64 mm** at **1 / 8 / 32 steps**, force RMSE
**0.81 / 1.43 / 2.06 N**, and return RMSE **0.03 / 7.08 / 10.48**.
On episodes 0/2/4/5 for critic fitting and 6/7 held out, the same 1000-round
comparison gives held-out value MAE **46.49** with model futures versus **5.17**
with physical futures; target absolute maxima are **99.72** versus **107.42**.
Commands: `/dataHDD/dong/tmp/wm_checkpoint_eval.py` and
`/dataHDD/dong/tmp/critic_gpu_resampled_probe.py --rounds 1000 --seconds 300
--seed 41 [--physical]`. Next: improve the world model with the unchanged
supervised loss, then repeat the paired value-learning comparison.

## 2026-11-04 — Fixed-data WM fitting and repeated value learning

Code `7675283`; A6000 GPU 0 runs
`/dataHDD/dong/tmp/fit_wm_critic_repair.py --updates 500 --seconds 3600 --microbatch 2048`
and `fit_wm_gpu_critic_repair.py` with the same arguments. Each continues the
existing WM/Adam for 500 rounds, with original eight-step loss, effective batch
2048, ten batches per round, and frozen policy. Gazebo uses recorded episodes
0–15 / held-out 16–17; GPU uses new recorded episodes 0–5 / held-out 6–7.
No physical data are collected during these fits. Times: **519 / 610 s**.

Gazebo held-out position RMSE changes **0.72→0.42 mm** at one step and
**1.78→1.44 mm** at eight steps; eight-step force RMSE **6.91→8.10 N**.
Both held-out terminal force peaks remain missed. Subsequent seed-41 critic
fitting for 1000 rounds has target absolute maximum **200.72**, value MAE
**62.96**; the maximum is a single combined force/workspace penalty.
GPU held-out position RMSE changes **3.22→0.57 mm** at eight steps and
**19.37→9.30 mm** at 32 steps; force RMSE changes **1.44→0.27 N** and
**2.06→0.32 N**.

`critic_gpu_after_wm_probe.py --rounds 1000 --seconds 300 --seed SEED`, seeds
41/42/43, finishes with target absolute maxima **15001.14 / 135.82 / 99.72**
and held-out MAE **2800.71 / 54.97 / 54.36**. Recorded-physical-future
comparisons (`critic_gpu_resampled_probe.py ... --physical`) give maxima
**107.42 / 104.53 / 102.61**, MAE **5.17 / 6.75 / 10.70**.
The tests identify unstable bootstrap on model-generated states; improved
average state-prediction error alone does not resolve it.

NDM `0171f5e` RunningNorm on the complete model-state sample, seed 41, still
finishes at target maximum **4811.88**, MAE **1219.57**. A physical-time
comparison uses 13 cutting steps (**0.65 s**) versus Cuboi's 32×0.02 s
(**0.64 s**): target maximum **99.72**, MAE **81.98**. Both alternatives fail
value accuracy; training retains 32 steps and its existing normalization.
Outputs: `runs/wm*_critic_repair_20261104`, `critic_gpu_after_wm_*`,
`critic_gpu_ndm_norm_*`, `critic_gpu_source_time_*`.

Evaluation fix `235e118`: `check_control_model.py` now counts the entering
task-terminal reward once when computing predicted returns; raw state-error
rollouts continue for the requested horizon. Corrected GPU held-out 32-step
return RMSE is **10.79 before / 59.56 after** fitting, from
`task_return_evaluation.json`. The old evaluator repeated terminal bonuses.
4090 full tests: **131 passed, 2 skipped**, 5.14 s. Fixed-recording experiments
leave policy frozen; the online joint-update result follows below.

## 2026-11-04 — Online joint updates with absorbing task termination

Code `5458efb`; A6000 GPU 0 runs
`/dataHDD/dong/tmp/run_critic_online_validation.py`, resuming the verified
500-round WM-only checkpoint described in METHOD. Original H32, 1024 worlds,
effective policy batch 4096, WM batch 2048, and update schedule are retained.
Budget: 30 policy updates or 3400 s computation, with external termination at
3540 s and kill at 3600 s. The run completes normally in **3128.32 s**:
**114 vector steps, 116736 new transitions, 30 joint updates**; optimizer steps
finish at WM **5300**, policy **30**, critic **600**.

All recorded values are finite. Physical pre-action critic values across
policy versions 0–29 reach absolute maximum **33.73**; the last mean is
**−8.01**, absolute maximum **10.66**. Training-target means range
**−34.33 to −4.61**, ending at **−13.47**; final-minibatch critic losses range
**0.97–28.43**. This online run has no sustained value growth. The physical
rollouts cover **5.7 s** each, with **0 completed or truncated episodes**;
complete-episode Monte Carlo value errors remain to be measured.

After update 30, sampled recorded-action replay rollouts at H8/H32 have
position RMSE **0.273/2.089 mm**, force RMSE **0.379/0.454 N**, and undiscounted
return RMSE **0.0693/0.4509**. Each horizon uses 64 training-replay windows
with task-absorbing termination. The last physical step, using WM529, has
one-step position/force RMSE **0.0557 mm/0.226 N**. Recorded timing includes
the audit: physics **2422.37 s**, WM fitting **260.44 s**, policy/critic plus
rollout diagnostics **413.76 s**.

Output: `runs/critic_online_terminalfix_20261104`; checkpoints at joint updates
1/10/20/30 and `last.pt`. W&B run `t54srl5x` finishes and syncs successfully.
4090 full tests: **131 passed, 2 skipped**, 5.16 s. The frozen-checkpoint
physical value evaluation follows below.

## 2026-11-04 — Frozen policy30 complete physical episodes

Code `17225b9`; A6000 GPU 0 runs
`/dataHDD/dong/tmp/evaluate_critic_complete_gpu.py` on the final online
`last.pt`, eight worlds, zero action noise, 900 s wall ceiling. All eight
same-initial-state episodes succeed in **152–153 steps / 7.60–7.65 s**;
wall time **277.09 s**, return **34.623–34.987**, peak force **9.403 N**,
released springs **279–281**. Collisions/truncations: **0/0**. Policy and
critic tensors remain exactly equal to the checkpoint; optimizer updates **0**.

Across **1221 transitions**, critic versus complete physical discounted
returns at gamma **0.99** has **MAE 41.5014, RMSE 47.8341, bias −41.5014**.
Initial value is **−34.0605**, actual initial return **−14.8874 to −14.5814**.
The final transition reward is about **99.687**, while its pre-action critic
is about **13.6**. First-114-step MAE is **29.5508**; later-step MAE **76.7731**.
The preceding online run collected 114 steps per world and zero terminations.
Task execution succeeds; value estimates still underestimate complete returns.

Independent CPU replay exactly reproduces rewards, termination flags and
success counters; independent discounted-return recursion agrees within
**1e-12**, reconstructed-history critic values within **3.82e-6**. The earlier
eight-world policy206 recording used action noise **0.1**, WM706 and the prior
terminal rule; this evaluation uses zero noise, WM530 and policy30 trained
with absorbing task termination. Both use the same task/material and backend.

Output: `runs/critic_complete_policy30_20261104`; per-step force/value curves
are in `force_value_curves.csv`, and world0 physical video frames in
`video_world0.npz`. Stored data are about **1.9 MB**. The following run adds natural terminal
episodes and repeats the frozen complete-return evaluation.
4090 full tests: **131 passed, 2 skipped**, 5.11 s.


## 2026-11-04 — Natural-terminal training and frozen policy52

Code `4638442`; A6000 GPU 0 runs
`/dataHDD/dong/tmp/run_critic_terminal_coverage.py`, then
`/dataHDD/dong/tmp/evaluate_critic_terminal_coverage.py`. Training resumes
policy30/WM530 with 1024 worlds and the unchanged H8/H32 schedule. Checkpoints
restore learner state; physical state and replay restart, including the source
40-step refill. The endpoint is ten joint updates after the first natural
termination, capped at 250 vector steps or 4900 s; frozen evaluation is capped
at 900 s within a total 6000 s budget.

Training finishes in **2540.90 s**, after **94 new vector steps / 96256
transitions** and **22 joint updates**, at policy **52**, WM **552**. The first
natural termination occurs at step **70**, policy **42**. Completed episodes:
**2872 = 1751 successes + 1121 collisions**, zero truncations. All 1024 first
episodes succeed in 70–80 steps. Physical critic values remain finite, with
absolute maximum **82.7721**. W&B run: `4s8uxh4o`.

The WM optimizer samples **2706 terminal labels** after the first termination.
All come from successful episodes: the 1121 collisions last only one or two
steps (**273 / 848 episodes**) and cannot form the source eight-step,
same-episode training windows. Independent reconstruction of all 22 rounds'
window counts matches the sampler. The failures are recorded but excluded from
WM fitting by this sequence-length requirement.

Frozen policy52 completes **8/8** zero-noise episodes in **18 steps / 0.90 s**;
wall time **35.12 s**, return **92.1573–92.1575**, peak force **42.595 N**,
released springs **287**. Across 144 transitions, gamma 0.99 value errors are
**MAE 141.6256, RMSE 142.1439, bias −141.6256**. Initial value is **−43.9075**,
actual initial discounted return **76.9421–76.9423**. The same-condition
policy30 evaluation had MAE **41.5014**: task execution improves while value
accuracy worsens. Independent CPU recomputation matches rewards and task flags
exactly, returns within **1e-12**, and critic values within **1.15e-5**.

At the saved world0 initial state, the unchanged H32 imagination predicts an
out-of-workspace failure at step 18, reward **−100.7226**, survival-weighted discounted reward
**−95.7152**. The physical episode instead succeeds at step 18, terminal reward
**99.719**, discounted return **76.9421**. Post-terminal reward and bootstrap
are zero. This separates the repaired absorbing-terminal implementation from
the model's erroneous future terminal state and the critic's value error.

Outputs: `runs/critic_online_terminal_coverage_20261104` and
`runs/critic_complete_terminal_coverage_20261104`; complete force/value curves,
physical frames and `model_terminal_diagnosis.json` are saved. Next: admit
recorded short terminal sequences to model fitting with a valid-step mask,
verify full-length loss equivalence and terminal coverage, then repeat the
same frozen physical-return check.
4090 full tests: **131 passed, 2 skipped**, 5.16 s.

## 2026-11-04 — Unified repository

Code `cb17d5f`, retaining parents `57286e4` (learning) and `5279f77`
(physics). RTX 4090, `/dataHDD/dong/Projects/Cutting_DEV`:

- `CUTTING_BACKEND_ROOT="$PWD/backends/mujoco" PYTHONPATH=. .venv/bin/python
  -m pytest tests -q`: **133 passed**, zero skips, **121.26 s**.
- From `backends/mujoco`, `PYTHONPATH=. ../../.venv/bin/python -m pytest
  tests -q`: **19 passed**, zero skips, **2.70 s**.
- `python3 scripts/checkout_sliceit.py --output external/SliceIt-online`:
  all **8** repositories fetched from their public origins; all patched Git
  trees match `integrations/sliceit/patches/sources.json`.

`scripts/benchmark_gpu_env.py --worlds 1 --wall-seconds 600 --output
runs/integration_reference_20261104` uses the bundled backend by default.
The complete physical reference cut succeeds in **40 steps / 2.00 s simulated
time**, **302.33 s wall time**, return **80.2902**, peak force **9.9761 N**,
**288** released springs, zero collision or truncation. SAC continues on the
same GPU. The run manifest records the pre-commit integration snapshot
(`57286e4`, dirty); its runtime source is committed in `cb17d5f`.


## 2026-11-04 — Short terminal sequences enter actual WM training

Code `2495171`; A6000 GPU 0 runs
`runs/critic_mask_validation_20261104/run.py`. Policy52/WM552 resumes with
1024 environments, fresh physical state/replay, the source 40-step refill,
H8/H32, effective batches 2048/4096 and original optimizer/task settings.
The endpoint is ten joint updates or 1800 s training, followed by eight frozen
physical episodes within a 2700 s total budget.

Training reaches policy **62**, WM **562** after **64 vector steps / 65536
new transitions**, in **1701.66 s**. Completed episodes: **3409 = 3072
successes + 337 collisions**. All collisions last one or two steps (**57 / 280**).
Ten WM rounds sample **204800 windows, 1320053 valid transitions, 88797 terminal
labels and 2433 collision labels**. In the first round, **269 collision labels**
come from 307 already-recorded one/two-step failures. These short failures now
enter model fitting. The **77903 short windows** also include terminal suffixes
of longer episodes. Actual short state/action/next-state records are saved.

Frozen policy62 completes **8/8** zero-noise cuts in **20 steps / 1.00 s**,
wall time **39.03 s**, mean return **90.814804**, peak force **30.658 N**,
released springs **287–288**. Policy/critic weights remain unchanged and optimizer
updates are zero. Across **160 transitions**, gamma 0.99 value errors are
**MAE 153.418925, RMSE 154.442620, bias −153.418925**. Initial value is
**−63.708687**, mean actual initial discounted return **74.074448**. The earlier
policy52 complete evaluation had MAE **141.625561**. Value accuracy has not
improved at this ten-update endpoint.

On the same saved policy52 episodes with policy52 fixed during imagination,
WM552 predicts a workspace failure at step **18**, survival-weighted discounted
reward **−95.715187**; WM562 predicts it at step **15**, reward **−94.888535**.
Teacher-forced analytic terminal mismatches on world0 decrease **1 to 0**.
With current policy62, WM562 also predicts a step15 workspace failure while the
physical episode succeeds at step20. Its imagined discounted reward is
**−94.691711**, versus physical **74.074467**.

Recorded-action replay isolates accumulated model error: using the exact
physical policy62 actions, WM562 still predicts failure at step15. At that step,
one-step end-effector position error is **[+0.304, −0.180, −0.255] mm**; continuous
prediction from the initial state reaches **[−23.640, −11.070, −25.971] mm**.
Predicted goal-x error **0.051697 m** exceeds the existing **0.05 m** workspace
limit; actual goal-x error is **0.028057 m**. Recorded-action joint/pose drift therefore crosses the workspace boundary and
produces an erroneous negative terminal reward; physical critic values remain
negatively biased. Absorbing termination zeros subsequent rewards/bootstrap.

Independent CPU reconstruction exactly matches rewards, task flags, success
counts and short-episode state chains. Complete-return recursion agrees within
**1e-12**, critic values within **1.53e-5**, and H32 survival-weighted objectives within **3.06e-5**.
Before this run, 4090 suites pass **140 root tests + 19 backend tests**, zero
skips; independent chronology reconstruction checks **150 ring-buffer snapshots**.

Total orchestration time **1759.85 s**; processes finish and GPU 0 is released.
W&B run `5h7xrk7c`; artifacts are under `runs/critic_mask_validation_20261104`:
`training/short_episodes`, `evaluation/force_value_curves.csv`, complete physical
frames, fixed-policy comparisons and `recorded_action_drift.json`. Next: audit
joint-state integration, normalization and history inputs against the source
using these fixed real-action rollouts; keep the critic objective and physical
task unchanged while resolving multi-step model error.

## 2026-11-05 — Frozen WM train/held-out fit

Code `b27a566`, A6000 GPU0, `runs/wm_frozen_fit_20261105/run.py`.
New data: 32 environments × 4 episodes; fixed groups yield 96 train episodes
(1981 transitions, 93 successes/3 workspace failures) and 32 held-out episodes
(648 transitions, all successful). Policy62, critic and both optimizers remain
frozen. WM562→662 completes 100 original H8 rounds in **1165.81 s**, within
the 2700 s total budget, retaining batch2048, ten Adam steps/round and source LR.

| Extra WM rounds | Held-out H1 position RMSE | H8 endpoint RMSE | Full-trajectory RMSE | Correct first terminal time/type |
|---:|---:|---:|---:|---:|
| 0 | 1.234 mm | 3.158 mm | 19.418 mm | 0/32 |
| 100 | 2.483 mm | 4.758 mm | 7.148 mm | 0/32 |

Final predicted goal-z error is **−16.39 to −6.52 mm**, actual **−0.392 to
−0.298 mm**, against the existing **2 mm** threshold. All 32 successes are
missed; predicted success count is 0, actual 11. Train/held-out physical-state
losses rise from round50's **0.015591/0.015923** to **0.023258/0.023553** while
terminal BCE decreases. Independent CPU reconstruction matches all 30 metric
groups and 640 first events; all frozen tensors match. Next: separate state-loss
and BCE gradients on the same saved batches. Artifacts:
`runs/wm_frozen_fit_20261105`; [W&B 7jnyx9n4](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/7jnyx9n4).

## 2026-11-05 — Shared physical evaluator and source SAC adapter

Implementation `47f4e60`; manifests record the pre-commit `b27a566` dirty
snapshot. `evaluate_benchmark.py --method METHOD --episodes 1 --wall-seconds
600 --output runs/fixed_benchmark_20261105/METHOD` checks identical robot/material
assets and task settings. SAC supplies its source Python/config/checkpoint;
SuperMimic loads policy62 from `critic_mask_validation_20261104/training/last.pt`.

| Method | Endpoint | Steps / simulated seconds | Return | Peak material force | Released springs |
|---|---|---:|---:|---:|---:|
| Policy62, A6000 | Success | 20 / 1.000 | 90.81489 | 30.65784 N | 288 |
| Gazebo SAC ckpt-10, 4090 | Force-limit failure | 1 / 0.042 | −100.60348 | 63.42530 N | 0 |
| Constant zero, A6000 | Wall budget; incomplete | 346 / 17.300 | −144.80551 (partial) | 5.83108 N | 282 |

All evaluations preserve weights and perform zero optimizer updates. SAC is a
Gazebo-trained checkpoint transferred to MuJoCo Warp; same-environment SAC
training is next. Wrench-x reaches **52.1676 N**, above its **50 N** limit.
Constant control ends **2.049 mm** above target with no physical terminal event.
An earlier shared-4090 constant run reaches 88 steps at its 600 s budget.
Policy62 uses **39.40 s** total wall time on A6000; hardware/concurrent workload
is recorded separately. Runtime packages match on both hosts.

The new source SAC bridge runs **7 physical steps/7 original SAC updates** in
a 60 s check, with warmup1000 unchanged. Reload restores all four optimizer
counters to 7; replay state/action alignment and frozen inference match.
The time-limit replay adapter preserves source-buffer done flags **[1,0,0]**
for natural termination, pure truncation and continuation through save/reload.
4090 checks: **155 root + 19 backend pass**; two legacy Gym cases run in the
Python3.8 bridge suite (**10 pass**, GPU-worker case covered in Python3.12).
Independent review covers task/reset identity, terminal frames, saved exception
prefixes and source replay ordering. Artifacts: `runs/fixed_benchmark_20261105`
and `runs/sac_gpu_bridge_check_20261105`.

## 2026-11-05/06 — Critic-target variants from policy62/WM562

Code: branch commits `7ac4ef0` (critic_target real, replay snapshots),
`0c091dd` (real_wm, actor_horizon), `ae056e8` (actor_bc_alpha). A6000 GPU 0,
1024 worlds, resume from `critic_mask_validation_20261104/training/last.pt`,
fresh critic pre-fitted 100 rounds on the refilled replay, source schedule,
then frozen `evaluate_benchmark.py --method supermimic --episodes 8` and
`runs/critic_real_c3_20261105/value_eval.py` (values vs complete physical
discounted returns, gamma 0.99; WM-imagined vs physical return from the same
starts; held-out = 32 episodes of `wm_frozen_fit_20261105`). The analysis script
reproduces the recorded policy62 numbers (benchmark MAE 153.419, held-out 152.707).

| Run (W&B) | Variant | Updates | Frozen 8 ep. | Steps | Return | Peak force | Value MAE / bias (benchmark) | Held-out MAE | max real target |
|---|---|---:|---|---:|---:|---:|---|---:|---:|
| policy62 | source | 62 | 8/8 | 20 | 90.81 | 30.66 N | 153.4 / −153.4 | 152.7 | – |
| C3 `m0rtgke0` | real, h32 | 150 | 8/8 | 33 | 84.46 | 15.42 N | 74.4 / +70.4, values 19.6–246 | 30.5 | 256.6 |
| V2 `tbsa78z9` | real_wm, h8 | 84 (stopped) | 0/8 collision | 1 | −100.60 | 59.43 N | 117.9 / +117.9 | 11.6 | 106.8 |
| V3 `dmtqrroa` | real_wm, h8, BC 2.5 | 72 (stopped) | 8/8 | 17 | 92.67 | 47.88 N | 10.7 / +6.1, values 60–123 | 4.0 | 102.8 |
| V4 `twr8mk8r` | real_wm, h32 | 115 (stopped) | 0/8 workspace | 26 | −112.48 | 19.11 N | 53.8 / +53.5, values −93…1 | 16.7 | 168.3 |
| V5 `dpfjiwgl` | real_wm, h8, BC 0.25 | 103 (1 h budget) | 8/8 | 18 | 92.18 | 39.35 N | 1.7 / −1.0, values 68–101 | 3.1 | 106.9 |

Measured:

- C3: bootstrap weight `survival_H` is 0 for updates 63–105, so the actor
  follows the source objective (imagined objective −40…−87 against physical
  ~87). Training-episode success drops from 98% to 0–80% (workspace failures),
  then recovers with slower cuts (66–97 steps). From update ~111 imagined paths
  survive H32 (`survival_H` 0.3–0.56), bootstrap reaches 127 per sample and the
  critic's endpoint value on WM states 150–221; recorded-window targets exceed
  110 (up to 256.6) once full 32-step windows bootstrap on overestimated values.
- V2: the H8 bootstrap carries ~55 of J ≈ 60; policy gradient norm 40–93 (C3
  2–18); mean |action| 0.95 → 1.37 within ten updates (C3 0.88 → 0.60).
  Training episodes switch to 1–2-step force-limit collisions (99% by
  iteration 1645). For 3316 recorded early-failure transitions WM562/WM584
  predict collision in 22%/22% (predicted peak/limit 0.67/0.83, recorded 1.04);
  from the frozen collision start the WM predicts no terminal on 8/8 paths and
  an objective of +29.6 against −100.6.
- V3: BC MSE stays ≤0.07 and mean |action| still rises 0.94 → 1.21 by update 72
  (V2 1.28): the alpha 2.5 term hardly constrains the actor. Training episodes
  at iteration 1625 collide in 83% (0.1 action noise); the deterministic policy
  at that point succeeds in 17 steps at 47.9 N, near the 50 N limit. H8
  imagination predicts 2 negative terminals on 648 held-out starts (H32 policy62: 450).

- V4 (one change against C3: WM-state critic samples, h32): bootstrap weight
  stays ~0 (updates 63–115) and the physical policy degrades as in C3 (training
  success 0% from iteration 1680, workspace failures after ~28 steps). Recorded
  targets exceed 110 (max 168.3) from update ~109, before any bootstrap weight.
- V5 (one change against V3: alpha 0.25): mean |action| 0.94 → 0.99 (max 1.06),
  BC MSE ≤ 0.026, policy gradient norm ≤ 77; training episodes 90–99% success,
  force-limit collisions 1–10%, cut length 16–20 steps. Recorded-target MAE
  before each critic update 3.3–8.7, max |target| 106.9. Frozen: 8/8 in 18
  steps (policy62: 20), return 92.18 (90.81), peak 39.3 N (30.7 N). Imagined H8
  objective − physical return: −43.7 (benchmark), −45.1 (held-out); the WM still
  imagines 218/648 held-out workspace failures with policy103.

### V5 stability run

`runs/critic_real_v5s_20261106` resumes V5's `last.pt` with identical options
(no critic reset; refill 40 vector steps), 5 h wall, W&B `4bowhisp`; SIGUSR1
checkpoints at policy updates 150/200/250. Policy 103 → 374, WM 603 → 874,
iteration 2424, 18042 s.

| Checkpoint | Policy/WM | Frozen | Steps | Return | Peak force | Value MAE / bias (benchmark) | max value | Held-out MAE | Imagined H8 − physical |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| 1863 | 150/650 | 8/8 | 17 | 92.55 | 42.47 N | 1.54 / −1.51 | 99.8 | 1.81 | −43.1 |
| 1989 | 201/701 | 8/8 | 17 | 92.60 | 45.51 N | 3.54 / −2.20 | 101.1 | 3.11 | −45.5 |
| 2113 | 250/750 | 8/8 | 17 | 92.64 | 47.48 N | 5.18 / −4.32 | 100.7 | 4.21 | −46.4 |
| last 2424 | 374/874 | 8/8 | 17 | 92.54 | 35.11 N | 3.95 / −3.45 | 100.4 | 3.58 | −34.8 |

Training episodes (0.1 action noise), per 60 vector steps from iteration 1680:
success 94, 94, 90, 82, 76, 66, 60, 67, 72, 94, 99.7, 99.4, 99.4%; all
force-limit failures occur at step 1 or 2 (≈85% at step 2). Mean recorded-target
MAE before each critic update, per 30 updates: 0.90, 0.73, 0.96, 1.19, 1.41,
1.53, 1.56, 1.45, 1.32, 1.13; max |target| 100.8 throughout; mean |action|
1.06–1.10; BC MSE 0.0017–0.0035. The gap between the critic's mean value and
its imagined-rollout targets on imagined states stays 36–43 per 30 updates
(single updates 30–63) without a trend.

Measured: every evaluated checkpoint passes 8/8 with value MAE ≤ 5.2 and values
≤ 101.1; critic-target MAE and noisy-episode collisions rise until about update
280 and then fall back. Frozen peak force rises to 47.5 N at update 250 (limit
50 N) and is 35.1 N at the end. Inferred: the imagined-state gap reflects WM
imagination (imagined workspace failures on physically successful cuts, e.g.
218/648 held-out paths at policy 103), not critic error against physical returns.

### Ablation (a), 1 h

`runs/critic_real_abl_a_20261106/run` (W&B `02cdch2v`): critic_target real
(no WM-state samples), actor horizon 8, BC 0.25, otherwise as V5. Policy 62 → 104.
Frozen 8/8, 18 steps, return 92.09, peak 30.83 N; benchmark value MAE 2.67
(bias −2.09, values 52–100), held-out MAE 5.15; max recorded target 105.1;
target MAE before update 3.9–7.2. Training episodes: success 98, 93, 54, 99,
99.9, 100% per 20 vector steps (collision dip at iteration 1640–1659).
Imagined H8 objective − physical return −45.1. At 42 updates this does not show
that WM-state samples are unnecessary; the user requested matched-length (≥250
updates) reruns of ablations (a) and (b). Ablation (b) started automatically
and was stopped at refill end (`critic_real_abl_b_20261106`, policy 62, W&B
`oiw4yvhv`, no policy update); it is not a result.

Attribution (measured runs, inferred conclusion): the 1–2-step collision
collapse occurs with real_wm + h8 (V2) but not with real_wm + h32 (V4) or real +
h32 (C3), so it needs the short horizon, through which the critic bootstrap
enters the actor gradient. real + h8 is not yet run. With alpha 0.25 the actor
still changes: shorter cuts (18 vs 20 steps), higher return and higher peak force
than policy62 at a mean |action| change of +0.05, so the regularizer limits but
does not reduce the update to imitation.

Inferred: V2/V3 failures come from the actor following the critic bootstrap
gradient into actions whose force consequences the WM under-predicts; the
critic's values on WM states were calibrated (model-state MAE 8–13 during V2).
C3's value failure comes from bootstrapping on WM endpoint states outside the
recorded-state critic's support once imagination stops terminating.

## 2026-11-07 — Source SAC collapse in the unified environment

`runs/sac_gpu_training_20261105` (GPU7, one environment, SliceIt tf2rl SAC with
the source yaml settings; the checkpoint's actor and critic optimizers record
learning rate 0.01, not the 1e-3 run-argument default; prioritized replay,
auto-alpha) succeeded on every episode up to environment step 87394.
From step 87395 every episode collides at step 1 (58–75 N, return −100.6091, no
spring released) until the run was stopped at step 95037. Frozen evaluation on
a fresh environment (`metrics_eval_20261106/sac_{85000,90000}`, 8 episodes each):

| Checkpoint | Success | Steps | Return | Peak force | Released springs |
|---|---:|---:|---:|---:|---:|
| ckpt-17, step 85000 | 8/8 | 18 | 92.026 | 49.6 N | 287/304 |
| ckpt-18, step 90000 | 0/8 (collision) | 1 | −100.609 | 74.5 N | 0 |

Measured: the 90000 policy outputs `[1, −1, 1, 1, −1, −1]`, saturated in all
six dimensions; training samples with exploration noise give identical returns.
Both checkpoints have only finite weights. Between them, Adam second-moment
estimates jump about 600× for one actor layer (0.18 → 109.7 norm) and about
200× for one critic layer (1.1 → 230.5). TensorBoard scalars at step 87000
(the last summary before the collapse) are ordinary: critic loss 7.5, alpha
0.010, mean log-prob 5.9. Inferred, not verified: one or more large gradient
spikes pushed the tanh-squashed actor into saturation, where its gradient
vanishes, and the following one-step collision data cannot pull it back. The
triggering batch is not recoverable from the logs.

The SAC baseline result for the single prism is the step-85000 checkpoint,
reported together with this collapse. A new SAC run is deferred to the parallel
multi-environment stage (see PLAN).

## 2026-11-07 — Original DiSECt gradients and the static bottom layer

`backends/mujoco/scripts/disect_gradcheck.py` on GPU7 with the official
NVlabs/DiSECt meshes (GitHub LFS, revision ff90165) and the three upstream
configs. Objective: mean knife force over a 0.6 s slicing trajectory (the
trajectory-optimization objective), backward gradient with respect to the
SlicingMotion parameters; central finite differences along one fixed random
direction in relative parameter coordinates. Bottom rule: `free` uses the
include box y < −0.002, `fixed` y < 0.002. Results in
`Cutting_DEV_metrics/runs/disect_gradcheck_20261107`.

| Object (config) | Bottom | Static particles | Mean / peak force | Released | Backward gradient | Analytic vs FD (ε = 1e-3) | FD (ε = 1e-2) |
|---|---|---:|---|---:|---|---|---:|
| Prism (`ansys_prism`) | free | 0/1187 | 7.37 / 13.73 N | 188/304 | NaN | — | 0.418 |
| Prism | fixed | 104/1187 | 7.88 / 13.98 N | 193/304 | finite | 0.022 vs −0.168 | 0.156 |
| Apple sphere (`ansys_sphere_apple`) | free | 0/704 | 10.97 / 21.03 N | 131/244 | NaN | — | 0.216 |
| Apple sphere | fixed | 77/704 | 16.13 / 37.58 N | 133/244 | finite | −0.957 vs −1.065 | −0.371 |
| Jello cylinder (`ansys_cylinder_jello`) | free | 0/937 | 1.01 / 1.93 N | 129/238 | NaN | — | −0.017 |
| Jello cylinder | fixed | 98/937 | 1.32 / 2.80 N | 132/238 | finite | −0.130 vs −0.106 | −0.025 |

Measured: all three meshes rest at y = 0.0008 m. The upstream prism and jello
configs use y < −0.002 and therefore make no particle static; only the apple
config (y < 0.002) fixes its bottom layer. SliceIt's `public_prism.json`
inherits y < −0.002. With a free bottom the forward pass is finite for every
object and every backward gradient is NaN; with the bottom layer static every
backward gradient is finite. A 0.05 s free-prism run already gives NaN
gradients. For the apple and jello the analytic directional derivative agrees
with the ε = 1e-3 finite difference within 10–25%. For the prism the finite
difference changes sign between ε = 1e-2 and 1e-3, so this check cannot judge
the prism gradient; a step-size sweep or float64 is needed.

## 2026-11-08 — real_wm crash on one-step-only batches (fixed)

Found in review (Codex) and reproduced: when every sampled window terminates
after its first transition, `real_wm` has no WM-predicted state and
`model_state_targets` raised `torch.cat(): expected a non-empty list of Tensors`
(trainer line 388); `real` on the same data returns all targets −100.6. The fix
keeps the batch's recorded samples, adds no model samples and reports the
model/recorded-state MAE only over non-empty subsets; failed trajectories, reward
and thresholds are unchanged. `real` and `real_wm` now produce identical critic
data on such batches (`test_real_wm_keeps_recorded_samples_when_no_window_has_a_model_state`).
Second review finding, fixed the same way: with a one-step critic window
(`PolicyTrainerCfg.horizon=1`, not `--actor-horizon`) the WM prediction list is
empty and `torch.stack([])` raised before that check; the check now runs before
the WM rollout (tests for windows 1 and 8). Runs use window 32 and never hit it.
The bug stops a run with an exception; it does not explain the earlier critic
value growth. Runs that finished without this exception (V5 and the rest) are
unaffected. Review notes kept for the method discussion: `real` targets are
λ-returns that bootstrap the critic, not complete-episode returns; `real_wm`
assigns recorded-state targets to WM-predicted states, which assumes the two are
close and can bias the critic where the WM mispredicts position, force or
termination. Choosing `real` removes that pairing only from the critic's training
samples: the actor still unrolls the WM, updates through BPTT and queries the
critic at predicted states, so WM prediction error still enters policy learning.
Changing the pairing is an algorithm-design change, not a code fix; it stays
documented with `real_wm` (V5) kept as a separate, labelled method.

## 2026-11-08 — Static vs free prism bottom in the MuJoCo benchmark

GPU7, `Cutting_DEV_metrics/runs/fixed_bottom_20261108`. Source scene
(`public-prism-ur5e-v1`, no static particle) against the fixed-bottom re-export
(`public-prism-fixed-bottom-ur5e-v1`, 104 static bottom particles), same frozen
policies. Episodes within one scene repeat to ≤ 0.02 N and ≤ 1 released spring,
so every difference below is systematic.

| Policy | Bottom | Success | Steps | Return | Peak force | Mean force | Released | Knife work | Impulse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference (4 ep.) | free | 4/4 | 40 | 80.29 | 9.97 N | 3.92 N | 288 | 0.314 J | 7.83 Ns |
| Reference | static | 4/4 | 40 | 79.46 | 11.24 N | 4.88 N | 302 | 0.354 J | 9.76 Ns |
| policy62 (4 ep.) | free | 4/4 | 20 | 90.82 | 30.66 N | 7.23 N | 288 | 0.885 J | 7.23 Ns |
| policy62 | static | 4/4 | 21 | 89.98 | 30.82 N | 8.40 N | 298 | 0.940 J | 8.82 Ns |
| SAC step 85000 (4 ep.) | free | 4/4 | 18 | 92.03 | 49.62 N | 7.99 N | 287 | 1.289 J | 7.19 Ns |
| SAC step 85000 | static | 4/4 | 18 | 91.57 | 50.29 N | 9.41 N | 298 | 1.331 J | 8.47 Ns |
| Constant zero (1 ep., 500 steps) | free | no | 500 | −193.83 | 5.83 N | 2.01 N | 280 | 0.136 J | 50.2 Ns |
| Constant zero | static | no | 500 | −221.33 | 6.03 N | 3.46 N | 291 | 0.167 J | 86.5 Ns |

Measured: with a static bottom, mean material force rises 16–25% for the three
cutting policies, impulse 18–25%, peak force 1–13%, and 10–14 more springs are
released; success and the ordering of methods by steps and peak force are
unchanged (policy62 needs one more step). SAC's peak material force reaches
50.29 N, above the 50 N limit value, without a force-limit termination (the
limit applies to the filtered sensor wrench). This contradicts the earlier
inference from the free prism's 1 mm vertical settling that the bottom condition
is negligible: although the free prism does not slide, the boundary condition
changes the forces. Decision (PLAN rule): the settings differ beyond noise, so
both are reported; a benchmark change to static bottoms requires re-evaluating
all methods in the static setting.
