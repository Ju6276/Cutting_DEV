# Current plan

## Numbered runs

- **1 — SAC + Gazebo + DiSECt**: resumed training in SliceIt
  `runs/sac_resume_20261103`, loading the retained TensorFlow checkpoint and
  replay buffer. The source run endpoint remains 100100 environment steps with
  the existing 72-hour execution deadline.
  W&B [xofe4ps8](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/xofe4ps8).
- **2 — SuperMimic + Gazebo + DiSECt**: stopped at iteration 872/update 316
  after critic divergence. Frozen episode return and bootstrap probes identify
  large value bias and terminal-head errors. The relative causal contributions
  remain to be established; single-world data reuse is a measured regime difference.
- **3 — SuperMimic + MuJoCo Warp + DiSECt**: bounded validation finished on
  A6000 GPU 0 at policy62/WM562. `critic_mask_validation_20261104` collects
  3409 complete episodes and samples 2433 collision terminal labels, including
  the previously excluded one/two-step failures. Frozen evaluation succeeds
  8/8 in 20 steps; value MAE remains high at 153.418925. No process remains
  from this run.
  W&B [5h7xrk7c](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/5h7xrk7c).
- Run 1 training stays on ROS 11321/Gazebo 11355. Frozen SAC evaluation uses
  separate ROS 11323/Gazebo 11357 and the source ten-episode deterministic evaluator.

## Approved runs — 2026-11-07, A6000 GPU 0 and GPU 1 in parallel

Both start from policy62/WM562 (`critic_mask_validation_20261104/training/last.pt`,
iteration 1561) with the final baseline `--critic-target real --actor-horizon 8
--actor-bc-alpha 0.25 --critic-reset --critic-prefit-rounds 100`, 1024 worlds,
and stop at `--max-iterations 860` (iteration 2421, the end of ablation a,
390 policy updates). Checkpoints at 150/200/250/300 updates and the final one
receive 8-episode frozen evaluation and `scripts/value_eval.py`. Budget: 12 h wall
per run; expected 7–9 h with both GPUs busy. One seed each.

- **bc_only** (`--actor-objective bc`), GPU 0. Question: does the model-based
  gradient contribute beyond behavior cloning toward replay actions? Decision:
  if bc_only matches ablation a in success, steps, return and force metrics at
  every frozen checkpoint, the h8 policy's quality is attributed to BC and the
  paper must not claim a model-based improvement for this task; if it is slower,
  weaker or fails, the model gradient contributes.
- **linear_force** (`--force-penalty linear --force-penalty-scale 50`), GPU 1.
  Question: does a non-saturating force penalty lower contact forces? Metric:
  peak force, knife work, safety margin and steps against ablation a, all scored
  with the benchmark's SliceIt reward. Decision: if peak force and safety margin
  drop clearly at matched success, consider the linear penalty for the final
  benchmark (then for every method); if only steps grow, report the trade-off.

Outcome (RESULTS 2026-11-08): bc_only stays at policy62 behaviour (20–21 steps)
while (a) cuts in 17–19 steps, so the model gradient contributes. linear_force
gives no clear peak-force reduction at matched success and its final checkpoint
fails 8/8 (workspace); not adopted, the SliceIt reward stays.

## Decisions — 2026-11-08

- **Benchmark bottom stays free** (user decision after the static/free comparison):
  `public-prism-ur5e-v1` remains the benchmark; the fixed-bottom contract is kept
  only as a diagnostic. Consequence: the original DiSECt adjoint is NaN with a free
  bottom (RESULTS 2026-11-07), so gradient-based DiSECt trajectory optimization and
  material fitting cannot run in the benchmark setting as is. Proposal (not
  approved): optimize or fit with a static bottom and evaluate the resulting
  trajectory in the free benchmark, reporting that mismatch.
- **Naming.** The final SuperMimic-style baseline is `real` + actor horizon 8 + BC
  0.25. V5 is `real_wm` + horizon 8 + BC 0.25. They are separate methods; their
  results are reported separately and never pooled.

## Diagnostic — 2026-11-08, GPU7: static vs free prism bottom

Question: does making the prism's bottom layer static (needed for finite DiSECt
gradients, RESULTS 2026-11-07) change the MuJoCo benchmark? Setting: the source
scene and the fixed-bottom re-export (`public-prism-fixed-bottom-ur5e-v1`), each
with the reference trajectory (4 episodes), policy62 (4), SAC step 85000 (4) and
constant-zero (1). Metric: success, steps, return, peak/mean material force,
released springs, knife work and impulse. Existing episodes already show no
horizontal drift of the free prism (bottom layer moves about 1 mm down, also
under constant-zero). Decision: if every metric agrees within run-to-run noise,
later benchmarks (multi-object stage) use static bottoms for all objects and the
existing single-prism results stand; otherwise report both settings.
Outcome (RESULTS 2026-11-08): mean force +16–25% and impulse +18–25% with a static
bottom; success and method ordering unchanged. Both settings are reported; the
choice of setting for the multi-object stage is open.

## Proposed — multi-object parallel stage (awaiting approval)

Current state: SuperMimic trains with 1024 identical worlds (one scene, one reset
state); SAC uses one environment. `BatchedDiSECt` packs copies of one exported
scene with a fixed per-world stride, so per-world material parameters need only
per-world writes of the replicated arrays, while different meshes per world are
not supported. Plan: one `BatchedDiSECt` per object, worlds grouped by object.

Precedent: Isaac Lab `MultiAssetSpawnerCfg` assigns one asset variant per
environment at scene creation; ManiSkill3 places different objects in each
parallel sub-scene; Rudin et al. 2021 lay all terrains out in one mesh and move
robots between them instead of swapping terrain at reset; RoboNinja (RSS 2023)
trains on 300 generated multi-material objects and tests on 50 in-distribution
and 50 out-of-distribution ones. Sampling: object assignment fixed per world
group (equal group sizes); continuous parameters (material, initial poses)
sampled uniformly at every reset; adaptive schemes (PLR, Jiang et al. 2021;
ADR, OpenAI 2019) are later, separate changes.

Steps, one change each:

- **P0 — per-world materials are correct.** No training. Sweep material values
  across worlds with the reference trajectory and compare each world with a
  separate single-world run. Decision: per-world force and released springs
  match the single runs within GPU float noise.
- **P1 — parallel SAC, single prism.** 1024 worlds, no randomization, SAC
  learning rate 3e-4 (SAC paper default), environment-step budget matched to
  SuperMimic. Decision: 8/8 success and return ≥ 90 without saturation collapse
  (compare the one-environment step-85000 checkpoint).
- **P2 — randomized single prism.** Material and initial-pose ranges sampled per
  reset; held-out parameter ranges for testing; SAC and SuperMimic.
- **P3 — three objects.** Prism, apple sphere, jello cylinder with free
  bottoms (decision 2026-11-08), worlds grouped by object; held-out parameters
  (or one object) for out-of-distribution tests. Before training, check with the
  reference trajectory that the free sphere and cylinder neither roll nor slide
  out of the cut (the apple export currently has a static bottom and needs a
  free-bottom re-export).

Open decisions: material ranges (proposal: centered on the DiSECt calibrated
values, e.g. Young's modulus ×0.5–×2, to be checked against the literature);
whether the policy observes an object identifier; budgets after P0 timing.

## Proposed — real-robot calibration (awaiting approval)

Steps R0–R4 (software path, F/T calibration, controller real2sim, material
calibration, closed-loop evaluation) and open hardware items are in
[REAL2SIM](REAL2SIM.md).

## Current implementation

The world model predicts 36 coordinates: joint position/velocity, observed wrench
history and interval peak wrench. Ten frames supply 480 physical/command inputs.
The 860-coordinate rollout state also retains analytic observation/reward history
and success count. Particles remain in DiSECt and recorded data. Source widths,
effective batches 2048/4096, maximum eight-step model horizon, 32-step BPTT, learning rates,
update ratio and task thresholds are unchanged. Microbatch 64 controls storage.

Gazebo runs 1 and 2 use the repaired post-step and zero-time reset joint snapshots.
Shutdown independently saves artifacts, closes the environment within wall-time
bounds and finishes W&B; a run-specific guard cleans up remaining child processes.
The completed SAC task check had 31 transitions, 30 updates, return 84.00210,
reset pose/history errors 0 and force-read errors 0. Old failed SAC checkpoint
files were deleted; diagnostic logs and replay remain. New world-model metadata
uses `synchronous_paused_joint_snapshot_v1` and rejects incompatible recordings
or resumed checkpoints.

## Active work — fixed environment

1. **Frozen WM fit completed, A6000 GPU 0.** `runs/wm_frozen_fit_20261105` starts from
   policy62/WM562. Freeze policy, critic and their optimizer state. Collect four
   complete episodes in each of 32 environments using the existing 0.1 action
   noise; groups 0–23 supply 96 training episodes and groups 24–31 supply 32
   held-out episodes. The collected split has 93/96 training successes and three
   workspace failures; all 32 held-out episodes succeed. Fit the WM for at most
   100 rounds or 1500 seconds, using the existing H8 objective, effective batch
   2048, ten batches per round and Adam state. The total hard budget is 2700
   seconds. Compare H1/H8/full recorded-action error and first terminal event
   at rounds 0/10/25/50/100. All 100 rounds finished in 1165.81 seconds;
   policy/critic and their optimizers stay identical. Held-out full position
   RMSE changes 19.42 → 7.15 mm, but all 32 actual successful terminals are missed.
   Final predicted goal-z error is −16.39 to −6.52 mm, actual about −0.3 mm,
   with the original 2 mm goal threshold. The data/metrics/frozen state passed
   independent CPU reconstruction. GPU 0 was released after the probe.
   W&B [7jnyx9n4](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/7jnyx9n4).
2. **One evaluation environment.** `public-prism-ur5e-v1` fixes the existing
   UR5e + MuJoCo Warp + FDCC + DiSECt assets, reset and task. Constant control,
   SAC and the current WM policy use `scripts/evaluate_benchmark.py`. The old
   single-policy evaluator is replaced by this shared runner. Commands, metric
   definitions and the fixed configuration are in [BENCHMARK](BENCHMARK.md).
   Policy62 succeeds in 20 steps; transferred Gazebo SAC ckpt-10 fails the force
   limit at 0.042 s. Constant-zero control reaches 346 steps/17.3 s and 282
   released springs at its 600 s budget, with an incomplete physical episode.
3. **Original SAC adapter.** Keep SliceIt's tf2rl implementation, Wave actor,
   batch 1024, learning rate 0.01, auto-alpha, seed 123456, warmup 1000,
   endpoint 100100 and checkpoint interval 5000. A local process bridge joins
   its Python 3.8 learner to the Python 3.12 GPU environment. First validate
   the transition/terminal/reset chain and frozen complete episodes; then train
   SAC from scratch in this same environment for the algorithm comparison.
4. **Next: separate WM loss gradients on the saved data.** Train and held-out
   errors track each other. Between rounds 50 and 100 the physical-state loss
   increases while terminal BCE decreases; the final predicted success counter
   is zero against an actual value of eleven. On identical saved batches, measure
   the direction and magnitude of dynamics-parameter gradients from state loss
   and terminal BCE. This distinguishes competition between objectives from a
   shared prediction error before selecting a source-supported repair. Do not
   resume joint policy/critic training yet. The last measured critic return MAE
   is 153.42; after the model event/return correction, compare a fresh critic to
   complete physical returns before joint updates.
5. **Model-based comparison.** Connect DiSECt's knife trajectory optimizer to
   the same robot/controller evaluation. Its free-knife optimization results
   remain a separate experiment until that adapter executes the same task.

No new long training is launched during these bounded checks. Run1 keeps its
existing Gazebo process, endpoint and wall budget. Current machine routes and
local environment paths are in `runs/operations/MACHINES.md` on the Mac;
connection details and credentials stay outside Git.

## Frozen review — 2026-11-05

State frozen at tag `freeze-20261105-pre-critic`. The code review and critic
failure analysis are in [ANALYSIS](ANALYSIS.md). No new run is started until
the proposed rule change and critic experiment plan are approved.

## External implementation evidence

[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym/tree/0171f5eadee50492d6ff30fa04b55e16cc11b61e)
provides full-update equivalence checks, independent physical evaluation and
500 world-model-only iterations; all three are now applied here. Its RunningNorm
critic and analytic soft termination are separate method choices. The current
candidate keeps the Cuboi learned head and static scales. SliceIt physical
thresholds and reward remain fixed. Analytic termination alone cannot restore
a force-limit event that the dynamics model failed to predict.
