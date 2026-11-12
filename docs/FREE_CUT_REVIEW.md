# Free-cut task implementation review — 2026-11-12

For Claude's review. Parent: e529161. User authorized implementation and verification;
no long training is launched. The old SliceIt task remains the default.

## Implemented path

1. GPUEnvironment(task=free_cut_v1) and MultiConditionEnvironment config.task select
   the new contract. It requires explicit six-axis limits and net wrench feedback.
   All cut springs must connect the two FEM components, with no fixed particles or
   residual spring damping. Physical success means complete separation and no
   force/workspace failure. Failure takes precedence and is charged once.
2. Time limit is terminal (incomplete), not a bootstrap-able truncation. step_count
   is retained in compact WM/critic history, incremented analytically, scaled by
   the horizon. The policy still sees 10 x 24 observations and an 8-D task goal.
3. Progress is a supervised simulator quantity. The WM predicts a sigmoid fraction
   of the remaining progress; rho is monotone and bounded. Rewards use its actual
   increment. A completion head is trained against physical separation labels,
   not all termination labels. Expected completion reward and survival use the
   same probability. Force/workspace/deadline absorption overrides survival.
   This changes the task-specific heads; it is not an unchanged SuperMimic algorithm.
4. Speed/load/normalized-jerk penalties are per-second rates multiplied by dt.
   Speed tracking is uncoupled from load. Weights must all be explicitly supplied;
   interface_check.json contains provisional numbers, not training approval.
5. Randomized LocalGroup resets now set material, restore immutable exported
   material state, settle selected worlds, then reset controller/history clocks.
   Other worlds and their reset snapshots are preserved. Original velocities
   are integrated rather than manually zeroed. The contact guard covers all
   material substeps for the selected worlds.
6. Group/process adapters, training metadata, per-episode logging and the existing
   imagination/MPPI absorption helper understand the new contract. No old
   checkpoint is declared compatible; no new policy has been trained.

## Files to inspect

- cutting_world_model/free_cut.py: task contract, reward, structural asset checks.
- cutting_world_model/free_cut_state.py: progress, clock, completion reward/features.
- cutting_world_model/world_model.py: separation targets and expected completion reward.
- cutting_world_model/gpu_env.py, multi_env.py: physical settlement and grouped resets.
- backends/mujoco/cutting_gpu_backend/{environment,batched_disect}.py: masked settling
  from immutable export; selected-row reset snapshots.
- trainer.py / mpc.py: hard failure/deadline absorption; legacy branch unchanged.
- episode_log.py: incomplete terminal versus truncation; separate separation metric.
- tests/test_free_cut.py: task precedence, finite-horizon absorption, BPTT finite
  differences, one supervised WM optimizer step, GPU masked reset, local/process groups.
- scripts/check_free_cut_task.py: complete physical references, no policy optimizer.

## Verification scope

Final root: 298 passed / 5 skipped; backend: 36 passed. Four physical references completed
(press, saw, fast press, stall); the provisional reward preference failed.
Remaining two references were stopped after rejection, before the 1200 s cap.
See RESULTS for full scope and outcomes.
Runs are on GPU7 under runs/free_cut_v1_review_20261112.
The reference budget is 1200 s on one GPU; this is a task interface/reward check,
not a force-calibration experiment or algorithm comparison.

The remaining checks are deliberately explicit:
- No newly fitted WM yet exists, so predicted versus physical multi-step return
  ordering has not been measured. Unit gradients do not establish WM accuracy.
- Fixed trajectories can reject poor weights; passing cannot prove no reward exploit.
- Re-settling every randomized reset adds simulation cost. No throughput claim.
- The direct MPPI rollout understands the contract, but the historical benchmark
  factory is still tied to the old benchmark; no new-task MPC run is claimed.
- Current topology admission supports the tested free FEM scenes. Other exported
  assets must pass it; there is no per-object relaxation of separation.
- All force limits/friction remain simulation conventions, not Mini45 calibration.



## Provisional reward finding

The first physical preference check rejects the interface-check weights:
press at 3 cm/s completes in 43 steps (return 72.77), while a 9 cm/s reference
completes in 17 steps (return 93.33). Both obey the simulated limits. This is
not a task-settlement bug: with gamma=0.99 and R_s=100, the earlier completion
bonus alone is worth about 19.5 more; a speed-penalty rate of 1 per second
cannot offset it. The weights therefore do not implement the desired speed
preference. Do not start training with interface_check.json.

This round does not change discounting, failure magnitudes or material parameters
to make the preference check pass. The code/reward contract is handed off for
review, with the rejected candidate and exact traces preserved. Before a training
plan, choose explicit reward weights against the task priority and rerun the
same fixed trajectories. The checker now also writes speed/utilization/jerk
components for future runs so candidate weights can be rescored without
resimulating; the first report predates those extra scalar fields.


Force-output fitting retains the source linear residual output. Magnitudes are
used at task/termination evaluation, not imposed with abs() inside the physical
integration: the latter would give zero force-fitting gradients at the source's
zero-initialized last layer. A dedicated regression checks nonzero supervised
force gradients from that initialization. No claim about fitted prediction
accuracy follows from this check.
