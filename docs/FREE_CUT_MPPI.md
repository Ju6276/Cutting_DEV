# Frozen MPPI on free_cut_v3

The user requested this integration on 2026-11-12. Branch: codex/free-cut-mppi.
GPU7's active Claude worktree and training are unchanged. A6000 worktree:
 /dataHDD/dong/Projects/Cutting_DEV_mppi_v3

## What is connected

scripts/evaluate_free_cut.py loads the exact training conditions and a frozen
checkpoint, selects one object group and runs one physical world. The planner is
the existing pinned UM-ARM-Lab/pytorch_mppi implementation (SOURCES). No optimizer
or task reward is replaced. Each control step samples candidate actions, rolls
them through the WM, executes the first planned action, and incorporates the
physical state/history into the next replan.

The free_cut_v3 reward, gamma=1, bounded completion hazard, workspace failures,
force-limit priority and finite-time terminal behavior come from the same WM/task
used in training. Only environment count, selected group, evaluation seed and
local rather than multiprocess execution differ. The full training distribution
must equal checkpoint metadata; physical/task metadata and the selected group's
contact model are checked again after environment construction. Older pose-task
checkpoints, changed reward/horizon/force limits and unmatched physics are rejected.
Existing fixed-prism evaluate_benchmark.py remains available.

Initial clearance is drawn once for the physical world, material and target speed
are drawn per episode as in training. Conditions are recorded in every episode.
This is not the old fixed-export reset, so the recorder does not require bitwise
identical randomized resets. Termination is recorded before resetting; separation,
success, force failure, workspace failure and timeout are distinct. A wall-budget
stop is incomplete, never counted as task success or a completed timeout.

MPPI sees the WM state, including simulator cut_progress and robot/control history.
The policy has observation history, remaining time and the task goal, not direct
cut_progress. This is a privileged-state planning baseline, not an equal-information
or real-robot deployable planner. The new progress observation would need an
estimator for real use. Exact WM pairing does not remove this information difference.

## Example (A6000)

Use the isolated /dataHDD/dong/Envs/cutting-mppi-v3 Python environment. It reads the
existing CuttingWorldModel environment's packages through cutting-base.pth and adds
the pinned planner dependencies without modifying that base environment.
Set CUDA_VISIBLE_DEVICES, CUDA_HOME, compiler PATH and cache directories as recorded
in runs/free_cut_mppi_20261112/run_mppi.sh. That script is the executed command.

    .venv/bin/python scripts/evaluate_free_cut.py \
      --method wm-mppi \
      --conditions configs/free_cut/short_train_cucumber.json \
      --group free_cucumber_003 \
      --checkpoint PATH/last.pt --paired-checkpoint PATH/last.pt \
      --mpc-horizon 8 --mpc-samples 512 --mpc-refinements 1 \
      --mpc-temperature 1 --mpc-noise-std 0.5 --mpc-discount 1 \
      --mpc-terminal-value none --episodes 1 --seed 0 --wall-seconds 900 \
      --output runs/UNIQUE_DIRECTORY

The same entry accepts --method supermimic or constant-zero. The former reads
the exact paired policy checkpoint; constant-zero is a fixed-action reference.
This entry does not silently connect legacy SAC checkpoints or prescribed
knife trajectories to the new task.

## Outputs and present validation scope

manifest.json records the condition distribution, its selected execution config,
asset/checkpoint hashes, actual checkpoint identity, solver settings and information
access. planning.json records predicted return, force utilization, survival,
effective sample count and planning time per control step. report.json records
physical task outcomes, force/impulse and inference/physics/reset time.
Episode NPZs contain T+1 states and T transitions; a duplicated state/info field
(e.g. cut_progress) uses transition_cut_progress for the transition sequence.

The available snapshot has iteration 894, WM updates 341, policy/critic updates 0.
Checkpoint SHA256: 04805daa38aa2fb6c35d8e53f114d0b95d0110e910b86a4be19d3ed3596219cf.
It is copied from the completed first short-run output, not the file being written
by the resumed training. This run checks integration; it cannot establish a
trained-policy comparison or the quality of a fully pretrained WM.
No solver search, model fitting, new force limit or automatic long training.
Results and actual completion status are appended after the bounded physical run.

## Physical integration results, 2026-11-12

A6000 GPU0. Both GPUs were idle before this task (43 / 17 MiB, 0%, no compute
processes). Other users' CPU Python processes were present and were not touched.
GPU7's existing training was left on its original checkout.

Two executions of the same case, same WM341 and same solver/seed (not two training
seeds, not a success-rate experiment). Both severed all 65 springs and ended
successfully with no force-limit, workspace or timeout event; zero optimizer updates
and checkpoint weights unchanged. The second run verifies the final NPZ recorder.
No reward, material, limits or solver tuning between runs.

| Run | Steps | Simulated seconds | Return | Peak net material force N | Impulse N s | Mean planning seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mppi_ready | 34 | 1.70 | 105.125 | 6.212 | 1.432 | 0.188 |
| mppi_final | 35 | 1.75 | 105.218 | 7.923 | 1.417 | 0.184 |

Sampled condition: Young's modulus x1.49723, density x0.86638, spring stiffness
x0.81350; target feed 0.0332536 m/s; initial lift 0.0098250 m relative to the export.
Both use the unchanged simulation limits [50,50,50,5,5,5], gamma=1 and horizon180.
Peak sensor-limit utilization: 0.13870 / 0.10745 respectively; this is a different
quantity from the net material force norm. Ground/knife friction remains uncalibrated.

The final physical run took 73.68 wall seconds including initialization, 51.21 s
physics, 6.06 s reset and 6.44 s inference. Planning alone averages 0.184 s per
50 ms control interval: approximately 3.7 times the interval, not real-time control.
The first run included fresh DiSECt compilation and took 149.93 s total.
Reported differences between these repeats are not an algorithmic improvement;
only recording/provenance code changed, and the simulator is not bitwise repeatable.

Final episode NPZ has 36 state rows and 35 action/reward/transition-progress rows.
Numerical alignment of state progress after each action and transition progress
was checked directly. The first run used the old NPZ writer which overwrote the
T+1 progress array with T transition entries; its scalar outcome/force report is
unaffected, but mppi_final is the authoritative trajectory artifact.

Runtime setup initially failed before stepping physics because the base environment's
Ninja was absent from PATH. Adding that bin directory and CUDA_HOME fixed the setup.
Planner dependencies were installed in a new venv; no base environment or training
process was modified. No successful evaluation was omitted.

Artifacts on A6000:
 /dataHDD/dong/Projects/Cutting_DEV_mppi_v3/runs/free_cut_mppi_20261112/
GPU7 review copy:
 /dataHDD/dong/Experiments/Cutting_DEV/runs/free_cut_mppi_20261112/a6000/
Use mppi_final/report.json, manifest.json, planning.json and episode_000.npz.
No project files or artifacts were downloaded to the Mac.

## Final code validation and handoff

GPU7 final root suite: 315 passed / 8 skipped (332.47 s).
Backend suite: 38 passed (5.97 s).
Dedicated planner/benchmark suite: 29 passed (1.38 s), including the three MPPI
tests skipped by the base root environment. Five optional historical dependency
tests remain skipped. No test tolerance or task threshold was relaxed.

The first backend invocation mistakenly used the repository root as its working
directory; 12 tests could not find relative assets. It was rerun from backends/mujoco
as required, and all 38 passed. Both logs are retained, including backend_wrong_cwd.log.
The final A6000 source-file hashes and all three current physics asset hashes were
independently compared with the GPU7 review checkout and match. Changed documentation
links and git diff --check passed.

Main files for review:
- scripts/evaluate_free_cut.py: versioned condition/checkpoint/asset manifest and bounded entry.
- cutting_world_model/free_cut_evaluation.py: one-group environment and strict metadata projection.
- cutting_world_model/mpc.py: shared checkpoint validation and explicit privileged progress access.
- cutting_world_model/benchmark.py: randomized reset support, complete terminal labels, NPZ collision fix.
- tests/test_free_cut_mppi.py: metadata rejection, exact v3 rollout, actual MPPI commands, trajectory alignment.

No merge to main and no long training. At the final GPU check, our evaluation had
exited; A6000 GPU0 was idle. GPU1 had acquired a process owned by user tu (~262 MiB
process memory, 287 MiB device memory); it was not interrupted.
