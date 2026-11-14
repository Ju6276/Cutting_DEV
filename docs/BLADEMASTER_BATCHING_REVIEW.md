# BladeMaster grouped GPU batch review

## Implementation

The three object groups retain their existing materials, sampling, time steps,
board contact, blade geometry and task reward. Each group executes one GPU graph
whose particle/grid/contact kernels include an explicit world dimension. The
group loop has three entries; it does not call 256 independent solvers.

The external source is commit `e2e5221` on `codex/ur5e-batched`, in the worktree under
`/dataHDD/dong/Projects/BladeMaster-Warp-batched-20261114`.
`generate_batch_kernels.py` generates world-indexed material/contact/tool/grid
constraint kernels from the existing forward source. The transformations add
array indices, atomic indices, function world arguments, launch indices and an
active-world mask; constitutive and contact expressions stay unchanged.
Generated files are checked in alongside the generator.

The wrapper preserves single-tool view aliases; world-specific branch masks
cover pair-list construction and blade constraint iterations. Every world has
its own contact-history parity. Before an odd-length coupling interval, history
tables are aligned only for active worlds whose saved parity differs from the
graph parity. Inactive material arrays remain unchanged.

Reset copies the actual settled position, velocity, deformation, plasticity and
board-friction history for the selected worlds. Scratch grids and initially
empty inter-fragment pair tables are rebuilt. A reset snapshot must precede any
cutting. Settling remains 0.2 seconds at the same nominal material.

The supported interface is float32, one externally held tool per world,
compact grid storage and list pair search. The single-world forward path remains
available for comparison. The physics backend has no gradient requirement; WM
and policy differentiation stays in PyTorch.

## Configuration and observations

`configs/blademaster/three_intact_batched256.json` contains 86 cucumbers,
85 half potatoes and 85 prisms. It retains the previous nominal task/reward and
sets `execution=batched`, `force_observation=sensor_wrench6_v1`.
The policy observation has 34 entries, including all six sensor force/torque
components at the existing history cadence. Object labels are for logging;
the policy receives no object identity.

Source metadata includes the external Git revision, geometry asset identities
and execution/configuration information. Old 25-entry observations and old
checkpoints are a different interface.

## Capacity and physical equivalence

RTX 4090, 256 environments:
- Initialization: 37.8 seconds.
- Six complete 50 ms control steps: 11.67 seconds for the first; 9.61–9.65
  seconds thereafter. Nonzero material reaction appeared by step six.
- Process memory: 17,862 MiB before the learner.
- Physical arrays plus reset snapshot: about 17.03 GB in the initial probe.
  Retaining settled friction history adds approximately 84 MB at 256 worlds.
- 384 worlds, equally grouped, would require 25.59 GB for those physical arrays
  alone, exceeding the 24 GiB card before robot/learner allocations.

Two-world versus original scalar contact test, 47 substeps per 2 ms exchange:
peak reaction 1.1584 N, maximum wrench-component difference 3.55e-5, maximum
particle-position difference 5.96e-8 m. Moving-knife side labels are identical;
the untouched world's reaction is zero. Pausing/resuming one world across an
odd interval, and resetting one world, preserve the other world's state.

## Validation commands

From the Cutting_DEV worktree, after sourcing `runs/sac_env.sh`:

```sh
PYTHONPATH=.:backends/mujoco /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python \
  scripts/check_blademaster_multi.py \
  --config configs/blademaster/three_intact_batched256.json \
  --full-network --skip-reference --seconds 3400 \
  --output runs/bm_batch_learning256_20261114
```

External-source regressions:

```sh
PYTHONPATH=/dataHDD/dong/Envs/blademaster-batch-test-deps:. \
  /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python \
  -m pytest tests/test_batched.py tests/test_reference_cloud.py -q
```

The full-network preflight uses real physical transitions, the source-sized
WM/policy/critic, the production replay capacity, h8 actor updates and optimizer/
replay restoration. No long training is started by this check.


## Completed physical and learning checks

`runs/bm_batch_full_preflight256_20261114/report.json` retains the complete
reference-control results: all 86 cucumbers, 85 half potatoes and 85 prisms
reached the pose goal without force/outside/timeout failure. Cucumber/prism
completed at step 50, half potato at step 51–52; each final native material graph
had two components. These are separate pose and material observations. That
probe was stopped after 24 mapped collection steps because the full critic
requires 32-step windows; it is not the completed learner result.

The fresh corrected run is `runs/bm_batch_learning256_20261114/report.json`:

- 256 real worlds, 40 mapped steps, 10,240 physical transitions.
- Full WM architecture and batch 2048/window 8/10 minibatches; full policy
  batch 4096, actor horizon 8; real critic 4096 windows of **32 steps**
  (131,072 valid samples), microbatch 8192.
- First update: 10 WM optimizer steps, 1 policy step and 20 critic steps; all
  three parameter sets changed and remained finite.
- Saved weights, optimizers and replay; restored them and completed another
  WM fit and actor/critic update.
- Production replay capacity 1000 per world; six-wrench policy observation 34.
- Selected-world reset left all unselected material state and observations
  exactly unchanged. The dedicated external regression also checks settled
  board-friction history restoration.
- Driver peak **22,624 MiB**, including the restore check with two replay
  allocations. Fresh run elapsed 926.7 seconds including compilation.
- These bounded updates explicitly skip WM warmup to exercise both optimizers;
  the planned training retains its separate 500-update WM pretraining stage.

The first physical reference probe predates the settled-friction-history reset
correction. Its initial reference trajectories remain recorded separately; the
fresh 256-world collection/update run and dedicated reset test use the correction.

`runs/bm_batch_snapshot_check_20261114` contains actual three-world material and
robot geometry exported through the parent's snapshot helper after one physical
step. It is a snapshot API check, not a screenshot of long training.

## Regression coverage and source distribution

- Cutting_DEV root suite: **420 passed, 5 skipped**, 241.08 s, optional MPPI
  environment and GPU backend enabled.
- Bundled MuJoCo backend: **39 passed**, 5.61 s.
- External forward batch/contact/reset/generator/reference-cloud regressions:
  **4 passed**, 19.58 s on GPU7.
- External all-repository CPU attempt: six passed before an existing affine
  differentiation test requested unavailable `cuda:0`. The full 2,164-test GPU
  suite was not run; the integration owner requested GPU release to the
  DiSECt capacity check after the focused forward and root/backend regressions.

External source is pushed to `Ju6276/BladeMaster-Lab`, branch
`codex/ur5e-batched`. `integrations/blademaster/0002-batched-worlds.patch` records
the same change after `0001-intact-reference-cloud.patch`, based on `4b2d983`.
The BladeMaster source remains an external configurable `source_root`; its
license is retained in that repository. Generated batching modules are included
in the patch, alongside the generator and its reproducibility test.
