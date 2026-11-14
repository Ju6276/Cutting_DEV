# BladeMaster UR5e backend integration review

Branch: `codex/blademaster-ur5e-backend`, based on `f9abb84`.
This is an additional physical backend, not a replacement for the DiSECt experiments.

## Interface and identity

`BladeMasterEnvironment` connects the existing UR5e MuJoCo Warp robot,
FDCC controller, six-dimensional action mapper and source wrench filter to
BladeMaster's `ExternalToolPort`. The backend source path is explicit in
`configs/blademaster/banana_ur5e.json`. Source commit, dirty state and the entire
material configuration are recorded in task metadata.

The initial supported material is the reproduction's **synthetic banana with
paper material parameters**, not a real scanned or calibrated banana. No DiSECt
mesh, prepared cut plane or cutting springs are constructed. BladeMaster's
forward cutting operation controls the material topology.

Each world owns an independent solver. Worlds advance sequentially; masked reset
reconstructs and settles only selected solvers. This is a functional small-batch
interface, not the existing high-throughput packed DiSECt implementation.
Checkpoint/replay resume restores learning state and starts fresh physical
episodes, following the existing trainer. It is not an exact mid-episode
BladeMaster solver snapshot.

## Frames, mounting, exchange and force feedback

- BladeMaster material coordinates are right-handed, z up, metres.
- The original UR5e simulated-knife origin establishes the initial mounting
  translation. A fixed EE-to-BladeMaster-wedge transform sets the wedge's local
  z vertical and its local y along the cutting edge at reset. The wedge shape
  is the BladeMaster preset, not the DiSECt visual mesh.
- Robot integration is 1 ms; force exchange is 2 ms; an RL action lasts 50 ms.
  BladeMaster advances the exact integer number of material substeps in each
  exchange. Banana currently uses 62.5 microseconds.
- Each exchange uses the actual robot FK pose and finite-difference tool twist.
  The external tool is held at the supplied pose over those material substeps
  (upstream `tool_substep=hold`); this discrete coupling is recorded explicitly.
- ExternalToolPort returns the material-on-tool reaction averaged over the
  interval. Its moment is shifted from the tool origin to the material origin,
  transformed to the robot base, then shifted/rotated to the sensor origin.
  Thus the sensor moment includes the lever arm cross force.
- The BM wrench plus the robot's existing sensor signal enters the original
  tare/filter and FDCC feedback. It is **not also applied as an external MuJoCo
  wrench**: this retains the project's signal coupling, rather than claiming
  a fully coupled rigid-body contact simulation.
- A finite-wedge board interlock checks every 1 ms robot step using a
  conservative bound over the revolute-joint interpolation, including mid-step
  rotation. It projects unsafe motion to the last safe fraction and holds at
  contact. The reach bound is the sum of URDF link offsets plus the farthest
  mounted blade vertex; deviation from the endpoint chord is bounded by
  reach times squared total joint rotation divided by eight. Prismatic chains
  are explicitly unsupported. It prevents material-tool poses
  below the board; it does not synthesize a board reaction force.

## Independent task and learned state

The task identity is `blademaster_pose_v1`. DiSECt's constructor rejects it,
and WM state creation requires `physical_backend=mujoco_warp_blademaster`.
Training metadata checks prevent silently resuming DiSECt weights/replay.

Success retains pose arrival/counting, failure precedence and the finite board
boundary. It gives +100; force/workspace failure gives -100 once; finite-horizon
timeout terminates without bootstrap. Limits [50,50,50,5,5,5] in this example
are interface-development settings, not a verified Mini45 hardware threshold.

There is no spring-release fraction. The auxiliary progress is explicitly:
`p[t] = max(p[t-1], clamp(1 - blade_board_gap[t] / initial_depth_span, 0, 1))`.
It measures maximum blade travel toward the board, **not separation**.
The running reward keeps the v3 speed/load/jerk formula with this different
progress variable. Therefore numerical rewards are not a matched DiSECt
benchmark comparison.

The WM predicts joint position/velocity and wrench history/peaks; it computes
blade-depth progress from differentiable FK after the common board projection.
There is no learned fracture-progress output or privileged BM damage input.
The same existing BPTT trainer, analytic pose settlement, real-target critic and
BC option are reused. Policy inputs remain the 25-dimensional history and
eight-dimensional pose/speed goal.

## Run

Requires the remote reproduction source and the robot assets in
`backends/mujoco/runs`. The MPPI Python environment already contains SciPy,
needed by BladeMaster scenarios. Verified dependency versions: Python environment
Torch 2.4.1+cu121, Warp 1.18.0, MuJoCo 3.16.0, SciPy 1.18.1. The source path
is configurable; install this project's backend requirements and the external
reproduction's dependencies before using another environment. No BladeMaster
source is vendored by this change. The external reproduction at 4b2d983 has no
top-level LICENSE file; its vendored dependency licenses are under
`vendor/DexLab-LICENSE`, `vendor/libfranka/LICENSE`,
`vendor/mujoco_menagerie/franka_emika_panda/LICENSE` and
`vendor/serl_controller/LICENSE`. These do not establish a license for all of
the reproduction source; that provenance should remain explicit when sharing.

```sh
source runs/sac_env.sh
PYTHONPATH=.:backends/mujoco /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python \
  scripts/check_blademaster_backend.py --output runs/blademaster_ur5e_20261114 \
  --seconds 900
```

This bounded check runs physical reference motion, then collects separate real
six-dimensional actions for replay, updates WM/actor/critic, saves/restores
weights, optimizer state and replay, and performs a resumed update.
The reference trajectory bypasses the action mapper and is **not** inserted
as fake policy-action training data.

The general entry point is `scripts/train_gpu.py --blademaster-config
configs/blademaster/banana_ur5e.json`; `--conditions` is mutually exclusive.
Do not launch a long run from this example without a budgeted plan.

## Validation

GPU7 shared-GPU validation, 2026-11-14:

- Real reference episode: 37 steps to pose completion, material peak 0.809 N,
  final blade-board gap 75.4 micrometres; no force failure or board penetration.
  This is pose completion, not a material-separation measurement.
- 48 real mapped actions collected separately: peak material force 0.856 N.
- Tiny-network integration check: 3 WM optimizer updates, 1 actor update and
  2 critic minibatch updates changed all three networks with finite parameters.
  State dimension 890; learned physical delta dimension 36. Saved optimizer
  counts/replay restored exactly; one further WM update completed.
- 2-world masked reset: all unselected observations and material x/v/F/Fp
  unchanged exactly; selected material position reset error 0.138 micrometres;
  subsequent batched step completed.
- General train_gpu entry: one real iteration completed and saved last.pt plus
  replay.pt, with the separate BladeMaster task metadata.
- 14 focused tests passed: existing pose/episode-log tests plus BM analytic
  progress, BPTT, backend identity, logging, wrench virtual-work and swept-board
  regression (safe endpoints with an unsafe interior arc).
- The lifecycle/training-stage regression tests also passed (36 tests).
- After adding the 1 ms swept guard, the actual reference was rerun: pose
  success at step 37, peak 0.8086 N, final board gap 75.2 micrometres. Evidence:
  `runs/blademaster_ur5e_20261114/swept_reference.json`.
- Full root regression was stopped at the parent's request to avoid contention:
  143 passed, no failures before interruption. Parent runs full root/backend
  suites after integration; this partial run is not a full-suite pass.

Evidence: `runs/blademaster_ur5e_20261114/{report,task,masked_reset}.json`,
`checkpoint.pt`; general entry under `runs/blademaster_entry_20261114/`.
Timings were shared-GPU and are not throughput measurements. Passing tests
verifies adapter execution, not calibrated forces, reliable separation or learned
policy performance. Other presets have loaders but have not received this physical
integration check; the checked runnable example is banana.

Integrated validation in the shared cutting worktree: **366 root tests passed, 5 skipped; 38 backend tests passed**. Logs: `runs/half_potato_integration_verified_root_20261114.log` and `runs/half_potato_integration_backend_20261114.log`. The earlier partial agent suites are not the final acceptance record.
