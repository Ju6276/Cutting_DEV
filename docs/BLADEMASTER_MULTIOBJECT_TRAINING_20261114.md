# BladeMaster three-object engineering training — 2026-11-14

## Scope and approved budget

Question: can the existing WM + BPTT training loop run end to end with one shared
policy on three **initially intact** objects in the BladeMaster forward solver?

This is engineering training, not a matched comparison against the DiSECt run:
the physical backend, object preparation, progress variable, distribution and
world count differ. No new critic target network, reward or RL algorithm is added.

- GPU7 only; 3 worlds (one cucumber, one half potato, one prism), independent
  material solvers stepped sequentially, shared UR5e/WM/policy/critic code.
- Fixed nominal materials; 3 cm/s feed goal; one shared 55 mm initial blade-board
  gap. No new randomization mechanism in this first backend run.
- From scratch, seed 0; 500 WM-only updates, then real-target critic, h8 BPTT,
  BC alpha 0.25. Existing source architecture, effective batch sizes and optimizer
  settings are retained; microbatch 64 is storage chunking.
- Stop at **5,000 vector steps (15,000 physical transitions) or 24 hours**,
  whichever comes first. No automatic extension. Save every 250 iterations,
  plus final/interruption checkpoint and replay. Outer watchdog allows at most
  120 seconds for save/cleanup following its 24-hour termination request.
- Offline W&B/local progress and per-object episode logs; do not expose credentials.
- Final interpretation is based on pose completion, force events, timeouts,
  speed and native material components separately. No performance claim is made
  from the preflight or from whichever metric improves first.

The parent accepted this configuration and conditionally authorized starting
after physical/update/reset/resume checks and root/backend regression suites.
The run must stay on its launched checkout/version.

## Intact geometry and units

The sampling input is the original volume mesh, **not a DiSECt scene.pt**:
- Cucumber: OmniObject3D-derived uncut `cucumber_003.msh`.
- Half potato: the capped, flat-bottom preparation
  `potato_003_half_20261114.msh`; this is a prepared half food, not an untouched
  whole potato or a native BladeMaster object.
- Prism: original DiSECt public `prism.rst`, reading node IDs/volume elements
  directly. No pre-cut topology is imported.

`scripts/prepare_blademaster_mesh.py` samples uniform subcell-centre particles
inside the tetrahedral body. At grid spacing 6 mm, each particle volume is
(3 mm)^3, exactly the BladeMaster convention. The source y-up coordinates rotate
as [x,y,z] -> [x,-z,y], a proper rotation; horizontal bounding boxes are centred,
and the lowest source point lies on the board. Neither spring arrays nor cut
side labels enter the asset. All three initial MPM connectivity graphs have one
component and zero initial damage.

| Object | Particles | Sampling volume / source volume | Mass |
|---|---:|---:|---:|
| Cucumber | 6,258 | 0.992407 | 0.160518 kg |
| Half potato | 2,777 | 0.998391 | 0.082552 kg |
| Prism | 18,513 | 0.999702 | 0.414876 kg |

Assets and provenance/hashes live under
`/dataHDD/dong/Assets/Cutting_DEV/blademaster_intact_20261114/`.
They remain server-side and outside Git.

The external solver extension is commit `db06c1a` on
`/dataHDD/dong/Projects/BladeMaster-Warp-ur5e-mesh-20261114`.
`integrations/blademaster/0001-intact-reference-cloud.patch` reproduces it on
`4b2d983`. It adds only validated reference-point input to solver construction;
default analytic-object sampling is unchanged. No material update/contact/cutting
kernel is changed.

## Parameters, blade and task

Only Young's modulus, Poisson ratio and density transfer from the current scene
configs. These are **simulation candidates, not calibration on these scans**.

| Object | E | nu | density | BM substep |
|---|---:|---:|---:|---:|
| Cucumber | 2.6 MPa | 0.37 | 950 kg/m3 | 42.553 us |
| Half potato | 2.0 MPa | 0.45 | 1,101 kg/m3 | 35.714 us |
| Prism | 40 kPa | 0.45 | 830 kg/m3 | 62.5 us |

The dt is capped by the upstream banana half-CFL rule and divides 2 ms exactly.
Fracture energy 700 J/m2, yield stress 30 kPa, friction and remaining parameters
are inherited from the reproduction's paper-banana preset. DiSECt spring stiffness
is **not** converted into fracture energy.

The common wedge is 120 mm long and 80 mm tall (half dimensions
[0.0005,0.06,0.04] m). The old 80 mm edge could not cover the 100 mm prism, so
this explicit shared blade configuration replaces it for all three objects.
The loader checks initial width plus particle radius against edge length.

All worlds use the same fixed EE-to-blade transform, board and target:
initial EE XYZ approximately [0.138049,0.505173,0.221000] m, target
[0.138049,0.505173,0.166001] m. Full quaternions and finite-blade geometry are in
the run task metadata. The initial blade gap is 55 mm, hence clearance above
each object's top differs (not object-ID information fed to the policy).

`blademaster_pose_v1` retains +100 pose completion with failure precedence,
force/workspace/timeout termination and the swept finite-blade board interlock.
Force limits [50,50,50,5,5,5] are development task settings. The running progress
is analytic maximum knife depth, not material separation. Policy observations
remain history of 25 values plus the pose/speed goal; material identity is only
logged.

Native BM components and damage saturation are separately recorded at episode
end. In the inherited prescribed/external-tool mode (`free_blade=False`),
side-code assignment follows the traversed blade region without waiting for
every damage variable to reach 1. Thus mechanical splitting can coexist with
damage maxima below 1; neither is a calibrated fracture measurement. There is
no initial prepared cut plane.

## Preflight evidence

`runs/blademaster_multi_preflight_20261114/report.json`:
- All three reference episodes completed the pose task without force or
  workspace failure; cucumber/prism took 50 steps, half potato 52.
- Native final component sizes: cucumber [2762,3496], half potato [1368,1409],
  prism [9537,8976].
- Peak material reaction: 5.52, 8.71, 1.57 N respectively. Peak six-axis
  force-limit utilization: 0.137, 0.219, 0.075.
- Minimum material det(F) stayed above 0.986. End-state lowest actual particle
  centre was at least 1.423 mm above the board (contact radius 1.5 mm);
  this diagnostic uses BM material particles, not DiSECt ghost coordinates.
- 72 real mapped-action transitions; WM/actor/critic parameters changed with
  finite values; optimizer/replay restored and WM plus policy updates continued.
- Resetting one world left all other observations and material x/v/F/Fp/codes
  unchanged exactly.
- Mean reference physical step about 2.175 s; full preflight 181.8 s. This does
  not include full-size source-network training cost, so pure-physics estimates
  are not the promised completion time.
- Three-world initialization settles for 0.2 simulated seconds. Initial p99
  material speeds: cucumber 2.61 mm/s, half potato 0.0058 mm/s, prism 5.12 mm/s;
  this is a recorded transient, not a claim of exact rest.

A separate batching correction ensures that board projection of one world does
not zero other worlds' velocities while recomputing derived MuJoCo state.
`Robot.set_joint_positions(mask=...)` preserves unselected q/qd before forward;
its default behaviour remains unchanged. The DiSECt board path already used a
selected-world update and is not modified.

Schedule persistence is restored from `75d4022`; integer metric counting is
restored by commit `905afb9`. These affect resume and logging,
not loss functions or policy objectives.

## Commands and final state

```sh
source runs/sac_env.sh
PYTHONPATH=.:backends/mujoco /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python \
  scripts/check_blademaster_multi.py --output runs/blademaster_multi_preflight_20261114

PYTHONPATH=.:backends/mujoco /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python \
  scripts/train_gpu.py --blademaster-config configs/blademaster/three_intact_nominal.json \
  --output runs/blademaster_three_train_20261114 --max-iterations 5000 --max-hours 24 \
  --microbatch 64 --critic-microbatch 8192 --wm-only-updates 500 --critic-target real --actor-horizon 8 \
  --actor-bc-alpha .25 --checkpoint-every 250 --wandb-mode offline --seed 0
```

Validation completed before launch:
- Root suite: **392 passed, 5 skipped**, including multi-object geometry tests.
- Backend suite: **39 passed**, including actual two-world masked placement.
- External reference-cloud regression: **1 passed**.
- Logs: `runs/bm_multi_root_tests_20261114.log` and
  `runs/bm_multi_backend_tests_20261114.log`.
- The preflight Torch allocation counter excludes Warp/driver allocations; it
  must not be reported as total GPU memory.
- Initial particle speed quantiles do not establish centre-of-mass motion or
  exact equilibrium. No COM velocity snapshot was saved; the definition remains
  0.2 seconds of actual settling, without claiming complete rest.

The main entry saves a final checkpoint/replay in its training-finalization
block and closes the environment in an enclosing `finally`. Both the internal
wall-clock check and SIGTERM stop request are checked between learner updates;
the external watchdog's 120-second kill grace is for saving and closing, not
additional training. Launch identity and initial counters follow in a separate
run record.

## Launched run

- Frozen training/backend code: `cf20ef0` (clean at startup), on top of
  `905afb9`; masked robot fix is the preceding `600d470`.
- External BladeMaster source: `db06c1a`, branch
  `codex/ur5e-intact-mesh` in `Ju6276/BladeMaster-Lab`.
- Host: `cutting-gpu7`; Python PID **1019882**, watchdog PID **1019881**.
- Start: 2026-11-14 06:17 server log time.
- Run: `runs/blademaster_three_train_20261114/`.
- Launcher and stdout: `runs/blademaster_three_train_20261114.launch.sh` and
  `runs/blademaster_three_train_20261114.log`.
- Initial live check: 9 vector steps, 27 physical transitions, all three worlds;
  no policy updates yet (the required 500-update WM stage precedes joint updates).
- Driver process memory at initialization: **1,748 MiB** from
  `nvidia-smi --query-compute-apps`. This is not a promised peak for joint training.
- Main-entry save/replay resume smoke passed separately: iterations 1 -> 2,
  transitions 3 -> 6, final status completed in both runs, under
  `runs/blademaster_multi_entry_20261114` and
  `runs/blademaster_multi_entry_resume_20261114`.
- Launch budget and command are exactly the approved command above, including
  explicit critic microbatch 8192. There is no automatic continuation beyond
  the first reached 5,000-vector-step or 24-hour limit.

This is a launch record, not a training-success or convergence report.

First full-size WM update verified: iteration 42, 126 transitions, 10 WM
optimizer steps, state loss 0.03954, terminal BCE 0.61515, total loss 0.65469,
gradient norm 6.3162, terminal accuracy 0.752997. All are finite; iteration 44
then began the next WM update. First update took 179.3 seconds including Torch
compilation. Subsequent minibatches were roughly 8 seconds each, so the 500-WM
pretraining stage can take roughly 12 hours; this is an early estimate, not an
extension of the fixed budget. Driver memory reached 2,368 MiB during this WM
update; joint-training peak remains unmeasured.
