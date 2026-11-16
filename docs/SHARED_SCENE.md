# Shared cutting scene — 2026-11-16

Authoritative input: [scene_v1.json](../configs/shared/scene_v1.json).
Both backend configurations are generated from this file; do not independently
edit their generated JSONs. Case sampling is identical in checks and eventual
training. Old runs retain their frozen configuration and are not paired runs.

## Starting the cut as on the intended robot

1. Place the object, let it settle using its actual material parameters.
2. Establish the board frame, cut location and object height.
3. Position the actual mounted blade above that individual object with a sampled
   5–30 mm clearance. This is free space, not placement in contact with food.
4. Cut using motion and force history. No camera image is supplied to the policy.

Current implementation uses the settled **whole object's highest physical particle**
as a conservative surface-height proxy. It is not a local surface estimator.
A real robot would obtain placement/height before cutting from perception or
measurement. Camera localization, its errors, and calibration are NOT implemented
or validated by this reset helper. The 5–30 mm distribution is the existing
engineering training range, not a measured best real-robot clearance.

A case is identified by object name, case seed and world index. Independent seeded
sampling generates explicit per-world clearances before either backend starts;
changing backend RNG use or object-group order does not change a case.
The sampled gap stays fixed for that world. After reset/settling the knife is
repositioned to the new surface height with that same gap. Other worlds retain
their state exactly. This follows the former main run's once-per-world sampling;
it does not mean a new gap is sampled every episode.

There is no separate fixed-5-mm debug task. Checks replay selected cases from
this same definition. Small worlds_per_object is a resource setting, not a
different initialization/reward implementation. Material state is not teleported
to fit the knife. If the requested start is outside the robot workspace, setup
fails rather than expanding the workspace silently.

## Complete shared configuration

| Item | Both backends under scene_v1 |
|---|---|
| Objects / allocation | Cucumber 003, flat-bottom half potato, prism; equal world count per object |
| Object sampling | Fixed geometry group per world; one shared learner, not sequential object training |
| Robot and controller | Same UR5e asset, FDCC parameters, action mapping and sensor frame |
| Physical knife / mount | disect_knife_rigid_mount_v1; same finite blade and rigid EE transform |
| Cut location / orientation | Prepared cut location; nominal vertical knife, no automatic cut-site search |
| Start | Per-object settled top + uniform 5–30 mm; paired explicit cases |
| Settling | 1.0 s simulated time with that episode's actual material parameters |
| Goal | Shared board-aligned prescribed pose, final XY radius plus z/pitch tolerances and arrival counter |
| Board | Height 0; finite-blade stop at 1 micrometre; support plane has no finite edge |
| Workspace | EE in robot-base frame: x[.038,.238], y[.405,.605], z[.096,.269] m |
| Force / torque limits | [50,50,50,5,5,5], componentwise; simulation settings, not Mini45 calibration |
| Observations | 34 values per frame, full six-component sensor wrench; history length is learner configuration |
| Outputs | Six network outputs; pitch disabled, five effective channels |
| Pitch option | Default off; when explicitly enabled, target angle ±5 degrees, rate ≤5 degrees/s |
| Final position | Robot EE XY distance from prescribed target <40 mm, plus original height/pitch/arrival conditions |
| Optional path constraint | task.tool_motion.enabled: false = final only (default); true = EE XY distance 25 mm free / 60 mm failure / weight 20 |
| Reward | pose_depth_v2: geometric maximum-depth progress 10, speed 20, load 1, jerk .1; success +100, failure −100 |
| Completion | Pose arrival without higher-priority failures; material separation recorded independently |
| Episode / discount | 180 control steps / 9 s; gamma 1 |
| Target speed | Uniform 1–5 cm/s, resampled per world per episode; both reference controllers use it |
| Young's modulus / density | Per-episode log-uniform scales .5–2 / .8–1.25 |
| Cutting resistance | Per-episode log-uniform scale .5–2: DiSECt spring stiffness, BladeMaster cutting energy |
| Friction coefficients | Board .2, knife .2; contact laws still differ; no calibration claimed |
| Nominal cucumber | E=2.6 MPa, nu=.37, density=950 kg/m³ |
| Nominal half potato | E=2 MPa, nu=.45, density=1101 kg/m³ |
| Nominal prism | E=.04 MPa, nu=.45, density=830 kg/m³ |

Workspace and tool-position thresholds are **fixed physical distances**, not scaled
with object size. Rendered board edges are not a physical boundary. The current
configuration has no particle-displacement penalty: fragment opening is not charged.
Tool distance is measured from the prescribed goal in robot-base XY, not from a
moving object or from the rendered board edge. Both modes retain force/workspace/board limits.

## Explicit differences and remaining work

- Material physics: FEM / pre-inserted cutting springs versus MPM / native damage.
  Equal E, density or friction coefficients do not imply equal resulting forces.
- Geometry discretization: each backend samples the same intended object with its
  own particle/discretization scheme; settled heights/deformation can differ.
- Material timesteps differ. Robot/material exchange is 2 ms; DiSECt interpolates
  tool motion within that interval, BladeMaster currently holds the supplied pose.
- Material episodes use the same per-world deterministic sampler in both backends.
  Each reset draws E, density, a cutting-resistance multiplier and target speed.
  Cutting-spring stiffness and cutting energy retain their different meanings;
  matching their scale distributions does not equate fracture physics.
- BladeMaster now applies per-world GPU coefficient buffers, restores selected
  worlds to their raw state and settles them with the new material. It does not
  reuse a nominal-material settled snapshot. The group timestep is chosen for
  maximum E / minimum density so changes do not require new captured graphs.
- Checkpoints include sampler configuration, next-episode counters and values.
  Resume continues the streams and starts fresh physical episodes, following the
  existing trainer's reset-on-resume convention.
- A long run still needs resource/capacity and budget selection, plus online W&B.
- Enabled pitch has mapping/WM gradient tests, but active tilted blade/material/
  board-contact validation remains pending. This turn probes vertical reference
  motion with pitch off; the sawing action mapping retains regression coverage.

## Preparation and replay

Run on GPU7 in /dataHDD/dong/Projects/Cutting_DEV_board; keep assets on servers.
The preparer preserves original scenes, exports CUDA-resident variants with shared
friction coefficients, records scene/geometry checksums, and writes the two backend
configs. Hash checks reject edits to prepared assets or a mismatched contract.

~~~bash
source runs/sac_env.sh
export PYTHONPATH=.:backends/mujoco
PY=/dataHDD/dong/Envs/cutting-mppi-20261108/bin/python
$PY scripts/prepare_shared_scenes.py --contract configs/shared/scene_v1.json --output runs/NEW_SHARED_SCENE/assets
$PY scripts/check_pose_depth.py --backend disect --shared-contract configs/shared/scene_v1.json --prepared runs/NEW_SHARED_SCENE/assets/manifest.json --output runs/NEW_SHARED_SCENE/disect.json --seconds 900
$PY scripts/check_pose_depth.py --backend blademaster --shared-contract configs/shared/scene_v1.json --prepared runs/NEW_SHARED_SCENE/assets/manifest.json --output runs/NEW_SHARED_SCENE/blademaster.json --seconds 900
~~~

Reports contain requested and actual blade gaps, settled tops, EE/goal poses,
loaded friction, pitch settings, observation width, physical-versus-WM reward
agreement and exact preservation of non-reset worlds. Numeric results belong in RESULTS.


## Episode case replay

The shared episode_conditions block defines the ranges and seed. A draw is keyed
by object name, world index and that world's episode counter. Resetting world A
does not consume world B's random stream. Per-episode parameters, speed and counter
are recorded in episode logs; they are not policy or WM material inputs.
For the same case and episode number, both backends receive the same multipliers
and speed. Native nominal fracture parameters remain backend-specific.

## Two tool-position modes

Set common.task.tool_motion.enabled to false (final only) or true (path and final),
then regenerate both backend inputs. The prepared manifest checks the entire
contract, including this switch. Final XY radius remains 40 mm in both modes;
this extends the former 40 mm single-axis criterion to the horizontal plane.
Height tolerance 2 mm, pitch-error tolerance .04 rad and >10 qualifying arrival
samples are retained. Qualifying samples are cumulative, not necessarily consecutive.
When path control is off, passing 60 mm alone neither adds cost nor ends the task;
the existing robot workspace still applies. Failure to return to the final region
before the horizon gives timeout, not a successful cut.

With path control on, each 50 ms endpoint subtracts
20/180 * clamp((distance-.025)/(.060-.025),0,1)^2.
Above 60 mm the task ends with failure -100 once; exactly 60 mm remains inside.
These are configurable engineering settings. No new policy has been trained
with either revised final-position mode.

Reward-equation audit: append --tool-reward-audit to either check_pose_depth.py
command above. It evaluates both switches using measured transitions and
counterfactual goal offsets; it does not assert learned motion improvement.
