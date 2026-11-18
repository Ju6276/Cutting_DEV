# Real-robot deployment interface and calibration plan

**2026-11-17 hardware update:** the first physical platform is UR10e with an
external ATI Mini45-E, user-confirmed SI-580-20 calibration, and a Mech-Mind camera
(model pending). Start with the [UR10e FDCC setup handoff](UR10E_FDCC_SETUP_ZH.md):
no-knife motion and compliance tests, without RL, MPC or vision. This is a plan,
not completed hardware validation. Later stages below are conditional.

Goal: make the simulated benchmark (UR5e + FDCC compliance controller + DiSECt
material) match the real setup closely enough that (1) material parameters can be
fitted from real cutting data, (2) policies trained in simulation can be evaluated
on the real robot with the same metrics, and (3) the remaining sim–real gap is
measured, not assumed. Nothing here runs before the open items are settled.

## Deployment interface — updated 2026-11-17

The current shared pose_depth_v3 policy interface uses a policy with ten 34-value frames
and an 8-value goal (position3, quaternion4, commanded speed1). Per frame:
selected pose error3, velocity3, jerk3, previous action6, three samples of the
six-axis normalized sensor wrench18, remaining episode time1. The policy has
six outputs: two stiffness settings, two controller gain settings, accumulated
sawing target displacement and optional pitch. Pitch is currently disabled.
The outputs pass through the same SlicingActionMapper and FDCC semantics;
they are not joint torques or a six-axis Cartesian command.

Deployment path: robot joint/FK state + compensated sensor wrench -> timestamped
filter/history and goal construction -> frozen policy -> action mapper ->
Cartesian compliance controller -> UR hardware interface. Policy updates are
50 ms in the present simulation. Hardware controller rate, latency, filtering
and stop response must be measured and matched; no 1 kHz real-driver claim.
Only the policy is needed online for this route. WM and critic train the policy;
they need not be deployed. Image input is absent; pre-cut localization may
supply the board frame, desired cutting location and estimated object height.

The repository has simulation inference and pinned SliceIt/ROS integration
sources, but no completed, validated current-policy real-robot deployment runner.
The physical URDF/mount/sensor frames, gravity compensation, controller parameter
mapping, communication timeout and real board/force stop remain deployment work.
The simulated finite-blade stop must not be copied as a claim that real contact
can be undone or that hardware stopping distance is zero.

User-selected modes: constrain final knife/robot position only, or additionally
constrain path deviation. Both now use robot-base XY EE distance to the prescribed
goal (final radius40 mm; optional path25 mm free/60 mm failure), observable from
calibrated robot kinematics. Particle/fragment opening is no longer a penalty in
the shared configuration. These defaults still require physical validation;
they are not sensor calibration or guarantees of real stopping distance.

Controller references: [FZI Cartesian controllers](https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers),
[UR hardware interface](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/rolling/doc/ur_robot_driver/ur_robot_driver/doc/hardware_interface.html).
The matching ROS version and controller integration are selected during deployment,
not by mixing the historical ROS1 patch set with ROS2 commands.

## Hardware and open items

| Item | Status |
|---|---|
| UR10e | User-confirmed physical platform; simulation currently uses UR5e. Robot calibration, tool installation and controller transfer require validation. |
| Mech-Mind camera | User-confirmed; model and extrinsics pending. Not required for initial FDCC tests. |
| ATI Mini45-E | User-confirmed SI-580-20: Fx/Fy ±580 N, Fz ±1160 N, Mx/My/Mz ±20 N·m. Installed calibration file, acquisition interface and mounting transform remain to verify. Sensor ranges are not task stop limits; simulation 50 N / 5 N·m is not a hardware prescription. |
| Knife and mount | **Open:** CAD or measured geometry and mass, to replace the simulated knife frame and tool payload. |
| Object placement | Current task uses freely placed objects. Measure the real support/friction and initial pose; historical bottom-held contracts remain separate. |
| Foods / specimens | **Open:** proposal below. |

## Prerequisite: force quantity and completion

Current shared scenes use net contact force/moment in the sensor frame, with
corrected contact coordinates and timing (METHOD). The scalar proxy belongs to
historical configurations. Interface tests do not calibrate the physical sensor
or mount. Current completion is prescribed pose arrival without higher-priority
failure, using the board-aligned task; material separation remains independent.
Neither a prescribed spring-release fraction nor board height proves separation.

## Steps (one change each)

### R0 — Software path and geometry

- UR10e via the matching Universal Robots ROS 2 driver and the FZI `cartesian_controllers`
  FDCC implementation (Scherzinger et al., IROS 2017). Identify and validate
  parameters on this hardware; do not copy UR5e simulation gains as approved settings.
- Record the knife-tip and sensor frames on the real tool (CAD plus a touch-off
  measurement) and update the simulated chain only if they differ.
- Measure the command-to-motion latency and the F/T sampling rate.
- Recorder: write real episodes with the same fields as simulated episodes
  (joint state, wrench history, commands, targets, timestamps), so the existing
  episode metrics and value analyses read them unchanged.

Decision: proceed when the robot executes the reference knife trajectory in air
under the compliance controller and the recorder output loads in the benchmark
metric code.

### R1 — Force/torque sensor calibration

- Bias and tool payload: hold the tool still in 12 or more orientations, then
  fit the bias, tool mass and centre of mass by least squares. Wrist-sensor
  studies report that ordinary least squares is noise-sensitive because the
  measured force appears in both regressor and target, and use total least
  squares instead (arXiv 1904.06158). The ROS `force_torque_tools` package
  implements the least-squares variant.
- Validate the gravity compensation with known masses at new orientations.
- Noise floor at rest and during free motion. Compare with the simulated wrench
  filter and tare (`tensor_filter_controller`); adopt the real filter in the
  simulation only if they differ.

Decision: residual force after compensation, in free motion, is below a
threshold fixed from the measured noise floor before any cutting data is used.

### R2 — Robot and controller real2sim

- Free-space step and ramp commands in each Cartesian axis; pressing into a
  rigid surface at several target forces. Repeat the same commands in the
  simulation.
- Compare end-effector trajectories and contact wrench responses; identify
  latency and any robot-model mismatch. Align joint torque limits with the real
  UR10e here (deferred from the simulation work).

Decision: simulated and real tracking agree within a tolerance set after the
first repeated real measurements (real repeat spread is the reference).

### R3 — Material calibration

- **Specimens.** Start with a controlled phantom (gelatin of fixed concentration,
  close to the jello config) plus two foods with different stiffness (proposal:
  potato and apple, which have published compression data and DiSECt
  calibrations). Cut specimens with a mould to the simulated prism dimensions.
  Current simulation direction is free placement with passive sliding. The old
  bottom-held fixture proposal is retired. Establish the real support/friction
  and safe tool/board boundary before claiming matching sim-to-real conditions.
- **Measurements per material.**
  1. Slow indentation/compression with the robot and the Mini45, giving an
     elastic-modulus prior. Food is viscoelastic and moduli depend on geometry
     and loading rate, so this is used as a prior only.
  2. Vertical cuts at three constant speeds, at least 5 repetitions each.
  3. Slicing (sawing) motions for validation only.
- **Fit.** Infer DiSECt parameters by gradient descent on the force-profile
  error, as in DiSECt (Heiden et al., RSS 2021: apple MAE 0.253 N, potato
  0.379 N on the Jamdagni and Jia data). SliceIt reports that one real slice per
  food item suffices for its fit. Here: fit on one speed, validate on held-out
  speeds and on the slicing motions. Earlier free-bottom optimization trials produced
  nonfinite adjoints; that observation does not establish that all gradient fitting
  requires a static bottom. Use matching fixture conditions and validate the chosen
  fitting procedure before adopting it.
- **Variability.** Real repetitions define the noise band. The randomization
  ranges for a future multi-object task come from the fitted
  parameter spread across specimens, not from guessed multipliers.

Decision: validation force-profile error within the real repeat spread for each
material; otherwise report the gap and which phase (pressing, fracture, friction)
it comes from.

### R4 — Closed-loop evaluation on the real robot

- Run the reference trajectory, then frozen policies (to be selected from the
  version-compatible WM+BPTT runs and the vector SAC baseline; the old
  BENCHMARK SAC at step 85000 is historical) on calibrated materials, with a workspace bound and an emergency
  stop. **Open:** how the force limit is detected and enforced on the real robot
  (sensor, filter, rate, stop mechanism, reaction time); no hardware guarantee is
  assumed until this is measured.
- Metrics follow the current task definition in TASK_FREE_CUT (success = goal-pose counter,
  terminal reasons separately); material separation is recorded as its own
  measurement, not as success. Also steps/time, sensor peak and mean force,
  limit exceedances. Compare with the simulation of the same calibrated material.

## Later

- A second arm or another robot is outside the initial UR10e FDCC validation.
  Holding with a compliant arm adds its own dynamics (Mitsioni et al., 2019).
- Skin or hard-core objects (low priority, per the user).

## References

- Heiden et al., DiSECt, RSS 2021 — [arXiv 2105.12244](https://arxiv.org/abs/2105.12244); extended version [arXiv 2203.10263](https://arxiv.org/abs/2203.10263)
- SliceIt! (ICRA 2024) — [arXiv 2404.02569](https://arxiv.org/abs/2404.02569), [code](https://github.com/omron-sinicx/sliceit)
- Jamdagni and Jia, robotic slicing of fruits and vegetables (ICRA 2021) — [project page](https://robolab-iastate.github.io/ResearchRoboticCutting)
- ATI Mini45 calibrations — [ATI](https://www.ati-ia.com/Products/ft/ft_models.aspx?id=Mini45)
- FDCC and `cartesian_controllers` — [arXiv 2009.11888](https://arxiv.org/abs/2009.11888) (follow-up); UR10e evaluation [arXiv 2303.12440](https://arxiv.org/abs/2303.12440)
- Wrist F/T calibration with total least squares — [arXiv 1904.06158](https://arxiv.org/abs/1904.06158); payload identification [Sensors 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8777628/); [force_torque_tools](https://index.ros.org/r/force_torque_tools/)
- Mitsioni et al., data-driven MPC for food cutting — [arXiv 1903.03831](https://arxiv.org/abs/1903.03831)
