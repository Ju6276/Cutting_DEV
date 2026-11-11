# Real-robot calibration (real2sim) plan — draft, awaiting approval

Goal: make the simulated benchmark (UR5e + FDCC compliance controller + DiSECt
material) match the real setup closely enough that (1) material parameters can be
fitted from real cutting data, (2) policies trained in simulation can be evaluated
on the real robot with the same metrics, and (3) the remaining sim–real gap is
measured, not assumed. Nothing here runs before the open items are settled.

## Hardware and open items

| Item | Status |
|---|---|
| UR5e | Available; same arm model as the simulation. Primary platform. |
| Diana7 | Available; not used in the first pass (see "Later"). |
| ATI Mini45 | Available. **Open:** calibration (SI-145-5: ±145 N Fx/Fy, ±290 N Fz, 1/16 N resolution; SI-290-10: ±290/580 N, 1/8 N), interface (Net F/T or DAQ), ROS version. The task force limit is not the inherited 50 N; it is set from this sensor's calibration, its mounting (axes, distance to the blade) and the experiment requirements (PLAN). |
| Knife and mount | **Open:** CAD or measured geometry and mass, to replace the simulated knife frame and tool payload. |
| Object placement | The multi-object task holds each object by its bottom 6 mm except near the cut plane (scanned round objects slid away otherwise; RESULTS 2026-11-11); the benchmark prism has a static bottom layer. **Open:** the real holding method (fixture, stop or board friction) that matches this. |
| Foods / specimens | **Open:** proposal below. |

## Steps (one change each)

### R0 — Software path and geometry

- UR5e via the Universal Robots ROS driver and the FZI `cartesian_controllers`
  FDCC implementation (Scherzinger et al., IROS 2017), with the same
  `configs/public_prism/controller.json` parameters as the simulation.
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
  UR5e here (deferred from the simulation work).

Decision: simulated and real tracking agree within a tolerance set after the
first repeated real measurements (real repeat spread is the reference).

### R3 — Material calibration

- **Specimens.** Start with a controlled phantom (gelatin of fixed concentration,
  close to the jello config) plus two foods with different stiffness (proposal:
  potato and apple, which have published compression data and DiSECt
  calibrations). Cut specimens with a mould to the simulated prism dimensions and
  rest them on the board unclamped, as in the free-bottom benchmark.
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
  speeds and on the slicing motions. Gradient fitting needs a static bottom (free
  bottom gives NaN adjoints); fitting with a static bottom against free-resting
  real specimens is a mismatch to report, or a gradient-free fit is needed.
- **Variability.** Real repetitions define the noise band. The randomization
  ranges for the multi-object stage (PLAN, P2/P3) come from the fitted
  parameter spread across specimens, not from guessed multipliers.

Decision: validation force-profile error within the real repeat spread for each
material; otherwise report the gap and which phase (pressing, fracture, friction)
it comes from.

### R4 — Closed-loop evaluation on the real robot

- Run the reference trajectory, then frozen policies (the current working
  baseline and SAC from the unified environment, step 85000; names as in
  BENCHMARK) on calibrated materials, with a workspace bound and an emergency
  stop. **Open:** how the force limit is detected and enforced on the real robot
  (sensor, filter, rate, stop mechanism, reaction time); no hardware guarantee is
  assumed until this is measured.
- Metrics follow BENCHMARK "Task definitions" (success = goal-pose counter,
  terminal reasons separately); material separation is recorded as its own
  measurement, not as success. Also steps/time, sensor peak and mean force,
  limit exceedances. Compare with the simulation of the same calibrated material.

## Later

- Diana7: a second platform for transfer, or a second arm holding the object.
  Holding with a compliant arm adds its own dynamics (Mitsioni et al., 2019).
- Skin or hard-core objects (low priority, per the user).

## References

- Heiden et al., DiSECt, RSS 2021 — [arXiv 2105.12244](https://arxiv.org/abs/2105.12244); extended version [arXiv 2203.10263](https://arxiv.org/abs/2203.10263)
- SliceIt! (ICRA 2024) — [arXiv 2404.02569](https://arxiv.org/abs/2404.02569), [code](https://github.com/omron-sinicx/sliceit)
- Jamdagni and Jia, robotic slicing of fruits and vegetables (ICRA 2021) — [project page](https://robolab-iastate.github.io/ResearchRoboticCutting)
- ATI Mini45 calibrations — [ATI](https://www.ati-ia.com/Products/ft/ft_models.aspx?id=Mini45)
- FDCC and `cartesian_controllers` — [arXiv 2009.11888](https://arxiv.org/abs/2009.11888) (follow-up); UR10e evaluation [arXiv 2210.13421](https://arxiv.org/abs/2210.13421)
- Wrist F/T calibration with total least squares — [arXiv 1904.06158](https://arxiv.org/abs/1904.06158); payload identification [Sensors 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8777628/); [force_torque_tools](https://index.ros.org/r/force_torque_tools/)
- Mitsioni et al., data-driven MPC for food cutting — [arXiv 1903.03831](https://arxiv.org/abs/1903.03831)
