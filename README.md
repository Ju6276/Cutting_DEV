# Cutting DEV

Force-conditioned food cutting: learned world model + BPTT policy,
UR10e / ATI Mini45 / IKEA knife, with DiSECt or native BladeMaster material physics.
Current task: pose_depth_v3 / project_continue_v2, cumulative pose arrival;
no permanent board freeze and no simulated board reaction.

## Documentation

[Start here](docs/README.md). Five daily guides:

- [Research](docs/RESEARCH.md): question, paper comparisons, next work.
- [Method](docs/METHOD.md): algorithms, inputs/outputs, task, rewards and scenes.
- [Operations](docs/OPERATIONS.md): machines, dependencies, commands and data.
- [Hardware](docs/HARDWARE.md): UR10e, ATI, Nano, real deployment and calibration.
- [Results](docs/RESULTS.md): dated measurements and evidence.

WM+BPTT is compared with SAC and frozen WM+MPPI. Analytic MPC remains an
engineering/controller-chain test tool; all methods use the shared FDCC execution layer.
Five foods: cucumber, half potato, peeled banana, tofu and luncheon.
Pose completion does not certify material separation.

Code: cutting_world_model/ (learning/task), backends/mujoco/ (robot/physics),
scripts/ (entries), configs/ (contracts), integrations/ (source provenance),
[hardware/ur10e_ros1/](hardware/ur10e_ros1/README.md) (real UR10e controller and air commissioning).
Start from the [runtime guide](docs/OPERATIONS.md); assets and runs remain outside Git.
[Sources](docs/reference/SOURCES.md) · [Licenses](NOTICE.md).
