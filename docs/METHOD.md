# Method and implementation choices

| Component | Source | Cutting implementation and reason |
|---|---|---|
| Training method | SuperMimic, Figure 2, Eq. 1–3, Algorithm A.1 | Fit state-transition windows, then optimize a policy through a frozen model and return to physical simulation for new data. Policy updates follow prediction checks. |
| Dynamics network | SuperMimic-Reproduced `ba610534`, `world_model/dynamics.py` | Import the original residual MLP with its three 2048-unit SiLU layers and zero final initialization. Inputs and outputs use the full cutting state instead of G1 features. |
| State update | G1 implementation predicts dynamics residuals with analytic rigid-body integration | Initial cutting adapter predicts additive residuals of particle position/velocity, spring stiffness, knife pose/twist and force. It reports physical-unit errors and invalid spring/quaternion predictions directly. |
| Normalization | Cutting adapter | Fit mean and standard deviation on training trajectories. Exactly constant features use scale 1; held-out trajectories do not enter normalization. |
| Physical data source | SliceIt DiSECt `db7de0f`, public prism configuration | Use existing FEM, contact and spring damage formulas, Gazebo knife geometry and the previously aligned initial height. Physics dt is 1e-5 s and control dt is 0.05 s, matching the running SliceIt bridge and agent. |
| Cut topology | DiSECt preinserts the cut plane and connects its sides with weakening springs | Keep that topology and include remaining spring stiffness in the state. Every dataset contains its connectivity and initial spring damping. |
| Action interface | SliceIt SAC controls trajectory/compliance; DiSECt accepts knife pose/twist | First fit the knife–object dynamics using prescribed twist. Mapping the six SAC actions through the robot and controller is a subsequent adapter. |
| Coupled data | SliceIt environment and public ROS bridge | A separate Gazebo/DiSECt instance executes the original six actions and records before/after states with the original reward and termination. Controller history and force history are recorded alongside the 24-D observation. The snapshot service adds recording only. |
| State representation | SuperMimic uses simulator state and context | Record accessible physical and controller state. Internal Gazebo solver and C++ controller buffers are not exposed in this representation; held-out multistep prediction measures their effect. |
| Geometry coverage | G1 has fixed robot dimensions | First use one fixed prism discretization. Additional geometry/material coverage requires corresponding data and a geometry-conditioned representation. |

References: [SuperMimic paper](https://arxiv.org/pdf/2610.28480),
[implementation](https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2),
[DiSECt](https://github.com/NVlabs/DiSECt),
[SliceIt](https://github.com/omron-sinicx/sliceit).
