# Backend and source mapping

The task's `UR3eDisectSlicingEnv-v1` identifier is a legacy class name. Its current `b_bot` XML uses **UR5e** geometry and inertia, matching the robot specified by SliceIt. This importer reads that exact XML from the running task capture and preserves the six named arm joints, fixed attachments, collision meshes, mass, COM and compound rigid-body inertia.

| Item | Current SliceIt / source | This backend probe | Reason |
|---|---|---|---|
| Robot source | `task.json:robot_xml`, `b_bot_base_link` → `b_bot_knife_center` | Same UR5e subtree, expressed in base frame | Keep the actual robot while evaluating the GPU backend |
| Physics | Gazebo robot + DiSECt material | MuJoCo Warp 3.16.0 robot; Warp 1.18.0 | Same versions as SuperMimic lock at `ba61053`; direct physics API isolates backend cost |
| Fixed inertia | Knife link has principal inertia `[0, 1e-6, 1e-6]` | Exact fixed-group mass/COM/inertia aggregation; no generated mesh inertia | MuJoCo requires positive inertia eigenvalues; rigid-group compound inertia is positive |
| Scene | ROS workcell and deformable cutting object | Arm, gripper, camera attachments and knife collision geometry, fixed base | First robot stepping and FK stage |
| Controls | Cartesian compliance, six slicing actions, 50 ms agent / 2 ms robot | Six direct motor torques, 0.1 Nm sinusoid, 2 ms physics | Verify joint/control mapping without replacing the original controller with invented PD |
| Integration/contact | Gazebo integration/contact settings | MuJoCo implicitfast, native mesh collision and defaults | Backend comparison; controller/force coupling requires a separate implementation |
| Learning gradients | SuperMimic BPTT through learned world model | Physical GPU simulator supplies transitions | MuJoCo Warp automatic differentiation is currently unavailable |

For fixed bodies expressed in one rigid group's frame, combine `M = Σm`, `c = Σ(m p)/M`, and `I = Σ(R I_local Rᵀ + m (||p−c||² Id − (p−c)(p−c)ᵀ))`. All original fixed link transforms and collision geometry remain. The importer disables inertia inference from meshes, so massless attachment geometries do not acquire new physical mass. Binary MJB storage preserves the compiled transform precision.

Sources: [MuJoCo static body fusion and inertia rules](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-fusestatic), [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), [mjlab architecture](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html), [pinned mjlab dependencies](https://github.com/mujocolab/mjlab/blob/8bb2bc9e187797689ec82c78c35019afd2fd0376/pyproject.toml).

# Full cutting batch integration

## Single-world source coupling

`cutting_gpu_backend/controller` ports FZI FDCC at `f4359b9`: virtual inertia,
20 ms internal integration, 0.9 internal velocity decay, position commands,
and original action mapping. Hardware updates are 1 ms, material exchange
2 ms, and agent actions 50 ms. The 2 ms torque benchmark above is separate.

| Item | Source | Coupled runtime |
|---|---|---|
| Position execution | Gazebo `SetPosition(..., true)` plus ODE zero-velocity motor with `fmax=URDF effort` | Assign qpos, retain generalized qvel, and use MuJoCo bounded friction constraints with the same effort bounds; reference acceleration `-qvel/dt` |
| Motor constraint | ODE velocity constraint, effort `[150,150,150,28,28,28]` Nm | MuJoCo `solref=[0,-1/dt]`, impedance 1 subject to MuJoCo's built-in maximum 0.9999; six extra scalar constraints |
| Self collision | ODE excludes joint-connected rigid bodies; task enables other arm self collisions | Explicit exclusions across adjacent fixed-body groups, preserving all non-adjacent collisions |
| Material | DiSECt `db7de0f`, 10 µs steps in ROSVisualizer | Same DFlex FEM/contact/spring/integration equations; exported Model/State loaded without ROS or Qt |
| Force signal | Sum of per-contact force magnitudes along DiSECt +y; source wrench transform and 200-sample Butterworth window | Same aggregation/filter, converted into the wrist sensor frame, added to the MuJoCo native rigid-body sensor with Gazebo's child-to-parent sign |
| Live ROS wiring | Both existing masters publish `/disect/wrench` with no subscriber; controller FT input has no publisher | Direct tensor feedback connects material force to the controller and observation; existing runs retain their original wiring |

The native wrist reading uses MuJoCo `rne_postconstraint` and the force/torque
sensor equations at the wrist body origin. Filter startup holds the initial
position command for its 200-sample window at 500 Hz and then tares the reading.
A shadow controller
at the same state removes only the filtered material contribution to measure
its effect on the controller update. Replay actions come from the recorded
SliceIt episode. `--hold-last-action` repeats the final recorded action up to
the original 500-action episode limit or task termination; it is a controller
exercise, with no policy training in this script.
`--scene-motion` instead maps the public DiSECt velocity into the robot base
frame and advances the end-effector reference for its configured 1.3 s, then
holds it. The captured FDCC defaults supply all gains; the original task goal
and force limits still determine termination. This is a prescribed-trajectory
integration check, separate from recorded-action replay and policy evaluation.

Sources: [Gazebo position hardware](https://github.com/ros-simulation/gazebo_ros_pkgs/blob/noetic-devel/gazebo_ros_control/src/default_robot_hw_sim.cpp),
[MuJoCo bounded friction constraint](https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-loss),
[MuJoCo impedance bounds](https://mujoco.readthedocs.io/en/stable/modeling.html#impedance).

[DiSECt](https://github.com/NVlabs/DiSECt) uses its bundled **DFlex** FEM and virtual-node cutting-spring implementation. Its `CuttingSim.simulation_step` imposes knife motion, then integrates material state; `model.knife_link_index` selects one knife, and `state.knife_f` holds per-contact force contributions. Isaac Lab and mjlab are robot-learning frameworks; switching to either does not port these material kernels or cutting state.

The remaining implementation is concrete:

1. Complete the single-world task endpoint with the implemented source controller, 50 hardware updates and 25 material exchanges per agent action. Compare recorded command-to-joint/knife behavior and sensor response.
2. Add an environment dimension or packed environment offsets to particles, tetrahedra, virtual nodes, cutting springs, contact force arrays, knife indices and material parameters. Replace singleton knife assumptions and global reductions with per-world operations. Reset only completed worlds.
3. Extend the single-world knife motion and filtered reaction-wrench exchange to independent batches. Preserve the source control-feedback coupling; applying material force directly to the rigid-body equations is a separate behavior change.
4. Keep current cut-surface/topology generation semantics per episode. DiSECt builds a cut surface from a prescribed trace before integration; arbitrary online topology generation requires explicit work beyond batching state arrays.
5. Reproduce the existing observation history, force aggregation, reward, termination, randomization and reset. Measure complete environment-step throughput with its real FEM substeps, memory and training updates at 1/32 worlds before scaling.

The robot benchmark measures rigid-body throughput. Thousands of complete cutting worlds require the material/controller/coupling implementation and a measured end-to-end resource curve. MuJoCo flex and Isaac deformables provide different material models; replacing DiSECt with either changes this task's physics.
