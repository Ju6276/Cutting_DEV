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

[DiSECt](https://github.com/NVlabs/DiSECt) uses its bundled **DFlex** FEM and virtual-node cutting-spring implementation. Its `CuttingSim.simulation_step` imposes knife motion, then integrates material state; `model.knife_link_index` selects one knife, and `state.knife_f` holds per-contact force contributions. Isaac Lab and mjlab are robot-learning frameworks; switching to either does not port these material kernels or cutting state.

The remaining implementation is concrete:

1. Preserve the current Cartesian compliance controller's pose/force coordinate transforms, integration state, velocity caps, stiffness/damping commands and 25 robot steps per agent action. Validate its command-to-joint/knife response against captured Gazebo episodes.
2. Add an environment dimension or packed environment offsets to particles, tetrahedra, virtual nodes, cutting springs, contact force arrays, knife indices and material parameters. Replace singleton knife assumptions and global reductions with per-world operations. Reset only completed worlds.
3. Exchange batched knife pose/velocity and material reaction wrench on GPU at each coupling interval, with correct xyzw/wxyz conversion, frame transforms and torque reference point. Preserve the current SliceIt coupling direction first; dynamic two-way reaction is a separate behavior change.
4. Keep current cut-surface/topology generation semantics per episode. DiSECt builds a cut surface from a prescribed trace before integration; arbitrary online topology generation requires explicit work beyond batching state arrays.
5. Reproduce the existing observation history, force aggregation, reward, termination, randomization and reset. Measure complete environment-step throughput with its real FEM substeps, memory and training updates at 1/32 worlds before scaling.

The robot benchmark measures rigid-body throughput. Thousands of complete cutting worlds require the material/controller/coupling implementation and a measured end-to-end resource curve. MuJoCo flex and Isaac deformables provide different material models; replacing DiSECt with either changes this task's physics.
