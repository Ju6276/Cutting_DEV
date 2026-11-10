# SliceIt — paper and source review
Reviewed 2026-11-10. [Paper](https://arxiv.org/abs/2404.02569), [official repository](https://github.com/omron-sinicx/sliceit).

## Evidence and scope
Read the method, calibration, simulator coupling, simulation/robot experiments and conclusion (PDF pp. 2–6). Read the deployed policy wrapper and the source-version task/config below. This is not a fresh reproduction of training or an audit of every dependency.

## Actual method and evidence
DiSECt provides deformable material response; Gazebo provides the robot; the bridge exchanges motion and reaction-force signals. A SAC actor adjusts six reference/compliance actions. The paper uses a temporal force encoder, so “SliceIt has no force history” is wrong. Our saved SAC observation wrapper must be described separately from the paper.

Real calibration uses initial Optuna/TPE search followed by Adam. Table I varies cutting spring stiffness/softness and contact stiffness/damping/friction parameters; a Young modulus alone is not a calibrated food. Training mixes calibrated tomato, cucumber and potato with parameter variation; carrot is a transfer test. Thus a shared controller, multiple foods, force feedback and compliance actions already have a direct cutting precedent.

The paper compares the dual simulator with a robot-only approximate contact model and reports simulated/real completion, forces and time. Its real evaluation uses relatively few physical food specimens; slices from one specimen are not independent material draws. The reported roughly 40 h versus 4 h training cost also matters when choosing this as a sample-efficiency comparator.

## Source correspondence
Source provenance is pinned in integrations/sliceit/patches/sources.json. Inspected the upstream ur_python_utilities version **7cd7cb8ce491b7953fce61ca891382e893e2ec43**, using git show rather than the locally patched checkout:
- ur3e_openai/src/ur3e_openai/task_envs/force_control/disect_slicing.py: reset/pose coupling, success counter, workspace and force/torque termination.
- ur3e_rl/config/simulation/slicing_3d_disect.yaml: agent period 0.05 s, six actions, learning rate 0.01, batch 1024, force-history options, goal tolerances and component wrench limits.
- SliceIt slicing_rl_policy.py: loading and executing the actor through the environment.

The YAML enables force history, a wave/temporal actor and wrench_hist_size=3. This does not establish that every historical local SAC run exposed the same history. The paper reward expression and released implementation also differ; METHOD and the benchmark contract govern our execution.

## Consequence for this project
Use SliceIt as the closest task/controller precedent. Multi-material training makes the experiment relevant but does not create a new method. A new contribution must explain and demonstrate what learned dynamics adds beyond this force-feedback SAC. Do not relabel the patched MuJoCo benchmark as an exact reproduction of the complete original Gazebo system.
