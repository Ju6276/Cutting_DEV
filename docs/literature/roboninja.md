# RoboNinja — optimized cutting demonstrations and adaptive policy
Reviewed 2026-11-10. [Paper](https://arxiv.org/abs/2302.11553), [code](https://github.com/real-stanford/roboninja).
Code inspected at **0d90994d1a23d5f595b53f83ba0abff3643bd848**.

## Evidence and scope
Read main method and simulation/robot evaluations, ablations and limitations (PDF pp. 3–10); inspected the teacher optimization loop, student dataset/training and adaptive controller. The underlying Taichi material solver was not independently revalidated.

## Algorithm
A differentiable MLS-MPM simulator supplies an optimization teacher. With known hard-core geometry, Adam optimizes a planar knife trajectory for removed material, collision and energy terms; it uses reverse differentiation through simulated material motion. The student is trained by supervised imitation of those optimized trajectories.

At deployment the hidden core is estimated from encountered collision positions. A U-Net predicts a core map; the policy consumes three raster channels: estimated core signed-distance map, knife-pose map and a constant tolerance map. These 3×256×256 inputs are generated geometric maps, not camera images. The action is a two-component incremental direction/knife-angle command. Collision feedback causes retreat and an adaptive safety tolerance.

## Source trace
- roboninja/workspace/optimization_workspace.py: forward simulation and reverse step_grad calls, trajectory optimizer and restricted initial-position gradients.
- roboninja/dataset/close_loop_policy_dataset.py: optimization.pkl expert trajectories, augmentation, raster inputs and two-dimensional targets.
- roboninja/workspace/close_loop_policy_workspace.py: MSE supervised student objective.
- roboninja/cutting_policy/cutting_policy_adaptive.py: collision-dependent tolerance and construction of policy input maps.

This is not SAC or a learned neural dynamics model used for online BPTT.

## Data and metrics
Training uses 300 hard-core geometries; evaluation separates seen-category/new-instance and new-category objects. The paper measures cutting completion, removed mass, collisions and simulated energy. Real experiments include printed cores with soft surrounding media and food, using load-cell collision feedback. Some robot procedures add oscillation and prepare the outer skin. Camera recording is not evidence that RGB drives this policy.

## Implication
A differentiable simulator can provide demonstrations rather than train a world model. Hidden material geometry, tactile/collision adaptation and a learned cutting policy already exist here. Our distinction would have to be the learned robot/force transition model and its use for control, with different task/force metrics; generic “adaptive multi-material cutting” is not new.
