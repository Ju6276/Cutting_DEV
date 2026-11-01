# Experiment order

The selected target is the original SliceIt robot and compliance-control task.
Gazebo and DiSECt remain the data source. Preserve the six SliceIt actions,
24-dimensional policy observation, reward, and termination conditions.
The new learner replaces SAC with SuperMimic-style world-model policy updates.

The existing SAC baseline keeps its 24-dimensional observations. The new
branch uses SuperMimic-style model-based actor–critic updates through imagined
rollouts. Its world-model state and its policy observation are separate:
the world model receives deformation and damage history, while the policy can
use the same robot/sensor observations as SAC. The current implementation
contains the state collector and dynamics adapter; policy/critic updates are
the fourth experiment below.

1. Record coupled Gazebo/DiSECt transitions at env.step boundaries, including
   particle state, damage, joint state, force history, previous velocities and
   accelerations, accumulated controller command, observations and rewards.
2. Collect distinct cutting trajectories; split whole trajectories into training
   and validation sets. Fit 8-step windows with the upstream MLP. Compare force,
   displacement and remaining spring stiffness against state persistence.
   Report negative stiffness, stiffness increases and quaternion norm errors.
3. Fit reward and terminal-probability heads on the original environment labels.
   Validate predictions on held-out complete episodes before policy training.
4. Reuse short imagined rollouts, frozen-model policy gradients and a terminal
   critic. Replay resulting policies in Gazebo/DiSECt and compare against SAC
   using total wall time including data collection and model training.

The running SAC is the independent baseline. Coupled collection starts with
three episodes using the original action-space warmup sampler, a 15-minute
wall cap, and per-step progress output. The supervisor stops its own children
on completion, failure or timeout. ROS 11322 and Gazebo 11356 are separate from
the SAC baseline's 11321/11355. Collection and subsequent model fitting run
sequentially to limit competition for the 4090.
