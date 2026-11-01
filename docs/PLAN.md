# Experiment order

1. Record a full public-prism cut with particle velocity, damage history and
   knife pose. Verify state/action alignment and restart consistency at a
   partially damaged state. Record cost alongside the existing SAC throughput.
2. Collect distinct cutting trajectories; split whole trajectories into training
   and validation sets. Fit 8-step windows with the upstream MLP. Compare force,
   displacement and remaining spring stiffness against state persistence.
   Report negative stiffness, stiffness increases and quaternion norm errors.
3. Add the robot/controller state transition and preserve the original SliceIt
   action definition, rewards and termination conditions. The current 24-D
   policy observation remains distinct from the world model's full state.
4. Reuse short imagined rollouts, frozen-model policy gradients and a terminal
   critic. Replay resulting policies in Gazebo/DiSECt and compare against SAC
   using total wall time including data collection and model training.

The running SAC is the independent baseline. Initial collector budget: one
1.3-second cut, 26 control steps, 10-minute wall cap. Failures preserve the last
progress record. Large model training starts after the data interface checks.
