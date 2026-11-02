# Current plan

Long training remains stopped. The active work is synchronizing the corrected
SliceIt interfaces and checking them with independent reviewers and bounded runs.

1. Decode force history consistently in the Torch reward; capture ROS reset
   state at the returned observation's sampling point. Preserve SuperMimic's
   replay, policy architecture, BPTT/GAE objectives and training settings.
2. Verify new ROS episodes against corrected SliceIt, and run the GPU collector
   through one complete learning update and the following physics step.
3. Before ROS world-model training, export native FDCC/controller memory into
   the state. Current GPU training already has the recurrent layout.
4. Diagnose the old critic's large terminal bootstrap separately, with per-step
   imagined states, values and targets. Passing interface checks establishes
   data/formula alignment; learning quality requires physical policy evaluation.

The old Gazebo runs had disconnected material-force topics. The corrected launch
connects the relay, filter and controller. Old task-interface checkpoints and
recorded rewards use the prior history interpretation; collect new data for the
current interface revision.
