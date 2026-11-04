# SliceIt integration

Run `python3 scripts/checkout_sliceit.py` from the project root to reconstruct
the pinned SliceIt source and all eight installed Git repositories. It restores
the public prism configurations and the local bridge/controller/stepping fixes.
`--local-source /path/to/SliceIt` uses an existing checkout's Git objects.

The checkout requires ROS Noetic, a built catkin underlay, the SliceIt Python
3.8 environment, and the public DiSECt dataset. The scene uses the public prism
material and the scene-aligned Gazebo knife. Author food-calibration data remain
a separate download.

SAC and Gazebo WM entry points are documented in [RUNNING](../../docs/RUNNING.md).
Sources and exact revisions: [lock](patches/sources.json).
