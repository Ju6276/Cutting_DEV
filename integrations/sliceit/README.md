# Historical SliceIt / Gazebo reproduction

This directory preserves pinned upstream sources and local integration patches.
It is not required to switch the current MuJoCo task to Gazebo.
Current training/evaluation: [Operations](../../docs/OPERATIONS.md).

scripts/checkout_sliceit.py reconstructs the source trees in
[the lock](patches/sources.json), including public-prism and bridge/controller fixes.
Its --local-source option uses existing Git objects. Rebuilding this historical
stack requires ROS Noetic, the SliceIt Python environment, catkin and source assets;
it is not a new current-baseline training instruction.
Attribution and patch scope: [Sources](../../docs/reference/SOURCES.md).
