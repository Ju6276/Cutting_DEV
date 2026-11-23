#!/usr/bin/env bash
# Rebuild the recorded controller source. Never starts ROS or the robot driver.
set -euo pipefail
module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$module_dir/../.." && pwd)"
ws="${1:?Usage: build.sh /absolute/path/to/NEW_catkin_workspace}"
[[ "$ws" = /* ]] || { echo "Use an absolute workspace path" >&2; exit 2; }
[[ ! -e "$ws/src/cartesian_controllers" ]] || { echo "Controller source already exists; use a new workspace" >&2; exit 2; }
set +u  # ROS setup scripts read optional unset variables.
source /opt/ros/noetic/setup.bash
set -u
mkdir -p "$ws/src"
git clone https://github.com/cambel/cartesian_controllers.git "$ws/src/cartesian_controllers"
git -C "$ws/src/cartesian_controllers" checkout --detach f4359b966bb3b8d8a07d829df463da814c2bee33
git -C "$ws/src/cartesian_controllers" apply "$repo_dir/integrations/sliceit/patches/2-cartesian_controllers.patch"
git -C "$ws/src/cartesian_controllers" apply "$module_dir/controller/compliance_range.patch"
cd "$ws"
catkin_make -j2 -DCMAKE_BUILD_TYPE=Release
echo "Built only. Source $ws/devel/setup.bash explicitly when preparing a session."
