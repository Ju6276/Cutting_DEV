#!/usr/bin/env bash
# Read-only ROS capture followed by an offline analytic diagnostic.
set -eo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
output=${1:-"$repo/runs/ur10e_mpc_assumed_$(date +%Y%m%d_%H%M%S)"}
source "${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
"${ROS_PYTHON:-/usr/bin/python3}" "$repo/scripts/hardware/capture_mpc_feedback.py" --output "$output"
"${MPC_PYTHON:-$repo/.venv/bin/python}" "$repo/scripts/hardware/replay_mpc_assumed.py" \
  --repo "$repo" --snapshot "$output" --output "$output/results"
