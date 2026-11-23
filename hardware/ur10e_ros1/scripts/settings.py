"""Filesystem configuration only; importing this module never talks to the robot."""
if not __debug__:
    raise RuntimeError("Hardware entries require runtime checks; do not use python -O")
import os
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
REPOSITORY = MODULE.parents[1]
ROBOT_IP = os.environ.get("CUTTING_ROBOT_IP", "192.168.0.123")
RECORDS = Path(os.environ.get("CUTTING_REAL_RUNS", str(Path.home() / "ur10e_fdcc_records" / "current")))
PLANNER_PYTHON = os.environ.get("CUTTING_PLANNER_PYTHON", str(REPOSITORY / ".venv-mpc-cu118/bin/python"))
KNIFE_MESH = Path(os.environ.get("CUTTING_KNIFE_MESH", str(Path.home() / "ur10e_fdcc_records/alignment_candidate_20261121/platform/knife.stl")))

def record_directory(name):
    path = RECORDS / name
    path.mkdir(parents=True, exist_ok=True)
    return path
