"""Controller parameters for the robot actually selected by an environment."""
import json
from pathlib import Path


def controller_parameters(metadata, backend_root):
    platform = metadata.get("robot_platform")
    if platform is not None:
        root = platform.get("asset_root")
        if not root:
            raise ValueError("Recorded robot platform lacks asset_root")
        from .robot_platform import RobotPlatform
        return RobotPlatform.load(root).controller_parameters
    return json.loads((Path(backend_root) / "runs/controller_source/controller_params.json").read_text())
