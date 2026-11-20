"""Explicit robot assets and FDCC frames, independent of the material backend.

The legacy UR5e bundle and estimated user UR10e bundle share this loader. A robot
bundle is not evidence that the new knife's material-contact model is integrated.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass
class RobotPlatform:
    metadata: dict
    model_path: Path
    controller_parameters: dict
    q: np.ndarray
    qd: np.ndarray

    @classmethod
    def load(cls, root):
        root = Path(root)
        manifest = root / "platform.json"
        if not manifest.exists():
            task = json.loads((root / "runs/source/task.json").read_text())
            with np.load(root / "runs/source/episode_000.npz") as episode:
                q, qd = episode["joint_q"][0].copy(), episode["joint_qd"][0].copy()
            parameters = json.loads((root / "runs/controller_source/controller_params.json").read_text())
            return cls(task, root / "runs/model_coupled/robot.mjb", parameters, q, qd)
        description = json.loads(manifest.read_text())
        runtime = description.get("runtime")
        if not runtime:
            raise ValueError("Robot assets are not compiled; run compile_robot_platform.py")
        for filename, expected in runtime["sha256"].items():
            actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("Robot asset hash mismatch: " + filename)
        metadata = {key: description[key] for key in ("robot_base", "robot_tip", "joint_names")}
        metadata["robot_xml"] = (root / "robot.urdf").read_text()
        metadata["robot_platform"] = {key: description[key] for key in
                                     ("revision", "status", "configuration", "source_hashes")}
        parameters = json.loads((root / "controller.json").read_text())
        q = np.asarray(description["q_initial_rad"], dtype=float)
        if q.shape != (len(metadata["joint_names"]),) or not np.isfinite(q).all():
            raise ValueError("Invalid platform reset joints")
        return cls(metadata, root / "robot.mjb", parameters, q, np.zeros_like(q))

    def controller(self, device="cpu"):
        from cutting_gpu_backend.controller.fdcc import SerialChain, ControllerConfig, FDCCController
        m = self.metadata
        chain = SerialChain(m["robot_xml"], m["robot_base"], m["robot_tip"], m["joint_names"]).to(device)
        controller = FDCCController(chain, ControllerConfig.from_parameters(self.controller_parameters))
        return chain, controller

    def robot(self, worlds=1):
        from cutting_gpu_backend.robot import Robot
        return Robot(self.model_path, self.q, self.qd, worlds,
                     sensor_sign=-1. if "robot_platform" in self.metadata else 1.)

    def require_cutting_ready(self):
        if "robot_platform" in self.metadata:
            raise ValueError("This UR10e/ATI robot bundle is not yet a validated cutting scene. "
                             "Do not reuse the UR5e blade SDF or BladeMaster wedge for the IKEA blade.")
