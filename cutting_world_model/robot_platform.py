"""Explicit robot assets and FDCC frames, independent of the material backend.

The legacy UR5e bundle and estimated user UR10e bundle share this loader. A robot
bundle is not evidence that the new knife's material-contact model is integrated.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def frame_poses(metadata, q):
    """Nominal URDF FK for asset/export tools without the training Torch runtime."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    q=np.asarray(q,dtype=float)
    if q.shape!=(len(metadata['joint_names']),) or not np.isfinite(q).all():
        raise ValueError('Joint vector does not match platform')
    values=dict(zip(metadata['joint_names'],q))
    poses={metadata['robot_base']:np.eye(4)}
    pending=list(ET.fromstring(metadata['robot_xml']).findall('joint'))
    while pending:
        progressed=False
        for joint in pending[:]:
            parent=joint.find('parent').get('link')
            if parent not in poses:continue
            origin=joint.find('origin');transform=np.eye(4)
            if origin is not None:
                transform[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                transform[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            motion=np.eye(4);kind=joint.get('type')
            if kind!='fixed':
                if joint.get('name') not in values:raise ValueError('Unspecified moving joint '+joint.get('name'))
                node=joint.find('axis')
                axis=np.fromstring(node.get('xyz','1 0 0') if node is not None else '1 0 0',sep=' ')
                axis/=np.linalg.norm(axis);value=values[joint.get('name')]
                if kind in ('revolute','continuous'):motion[:3,:3]=Rotation.from_rotvec(axis*value).as_matrix()
                elif kind=='prismatic':motion[:3,3]=axis*value
                else:raise ValueError('Unsupported joint '+kind)
            poses[joint.find('child').get('link')]=poses[parent]@transform@motion
            pending.remove(joint);progressed=True
        if not progressed:
            # Joints outside the chosen base subtree (e.g. world -> base) do not
            # contribute to any descendant pose. Never invent their transform.
            break
    if metadata['robot_tip'] not in poses:raise ValueError('Robot tip unreachable from base')
    return poses

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
        # Stable execution identity, not the location of the bundle. These
        # hashes were checked above and include the real FDCC parameter file.
        metadata["robot_platform"]["runtime_sha256"] = dict(runtime["sha256"])
        metadata["robot_platform"]["controller_sha256"] = runtime["sha256"]["controller.json"]
        geometry = root / "cutting_geometry.json"
        if geometry.exists():
            from .cad_knife import load
            from scipy.spatial.transform import Rotation
            import xml.etree.ElementTree as ET
            meta, _ = load(str(root))
            xml=ET.fromstring(metadata["robot_xml"])
            def fixed(name,parent,xyz,rpy):
                ET.SubElement(xml,"link",name=name)
                j=ET.SubElement(xml,"joint",name=parent+"__"+name,type="fixed")
                ET.SubElement(j,"parent",link=parent);ET.SubElement(j,"child",link=name)
                ET.SubElement(j,"origin",xyz=" ".join(map(str,xyz)),rpy=" ".join(map(str,rpy)))
            pose=meta["knife_mount_from_disect"]
            fixed("knife_material","knife_mount",pose[:3],Rotation.from_quat(pose[3:]).as_euler("xyz"))
            # Board axes: material x -> base y, y -> base z, z -> base x.
            # Its centre is the nominal blade centre; task alignment later changes only height.
            nominal_meta=dict(metadata,robot_xml=ET.tostring(xml,encoding="unicode"))
            center_frame=description["configuration"].get("cutting_center_frame","knife_material")
            board_origin=frame_poses(nominal_meta,description["q_initial_rad"])[center_frame][:3,3].tolist()
            board_origin[2]=float(description["configuration"]["board_height_m"])
            fixed("cutting_board_disect",metadata["robot_base"],board_origin,
                  Rotation.from_matrix([[0,0,1],[1,0,0],[0,1,0]]).as_euler("xyz"))
            metadata["robot_xml"]=ET.tostring(xml,encoding="unicode")
            metadata["material_tool_frame"]="knife_material"
            metadata["cad_tool"]=meta
            metadata["robot_platform"]["asset_root"]=str(root.resolve())
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

    def robot(self, worlds=1, *, native_sensor=False):
        from cutting_gpu_backend.robot import Robot
        return Robot(self.model_path, self.q, self.qd, worlds,
                     sensor_sign=1. if native_sensor else (-1. if "robot_platform" in self.metadata else 1.))

    def require_cutting_ready(self):
        if "robot_platform" in self.metadata:
            raise ValueError("This UR10e/ATI robot bundle is not yet a validated cutting scene. "
                             "Do not reuse the UR5e blade SDF or BladeMaster wedge for the IKEA blade.")


def model_path_for_task(root, metadata):
    """Recorded platform selects its robot model; legacy tasks keep their original bundle."""
    platform=metadata.get("robot_platform")
    if platform is not None:
        if not platform.get("asset_root"):
            raise ValueError("Recorded robot platform lacks asset_root")
        return RobotPlatform.load(platform["asset_root"]).model_path
    return Path(root)/"runs/model_coupled/robot.mjb"
