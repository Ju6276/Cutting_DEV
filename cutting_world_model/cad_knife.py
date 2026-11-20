"""Verified CAD blade geometry shared by robot, material, board and renderer."""
import hashlib,json
from pathlib import Path
from functools import lru_cache
import numpy as np
REVISION="ikea_cad_mesh_v1"
@lru_cache(None)
def load(root):
 root=Path(root);meta=json.loads((root/"cutting_geometry.json").read_text())
 if meta["revision"]!=REVISION:raise ValueError("Unsupported CAD blade revision")
 path=Path(meta["geometry"])
 if hashlib.sha256(path.read_bytes()).hexdigest()!=meta["geometry_sha256"]:raise ValueError("CAD knife hash mismatch")
 with np.load(path) as data:arrays={k:data[k].copy() for k in data.files}
 return meta,arrays
def mount(task,like):
 from cutting_gpu_backend.coupling import fixed_pose_from_urdf
 return fixed_pose_from_urdf(task["robot_xml"],task["robot_tip"],task["material_tool_frame"],device=like.device,dtype=like.dtype)
