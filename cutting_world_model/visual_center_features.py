"""Numpy-only point-cloud normalization shared by rendering and inference."""
import numpy as np
from .visual_cut import blade_points

REVISION = "visible_cloud_center_v1"
SCALE_M = .1
POINT_COUNT = 512


def heading(config):
    if "prescribed_ee_goal" in config:
        raise ValueError("Center selection must not receive a prescribed goal")
    if not np.allclose(config["board_normal_base"], [0, 0, 1], atol=1e-6):
        raise ValueError("Center pilot requires horizontal calibrated board")
    quat = np.asarray(config["tool_orientation_xyzw"], float).copy()
    reference = np.asarray(config.get("policy_quaternion_reference_xyzw", quat), float)
    if quat.shape != (4,) or reference.shape != (4,) or not np.isfinite([quat, reference]).all():
        raise ValueError("Invalid quaternion")
    if not np.isclose(np.linalg.norm(quat), 1, atol=1e-5) or not np.isclose(np.linalg.norm(reference), 1, atol=1e-5):
        raise ValueError("Unit quaternions required")
    if quat @ reference < 0:
        quat = -quat
    vertices = blade_points(np.r_[np.zeros(3), quat], config["vertices_in_ee"])
    _, _, vh = np.linalg.svd(vertices-vertices.mean(0), full_matrices=False)
    edge = vh[0].copy()
    edge[2] = 0
    edge /= np.linalg.norm(edge)
    normal = np.cross([0, 0, 1], edge)
    if normal[1] < 0:
        normal = -normal
    if not np.allclose(normal, [0, 1, 0], atol=1e-3):
        raise ValueError("Trained center pilot requires fixed +base Y cut normal")
    return quat, np.array([0., 1., 0.])


def cloud_input(points, board_z, seed=0):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 50 or not np.isfinite(p).all():
        raise ValueError("At least 50 finite visible points required")
    origin = np.array([p[:, 0].mean(), p[:, 1].mean(), board_z])
    # Origin uses visible points, never full mesh center or task target.
    indices = np.random.default_rng(seed).choice(len(p), POINT_COUNT, replace=len(p)<POINT_COUNT)
    features = ((p[indices]-origin)/SCALE_M).astype(np.float32)
    return features, origin



TABLE_FRAME = "table_aligned_base_v1"
BASE_FRAME = "base_axes_v1"

def table_cloud_input(points, config, board_offset_m=0., seed=0):
    """Current horizontal table, fixed axes parallel to the robot base.

    Explicitly remove the calibrated table origin before cloud centering.
    This changes no shape scale and consumes no target/mesh information.
    """
    if not np.allclose(config["board_normal_base"],[0,0,1],atol=1e-6):
        raise ValueError("Table input requires a horizontal board")
    if not np.allclose(config.get("board_x_base",[1,0,0]),[1,0,0],atol=1e-6):
        raise ValueError("Current fixed-heading table axes must align with robot-base axes")
    table_origin=np.asarray(config["board_origin_base"],dtype=float).copy()
    if table_origin.shape!=(3,) or not np.isfinite(table_origin).all():
        raise ValueError("Invalid table origin")
    if not np.isfinite(board_offset_m):raise ValueError("Invalid board offset")
    table_origin[2]+=float(board_offset_m)
    features,origin_table=cloud_input(np.asarray(points)-table_origin,0.,seed)
    return features,origin_table,table_origin
