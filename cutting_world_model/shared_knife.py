"""Canonical physical blade and nominal EE mounting shared by both backends."""

import numpy as np
import torch
from cutting_gpu_backend.coupling import SliceItCoupling, compose_pose, inverse_pose

REVISION = "disect_knife_rigid_mount_v1"
# Exact float32 vertices of the current prepared DiSECt scenes, in metres.
TOP = 0.029500000178813934
TIP = -0.02954000048339367
SHOULDER = -0.029500000178813934
WIDTH = 0.0017500000540167093
NECK = 0.000039999998989515007
EDGE = 0.07500000298023224
CENTER = (TOP + TIP) / 2
HALF = (WIDTH, EDGE, (TOP - TIP) / 2)
BEVEL = SHOULDER - TIP


def vertices_disect():
    section = [
        (-WIDTH, TOP),
        (WIDTH, TOP),
        (-NECK, SHOULDER),
        (NECK, SHOULDER),
        (0, TIP),
    ]
    return np.array([[x, y, z] for z in (-EDGE, EDGE) for x, y in section])


def vertices_blade():
    return np.array(
        [
            [0, y, -HALF[2]] if i == 0 else [x, y, z]
            for y in (-EDGE, EDGE)
            for i, (x, z) in enumerate(
                [
                    (0, -HALF[2]),
                    (NECK, -HALF[2] + BEVEL),
                    (WIDTH, HALF[2]),
                    (-WIDTH, HALF[2]),
                    (-NECK, -HALF[2] + BEVEL),
                ]
            )
        ]
    )


def disect_from_blade(like):
    # [x_D,y_D,z_D] = [x_B,z_B,-y_B]; a proper -90deg rotation around x.
    return like.new_tensor([0, CENTER, 0, -(2**-0.5), 0, 0, 2**-0.5])


def canonical_mount(root, chain, knife_chain, task):
    from pathlib import Path

    if task.get("cad_tool"):
        from .cad_knife import mount
        ee_d=mount(task,chain.origin_position)
        d_b=ee_d.new_tensor(task["cad_tool"]["disect_from_blademaster"])
        from cutting_gpu_backend.coupling import fixed_pose_from_urdf
        base_d=fixed_pose_from_urdf(task["robot_xml"],task["robot_base"],"cutting_board_disect",device=ee_d.device,dtype=ee_d.dtype)
        return ee_d, compose_pose(ee_d,d_b), base_d
    q = torch.as_tensor(
        np.load(Path(root) / "runs/source/episode_000.npz")["joint_q"][0],
        device=chain.origin_position.device,
        dtype=chain.origin_position.dtype,
    )[None]
    ee = chain(q, mass_matrix=False)[0][chain.tip]
    source = knife_chain(q, mass_matrix=False)[0]["b_bot_knife_sim"]
    coupling = SliceItCoupling.from_urdf(
        task["robot_xml"], 0.059, device=q.device, dtype=q.dtype
    )
    base_d = compose_pose(coupling.base_from_disect, coupling.reset_pose(source))
    ee_d = compose_pose(inverse_pose(ee), base_d)[0]
    ee_b = compose_pose(ee_d, disect_from_blade(q))
    return ee_d, ee_b, coupling.base_from_disect
