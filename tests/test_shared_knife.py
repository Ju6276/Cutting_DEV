"""Geometry, mount and policy-history provenance checks independent of GPU material."""

import json
from pathlib import Path
import numpy as np
import torch
from cutting_world_model.shared_knife import (
    HALF,
    BEVEL,
    NECK,
    vertices_disect,
    vertices_blade,
    disect_from_blade,
    canonical_mount,
)
from cutting_gpu_backend.coupling import compose_pose
from cutting_gpu_backend.controller.fdcc import SerialChain, quat_apply
from cutting_world_model.blademaster_backend import blade_vertices
from cutting_world_model.free_cut import validate_task


def test_both_profiles_are_same_vertex_set_under_proper_transform():
    b = torch.tensor(vertices_blade(), dtype=torch.double)
    transform = disect_from_blade(b)
    d = quat_apply(transform[3:].expand(len(b), -1), b) + transform[:3]
    expected = torch.tensor(vertices_disect(), dtype=torch.double)
    assert torch.cdist(d, expected).min(1).values.max() < 1e-12
    np.testing.assert_allclose(
        blade_vertices(HALF, BEVEL, NECK), vertices_blade(), atol=1e-12
    )
    assert (
        abs(
            float(
                torch.det(
                    torch.stack(
                        [
                            quat_apply(transform[3:], e)
                            for e in torch.eye(3, dtype=torch.double)
                        ],
                        1,
                    )
                )
            )
            - 1
        )
        < 1e-12
    )


def test_shared_mount_agrees_at_multiple_robot_poses():
    root = Path("backends/mujoco")
    task_path = root / "runs/source/task.json"
    if not task_path.exists():
        import pytest

        pytest.skip("Remote robot assets required")
    task = json.loads(task_path.read_text())
    chain = SerialChain(
        task["robot_xml"], task["robot_base"], task["robot_tip"], task["joint_names"]
    ).to("cpu", torch.double)
    knife = SerialChain(
        task["robot_xml"], task["robot_base"], "b_bot_knife_sim", task["joint_names"]
    ).to("cpu", torch.double)
    md, mb, base = canonical_mount(root, chain, knife, task)
    q = torch.tensor(
        np.load(root / "runs/source/episode_000.npz")["joint_q"][0], dtype=torch.double
    )[None].repeat(4, 1)
    q[1:, 1] += 0.03
    q[2:, 4] -= 0.02
    q[3, 0] += 0.01
    ee = chain(q, mass_matrix=False)[0][chain.tip]
    pd = compose_pose(ee, md.expand(4, -1))
    pb = compose_pose(ee, mb.expand(4, -1))
    vd = torch.tensor(vertices_disect(), dtype=torch.double)
    vb = torch.tensor(vertices_blade(), dtype=torch.double)
    world_d = (
        quat_apply(pd[:, None, 3:].expand(-1, 10, -1), vd.expand(4, -1, -1))
        + pd[:, None, :3]
    )
    world_b = (
        quat_apply(pb[:, None, 3:].expand(-1, 10, -1), vb.expand(4, -1, -1))
        + pb[:, None, :3]
    )
    assert torch.cdist(world_d, world_b).min(-1).values.max() < 1e-12
    # Along-edge motion must follow the source saw axis (robot base X).
    edge = quat_apply(pb[0, 3:], pb.new_tensor([0.0, 1.0, 0.0]))
    assert abs(float(edge[0])) > 0.999


def test_shared_tool_is_explicit_versioned_task():
    for folder in ("free_cut", "blademaster"):
        config = json.loads(
            Path("configs", "archive", "scene_checks_20261116", folder, "shared_knife_check.json").read_text()
        )
        assert (
            validate_task(config["task"])["tool_revision"]
            == "disect_knife_rigid_mount_v1"
        )
        old = json.loads(
            Path("configs", "archive", "scene_checks_20261116", folder, "depth_three_object_check.json").read_text()
        )
        assert "tool_revision" not in old["task"]
