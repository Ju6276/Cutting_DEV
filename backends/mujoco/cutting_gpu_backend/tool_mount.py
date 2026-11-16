"""Rigid finite-tool board bound, independent of material backend."""

import torch
from .controller.fdcc import quat_apply


def gap_from_pose(pose, geometry):
    vertices = pose.new_tensor(geometry["vertices_in_ee"])
    q = pose[..., None, 3:].expand(*pose.shape[:-1], len(vertices), 4)
    v = vertices.expand(*pose.shape[:-1], len(vertices), 3)
    points = pose[..., None, :3] + quat_apply(q, v)
    origin = pose.new_tensor(geometry["board_origin_base"])
    normal = pose.new_tensor(geometry["board_normal_base"])
    return ((points - origin) * normal).sum(-1).amin(-1)


@torch.no_grad()
def swept_joint_fraction(old_q, new_q, fk, geometry, reach_bound, active):
    """Conservative finite-blade clearance along a revolute joint interpolation.

    Every link-offset vector rotates with angular speed bounded by sum(abs(dq)).
    Thus ||p''(alpha)|| <= reach_bound * sum(abs(dq))**2. A curve deviates
    from its endpoint chord by at most max||p''||/8. The serial-chain reach
    includes every fixed translation and the most distant blade vertex.
    """
    start = gap_from_pose(fk(old_q), geometry)
    delta = new_q - old_q
    curvature = reach_bound * delta.abs().sum(-1).square() / 8
    stop = geometry["stop_gap_m"]

    def bound(f):
        end = gap_from_pose(fk(old_q + f[:, None] * delta), geometry)
        return torch.minimum(start, end) - curvature * f.square()

    full = torch.ones_like(start)
    hit = active & (bound(full) <= stop)
    lo, hi = torch.zeros_like(start), full.clone()
    if bool(hit.any()):
        for _ in range(30):
            mid = (lo + hi) / 2
            safe = bound(mid) > stop
            lo = torch.where(safe, mid, lo)
            hi = torch.where(safe, hi, mid)
    return torch.where(hit, lo, full), hit
