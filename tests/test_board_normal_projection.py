"""project_continue_v2 board projection: remove only the inward normal displacement.

The corrected endpoint's end-effector pose is the proposed pose translated along the
board normal (zero rotation, zero tangential motion). The f12ca04 minimum-norm joint
correction (project_continue_v1, kept for evaluation) is run in the same harness to
show the drift it produced. Physics (continuing_joint_step) and the WM (project_board)
call the same endpoint function.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cutting_gpu_backend.controller.fdcc import SerialChain, quat_inverse, quat_mul
from cutting_gpu_backend.tool_mount import (_residual_jacobian, _twist_solve, continuing_joint_step,
                                            gap_from_pose, joint_limits, project_joint_endpoint,
                                            project_joint_endpoint_v1, project_joint_velocity)

STOP = 1e-6
# UR-like spatial 6R test arm (nonsingular at Q0), units m.
LINKS = [('z', (0, 0, .1)), ('y', (0, .1, 0)), ('y', (.4, 0, 0)), ('y', (.4, 0, 0)),
         ('z', (0, .1, 0)), ('y', (0, 0, .1))]
Q0 = [.3, -1., 1.2, -1.5, -1.4, .2]
SPEEDS = (.005, .02, .05)  # 0.5, 2, 5 cm/s
# Per-step Newton tolerance is 1e-12 (m, rad); 100 steps accumulate at most ~1e-10.
DRIFT_BOUND_M = 1e-9
ROTATION_BOUND_RAD = 1e-9


def arm_xml(limit=6.2832):
    xml = '<robot name="arm">'
    for i, (axis, origin) in enumerate(LINKS):
        a = {'z': '0 0 1', 'y': '0 1 0'}[axis]
        xml += ('<joint name="j%d" type="revolute"><parent link="l%d"/><child link="l%d"/>'
                '<origin xyz="%g %g %g"/><axis xyz="%s"/>'
                '<limit lower="%g" upper="%g" velocity="3.14"/></joint>') % ((i, i, i+1) + origin + (a, -limit, limit))
    xml += '<joint name="tip" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 .1 0"/></joint>'
    return xml + '</robot>'


def arm(xml=None, base='l0', tip='tool', names=None):
    names = names or ['j%d' % i for i in range(6)]
    chain = SerialChain(xml or arm_xml(), base, tip, names).double()
    return (lambda q: chain(q, mass_matrix=False)[0][chain.tip]), chain


def resting_geometry(fk, q, vertices=((-.05, 0., .02), (.05, 0., .02)), normal=(0., 0., 1.)):
    """Board plane placed so the blade rests at the projection's clearance (stop + 0.2 um)."""
    probe = dict(vertices_in_ee=[list(v) for v in vertices], board_origin_base=[0., 0., 0.],
                 board_normal_base=list(normal), stop_gap_m=STOP)
    height = gap_from_pose(fk(q), probe)[0].item() - (STOP + 2e-7)
    n = torch.tensor(normal, dtype=torch.double)
    return dict(probe, board_origin_base=(height*n/n.square().sum()).tolist())


def translate(fk, q, shift):
    """Exact joint configuration whose EE pose is fk(q) translated by `shift` (rotation kept)."""
    pose = fk(q).detach()
    target = torch.cat((pose[..., :3] + shift, pose[..., 3:]), -1)
    with torch.enable_grad():
        q = q.detach().clone().requires_grad_(True)
        for _ in range(20):
            residual, jacobian = _residual_jacobian(fk, q, target, False)
            if residual.abs().max() < 1e-14:
                break
            q = (q - _twist_solve(jacobian, residual)).detach().requires_grad_(True)
    return q.detach()


def rotation_angle(a, b):
    relative = quat_mul(quat_inverse(b[..., 3:]), a[..., 3:])
    return 2*torch.atan2(relative[..., :3].norm(dim=-1), relative[..., 3].abs())


def press(fk, chain, geometry, q, endpoint, seconds=1., dt=.01):
    """Pure inward commands for `seconds` at SPEEDS (one world each), projected every dt.

    10 ms command steps keep the suite fast; the adapters' 1-2 ms substeps are
    exercised by scripts/check_board_continue.py on real physics.
    """
    n = torch.tensor(geometry['board_normal_base'], dtype=torch.double)
    speeds = torch.tensor(SPEEDS, dtype=torch.double)
    q = q.expand(len(SPEEDS), -1).clone()
    start = fk(q)
    active = torch.ones(len(SPEEDS), dtype=torch.bool)
    with torch.no_grad():
        for _ in range(round(seconds/dt)):
            proposed = translate(fk, q, -(speeds*dt)[:, None]*n)
            q, hit = continuing_joint_step(q, proposed, fk, geometry, 2., active, endpoint=endpoint,
                                           limits=(chain.lower, chain.upper))
            assert hit.all()
    end = fk(q)
    move = end[:, :3] - start[:, :3]
    tangential = (move - (move*n).sum(-1, keepdim=True)*n).norm(dim=-1)
    gap = gap_from_pose(end, geometry)
    return tangential, rotation_angle(end, start), gap


def check_press(fk, chain, geometry, q0, label):
    tangential, rotation, gap = press(fk, chain, geometry, q0, project_joint_endpoint)
    print(label, 'v2 normal-only: tangential drift m', tangential.tolist(), 'rotation rad', rotation.tolist())
    assert (tangential < DRIFT_BOUND_M).all() and (rotation < ROTATION_BOUND_RAD).all()
    assert (gap >= STOP - 1e-9).all() and (gap <= STOP + 3e-7).all()
    old_t, old_r, old_gap = press(fk, chain, geometry, q0, project_joint_endpoint_v1)
    print(label, 'v1 minimum-norm: tangential drift m', old_t.tolist(), 'rotation rad', old_r.tolist())
    assert (old_gap >= STOP - 1e-9).all()
    return old_t, old_r


def test_pure_inward_press_has_no_tangential_drift_or_rotation():
    fk, chain = arm()
    q0 = torch.tensor([Q0], dtype=torch.double)
    geometry = resting_geometry(fk, q0)
    old_t, old_r = check_press(fk, chain, geometry, q0, 'test arm')
    # The f12ca04 minimum-norm correction drifts on the same commands.
    assert (old_t > 1e3*DRIFT_BOUND_M).all()


@pytest.mark.skipif(not os.environ.get('CUTTING_BACKEND_ROOT'), reason='UR5e source assets')
def test_pure_inward_press_on_the_ur5e_source_chain():
    root = Path(os.environ['CUTTING_BACKEND_ROOT'])
    task = json.loads((root/'runs/source/task.json').read_text())
    fk, chain = arm(task['robot_xml'], task['robot_base'], task['robot_tip'], task['joint_names'])
    q0 = torch.as_tensor(np.load(root/'runs/source/episode_000.npz')['joint_q'][0], dtype=torch.double)[None]
    geometry = resting_geometry(fk, q0)
    check_press(fk, chain, geometry, q0, 'UR5e source chain')


def test_lateral_motion_on_the_board_is_preserved_exactly():
    fk, chain = arm()
    q0 = torch.tensor([Q0], dtype=torch.double)
    geometry = resting_geometry(fk, q0)
    n = torch.tensor([0., 0., 1.], dtype=torch.double)
    limits = (chain.lower, chain.upper)
    for lateral in ((1e-3, 0., 0.), (0., -2e-3, 0.), (5e-4, 5e-4, 0.)):
        lateral = torch.tensor(lateral, dtype=torch.double)
        # Endpoint projection of one slide-while-pressing proposal (1 mm inward).
        q, hit = project_joint_endpoint(translate(fk, q0, lateral - 1e-3*n), fk, geometry, limits=limits)
        assert hit.all()
        torch.testing.assert_close(fk(q)[0, :3] - fk(q0)[0, :3], lateral, atol=1e-10, rtol=0)
        assert rotation_angle(fk(q), fk(q0)).item() < 1e-10
        # Outward/lateral proposals that stay clear are unchanged.
        up = translate(fk, q0, lateral + 1e-3*n)
        out, hit = project_joint_endpoint(up, fk, geometry, limits=limits)
        torch.testing.assert_close(out, up, atol=0, rtol=0)
        assert not hit.any()
    # Physical step path (projection + swept check) over 50 small slide-and-press steps.
    q = q0
    lateral = torch.tensor([2e-5, 0., 0.], dtype=torch.double)
    with torch.no_grad():
        for _ in range(50):
            q, hit = continuing_joint_step(q, translate(fk, q, lateral - 2e-5*n), fk, geometry, 2.,
                                           torch.ones(1, dtype=torch.bool), limits=limits)
            assert hit.all()
    torch.testing.assert_close(fk(q)[0, :3] - fk(q0)[0, :3], 50*lateral, atol=1e-10, rtol=0)
    assert rotation_angle(fk(q), fk(q0)).item() < 1e-10


def test_velocity_projection_removes_only_inward_normal_velocity():
    fk, chain = arm()
    q = torch.tensor([Q0], dtype=torch.double).repeat(3, 1)
    geometry = resting_geometry(fk, q[:1])
    torch.manual_seed(0)
    velocity = torch.randn(3, 6, dtype=torch.double) * .1
    with torch.enable_grad():
        point = q.clone().requires_grad_(True)
        _, jacobian = _residual_jacobian(fk, point, fk(q).detach(), False)
        rate = torch.autograd.grad(gap_from_pose(fk(point), geometry).sum(), point)[0]
    velocity[0] -= 5*rate[0]            # world 0 moves into the board
    velocity[1] += 5*rate[1]            # world 1 moves away: unchanged
    limited = torch.tensor([True, True, False])
    out = project_joint_velocity(q, velocity, fk, geometry, limited)
    before, after = [(jacobian @ v[..., None])[..., 0] for v in (velocity, out)]
    n = torch.tensor([0., 0., 1.], dtype=torch.double)
    assert (rate[0]*velocity[0]).sum() < 0
    # Inward clearance rate removed (to the 1e-12 Jacobian damping, relative ~1e-10).
    assert abs((rate[0]*out[0]).sum().item()) < 1e-10 * abs((rate[0]*velocity[0]).sum().item())
    torch.testing.assert_close(after[0, 3:], before[0, 3:], atol=1e-10, rtol=0)   # no angular change
    change = after[0, :3] - before[0, :3]
    torch.testing.assert_close(change - (change*n).sum()*n, torch.zeros(3, dtype=torch.double), atol=1e-10, rtol=0)
    torch.testing.assert_close(out[1:], velocity[1:], atol=0, rtol=0)


def test_physics_and_wm_use_the_same_projection():
    from test_board_continue import continuing, physical_chain
    from test_pose_depth_v3 import joints_for, joint_delta
    from cutting_world_model.pose_cut import project_board
    wm, s, g = continuing(); sp = wm.spec
    fk = physical_chain(sp)
    q = torch.cat([joints_for(sp, sp.field(s, 'joint_q'), dx, dy)
                   for dx, dy in ((0., -2e-3), (1e-3, -1.5e-3), (-5e-4, -1e-3 - 1e-6), (2e-4, -5e-4))])
    states = s.repeat(len(q), 1)
    fields = sp.integrate_physical(states, torch.cat([joint_delta(sp, s, x[None]) for x in q]))
    physical, hit = project_joint_endpoint(q, fk, sp.board_geometry, limits=sp.joint_limits)
    assert hit.tolist() == [True, True, True, False]
    torch.testing.assert_close(fields['joint_q'], physical, atol=1e-12, rtol=0)
    # Pure normal translation in the WM as well: x unchanged, rotation unchanged.
    torch.testing.assert_close(sp.fk(fields['joint_q'])[:, 0], sp.fk(q)[:, 0], atol=1e-12, rtol=0)
    assert (rotation_angle(sp.fk(fields['joint_q']), sp.fk(q)) < 1e-10).all()


def test_joint_limits_match_backend_chain():
    _, chain = arm()
    lower, upper = joint_limits(arm_xml(), 'l0', 'tool', ['j%d' % i for i in range(6)])
    torch.testing.assert_close(lower, chain.lower, atol=0, rtol=0)
    torch.testing.assert_close(upper, chain.upper, atol=0, rtol=0)


def point_fk(z):
    """Translation-only toy arm; z(q) gives the height."""
    def fk(q):
        pose = torch.cat((q[..., :2], z(q)[..., None], torch.zeros_like(q[..., :3]), torch.ones_like(q[..., :1])), -1)
        return pose
    return fk


POINT = dict(vertices_in_ee=[[0., 0., 0.]], board_origin_base=[0., 0., 0.], board_normal_base=[0., 0., 1.],
             stop_gap_m=STOP)


def test_singular_or_near_singular_normal_direction_raises():
    q = torch.tensor([[.1, .2, -1e-3, 0., 0., 0.]], dtype=torch.double)
    blind = point_fk(lambda q: q[..., 2]*0 - 1e-3)          # no joint moves the blade along the normal
    with pytest.raises(FloatingPointError, match='singular'):
        project_joint_endpoint(q, blind, POINT)
    weak = point_fk(lambda q: 1e-5*q[..., 2])                # 10 um per rad: near-singular
    q = torch.tensor([[.1, .2, -100., 0., 0., 0.]], dtype=torch.double)
    with pytest.raises(FloatingPointError, match='near-singular'):
        project_joint_endpoint(q, weak, POINT)


def test_joint_limit_binding_after_projection():
    # Redundant height: joints 2 and 3 both lift; minimum-norm shares the lift.
    fk = point_fk(lambda q: q[..., 2] + q[..., 3])
    inf = float('inf')
    upper = torch.tensor([inf, inf, inf, 0., inf, inf], dtype=torch.double)
    limits = (-upper, upper)
    # Joint 3 already at its upper limit: clamping removes half the lift -> penetration -> error.
    q = torch.tensor([[0., 0., -1e-3, 0., 0., 0.]], dtype=torch.double)
    with pytest.raises(FloatingPointError, match='joint limit'):
        project_joint_endpoint(q, fk, POINT, limits=limits)
    # Without the limit the same proposal is corrected exactly.
    out, hit = project_joint_endpoint(q, fk, POINT)
    assert hit.all() and gap_from_pose(fk(out), POINT).item() == pytest.approx(STOP + 2e-7, abs=1e-12)
    # A binding limit that still clears the stop (loses 0.1 um of the 0.2 um margin)
    # is accepted and respected; the removed displacement is then not purely normal.
    share = (STOP + 2e-7 + 1e-3) / 2
    upper = torch.tensor([inf, inf, inf, share - 1e-7, inf, inf], dtype=torch.double)
    out, hit = project_joint_endpoint(q, fk, POINT, limits=(torch.full((6,), -inf, dtype=torch.double), upper))
    assert hit.all() and out[0, 3].item() == upper[3].item()
    assert gap_from_pose(fk(out), POINT).item() == pytest.approx(STOP + 1e-7, abs=1e-12)


def test_analytic_chain_jacobian_matches_autograd_and_gives_the_same_projection():
    from cutting_gpu_backend.tool_mount import chain_pose_jacobian
    fk, chain = arm()
    q = torch.tensor([Q0, [-.4, -1.2, 1.4, -1.9, 1.3, -.6]], dtype=torch.double)
    target = fk(q).detach().clone()
    target[:, :3] += .01
    target[:, 3:] = torch.nn.functional.normalize(target[:, 3:] + .05, dim=-1)
    with torch.enable_grad():
        auto = _residual_jacobian(fk, q.clone().requires_grad_(True), target, False)
    analytic = _residual_jacobian(fk, q, target, False, chain_pose_jacobian(chain))
    torch.testing.assert_close(analytic[0], auto[0], atol=1e-14, rtol=0)
    torch.testing.assert_close(analytic[1], auto[1], atol=1e-12, rtol=0)
    geometry = resting_geometry(fk, q[:1])
    n = torch.tensor([0., 0., 1.], dtype=torch.double)
    proposed = translate(fk, q[:1], torch.tensor([3e-4, -2e-4, 0.], dtype=torch.double) - 5e-4*n)
    a, _ = project_joint_endpoint(proposed, fk, geometry)
    b, _ = project_joint_endpoint(proposed, fk, geometry, jacobian=chain_pose_jacobian(chain))
    torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
    v = torch.full_like(a, -.05)
    torch.testing.assert_close(project_joint_velocity(a, v, fk, geometry, torch.ones(1, dtype=torch.bool)),
                               project_joint_velocity(a, v, fk, geometry, torch.ones(1, dtype=torch.bool),
                                                      jacobian=chain_pose_jacobian(chain)), atol=1e-12, rtol=0)
