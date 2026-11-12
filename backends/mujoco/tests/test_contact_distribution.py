"""Run the production SDF contact kernel alone: no integration or other forces."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/disect"))


def contact_case(worlds=1, endpoint=True, x_override=None, v_override=None, triangular=False):
    import disect.dflex as df
    from disect.dflex.sim import eval_spring_sdf_contacts, eval_spring_tri_contacts
    from disect.dflex.adjoint import Tape
    df.config.no_grad = True
    device = "cuda:0"
    def tensor(x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=device)
    # Left goes inside -> outside. Right stored edge goes outside -> inside,
    # and the production kernel reverses its endpoints before searching.
    x = tensor([[.004, .01, 0.], [.03, .015, .01],
                [-.03, .015, .02], [-.004, .01, .01]]).repeat(worlds, 1)
    v = tensor([[.01, .02, .03], [-.03, .01, -.02],
                [.04, -.02, .01], [-.01, .03, -.04]]).repeat(worlds, 1)
    if x_override is not None:
        x = x_override.to(device=device, dtype=torch.float32).contiguous()
    if v_override is not None:
        v = v_override.to(device=device, dtype=torch.float32).contiguous()
    spring = torch.arange(2*worlds, dtype=torch.int32, device=device).reshape(-1, 2)
    edges = torch.arange(4*worlds, dtype=torch.int32, device=device).reshape(-1, 2)
    # Search interval collapses to u=0 on both sides in the analytic endpoint case.
    coords = tensor([0., 1.] if endpoint else [.8, .3]).repeat(worlds)
    poses = tensor([[0., 0., 0., 0., 0., 0., 1.]]).repeat(worlds, 1)
    twists = tensor([[0., .2, 0., 0., 0., .01]]).repeat(worlds, 1)
    params = tensor([.02, .04, .002, .005, .1])
    particle_f, knife_f = torch.zeros_like(x), torch.zeros(2*worlds, 3, device=device)
    coord, distance = torch.zeros(2*worlds, device=device), torch.zeros(2*worlds, device=device)
    normal = torch.zeros(2*worlds, 3, device=device)
    if triangular:
        assert worlds == 1  # inactive triangle path does not implement world batching
        coord.copy_(tensor([.2, .7]))
        distance.fill_(-.001)
        normal.copy_(tensor([[1., 0., 0.], [-1., 0., 0.]]))
        Tape().launch(
            func=eval_spring_tri_contacts, dim=worlds,
            inputs=[spring, edges, coords, worlds, x, v, 0, poses, twists,
                    coord, distance, normal, tensor([1000.]), tensor([10.]),
                    tensor([.01]), tensor([.5]), tensor([0.])],
            outputs=[particle_f, knife_f], adapter=device)
    else:
        Tape().launch(
            func=eval_spring_sdf_contacts, dim=worlds,
            inputs=[spring, edges, coords, worlds, x, v, 0, 1, 1, poses, twists, params,
                    tensor([1000.]*worlds), tensor([10.]*worlds),
                    tensor([.01]*worlds), tensor([.5]*worlds), 0.],
            outputs=[particle_f, knife_f, coord, distance, normal], adapter=device)
    # Independent CPU double-precision accounting of the emitted CUDA forces.
    x, v, particle_f, knife_f, coord, distance = [
        a.detach().cpu().double() for a in (x, v, particle_f, knife_f, coord, distance)]
    first = torch.arange(worlds)[:, None]*4 + torch.tensor([0, 3])
    second = torch.arange(worlds)[:, None]*4 + torch.tensor([1, 2])
    first, second = first.flatten(), second.flatten()
    point = x[first]*(1-coord[:, None]) + x[second]*coord[:, None]
    velocity = v[first]*(1-coord[:, None]) + v[second]*coord[:, None]
    def sum_world(a, n):
        return a.reshape(worlds, n, *a.shape[1:]).sum(1)
    force_residual = sum_world(particle_f, 4)-sum_world(knife_f, 2)
    moment_residual = sum_world(torch.linalg.cross(x, particle_f), 4)-sum_world(
        torch.linalg.cross(point, knife_f), 2)
    power_residual = sum_world((v*particle_f).sum(-1), 4)-sum_world((velocity*knife_f).sum(-1), 2)
    return dict(coords=coord, distance=distance, force=knife_f, point=point, contact_velocity=velocity,
                force_residual=force_residual, moment_residual=moment_residual,
                power_residual=power_residual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contact kernel")
@pytest.mark.parametrize("worlds,endpoint", [(1, True), (4, False)])
def test_sdf_contact_location_force_moment_and_virtual_power(worlds, endpoint):
    case = contact_case(worlds, endpoint)
    assert (case["force"].norm(dim=-1).reshape(worlds, 2).max(-1).values > .1).all()
    if endpoint:
        # The search interval [0,0] admits only the first endpoint, on both sides.
        torch.testing.assert_close(case["coords"], torch.zeros_like(case["coords"]), atol=0, rtol=0)
    else:
        upper = torch.tensor([.8, .7], dtype=torch.float64).repeat(worlds)
        assert ((case["coords"] >= 0) & (case["coords"] <= upper+1e-6)).all()
    # Absolute tolerances reflect float32 accumulation, in N, N*m and W respectively.
    for name, atol in (("force_residual", 2e-6), ("moment_residual", 2e-7), ("power_residual", 2e-7)):
        torch.testing.assert_close(case[name], torch.zeros_like(case[name]), atol=atol, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contact kernel")
def test_reported_coordinate_reconstructs_the_sdf_evaluation_point():
    case = contact_case(4, endpoint=False)
    # Query the same production SDF at the reconstructed point using zero-length edges.
    query = contact_case(4, endpoint=True,
                         x_override=case["point"].repeat_interleave(2, dim=0),
                         v_override=case["contact_velocity"].repeat_interleave(2, dim=0))
    torch.testing.assert_close(query["distance"], case["distance"], atol=2e-7, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contact kernel")
def test_triangle_contact_scatter_matches_its_position_and_velocity_interpolation():
    case = contact_case(triangular=True)
    assert (case["force"].norm(dim=-1) > .1).all()
    for name, atol in (("force_residual", 2e-6), ("moment_residual", 2e-7), ("power_residual", 2e-7)):
        torch.testing.assert_close(case[name], torch.zeros_like(case[name]), atol=atol, rtol=0)
