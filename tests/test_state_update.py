import torch

from cutting_world_model.state_update import compose_rotation, structured_update


def test_rotation_composition_has_unit_norm_and_finite_zero_angle_gradient():
    q = torch.tensor([[0., 0., 0., 1.]], dtype=torch.float64)
    rotation = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(lambda value: compose_rotation(q, value), rotation)
    result = compose_rotation(q, torch.tensor([[.4, -.2, .7]], dtype=torch.float64))
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(1, dtype=torch.float64))


def test_known_context_and_irreversible_damage():
    layout = {"target_pose": (0, 2), "step_count": (2, 3),
              "motion_command": (3, 9), "cut_spring_ke": (9, 12)}
    state = torch.tensor([[7., 8., 3., 0., 0., 0., 0., 0., 0., 0., 10., 20.]])
    proposal = torch.ones_like(state) * 99
    proposal[0, 9:] = torch.tensor([5., -5., 15.])
    action = torch.tensor([[0., 0., 0., 0., .5, -.5]])
    updated = structured_update(state, proposal, action, layout, [.002, .004])
    torch.testing.assert_close(updated[:, :3], torch.tensor([[7., 8., 4.]]))
    torch.testing.assert_close(updated[:, 3:9], torch.tensor([[.001, 0., 0., 0., .002, 0.]]))
    torch.testing.assert_close(updated[:, 9:], torch.tensor([[0., 0., 15.]]))
