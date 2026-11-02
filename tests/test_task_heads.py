import torch

from cutting_world_model.task_heads import SliceItHeads


def heads():
    return SliceItHeads({"target_dims": [0, 2, 4], "wrench_hist_size": 3,
        "max_distance": [.1, .1, .06, .785398, .785398, 1.5707],
        "normalize_velocity": True, "jerkiness_as_obs": True, "last_action_as_obs": True,
        "goal_threshold": [.04, .002, .04], "workspace_limit": [.05, .1, .174533],
        "successes_threshold": 10, "termination_on_negative_reward": False,
        "cost": {"collision": -100., "done": 100., "step": -1.},
        "w_dist": 1., "w_force": 1., "w_jerkiness": 1.},
        {"max_force_torque": [50., 50., 50., 10., 10., 10.]})


def test_continuous_reward_derivative():
    observation = torch.full((1, 24), .1, dtype=torch.float64, requires_grad=True)
    false = torch.zeros(1, dtype=torch.bool)
    torch.autograd.gradcheck(lambda value: heads().reward(value, false, false, false), observation)


def test_force_history_reward_and_gradient_match_physical_component_means():
    head = heads()
    head.history = 4
    wrench = (torch.arange(2 * 3 * 4 * 6, dtype=torch.float64).reshape(2, 3, 4, 6) / 100 + .1)
    wrench.requires_grad_(True)
    zero = torch.zeros(2, 3, 6, dtype=torch.float64)
    observation = head.observation(zero, zero, zero, zero, wrench)
    false = torch.zeros(2, 3, dtype=torch.bool)
    reward = head.reward(observation, false, false, false)
    component_means = wrench[..., [0, 2, 4]].mean(dim=-2)
    expected = (-torch.sigmoid(torch.linalg.vector_norm(component_means, dim=-1) / 2 - 3) - 1) / 4
    torch.testing.assert_close(reward, expected)
    actual_gradient = torch.autograd.grad(reward.sum(), wrench, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), wrench)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_collision_suppresses_success_bonus():
    obs = torch.ones(1, 24) * .1
    yes, no = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    head = heads()
    normal = head.reward(obs, no, no, no)
    torch.testing.assert_close(head.reward(obs, yes, yes, no) - normal, torch.tensor([-100.]))
    torch.testing.assert_close(head.reward(obs, no, yes, no) - normal, torch.tensor([100.]))


def test_original_success_counter_keeps_nonconsecutive_hits():
    head = heads()
    obs = torch.zeros(1, 24)
    false = torch.zeros(1, dtype=torch.bool)
    done, success, _, count = head.termination(obs, torch.tensor([9.]), false)
    assert not done.item() and count.item() == 10
    obs[0, 1] = 1
    done, success, _, count = head.termination(obs, count, false)
    assert count.item() == 10 and not success.item()
    obs[0, 1] = 0
    done, success, _, count = head.termination(obs, count, false)
    assert done.item() and success.item() and count.item() == 11
