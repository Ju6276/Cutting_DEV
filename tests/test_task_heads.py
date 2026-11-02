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
