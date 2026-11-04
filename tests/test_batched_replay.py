import torch
from cutting_world_model.online_replay import BatchedOnlineReplay


def test_independent_terminals_truncation_and_window_boundaries():
    replay = BatchedOnlineReplay(3, 2, 'cpu')
    action = torch.zeros(3, 6)
    goal = torch.zeros(3, 7)
    for t in range(12):
        state = torch.tensor([[t, 0.], [t, 1.], [t, 2.]])
        term = torch.tensor([t == 3, False, False])
        trunc = torch.tensor([False, t == 5, False])
        replay.add(state, action, action, goal, torch.ones(3), state + torch.tensor([1., 0.]), term, trunc)
    b = replay.buffer
    assert b.ep_counter.tolist() == [1, 1, 0]
    assert b.done[0, 4] == 1 and b.terminal[0, 4]
    assert b.done[1, 6] == 0 and b.terminal[1, 6]
    windows = b.sample_windows(128, 3)
    torch.testing.assert_close(windows.states[1:, :, 0] - windows.states[:-1, :, 0],
                               windows.valid_steps.float())
    torch.testing.assert_close(windows.states[:, :, 1], windows.states[:1, :, 1].expand(4, -1))
