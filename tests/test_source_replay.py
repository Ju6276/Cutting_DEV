import torch
from third_party.supermimic.replay_buffer import ReplayBuffer


def test_windows_keep_terminal_state_and_do_not_cross_reset():
    buffer = ReplayBuffer(1, 32, 1, 1, 0, "cpu")
    env = torch.tensor([0])
    for episode in range(2):
        for step in range(4):
            terminal = step == 3
            buffer.add(env, torch.tensor([[100 * episode + step]], dtype=torch.float32),
                       torch.zeros(1, 1), torch.ones(1, 1), torch.empty(1, 0), torch.zeros(1),
                       torch.tensor([float(terminal)]), torch.tensor([step]), torch.tensor([episode]),
                       torch.tensor([terminal]), torch.tensor([step == 0]))
        buffer.end_episode(env)
    batch = buffer.sample_windows(20, 3)
    torch.testing.assert_close(batch.states[1:] - batch.states[:-1], torch.ones(3, 20, 1))
    assert batch.done[-1].all()
    assert not batch.done[:-1].any()
