import numpy as np
import pytest

from cutting_world_model.training import EpisodeWindows


def episode(path, value, terminal_at=9):
    count = 11
    state = np.arange(count, dtype=np.float32) + value
    done = np.zeros(count - 1)
    done[terminal_at] = 1
    np.savez(path, time=np.arange(count) * .05,
             particle_q=np.tile(state[:, None, None], (1, 2, 3)),
             particle_qd=np.zeros((count, 2, 3)), cut_spring_ke=np.ones((count, 3)),
             knife_pose=np.zeros((count, 7)), knife_velocity=np.zeros((count, 6)),
             force=state[:, None], actions=np.tile(state[:-1, None], (1, 6)),
             rewards=state[1:], terminated=done, truncated=np.zeros(count - 1))


def test_windows_keep_episode_and_action_alignment(tmp_path):
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    episode(first, 0)
    episode(second, 100)
    windows = EpisodeWindows([first, second])
    states, actions, rewards, done = windows.batch([2, 3], "cpu")
    np.testing.assert_array_equal(states[:-1, :, 0], actions[:, :, 0])
    np.testing.assert_array_equal(states[1:, :, 0], rewards)
    assert states[:, 0, 0].tolist() == list(range(2, 11))
    assert states[:, 1, 0].tolist() == list(range(100, 109))
    assert done[-1, 0] == 1
    assert done[:, 1].sum() == 0


def test_reject_windows_crossing_terminal_reset(tmp_path):
    path = tmp_path / "invalid.npz"
    episode(path, 0, terminal_at=4)
    with pytest.raises(ValueError, match="episode boundary"):
        EpisodeWindows([path])


def test_normalization_fits_training_episodes_only(tmp_path):
    train_path, held_out = tmp_path / "train.npz", tmp_path / "held_out.npz"
    episode(train_path, 0)
    episode(held_out, 1000)
    train = EpisodeWindows([train_path])
    validation = EpisodeWindows([held_out])
    mean, scale, reward_mean, reward_scale = train.statistics()
    assert mean[0] == 5
    assert validation.statistics()[0][0] == 1005
    assert np.isfinite(scale).all() and (scale > 0).all()
    assert reward_mean == 5.5
