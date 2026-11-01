import numpy as np

from cutting_world_model.dynamics import FIELDS, pack_trajectory


def test_transition_alignment_and_history(tmp_path):
    data = {
        "particle_q": np.arange(36).reshape(3, 4, 3),
        "particle_qd": np.zeros((3, 4, 3)),
        "cut_spring_ke": np.array([[10, 20], [9, 18], [8, 0]]),
        "knife_pose": np.zeros((3, 7)),
        "knife_velocity": np.zeros((3, 6)),
        "force": np.arange(3)[:, None],
        "time": np.arange(3) * 0.05,
        "actions": np.ones((2, 6)),
    }
    path = tmp_path / "trajectory.npz"
    np.savez(path, **data)
    state, action, layout = pack_trajectory(path)
    assert len(state) == len(action) + 1
    for field in FIELDS:
        begin, end = layout[field]
        np.testing.assert_array_equal(state[:, begin:end], data[field].reshape(3, -1))
