import numpy as np

from scripts.record_episode import interpolate_steps


def test_display_never_anticipates_force_or_release_events():
    states = [
        {'sim_time': 0., 'joint_q': np.array([0.]), 'particles': np.array([[0., 0., 0.]]),
         'knife': np.array([[0., 1., 0.]]), 'force': 0., 'sensor': 0., 'released': 0},
        {'sim_time': .012, 'joint_q': np.array([1.]), 'particles': np.array([[1., 0., 0.]]),
         'knife': np.array([[1., 1., 0.]]), 'force': 55., 'sensor': 51., 'released': 3},
    ]
    frames = interpolate_steps(states, 4)
    np.testing.assert_allclose([x['sim_time'] for x in frames], [0., .003, .006, .009, .012])
    assert [x['force'] for x in frames] == [0., 0., 0., 0., 55.]
    assert [x['released'] for x in frames] == [0, 0, 0, 0, 3]
    np.testing.assert_array_equal(frames[2]['particles'], [[.5, 0., 0.]])
    np.testing.assert_array_equal(states[0]['particles'], [[0., 0., 0.]])
    assert frames[-1] is states[-1]
