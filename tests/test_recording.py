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


def test_recorder_uses_aligned_frame_pose_not_stale_material_link_cache():
    from types import SimpleNamespace
    import torch
    from scripts.record_episode import GeometryRecorder
    recorder=GeometryRecorder.__new__(GeometryRecorder)
    vertices=torch.tensor([[1.,0,0],[0.,1,0]],dtype=torch.float64)
    def stale_cache(_):
        raise AssertionError('body_X_sc is stale immediately after reset alignment')
    recorder.material=SimpleNamespace(model=SimpleNamespace(knife_tri_vertices=vertices),
                                      state_field=stale_cache)
    recorder.env=SimpleNamespace(device=torch.device('cpu'))
    recorder.steps=[]
    recorder.to_world=lambda x:torch.as_tensor(x).numpy()
    q=2**-.5
    frame=dict(joint_q=torch.zeros(1,6),particle_q=torch.zeros(1,2,3),
               knife_pose=torch.tensor([[2.,3.,4.,0.,0.,q,q]],dtype=torch.float64),
               cut_spring_ke=torch.ones(1,2),sim_time=torch.zeros(1,1),peak_wrench=torch.zeros(1,6))
    recorder(frame,None)
    np.testing.assert_allclose(recorder.steps[0]['knife'],[[2.,4.,4.],[1.,3.,4.]],atol=1e-12)
