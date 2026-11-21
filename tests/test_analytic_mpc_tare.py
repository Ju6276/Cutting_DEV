"""Analytic MPC must retain execution tare across candidates, time and worlds."""
import copy
from types import SimpleNamespace
import pytest
import torch

from cutting_gpu_backend.controller.fdcc import FDCCController
from cutting_world_model.shared_baselines import AnalyticFDCCModel, SharedMPPIPolicy, frame_metadata
from cutting_world_model.mpc import MPPIConfig
from test_board_continue import continuing


def fixture(worlds=1):
    wm, state, goal = continuing()
    meta = copy.deepcopy(wm.spec.metadata)
    meta['robot_xml'] = meta['robot_xml'].replace('</joint>',
        '<limit lower="-3" upper="3" velocity="2"/></joint>')
    labels = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    params = {'compliance_ref_link':'l6', 'ft_sensor_ref_link':'l6',
        'stiffness':dict(zip(labels,[1000]*3+[40]*3)),
        'solver':{'iterations':1,'forward_dynamics':{'link_mass':.1},'error_scale':.015},
        'pd_gains':{k:{'p':.1,'d':0} for k in labels},
        'force':{'hand_frame_control':False},
        # Nonzero gravity orthogonal to the toy robot's rotation axis.
        'gravity':{'x':0.,'y':-9.8,'z':0.},
        'tool':{'mass':.35,'com_x':.02,'com_y':-.03,'com_z':.07}}
    params['stiffness'].update(dict(zip(['sel_x','sel_y','sel_z','sel_ax','sel_ay','sel_az'],[1,1,.5,1,.5,1])))
    params['stiffness']['use_parallel_force_position_control']=True
    model = AnalyticFDCCModel(meta,params,substeps=1)
    sp = model.spec
    state = state.repeat(worlds,1)
    frame = {name:sp.field(state,name).clone() for name in sp.base_fields}
    frame['success_counter'] = frame['success_counter'][:,None]
    frame['joint_q'][:,0] += torch.arange(worlds,dtype=state.dtype)*.2
    frame['target_pose'] = sp.fk(frame['joint_q'])
    frame['target_speed'] = torch.full((worlds,1),.03,dtype=state.dtype)
    ctl = FDCCController(model.chain, model.controller_config)
    ctl.reset(frame['joint_q'])
    frame['controller_tare_compensation'] = ctl.tare_compensation.clone()
    return model, params, frame, meta


def test_prediction_keeps_actual_tare_when_robot_rotates(monkeypatch):
    model, _, frame, _ = fixture()
    sp = model.spec
    actual_tare = frame['controller_tare_compensation'].clone()
    # Assigning the explicit context also works as a regression against the
    # previous class: its predict() ignored this attribute and retared itself.
    model.tare_compensation = actual_tare.clone()
    state = sp.pack_batch(frame)
    controller = FDCCController(model.chain,model.controller_config)
    captured = []
    original = FDCCController.force_error
    def check_error(self, frames, reading, desired):
        out = original(self,frames,reading,desired)
        captured.append(out.clone())
        return out
    monkeypatch.setattr(FDCCController,'force_error',check_error)
    for angle in (.4,-.3,.6):
        # Four different candidate orientations share this one physical baseline.
        q = frame['joint_q'].repeat(4,1)
        q[:,0] += angle + torch.arange(4,dtype=q.dtype)*.03
        frames = model.chain(q,mass_matrix=False)[0]
        reading = controller._gravity_sensor_wrench(frames)-actual_tare
        batch = state.repeat(4,1)
        sp.field(batch,'joint_q').copy_(q)
        sp.field(batch,'joint_qd').zero_()
        sp.field(batch,'wrench_history').copy_(reading[:,None,:])
        sp.field(batch,'last_wrench').copy_(reading)
        goal = torch.cat((sp.fk(q),q.new_full((4,1),.03)),-1)
        model.predict(batch,q.new_zeros((4,6)),goal)
        torch.testing.assert_close(captured[-1],torch.zeros_like(captured[-1]),atol=1e-12,rtol=0)
    torch.testing.assert_close(model.tare_compensation,actual_tare,atol=0,rtol=0)


def test_nonzero_payload_refuses_missing_or_malformed_context():
    model, _, frame, _ = fixture()
    goal = torch.cat((frame['target_pose'],frame['target_speed']),-1)
    with pytest.raises(ValueError,match='physical controller tare'):
        model.predict(model.spec.pack_batch(frame),torch.zeros(1,6),goal)
    missing = {k:v for k,v in frame.items() if k!='controller_tare_compensation'}
    with pytest.raises(ValueError,match='controller_tare_compensation'):
        model.tare_from_frame(missing)
    for invalid in (torch.empty(0,6),torch.ones(1,5),torch.full((1,6),float('nan'))):
        with pytest.raises(ValueError,match='Finite'):
            model.set_tare_compensation(invalid)
    model.set_tare_compensation(frame['controller_tare_compensation'])
    saved = model.tare_compensation.clone()
    frame['controller_tare_compensation'].zero_()
    torch.testing.assert_close(model.tare_compensation,saved,atol=0,rtol=0)


def test_real_mppi_context_is_isolated_across_candidates_worlds_and_reset(monkeypatch):
    pytest.importorskip('pytorch_mppi')
    model, params, frame, meta = fixture(worlds=2)
    env = SimpleNamespace(worlds=2,device=torch.device('cpu'),current=frame,metadata=meta)
    metadata_without_context = frame_metadata(SimpleNamespace(metadata=meta,
        current={k:v for k,v in frame.items() if k!='controller_tare_compensation'}))
    assert frame_metadata(env)==metadata_without_context
    planner = SharedMPPIPolicy(env,MPPIConfig(horizon=3,samples=8,discount=1.),
                              controller_parameters=params,analytic_substeps=1)
    assert planner.rollouts[0].wm is not planner.rollouts[1].wm
    assert planner.spec.state_dim == model.spec.state_dim
    assert 'controller_tare_compensation' not in planner.spec.base_fields
    tare = frame['controller_tare_compensation'].clone()
    assert not torch.equal(tare[0],tare[1])
    captured = []
    original = AnalyticFDCCModel.predict
    def inspect(self,state,action,goal):
        captured.append((id(self),len(state),self.tare_compensation.clone()))
        return original(self,state,action,goal)
    monkeypatch.setattr(AnalyticFDCCModel,'predict',inspect)
    torch.manual_seed(11)
    assert torch.isfinite(planner.act(frame)).all()
    for w, rollout in enumerate(planner.rollouts):
        calls = [row for row in captured if row[0]==id(rollout.wm)]
        assert len(calls)>=3 and any(row[1]==8 for row in calls)
        for _,_,baseline in calls:
            torch.testing.assert_close(baseline,tare[w:w+1],atol=0,rtol=0)
    changed = {k:v.clone() for k,v in frame.items()}
    changed['controller_tare_compensation'][0] += .1
    changed['controller_tare_compensation'][1] += 99 # unselected row must be ignored
    planner.reset(changed,torch.tensor([True,False]))
    torch.testing.assert_close(planner.rollouts[0].wm.tare_compensation,
                               changed['controller_tare_compensation'][:1],atol=0,rtol=0)
    torch.testing.assert_close(planner.rollouts[1].wm.tare_compensation,tare[1:],atol=0,rtol=0)
    changed['controller_tare_compensation'][1] = tare[1]
    planner.observe(changed)
    before = planner.rollouts[0].wm.tare_compensation.clone()
    changed['controller_tare_compensation'].zero_()
    torch.testing.assert_close(planner.rollouts[0].wm.tare_compensation,before,atol=0,rtol=0)
    missing = {k:v for k,v in changed.items() if k!='controller_tare_compensation'}
    with pytest.raises(ValueError,match='controller_tare_compensation'):
        planner.observe(missing)
    with pytest.raises(ValueError,match='world count'):
        planner.observe({k:v[:1] for k,v in changed.items()})


def test_blademaster_frame_exports_detached_controller_tare():
    from cutting_world_model.blademaster_backend import BladeMasterBatch
    model, _, frame, _ = fixture(worlds=2)
    batch = BladeMasterBatch.__new__(BladeMasterBatch)
    batch.chain = model.chain
    batch.frames = model.chain(frame['joint_q'],mass_matrix=False)[0]
    batch.board_geometry = model.spec.board_geometry
    batch.robot = SimpleNamespace(q=frame['joint_q'],qd=frame['joint_qd'])
    batch.controller = SimpleNamespace(tare_compensation=frame['controller_tare_compensation'])
    batch.target = frame['target_pose']
    batch.wrench_queue = torch.zeros(2,25,6,dtype=torch.double)
    batch.task = {'environment':{'wrench_hist_size':3}}
    batch.peak = torch.zeros(2,6,dtype=torch.double)
    batch.mapper = SimpleNamespace(added_motion_command=torch.zeros(2,6,dtype=torch.double))
    batch.progress = torch.zeros(2,1)
    batch.reward_depth_span = torch.ones(2,1)
    batch.contacted = torch.zeros(2,dtype=torch.bool)
    batch.sim_time = torch.zeros(2)
    exported = batch.frame()['controller_tare_compensation']
    torch.testing.assert_close(exported,batch.controller.tare_compensation,atol=0,rtol=0)
    exported.zero_()
    assert batch.controller.tare_compensation.abs().max()>0

