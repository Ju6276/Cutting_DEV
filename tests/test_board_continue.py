"""Current V3 has a per-step board boundary, never an absorbing physical latch."""
import copy
import pytest
import torch
from test_pose_depth_v3 import model, put, joints_for, joint_delta, gap
from cutting_world_model.world_model import CuttingWorldModel

def continuing(motion='project_continue_v2'):
    old, state, goal = model()
    meta=copy.deepcopy(old.spec.metadata)
    meta['free_cut_task']['board']['motion'] = motion
    meta['free_cut_task']['tool_revision'] = 'disect_knife_rigid_mount_v1'
    wm=CuttingWorldModel(meta, hidden=(12,12)).double()
    wm.load_state_dict(old.state_dict())
    return wm,state,goal

def test_contact_state_can_lift_and_fields_are_predicted():
    wm,s,g=continuing(); sp=wm.spec
    assert not sp.board_latch
    s=put(sp,s,board_contact=[1.],last_wrench=[1.,2.,3.,.1,.2,.3])
    q=joints_for(sp,sp.field(s,'joint_q'),0.,.001)
    fields=sp.integrate_physical(s,joint_delta(sp,s,q))
    torch.testing.assert_close(fields['joint_q'],q)
    assert fields['board_contact'].item()==0
    assert gap(sp,sp.assemble(fields)).item() > .001
    # No model transition mask is enabled, including records starting at contact.

def test_downward_plus_lateral_motion_is_not_permanently_blocked():
    wm,s,g=continuing(); sp=wm.spec
    q=joints_for(sp,sp.field(s,'joint_q'),0.,-.002)
    fields=sp.integrate_physical(s,joint_delta(sp,s,q))
    assert fields['board_contact'].item()==1
    assert gap(sp,sp.assemble(fields)).item() >= sp.board_geometry['stop_gap_m']-1e-9
    s=sp.assemble(fields)
    old_pose=sp.fk(sp.field(s,'joint_q'))
    q=joints_for(sp,sp.field(s,'joint_q'),.001,-.0001)
    out=sp.integrate_physical(s,joint_delta(sp,s,q))
    assert (sp.fk(out['joint_q'])-old_pose)[0,0].abs() > .0005
    assert gap(sp,sp.assemble(out)).item() >= sp.board_geometry['stop_gap_m']-1e-9

def test_new_config_is_explicit_and_old_checkpoint_still_latches():
    from cutting_world_model.free_cut import validate_task
    wm,_,_=continuing()
    validate_task(wm.spec.metadata['free_cut_task'])
    assert model()[0].spec.board_latch
    bad=copy.deepcopy(wm.spec.metadata['free_cut_task'])
    bad['board']['motion']='unknown'
    with pytest.raises(ValueError): validate_task(bad)

def test_all_new_predictions_respect_peak_last_relation():
    wm,s,g=continuing(); sp=wm.spec
    s=put(sp,s,board_contact=[1.])
    fields=sp.integrate_physical(s,torch.zeros(1,sp.delta_dim,dtype=s.dtype))
    fields['last_wrench']=s.new_tensor([[55.,0.,0.,0.,0.,0.]])
    fields['peak_wrench']=torch.zeros_like(fields['last_wrench'])
    out=sp.apply_board_latch(s,fields,torch.tensor([False]),s.new_tensor([.001]))
    assert out['peak_wrench'][0,0]==55
    assert out['last_wrench'][0,0]==55
    assert not out['board_contact'].any()
    fields['peak_wrench'][0,0]=-100.
    out=sp.apply_board_latch(s,fields,torch.tensor([False]),s.new_tensor([.001]))
    assert out['peak_wrench'][0,0]==100, 'Projection must not erase a predicted force violation'


def test_projection_gradient_and_inactive_world():
    from cutting_gpu_backend.tool_mount import project_joint_endpoint, continuing_joint_step
    def fk(q):
        quat=torch.cat((q[...,3:]*0,torch.ones_like(q[...,:1])),-1)
        return torch.cat((q[...,:3],quat),-1)
    geometry=dict(vertices_in_ee=[[0.,0.,0.],[.02,0.,0.]],board_origin_base=[0.,0.,0.],
                  board_normal_base=[0.,0.,1.],stop_gap_m=1e-6)
    q=torch.tensor([[.01,.02,-.001,0.,0.,0.]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:project_joint_endpoint(x,fk,geometry)[0],(q,),
                                   eps=1e-7,atol=2e-5,rtol=1e-3)
    old=q.detach().repeat(2,1);old[:,2]=1.2e-6
    new=old.clone();new[:,0]+=.001;new[:,2]-=.001
    out,hit=continuing_joint_step(old,new,fk,geometry,0.,torch.tensor([True,False]))
    assert hit.tolist()==[True,False]
    torch.testing.assert_close(out[1],old[1])
    assert out[0,0]==new[0,0] and out[0,2]>=geometry['stop_gap_m']
    lifted=old.clone();lifted[:,2]+=.002
    out,hit=continuing_joint_step(old,lifted,fk,geometry,1.,torch.ones(2,dtype=torch.bool))
    torch.testing.assert_close(out,lifted)
    assert not hit.any()

def test_contact_transitions_are_fitted_not_masked():
    wm,s,g=continuing();sp=wm.spec
    s=put(sp,s,board_contact=[1.])
    a=torch.zeros(1,6,dtype=s.dtype)
    nxt,_,_=wm.step(s,a,g)
    target=nxt.detach().clone()
    target[:,sp.layout['last_wrench']]+=1
    loss,m=wm.masked_window_loss(torch.stack((s,target)),a[None],g,
                                  torch.zeros(1,1,dtype=s.dtype),torch.ones(1,1,dtype=torch.bool))
    assert m['nse_1step_count']==1
    assert m['dynamics_fit_count']==1
    assert m['dynamics_masked_count']==0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in wm.dynamics.parameters())


def test_cumulative_arrival_counter_and_failure_priority_retained():
    wm,s,g=continuing();sp=wm.spec
    sp.heads.env['successes_threshold']=10
    # Counter 0 cannot succeed simply because pose already equals goal.
    fields=sp.integrate_physical(s,torch.zeros(1,sp.delta_dim,dtype=s.dtype))
    a=torch.zeros(1,6,dtype=s.dtype)
    nxt,reward,done=sp.derive(s,fields,a,g)
    assert sp.field(nxt,'success_counter').item()==1
    assert not done.item()
    s=put(sp,s,success_counter=[10.])
    fields=sp.integrate_physical(s,torch.zeros(1,sp.delta_dim,dtype=s.dtype))
    nxt,reward,done=sp.derive(s,fields,a,g)
    assert done.item() and sp.pose_success(nxt).item()
    fields['peak_wrench']=s.new_full((1,6),1000.)
    nxt,reward,done=sp.derive(s,fields,a,g)
    assert done.item() and not sp.pose_success(nxt).item() and reward.item()<0

def test_new_and_historical_metadata_cannot_be_paired():
    from cutting_world_model.checkpoint import check_checkpoint_pair
    wm,_,_=continuing();old,_,_=model()
    with pytest.raises(ValueError,match='metadata differ'):
        check_checkpoint_pair({'metadata':wm.spec.metadata},{'metadata':old.spec.metadata})


def test_position_command_does_not_invent_joint_velocity():
    from cutting_gpu_backend.tool_mount import project_joint_velocity
    def fk(q):
        return torch.cat((q[...,:3],q[...,3:]*0,torch.ones_like(q[...,:1])),-1)
    geometry=dict(vertices_in_ee=[[0,0,0]],board_origin_base=[0,0,0],
                  board_normal_base=[0,0,1],stop_gap_m=1e-6)
    q=torch.zeros(3,6,dtype=torch.double);q[:,2]=1.2e-6
    v=torch.zeros_like(q);v[:,0]=.1;v[:,2]=torch.tensor([-.02,.02,0.])
    result=project_joint_velocity(q,v,fk,geometry,torch.ones(3,dtype=torch.bool))
    torch.testing.assert_close(result[:,0],v[:,0])
    torch.testing.assert_close(result[:,2],q.new_tensor([0.,.02,0.]),atol=1e-9,rtol=1e-7)


def test_wm_boundary_projection_has_finite_useful_gradients():
    wm,s,g=continuing();sp=wm.spec
    q=joints_for(sp,sp.field(s,'joint_q'),.0003,-.002)
    delta=joint_delta(sp,s,q).requires_grad_(True)
    fields=sp.integrate_physical(s,delta)
    value=sp.fk(fields['joint_q'])[...,0].sum()+fields['last_wrench'].sum()
    grad=torch.autograd.grad(value,delta)[0]
    assert torch.isfinite(grad).all()
    assert grad[:,sp.delta_slices['joint_q']].norm()>0
    assert grad[:,sp.delta_slices['last_wrench']].norm()>0


def test_zero_rotation_second_derivative_for_projection():
    from cutting_world_model.kinematics import quat_exp_second_order
    for x in (0.,1e-5,.1):
        v=torch.full((1,3),x,dtype=torch.double,requires_grad=True)
        assert torch.autograd.gradgradcheck(quat_exp_second_order,(v,),atol=1e-6,rtol=1e-4)


def test_current_diagnostic_replays_through_training_imagination():
    from test_pose_depth_v3 import FixedPolicy
    from cutting_world_model.policy import SliceItCritic
    from cutting_world_model.trainer import OnlineLearner
    from third_party.supermimic.trainer_config import WMTrainerCfg
    from scripts.diagnose_pose_predictions import analyse,event_flags,EVENTS
    wm,s,g=continuing();sp=wm.spec
    s=put(sp,s,board_contact=[1.])
    learner=OnlineLearner(wm,FixedPolicy().double(),SliceItCritic(sp.state_dim,8).double(),
                          actor_horizon=2,wm_cfg=WMTrainerCfg(compile=None))
    a=torch.zeros(1,6,dtype=s.dtype)
    with torch.no_grad():
        nxt,r,_=wm.predict(s,a,g)
        flags,terminal=event_flags(sp,nxt)
        rec=dict(alive=torch.ones(1,1,dtype=torch.bool),state=s[None],next_state=nxt[None],
                 goal=g[None],history_states=s.repeat(1,10,1,1),action=a[None],
                 depth_span=sp.field(s,'depth_span').reshape(1,1),terminated=terminal[None],
                 contact_start=torch.ones(1,1,dtype=torch.bool),
                 contact_next=sp.contact(nxt).reshape(1,1),reward=r[None])
        rec.update({'real_'+e:flags[e][None] for e in EVENTS})
        result=analyse(learner,rec,k=1)
    assert result['latch']['persistent_latch'] is False
    assert result['replay_first_step_max_normalized_gap'] < 1e-6


# ---- board_contact is a state property (end-of-step clearance <= 10 um) ----

def at_gap(sp, s, clearance):
    """Fixture state whose blade rests `clearance` above the board (fixture EE starts 1 mm above)."""
    q = joints_for(sp, sp.field(s, 'joint_q'), 0., -(1e-3 - clearance))
    s = put(sp, s, joint_q=q[0])
    assert gap(sp, s).item() == pytest.approx(clearance, abs=1e-12)
    return s


def physical_chain(sp):
    """Backend SerialChain of the same fixture robot (the backend parser needs joint limits)."""
    from cutting_gpu_backend.controller.fdcc import SerialChain
    m = sp.metadata
    xml = m['robot_xml'].replace('<axis xyz="0 0 1"/>',
                                 '<axis xyz="0 0 1"/><limit lower="-6.2832" upper="6.2832" velocity="3.14"/>')
    chain = SerialChain(xml, m['robot_base'], m['robot_tip'], m['joint_names']).double()
    return lambda q: chain(q, mass_matrix=False)[0][chain.tip]


CLEARANCES = (2e-6, 9e-6, 1.1e-5, 2e-5, 5e-4)
EXPECTED = [True, True, False, False, False]


def test_resting_state_with_zero_predicted_motion_reports_contact():
    wm, s, g = continuing(); sp = wm.spec
    zero = torch.zeros(1, sp.delta_dim, dtype=s.dtype)
    for clearance, expected in zip(CLEARANCES, EXPECTED):
        # Already on (or hovering above) the board and predicted not to move: no
        # projection acts in this step, yet the flag describes the pose.
        r = put(sp, at_gap(sp, s, clearance), board_contact=[float(not expected)])
        fields = sp.integrate_physical(r, zero)
        torch.testing.assert_close(fields['joint_q'], sp.field(r, 'joint_q'), atol=0, rtol=0)
        assert bool(fields['board_contact'].item()) is expected, clearance


def test_physics_and_wm_give_the_same_contact_flag_on_the_same_pose():
    from cutting_gpu_backend.knife_board import KnifeBoard
    from cutting_gpu_backend.tool_mount import BOARD_CONTACT_GAP_M, board_contact, gap_from_pose as physical_gap
    from cutting_world_model.pose_depth import BOARD_LATCH_GAP_M
    assert BOARD_CONTACT_GAP_M == BOARD_LATCH_GAP_M == 1e-5
    wm, s, g = continuing(); sp = wm.spec
    states = torch.cat([at_gap(sp, s, c) for c in CLEARANCES])
    q = sp.field(states, 'joint_q')
    fk = physical_chain(sp)
    torch.testing.assert_close(fk(q)[:, :3], sp.fk(q)[:, :3], atol=1e-12, rtol=0)
    # DiSECt adapter flag: KnifeBoard.register_stop with the adapter's rigid-blade
    # clearance; stale previous flags and `limited` must not matter.
    board = object.__new__(KnifeBoard)
    board.continuing = True
    board.contacted = torch.tensor([False, False, True, True, True])
    n = len(CLEARANCES)
    KnifeBoard.register_stop(board, torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool),
                             gap=physical_gap(fk(q), sp.board_geometry))
    fields = sp.integrate_physical(states, torch.zeros(n, sp.delta_dim, dtype=s.dtype))
    assert board.contacted.tolist() == fields['board_contact'][:, 0].bool().tolist() == EXPECTED
    assert board_contact(physical_gap(fk(q), sp.board_geometry)).tolist() == EXPECTED
    # Inactive worlds keep their flag; a missing clearance is an error, not a silent default.
    KnifeBoard.register_stop(board, torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool),
                             gap=torch.zeros(n, dtype=torch.double))
    assert board.contacted.tolist() == EXPECTED
    with pytest.raises(ValueError, match='clearance'):
        KnifeBoard.register_stop(board, torch.ones(n, dtype=torch.bool))


def test_historical_latch_register_stop_is_unchanged():
    from types import SimpleNamespace
    from cutting_gpu_backend.knife_board import KnifeBoard
    board = SimpleNamespace(contacted=torch.tensor([False, False, True, False]),
                            gap=lambda: torch.tensor([5e-6, 5e-6, 1., 1.]))
    # No `continuing` attribute / continuing False: absorbing latch, limited AND <= 10 um.
    KnifeBoard.register_stop(board, torch.tensor([True, False, False, True]))
    assert board.contacted.tolist() == [True, False, True, False]
    board.continuing = False
    KnifeBoard.register_stop(board, torch.tensor([False, False, False, False]), gap=torch.zeros(4))
    assert board.contacted.tolist() == [True, False, True, False]


# ---- semantics revision: f12ca04 continuation (project_continue_v1) is evaluate-only ----

def test_f12ca04_continuation_keeps_its_flag_for_evaluation_but_not_training():
    from cutting_world_model.checkpoint import check_checkpoint_pair
    from cutting_world_model.free_cut import refuse_retired_training, validate_task
    old, s, g = continuing('project_continue_v1'); new, _, _ = continuing()
    validate_task(old.spec.metadata['free_cut_task'])
    assert not old.spec.board_latch and not old.spec.current_continuation and new.spec.current_continuation
    with pytest.raises(ValueError, match='project_continue_v1 .* read/evaluate-only'):
        refuse_retired_training(old.spec.metadata['free_cut_task'])
    refuse_retired_training(new.spec.metadata['free_cut_task'])
    # Its checkpoints/replay cannot be paired with (or resumed under) current metadata.
    with pytest.raises(ValueError, match='metadata differ'):
        check_checkpoint_pair({'metadata': old.spec.metadata}, {'metadata': new.spec.metadata})
    # Recorded v1 rule still evaluates as recorded: resting without projection -> 0.
    r = at_gap(old.spec, s, 2e-6)
    zero = torch.zeros(1, old.spec.delta_dim, dtype=s.dtype)
    assert old.spec.integrate_physical(r, zero)['board_contact'].item() == 0
    assert new.spec.integrate_physical(r, zero)['board_contact'].item() == 1


def test_start_within_roundoff_below_plane_can_still_lift_and_move_laterally():
    from cutting_gpu_backend.tool_mount import continuing_joint_step, gap_from_pose as physical_gap
    def fk(q):
        return torch.cat((q[..., :3], q[..., 3:] * 0, torch.ones_like(q[..., :1])), -1)
    geometry = dict(vertices_in_ee=[[0., 0., 0.]], board_origin_base=[0., 0., 0.],
                    board_normal_base=[0., 0., 1.], stop_gap_m=1e-6)
    old = torch.zeros(3, 6, dtype=torch.double)
    old[:, 2] = -1e-6  # float32 roundoff below the plane, inside the 2 um penetration tolerance
    new = old.clone()
    new[0, 2] += .002          # lift
    new[1, 0] += .001          # lateral at the same height
    new[2, 2] -= .001          # deeper: must still be stopped at the stop gap
    out, hit = continuing_joint_step(old, new, fk, geometry, 1., torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(out[0], new[0])
    assert out[1, 0].item() == pytest.approx(.001, abs=1e-12)
    end = physical_gap(fk(out), geometry)
    assert (end >= geometry['stop_gap_m'] - 1e-9).all()
    assert hit.tolist() == [False, True, True]
