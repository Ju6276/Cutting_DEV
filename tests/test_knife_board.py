import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backends/mujoco'))
from cutting_gpu_backend.knife_board import KnifeBoard
from cutting_world_model.free_cut import FreeCutTask
from test_task_heads import heads


BOARD = dict(height_m=0., stop_gap_m=1e-6, coverage_tolerance_m=.0005)

def material():
    # Two tracked cut-interface witnesses, neither stiffness determines coverage.
    x=torch.tensor([[-.0001,0.,-.03],[.0001,0.,-.03],
                    [-.0001,.02,.03],[.0001,.02,.03]], dtype=torch.float64)
    v=torch.tensor([[-.002,.03,-.075],[.002,.03,-.075],[0.,0.,-.075],
                    [-.002,.03,.075],[.002,.03,.075],[0.,0.,.075]],dtype=x.dtype)
    m=SimpleNamespace(worlds=1,position=torch.tensor([[0.,.04,0.]],dtype=x.dtype),
                      rotation=torch.tensor([[0.,0.,0.,1.]],dtype=x.dtype))
    m.model=SimpleNamespace(knife_tri_vertices=v,cut_edge_indices=torch.tensor([[0,0],[1,1],[2,2],[3,3]]),
                            cut_edge_coords=torch.zeros(4, dtype=x.dtype),
                            cut_spring_indices=torch.tensor([[0,1],[2,3]]))
    m.state=SimpleNamespace(particle_q=x)
    m.state_field=lambda name: m.state.particle_q[None] if name=='particle_q' else torch.ones((1,2),dtype=x.dtype)
    return m

def test_board_swept_stop_translation_and_high_speed():
    for speed in (1.,20.,100.):
        m=material(); b=KnifeBoard(m,BOARD)
        twist=torch.tensor([[0.,-speed,0.,0.,0.,0.]],dtype=m.position.dtype)
        f,hit=b.fraction(twist,.05,torch.tensor([True]))
        assert hit.item()
        end=m.position+twist[:,:3]*.05*f[:,None]
        assert b.gap(end,m.rotation).item() >= BOARD['stop_gap_m']-1e-8
        assert b.gap(end,m.rotation).item() < 2e-6

def test_board_swept_stop_rotation_never_crosses_plane():
    m=material(); b=KnifeBoard(m,BOARD)
    twist=torch.tensor([[0.,0.,0.,50.,0.,0.]],dtype=m.position.dtype)
    f,hit=b.fraction(twist,.05,torch.tensor([True]))
    assert hit.item()
    for t in torch.linspace(0,float(f)*.05,101):
        angle=50*t
        q=torch.tensor([[torch.sin(angle/2),0.,0.,torch.cos(angle/2)]],dtype=m.position.dtype)
        assert b.gap(m.position,q).item() >= 0

def test_coverage_is_sweep_not_damage_or_only_arrival():
    m=material(); b=KnifeBoard(m,BOARD); b.reset(torch.tensor([True]))
    old=m.position.clone(); m.position[:,1]=.01
    b.observe(old,m.rotation,torch.tensor([True]))
    assert b.covered.tolist()==[[False,True]]
    old=m.position.clone(); m.position[:,1]=1e-6
    b.observe(old,m.rotation,torch.tensor([True]))
    assert b.covered.all()
    # The object sliding outside finite blade length cannot be counted.
    m=material(); b=KnifeBoard(m,BOARD); b.reset(torch.tensor([True]))
    m.state.particle_q[:,2]+=1.
    old=m.position.clone(); m.position[:,1]=1e-6
    b.observe(old,m.rotation,torch.tensor([True]))
    assert not b.covered.any()

def test_v4_success_board_coverage_force_and_separation_are_distinct():
    cfg=dict(version='free_cut_v4',horizon_steps=180,gamma=1.,initial_progress_fraction=.001,
             workspace=dict(frame='robot_base',point='ee',lower=[-1,-1,-1],upper=[1,1,1]),
             board=BOARD,weights=dict(progress=10.,speed=20.,load=1.,jerk=.1,success=100.,failure=100.))
    h=heads(); h.env['agent_control_dt']=.05
    task=FreeCutTask(cfg,h)
    obs=torch.zeros(5,25); peak=torch.zeros(5,6); peak[3,0]=1e5
    board=dict(knife_board_contact=torch.tensor([[1],[1],[0],[1],[1]],dtype=torch.bool),
               knife_path_complete=torch.tensor([[1],[0],[1],[1],[1]],dtype=torch.bool),
               knife_min_board_gap=torch.tensor([[1e-6],[1e-6],[.01],[1e-6],[-.01]]))
    ev=task.events(obs,peak,torch.tensor([.9,1.,1.,1.,1.]),torch.ones(5),pose=torch.zeros(5,7),board=board)
    assert ev['success'].tolist()==[True,False,False,False,False]
    assert ev['separated'].tolist()==[False,True,True,True,True]
    assert ev['terminated'].tolist()==[True,True,False,True,True]

def test_old_world_model_cannot_be_silently_used_with_new_task():
    from cutting_world_model.world_model import CuttingWorldModel
    with pytest.raises(ValueError,match='frozen-policy'):
        CuttingWorldModel({'free_cut_task':{'version':'free_cut_v4'}})


def test_rotation_bound_does_not_claim_contact_at_positive_clearance():
    m=material(); b=KnifeBoard(m,BOARD)
    b.register_stop(torch.tensor([True]))
    assert not b.contacted.any()
    m.position[:,1]=2e-6
    b.register_stop(torch.tensor([True]))
    assert b.contacted.all()


def test_knife_beside_object_does_not_complete_path():
    m=material(); b=KnifeBoard(m,BOARD); b.reset(torch.tensor([True]))
    m.state.particle_q[:,0] += .02
    old=m.position.clone(); m.position[:,1]=1e-6
    b.observe(old,m.rotation,torch.tensor([True]))
    assert not b.covered.any()


def test_inactive_world_is_not_intercepted():
    m=material(); b=KnifeBoard(m,BOARD)
    twist=torch.tensor([[0.,-100.,0.,0.,0.,0.]],dtype=m.position.dtype)
    f,hit=b.fraction(twist,.05,torch.tensor([False]))
    assert f.item()==1 and not hit.item()
    old=m.position.clone(); m.position[:,1]=1e-6
    b.observe(old,m.rotation,torch.tensor([False]))
    assert not b.covered.any()


def test_reset_clears_coverage_contact_and_minimum():
    m=material(); b=KnifeBoard(m,BOARD)
    b.covered[:]=True; b.contacted[:]=True; b.min_gap[:]=-1
    b.reset(torch.tensor([True]))
    assert not b.covered.any() and not b.contacted.any()
    assert b.min_gap.item()==pytest.approx(.04)
    assert torch.equal(b.last_mid,b.midpoints())
