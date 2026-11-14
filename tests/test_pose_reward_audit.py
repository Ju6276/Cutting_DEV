import torch
from scripts.audit_pose_reward import IndependentReferences, trajectory, components
from cutting_world_model.free_cut import FreeCutTask
from test_pose_cut import config
from test_task_heads import heads


def test_reference_speed_changes_duration_not_endpoint():
    distance=.047
    for case in ('press','fast','extreme_fast','slow'):
        ref=trajectory(case,distance,.03,.05,180)
        endpoint=ref.displacement(torch.tensor([9.],dtype=torch.float64))
        torch.testing.assert_close(endpoint,torch.tensor([[0.,-distance,0.]],dtype=torch.float64),rtol=0,atol=1e-12)
    still=trajectory('stall',distance,.03,.05,180)
    assert still.displacement(torch.tensor([9.])).eq(0).all()


def test_batch_references_keep_world_specific_times_and_paths():
    refs=[trajectory(case,.04,.03,.05,180) for case in ('press','fast','retract','saw')]
    time=torch.tensor([.2,.3,3.,.8],dtype=torch.float64)
    rot=torch.tensor([0.,0.,0.,1.],dtype=torch.float64)
    got=IndependentReferences(refs).displacement(time,rot)
    expected=torch.cat([r.displacement(time[i:i+1],rot) for i,r in enumerate(refs)])
    torch.testing.assert_close(got,expected)
    assert not torch.equal(got[0],got[1])
    assert got[2,1].abs()<.002


def test_reward_audit_decomposition_matches_task_with_success_and_failures():
    h=heads();h.env['agent_control_dt']=.05
    task=FreeCutTask(config(),h)
    generator=torch.Generator().manual_seed(5)
    obs=torch.randn(4,25,generator=generator,dtype=torch.float64)*.1
    old=torch.tensor([.1,.2,.9,.3],dtype=torch.float64)
    rho=torch.tensor([.2,.2,1.,.35],dtype=torch.float64)
    frame={'observation':obs,'cut_progress':rho[:,None],
           'peak_wrench':torch.randn(4,6,generator=generator,dtype=torch.float64)}
    info={'success':torch.tensor([True,False,False,False]),
          'collision':torch.tensor([False,True,False,False]),
          'outside':torch.tensor([False,False,True,False])}
    speed=torch.tensor([.01,.02,.03,.05],dtype=torch.float64)
    split=components(task,old,frame,speed,info)
    actual=task.reward(old,rho,obs,frame['peak_wrench'],speed,info)
    torch.testing.assert_close(sum(split.values()),actual)
