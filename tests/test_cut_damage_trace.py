import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import os
import sys
import torch

spec = importlib.util.spec_from_file_location(
    'damage_trace', Path(__file__).resolve().parents[1] / 'scripts/cut_damage_trace.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def inputs():
    # Three independent springs: stretched+contact, stretched+no contact, compressed+contact.
    x = torch.tensor([[0.,0.,0.], [2.,0.,0.], [0.,0.,0.], [2.,0.,0.],
                      [0.,0.,0.], [.5,0.,0.]])
    edges = torch.stack((torch.arange(6), torch.arange(6)), -1)
    springs = torch.arange(6).reshape(3,2)
    knife_f = torch.zeros(6,3); knife_f[0,0] = 1.; knife_f[4,1] = 1.
    return [x, x*0, edges, torch.zeros(6), springs, torch.ones(3), torch.tensor([10.,10.,10.]),
            torch.zeros(3), torch.zeros(3,3), torch.zeros(3), torch.tensor([3.,3.,3.]), knife_f]


def test_damage_requires_contact_and_positive_extension_and_does_not_mutate_inputs():
    args=inputs(); before=[v.clone() for v in args]
    old,expected,contact,stretch,force=m.damage_inputs(args)
    torch.testing.assert_close(expected,torch.tensor([7.,10.,10.]))
    assert contact.tolist()==[True,False,True]
    torch.testing.assert_close(stretch,torch.tensor([1.,1.,0.]))
    for a,b in zip(args,before):
        assert torch.equal(a,b)


def test_trace_accumulates_actual_loss_and_detects_a_wrong_update(tmp_path):
    args=inputs()
    material=SimpleNamespace(worlds=1,dt=.01,state=SimpleNamespace(cut_spring_ke=args[6]))
    trace=m.DamageTrace(material)
    trace.phase='press'
    values=m.damage_inputs(args)
    after=torch.tensor([7.,10.,10.])
    trace.record(*values,after)
    trace.snapshot(torch.tensor([.01]))
    trace.phase='hold'
    # Inject a broken update in the observer test, not in any real simulation.
    trace.record(*values,torch.tensor([8.,10.,10.]))
    trace.snapshot(torch.tensor([.02]))
    summary=trace.save(tmp_path/'trace.npz')
    assert summary['max_formula_error']==1.
    with np.load(tmp_path/'trace.npz') as saved:
        assert saved['press__damage_steps'].tolist()==[[1,0,0]]
        assert saved['press__contact_steps'].tolist()==[[1,0,1]]
        assert saved['press__eligible_steps'].tolist()==[[1,0,0]]
        assert saved['press__actual_loss'].tolist()==[[3.,0.,0.]]
        assert saved['hold__max_formula_error'].tolist()==[[1.,0.,0.]]


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'),
                    reason='Actual DiSECt kernel module requires the backend CUDA environment')
def test_actual_tape_hook_restores_even_on_error():
    sys.path.insert(0, str(Path(os.environ['CUTTING_BACKEND_ROOT']) / 'third_party/disect'))
    import disect.dflex.adjoint as df
    material=SimpleNamespace(worlds=1,dt=.01,state=SimpleNamespace(cut_spring_ke=torch.ones(1)))
    original=df.Tape.launch
    with pytest.raises(RuntimeError, match='intentional'):
        with m.DamageTrace(material):
            assert df.Tape.launch is not original
            raise RuntimeError('intentional')
    assert df.Tape.launch is original


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'),
                    reason='Actual DiSECt CUDA kernel')
def test_observer_preserves_actual_kernel_outputs():
    sys.path.insert(0, str(Path(os.environ['CUTTING_BACKEND_ROOT']) / 'third_party/disect'))
    import disect.dflex.adjoint as df
    from disect.dflex.sim import eval_cut_springs

    def execute(observe):
        args=[v.cuda() for v in inputs()]
        args[2]=args[2].int(); args[4]=args[4].int()
        force=torch.zeros_like(args[0])
        material=SimpleNamespace(worlds=1,dt=.01,state=SimpleNamespace(cut_spring_ke=args[6]))
        tracer=m.DamageTrace(material)
        from contextlib import nullcontext
        with tracer if observe else nullcontext():
            df.Tape().launch(eval_cut_springs,3,args,[force,args[6]],'cuda:0')
        return args[6].cpu(),force.cpu(),tracer

    plain,plain_f,_=execute(False)
    observed,observed_f,tracer=execute(True)
    assert torch.equal(plain,observed)
    assert torch.equal(plain_f,observed_f)
    torch.testing.assert_close(observed,torch.tensor([7.,10.,10.]))
    assert tracer.steps==1
    assert float(tracer.stats['rest']['max_formula_error'].max())==0.
