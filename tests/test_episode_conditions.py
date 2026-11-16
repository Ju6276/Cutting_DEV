import numpy as np
import pytest
from cutting_world_model.episode_conditions import EpisodeConditions, validate, metadata

CFG=dict(revision="paired_episode_conditions_v1",seed=7,
         youngs_modulus=[.5,2.],density=[.8,1.25],cut_strength=[.5,2.],target_speed=[.01,.05])
GROUPS=[dict(name="cucumber",worlds=3),dict(name="potato",worlds=2)]

def test_reset_order_does_not_change_per_world_case():
    a=EpisodeConditions(CFG,GROUPS);b=EpisodeConditions(CFG,GROUPS)
    first=a.sample([1]*5);b.sample([1]*5)
    a.sample([1,0,1,0,0])
    second=a.sample([0,1,0,0,1])
    b.sample([0,1,0,0,1]);other=b.sample([1,0,1,0,0])
    for key in second:np.testing.assert_array_equal(second[key],other[key])
    assert a.next_episode.tolist()==[2,2,2,1,2]
    for key in first:
        assert second[key][3]==first[key][3]
        assert (second[key][[0,1,2,4]]!=first[key][[0,1,2,4]]).all()

def test_restore_continues_next_episode_stream():
    a=EpisodeConditions(CFG,GROUPS);a.sample([1]*5);a.sample([1,0,0,1,0])
    b=EpisodeConditions(CFG,GROUPS);b.load_state_dict(a.state_dict())
    expected=a.sample([1,0,1,0,1]);actual=b.sample([1,0,1,0,1])
    for key in expected:np.testing.assert_array_equal(expected[key],actual[key])

def test_sampled_ranges_and_log_uniform_materials():
    sampler=EpisodeConditions(CFG,[dict(name="a",worlds=4000)])
    values=sampler.sample([1]*4000)
    for key,v in values.items():
        lo,hi=CFG[key]
        assert v.min()>=lo and v.max()<=hi
        scaled=(v-lo)/(hi-lo) if key=="target_speed" else (np.log(v)-np.log(lo))/(np.log(hi)-np.log(lo))
        assert abs(scaled.mean()-.5)<.03

@pytest.mark.parametrize("field,bounds",[("density",[0,1]),("cut_strength",[2,1]),
                                        ("target_speed",[.01,float("nan")])])
def test_invalid_ranges_rejected(field,bounds):
    with pytest.raises(ValueError):validate({**CFG,field:bounds})

def test_checkpoint_identity_is_checked_and_seed_not_a_network_feature():
    a=EpisodeConditions(CFG,GROUPS);a.sample([1]*5)
    b=EpisodeConditions({**CFG,"seed":8},GROUPS)
    with pytest.raises(ValueError):b.load_state_dict(a.state_dict())
    assert metadata(CFG)==metadata({**CFG,"seed":8})
    assert "seed" not in metadata(CFG)

def test_group_reorder_preserves_named_streams():
    a=EpisodeConditions(CFG,GROUPS);b=EpisodeConditions(CFG,list(reversed(GROUPS)))
    x=a.sample([1]*5);y=b.sample([1]*5)
    for key in x:
        assert dict(zip(a.ids,x[key]))==dict(zip(b.ids,y[key]))

def test_training_resume_starts_next_cases_and_legacy_retains_frame():
    from types import SimpleNamespace
    from cutting_world_model.episode_conditions import resume_environment
    original=EpisodeConditions(CFG,GROUPS);original.sample([1]*5)
    saved=original.state_dict()
    new=EpisodeConditions(CFG,GROUPS);new.sample([1]*5)
    env=SimpleNamespace(condition_sampler=new,current={"old":True},
                        reset=lambda:new.sample([1]*5))
    resumed=resume_environment(env,saved);expected=original.sample([1]*5)
    for key in expected:np.testing.assert_array_equal(resumed[key],expected[key])
    assert new.next_episode.tolist()==[2]*5
    with pytest.raises(ValueError):resume_environment(env,None)
    old=SimpleNamespace(current={"old":True})
    assert resume_environment(old,None) is old.current
