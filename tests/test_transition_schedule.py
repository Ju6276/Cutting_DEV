import pytest
from cutting_world_model.online_schedule import TransitionSchedule, OnlineSchedule

@pytest.mark.parametrize('worlds', [1,3,192,256,384,768,2048])
def test_updates_earned_from_transitions(worlds):
    s=TransitionSchedule(worlds)
    clock=[]
    for _ in range(20000//worlds+1):
        s.after_environment_step()
        clock.extend(s.update_iterations)
        eligible=max(0,s.transitions-10240)
        assert len(clock)==eligible//640
        assert s.transition_credit==eligible%640
    assert clock==[(10240+640*(i+1))/256 for i in range(len(clock))]

@pytest.mark.parametrize('split',[0,1,26,27,28,42,51])
def test_transition_resume_exact(split):
    s=TransitionSchedule(384)
    for _ in range(split): s.after_environment_step()
    t=TransitionSchedule.resume(s.state_dict(),384)
    for _ in range(50):
        assert s.after_environment_step()==t.after_environment_step()
        assert s.update_iterations==t.update_iterations
        assert s.state_dict()==t.state_dict()

def test_empty_replay_refills_without_losing_global_clock():
    s=TransitionSchedule(384)
    for _ in range(80): s.after_environment_step()
    t=TransitionSchedule.resume(s.state_dict(),384,replay_restored=False)
    start=t.transitions
    n=0
    for _ in range(60):
        n+=t.after_environment_step()
        assert n==max(0,t.transitions-start-10240)//640
    assert t.update_iterations[-1]>start/256

def test_resume_rejects_other_regimes_and_invalid_credit():
    with pytest.raises(ValueError): TransitionSchedule.resume(OnlineSchedule(worlds=384).state_dict(),384)
    s=TransitionSchedule(384).state_dict()
    with pytest.raises(ValueError): TransitionSchedule.resume(s,256)
    s['transition_credit']=640
    with pytest.raises(ValueError): TransitionSchedule.resume(s,384)



def test_default_budget_matches_lr_collection_clock():
    from third_party.supermimic.schedule_config import ScheduleCfg
    cfg = ScheduleCfg(total_iterations=2)
    source=OnlineSchedule(config=cfg,worlds=1)
    scaled=TransitionSchedule(config=cfg,worlds=1,warmup_transitions=0)
    for _ in range(2):source.after_environment_step();scaled.after_environment_step()
    assert source.finished and not scaled.finished
    for _ in range(510):scaled.after_environment_step()
    assert scaled.finished and scaled.transitions==512
