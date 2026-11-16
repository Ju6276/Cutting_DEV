import numpy as np
import pytest
from scripts.compose_wrench_video import sensor_series, video_indices

def fixture():
    limits=np.array([50,50,50,5,5,5.])
    physical=np.arange(18.).reshape(6,3)-9
    obs=np.zeros((1,34));obs[:,15:33]=(physical/limits[:,None]).reshape(1,18)
    row=dict(before={'observation':obs.copy()},active=np.array([True]),
             next_observation=obs.copy(),next_step_count=np.array([[1]]))
    task=dict(policy_observation_revision='sensor_wrench6_v1',
              controller={'max_force_torque':limits},environment={'agent_control_dt':.05})
    return row,task,physical

def test_signed_sensor_units_and_latest_sample():
    row,task,physical=fixture()
    times,wrench,limits=sensor_series([row],0,1,task)
    np.testing.assert_allclose(times,[0,.05])
    np.testing.assert_allclose(wrench,np.stack([physical[:,-1]]*2))

@pytest.mark.parametrize('damage',['reset','inactive','legacy','nonfinite','discontinuity'])
def test_rejects_misaligned_or_invalid_signals(damage):
    row,task,_=fixture(); rows=[row]
    if damage=='reset':row['next_step_count'][0]=0
    if damage=='inactive':row['active'][0]=False
    if damage=='legacy':task['policy_observation_revision']='source'
    if damage=='nonfinite':row['next_observation'][0,20]=np.nan
    if damage=='discontinuity':
        import copy
        extra=copy.deepcopy(row);extra['next_step_count'][0]=2
        extra['before']['observation'][0,16]+=1;rows.append(extra)
    with pytest.raises(ValueError):sensor_series(rows,0,len(rows),task)

def test_studio_timing_includes_final_hold_and_checks_count():
    times=np.array([0,.05,.10])
    result=video_indices(times,20,.5,24)
    np.testing.assert_array_equal(result[:4],[0,1,1,2])
    assert (result[4:]==2).all()
    with pytest.raises(ValueError):video_indices(times,20,.5,23)
