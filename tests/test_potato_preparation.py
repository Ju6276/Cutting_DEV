"""Stability windows must reject an instantaneous zero crossing or slow drift."""
import copy
from scripts.compare_potato_preparation import stable_windows

def row(com=(0.,0.,0.),velocity=(0.,0.,0.),rms=0.):
    return dict(com_m=[list(com)],com_velocity_mm_s=[list(velocity)],particle_rms_mm_s=[rms])

def test_requires_full_trailing_window():
    out=stable_windows([row() for _ in range(6)])
    assert out==[[False]]*5+[[True]]

def test_rejects_internal_motion_despite_zero_com_velocity():
    rows=[row(rms=2.) for _ in range(6)]
    assert stable_windows(rows)[-1]==[False]

def test_rejects_single_velocity_zero_crossing():
    rows=[row(velocity=(0.,2.,0.)) for _ in range(5)]+[row()]
    assert stable_windows(rows)[-1]==[False]

def test_rejects_small_speed_but_excess_accumulated_drift():
    rows=[row(com=(i*4e-6,0.,0.),velocity=(.08,0.,0.)) for i in range(6)]
    assert stable_windows(rows)[-1]==[False]

def test_worlds_are_checked_independently():
    rows=[]
    for _ in range(6):
        r=row()
        for k,v in row(velocity=(0.,1.,0.)).items():
            r[k]+=copy.deepcopy(v)
        rows.append(r)
    assert stable_windows(rows)[-1]==[True,False]
