import copy
import json
from pathlib import Path

import pytest
from scripts.prepare_blademaster_disect_assets import PRESETS, disect_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", list(PRESETS))
def test_elastic_mapping_preserves_disect_contact_and_damage(name):
    template=json.loads((ROOT/"configs/free_scenes/potato_003_half.json").read_text())
    old=copy.deepcopy(template)
    preset=PRESETS[name]
    cfg=disect_config(template,preset,Path("/tmp/example.msh"),.07)
    assert template==old
    young,nu=preset["young"],preset["poisson"]
    assert cfg["young"]==young and cfg["poisson"]==nu and cfg["density"]==preset["density"]
    assert cfg["mu"]==pytest.approx(young/(2*(1+nu)))
    assert cfg["lambda"]==pytest.approx(young*nu/((1+nu)*(1-2*nu)))
    for key,value in old.items():
        if "spring" in key or "sdf_" in key or key.startswith("ground"):
            assert cfg[key]==value
    assert cfg["static_vertices"]==dict(include=[],exclude=[])
    assert "cutting_energy" not in cfg and "yield_stress" not in cfg
    tip_initial=cfg["initial_y"]-.02954
    assert tip_initial-(.07+cfg["geometry"]["position"][1])==pytest.approx(.01)
    assert tip_initial+cfg["velocity_y"]*cfg["sim_duration"]<0


def test_generated_candidates_keep_source_size_and_distinct_task_identity():
    catalog=json.loads((ROOT/"configs/blademaster_disect/catalog.json").read_text())
    assert set(catalog)=={"banana","cake","tofu","luncheon"}
    for name,preset in PRESETS.items():
        cfg=json.loads((ROOT/"configs/blademaster_disect"/("pose_"+name+".json")).read_text())
        assert cfg["task"]["version"]=="free_cut_pose_v1"
        assert cfg["groups"][0]["scene"].endswith("bm_"+name+"_20261114/scene.pt")
        assert catalog[name]["source_preset"]["size"]==preset["size"]
        assert catalog[name]["fixed_particles"]==0
    # The 160 mm cake retains its diameter; slice position fits a 150 mm blade.
    radius=PRESETS["cake"]["size"][0]/2
    offset=PRESETS["cake"]["x_offset"]
    assert 2*(radius*radius-offset*offset)**.5<.15


def test_ground_overlap_is_measured_in_material_frame_not_robot_frame():
    from types import SimpleNamespace
    import torch
    from scripts.check_blademaster_disect_assets import ground_stats
    q=torch.tensor([[[0., .002, 0.], [0., -.0005, 0.], [0., -.002, 0.], [0., -.01, 0.]]])
    m=SimpleNamespace(state_field=lambda _: q,
                      model=SimpleNamespace(contact_ke=1000.,contact_kd=0.,contact_distance=.001))
    result=ground_stats(m)
    assert result["particles_below_board"]==[3]
    assert result["particles_below_minus_1mm"]==[2]
    assert result["particles_below_minus_5mm"]==[1]
    assert result["minimum_y_mm"]==pytest.approx([-10.])
