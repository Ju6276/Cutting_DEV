"""Reset ordering regression: settle from export, not last episode's shape."""
from types import SimpleNamespace
import pytest
import torch
from cutting_world_model.gpu_env import GPUEnvironment
from cutting_world_model import scene_alignment

@pytest.mark.parametrize("selected", [[True,False],[True,True],[False,False]])
def test_settle_aligns_restored_export_then_final_surface(monkeypatch, selected):
    mask=torch.tensor(selected)
    export=torch.tensor([.0708,.0708],dtype=torch.float64)
    settled=torch.tensor([.06515,.06611],dtype=torch.float64)
    clearance=torch.tensor([.005,.006],dtype=torch.float64)
    height=settled.clone()
    knife=height+clearance
    before_height=height.clone();before_knife=knife.clone()
    events=[]
    def restore_export(m):
        height[m]=export[m];events.append("export")
    def align(env,frame,m):
        knife[m]=height[m]+clearance[m];events.append("align")
        return frame
    def settle(seconds,m):
        # Matches CuttingBatch: raw geometry is restored inside material settling.
        restore_export(m)
        torch.testing.assert_close(knife[m]-height[m],clearance[m],rtol=0,atol=1e-12)
        height[m]=settled[m];events.append("settled")
    def reset(m):
        height[m]=settled[m]
        align(None,{},m)
        return dict(height=height.clone(),knife=knife.clone())
    env=SimpleNamespace(worlds=2,device=torch.device("cpu"),metadata={},reset=reset,
        physics=SimpleNamespace(material=SimpleNamespace(restore_export=restore_export),
                                frame=lambda:{},settle=settle))
    monkeypatch.setattr(scene_alignment,"align_reset",align)
    frame=GPUEnvironment.settle(env,1.,mask)
    torch.testing.assert_close(frame["knife"][mask]-frame["height"][mask],clearance[mask],rtol=0,atol=1e-12)
    torch.testing.assert_close(height[~mask],before_height[~mask],rtol=0,atol=0)
    torch.testing.assert_close(knife[~mask],before_knife[~mask],rtol=0,atol=0)
    assert env.metadata["settle_seconds"]==1.

def test_settling_contact_error_identifies_scene_and_original_world():
    from cutting_gpu_backend.batched_disect import BatchedDiSECt
    batch=object.__new__(BatchedDiSECt)
    batch.position=torch.zeros(4,3)
    batch.worlds=4
    batch._scene_path="food_scene.pt"
    batch.material_scales={"density":torch.ones(4)}
    batch.capture_rows=lambda mask: ({},{},{})
    scratch=SimpleNamespace(set_material_scales=lambda *args,**kwargs:None,
        restore_rows=lambda *args:None,advance_stationary=lambda _:torch.tensor([.25,0.]))
    batch._settle_batches={2:scratch}
    with pytest.raises(ValueError,match=r"scene=food_scene.pt; offending_worlds=\[1\].*total_offending=1"):
        batch.settle_selected(1.,torch.tensor([False,True,False,True]))
