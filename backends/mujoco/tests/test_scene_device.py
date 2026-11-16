from types import SimpleNamespace
import pytest
import torch
from cutting_gpu_backend.batched_disect import BatchedDiSECt, DiSECt

def test_cpu_serialized_scene_rejected_before_packed_kernels(monkeypatch):
    def load(self,*args,**kwargs):
        self.position=torch.zeros(1,3)
        self.state=SimpleNamespace(particle_q=torch.zeros(1,3))
        self.model=SimpleNamespace(particle_mass=torch.ones(1))
    monkeypatch.setattr(DiSECt,"__init__",load)
    with pytest.raises(ValueError,match="CUDA-resident"):
        BatchedDiSECt("fixture",1)
