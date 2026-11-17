from types import SimpleNamespace
import pytest
from cutting_world_model import cuda_memory


def test_disabled_does_not_access_cuda(monkeypatch):
    monkeypatch.setattr(cuda_memory, "torch", SimpleNamespace())
    assert cuda_memory.release_physics_cache(0) is None


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_reserve(value):
    with pytest.raises(ValueError):
        cuda_memory.release_physics_cache(value)


def test_only_unused_cache_is_released_when_free_memory_is_low(monkeypatch):
    calls = []
    free = [2 * 2**30, 4 * 2**30]
    cuda = SimpleNamespace(
        mem_get_info=lambda: (free.pop(0), 24 * 2**30),
        memory_reserved=lambda: 3 * 2**30,
        memory_allocated=lambda: 1 * 2**30,
        empty_cache=lambda: calls.append("release"),
    )
    monkeypatch.setattr(cuda_memory, "torch", SimpleNamespace(cuda=cuda))
    result = cuda_memory.release_physics_cache(3)
    assert calls == ["release"]
    assert result == dict(free_before_gib=2, free_after_gib=4,
                          unused_cache_before_gib=2, released=True)


@pytest.mark.parametrize("free,cached", [(4, 2), (1, 0)])
def test_no_release_when_unnecessary_or_no_unused_cache(monkeypatch, free, cached):
    def unexpected():
        raise AssertionError("Must not flush")
    cuda = SimpleNamespace(
        mem_get_info=lambda: (free * 2**30, 24 * 2**30),
        memory_reserved=lambda: (1 + cached) * 2**30,
        memory_allocated=lambda: 1 * 2**30,
        empty_cache=unexpected,
    )
    monkeypatch.setattr(cuda_memory, "torch", SimpleNamespace(cuda=cuda))
    assert not cuda_memory.release_physics_cache(3)["released"]


class FakeCuda:
    """Allocator counters driven by the test; peaks reset like torch.cuda."""
    def __init__(self):
        self.reserved = self.allocated = 0
        self.peak_reserved = self.peak_allocated = 0
        self.free = 20 * 2**20
    def is_available(self): return True
    def set(self, reserved, allocated):
        self.reserved, self.allocated = reserved * 2**20, allocated * 2**20
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        self.peak_allocated = max(self.peak_allocated, self.allocated)
    def reset_peak_memory_stats(self):
        self.peak_reserved, self.peak_allocated = self.reserved, self.allocated
    def max_memory_reserved(self): return self.peak_reserved
    def max_memory_allocated(self): return self.peak_allocated
    def memory_reserved(self): return self.reserved
    def memory_allocated(self): return self.allocated
    def mem_get_info(self): return self.free, 24 * 2**30


def test_phase_peaks_are_attributed_per_phase_and_run():
    cuda = FakeCuda()
    memory = cuda_memory.PhaseMemory(cuda)
    memory.enter('world_model'); cuda.set(10, 8); cuda.set(6, 4)
    memory.enter('physics'); cuda.set(7, 4)
    window = memory.take_window()
    assert window['world_model']['reserved_peak_mib'] == 10
    assert window['world_model']['allocated_peak_mib'] == 8
    assert window['physics']['reserved_peak_mib'] == 7  # reset at phase entry
    assert window['physics']['device_free_start_mib'] == 20
    assert memory.current == 'physics' and memory.take_window()['physics']['reserved_end_mib'] == 7
    memory.enter('world_model'); cuda.set(12, 9)
    memory.close()
    assert memory.run['world_model']['reserved_peak_mib'] == 12
    assert memory.run_peak_allocated_mib == 9
    metrics = cuda_memory.flat_metrics(memory.take_window())
    assert metrics['gpu_mem/world_model/reserved_peak_mib'] == 12


def test_phase_memory_disabled_without_cuda():
    cuda = SimpleNamespace(is_available=lambda: False)
    memory = cuda_memory.PhaseMemory(cuda)
    memory.enter('physics'); memory.close()
    assert memory.take_window() == {} and memory.run_peak_allocated_mib == 0
