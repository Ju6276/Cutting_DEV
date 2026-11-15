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
