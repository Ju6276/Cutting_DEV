"""Release only unused Torch cache before a separate CUDA physics allocator runs."""
import math
import torch


def release_physics_cache(reserve_gib):
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("CUDA cache reserve must be finite and nonnegative")
    if reserve_gib == 0:
        return None
    free, _ = torch.cuda.mem_get_info()
    cached = max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
    released = free < reserve_gib * 2**30 and cached > 0
    if released:
        torch.cuda.empty_cache()
    after, _ = torch.cuda.mem_get_info() if released else (free, 0)
    return dict(free_before_gib=free / 2**30, free_after_gib=after / 2**30,
                unused_cache_before_gib=cached / 2**30, released=bool(released))
