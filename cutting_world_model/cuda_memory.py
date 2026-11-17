"""Torch/Warp GPU memory sharing: cache release before physics and per-phase accounting."""
import math
import sys
import torch

MIB = 2**20


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


def warp_pool_mib():
    """Warp's stream-ordered pool (the allocator of wp.empty in MuJoCo Warp); None if unused."""
    wp = sys.modules.get("warp")
    if wp is None:
        return None
    try:
        if not wp.is_mempool_enabled("cuda:0"):
            return None
        return dict(warp_pool_used_mib=wp.get_mempool_used_mem_current("cuda:0") / MIB,
                    warp_pool_used_high_mib=wp.get_mempool_used_mem_high("cuda:0") / MIB)
    except Exception:  # Warp not initialized on CUDA: nothing to report.
        return None


class PhaseMemory:
    """Peak Torch reserved/allocated memory per named phase, plus device use at phase end.

    Torch peaks are host-side allocator counters (no synchronization). Each phase
    change closes the previous phase and resets the peak counters, so
    max_memory_allocated() alone no longer describes the whole run; use
    ``run_peak_allocated_mib``. Device use counts every allocator (Torch, Warp,
    CUDA context); free device memory is what a Warp allocation can still obtain.
    """

    def __init__(self, cuda=None):
        self.cuda = torch.cuda if cuda is None else cuda
        self.enabled = bool(self.cuda.is_available())
        self.current = None
        self.window = {}
        self.run = {}

    def enter(self, phase):
        if not self.enabled or phase == self.current:
            return
        self.close()
        self.cuda.reset_peak_memory_stats()
        self.start_free = self.cuda.mem_get_info()[0]
        self.current = phase

    def close(self):
        if not self.enabled or self.current is None:
            return
        free, total = self.cuda.mem_get_info()
        row = dict(reserved_peak_mib=self.cuda.max_memory_reserved() / MIB,
                   allocated_peak_mib=self.cuda.max_memory_allocated() / MIB,
                   reserved_end_mib=self.cuda.memory_reserved() / MIB,
                   allocated_end_mib=self.cuda.memory_allocated() / MIB,
                   device_used_end_mib=(total - free) / MIB,
                   device_free_start_mib=self.start_free / MIB,
                   device_free_end_mib=free / MIB)
        for table in (self.window, self.run):
            old = table.get(self.current)
            table[self.current] = row if old is None else {
                k: (min if k.startswith('device_free') else max)(old[k], v) for k, v in row.items()}
        self.current = None

    def take_window(self):
        """Per-phase extremes since the previous call (the open phase is closed first)."""
        phase = self.current
        self.close()
        window, self.window = self.window, {}
        if phase is not None:
            self.enter(phase)
        return window

    @property
    def run_peak_allocated_mib(self):
        return max((row['allocated_peak_mib'] for row in self.run.values()), default=0.)

    def summary(self):
        result = {phase: dict(row) for phase, row in self.run.items()}
        pool = warp_pool_mib()
        return {'phases': result, **({'warp': pool} if pool else {})}


def flat_metrics(window, prefix='gpu_mem/'):
    return {f'{prefix}{phase}/{key}': value
            for phase, row in window.items() for key, value in row.items()}
