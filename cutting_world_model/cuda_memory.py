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


CACHE_RELEASE_MODES = ("always", "reserve", "off")


def release_unused_cache(mode="always", reserve_gib=0.):
    """Return idle Torch allocator blocks to the device before a Warp-using phase.

    ``always`` releases whenever Torch holds unoccupied cache; ``reserve`` only when
    free memory is below ``reserve_gib`` (release_physics_cache); ``off`` does nothing.
    Live tensors are never touched.
    """
    if mode not in CACHE_RELEASE_MODES:
        raise ValueError(f"CUDA cache release mode must be one of {CACHE_RELEASE_MODES}")
    if mode == "off":
        return None
    if mode == "reserve":
        if not reserve_gib > 0:
            raise ValueError("reserve mode requires a positive reserve")
        return release_physics_cache(reserve_gib)
    free, _ = torch.cuda.mem_get_info()
    cached = max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
    if cached > 0:
        torch.cuda.empty_cache()
    after = torch.cuda.mem_get_info()[0] if cached > 0 else free
    return dict(free_before_gib=free / 2**30, free_after_gib=after / 2**30,
                unused_cache_before_gib=cached / 2**30, released=cached > 0)


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


class HeadroomGuard:
    """Fail loudly when free device memory before physics/reset cannot hold Warp's need.

    Warp (MuJoCo Warp collision, wp.empty) allocates from the CUDA stream-ordered pool,
    whose default release threshold (Warp does not change it) returns blocks to the
    device at synchronization, so each physics/reset needs its in-use high-water mark
    as free device memory. The requirement is that high-water (since process start,
    which includes the full-batch initial reset) plus ``margin_gib``. ``check`` raises
    only when armed (the trainer arms it after the first joint learning round); it
    never changes the number of worlds or any batch.
    """

    def __init__(self, margin_gib, cuda=None, warp=warp_pool_mib):
        if not math.isfinite(margin_gib) or margin_gib < 0:
            raise ValueError("GPU headroom margin must be finite and nonnegative")
        self.margin_mib = margin_gib * 1024
        self.cuda = torch.cuda if cuda is None else cuda
        self.warp = warp

    def check(self, where, armed):
        free = self.cuda.mem_get_info()[0] / MIB
        pool = self.warp()
        warp_high = None if pool is None else pool["warp_pool_used_high_mib"]
        required = (warp_high or 0.) + self.margin_mib
        row = dict(where=where, free_mib=free, warp_high_mib=warp_high,
                   required_mib=required, armed=bool(armed))
        if armed and free < required:
            raise RuntimeError(
                f"GPU headroom too low before {where}: {free:.0f} MiB free < "
                f"{required:.0f} MiB required (Warp high-water {warp_high} MiB + margin "
                f"{self.margin_mib:.0f} MiB). Stopping instead of risking a Warp allocation "
                f"failure; free the card or plan capacity (worlds are never reduced).")
        return row
