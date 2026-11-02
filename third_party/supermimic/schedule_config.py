"""ScheduleCfg extracted from upstream config.py; see README.md and LICENSE."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000
    updates_per_env_step: float = 0.4
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50
    checkpoint_interval: int = 4000
    max_hours: float | None = None
