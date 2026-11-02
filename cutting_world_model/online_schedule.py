"""Environment-step update cadence from SuperMimic-Reproduced Trainer.run."""
from third_party.supermimic.schedule_config import ScheduleCfg


class OnlineSchedule:
    def __init__(self, config=None, start_iteration=0):
        self.config = config or ScheduleCfg()
        self.iteration = start_iteration
        self.start_iteration = start_iteration
        self.accumulator = 0.

    def after_environment_step(self):
        self.iteration += 1
        updates = 0
        if self.iteration - self.start_iteration >= self.config.warmup_iterations:
            self.accumulator += self.config.updates_per_env_step
            while self.accumulator >= 1.:
                self.accumulator -= 1.
                updates += 1
        return updates
