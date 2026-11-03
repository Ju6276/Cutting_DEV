"""Source vector-step schedule, measured in batches of 4096 physical transitions."""
from third_party.supermimic.schedule_config import ScheduleCfg


class OnlineSchedule:
    reference_worlds = 4096  # SuperMimic-Reproduced ba610534 EnvCfg.num_envs.
    reference_replay_capacity = 1000  # Source rows per environment.

    def __init__(self, config=None, start_iteration=0, worlds=4096,
                 start_transitions=None):
        if not isinstance(worlds, int) or worlds < 1:
            raise ValueError('worlds must be a positive integer')
        self.config = config or ScheduleCfg()
        self.worlds = worlds
        self.iteration = start_iteration
        self.transitions = (start_iteration * worlds if start_transitions is None
                            else start_transitions)
        if self.transitions < 0 or start_iteration < 0:
            raise ValueError('Schedule counters must be nonnegative')
        self.start_source_iteration = self.source_iteration
        self.start_transitions = self.transitions
        self.accumulator = 0.
        self.checkpoint_due = False
        self.update_source_iterations = []

    @property
    def source_iteration(self):
        return self.transitions // self.reference_worlds

    @property
    def replay_capacity(self):
        total = self.reference_worlds * self.reference_replay_capacity
        return (total + self.worlds - 1) // self.worlds

    def state_dict(self):
        return {'unit': 'physical_transitions_v1', 'worlds': self.worlds,
                'iteration': self.iteration, 'transitions': self.transitions,
                'reference_worlds': self.reference_worlds,
                'replay_rows_per_world': self.replay_capacity}

    @classmethod
    def resume(cls, saved, worlds):
        if saved.get('unit') != 'physical_transitions_v1':
            raise ValueError('Checkpoint lacks the physical-transition schedule')
        if saved['reference_worlds'] != cls.reference_worlds:
            raise ValueError('Checkpoint reference environment count differs')
        return cls(start_iteration=saved['iteration'], worlds=worlds,
                   start_transitions=saved['transitions'])

    def after_environment_step(self):
        previous = self.source_iteration
        previous_refill = (self.transitions - self.start_transitions) // self.reference_worlds
        self.iteration += 1
        self.transitions += self.worlds
        updates = 0
        current = self.source_iteration
        self.checkpoint_due = (current // self.config.checkpoint_interval >
                               previous // self.config.checkpoint_interval)
        self.update_source_iterations = []
        refill = (self.transitions - self.start_transitions) // self.reference_worlds
        for step in range(previous_refill + 1, refill + 1):
            source_step = self.start_source_iteration + step
            if source_step > self.config.total_iterations:
                break
            if step >= self.config.warmup_iterations:
                self.accumulator += self.config.updates_per_env_step
                while self.accumulator >= 1.:
                    self.accumulator -= 1.
                    updates += 1
                    self.update_source_iterations.append(source_step)
        return updates
