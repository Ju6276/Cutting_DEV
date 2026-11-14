"""Environment-step update cadence from SuperMimic-Reproduced Trainer.run."""
from dataclasses import asdict
import math
import warnings

from third_party.supermimic.schedule_config import ScheduleCfg


class OnlineSchedule:
    replay_capacity = 1000  # Source rows per environment.

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
        self.start_iteration = start_iteration
        self.accumulator = 0.
        self.resume_provenance = "fresh"
        self.checkpoint_due = False
        self.update_iterations = []

    def state_dict(self):
        return {'unit': 'vector_steps_v1', 'state_version': 2, 'worlds': self.worlds,
                'iteration': self.iteration, 'transitions': self.transitions,
                'replay_rows_per_world': self.replay_capacity,
                'config': asdict(self.config), 'warmup_start_iteration': self.start_iteration,
                'accumulator': self.accumulator, 'resume_provenance': self.resume_provenance}

    @classmethod
    def resume(cls, saved, worlds, *, replay_restored=True):
        if saved.get('unit') != 'vector_steps_v1':
            raise ValueError('Checkpoint lacks the vector-step schedule')
        if saved['worlds'] != worlds:
            raise ValueError('Checkpoint environment count differs')
        version = saved.get('state_version', 1)
        if version not in (1, 2):
            raise ValueError('Unsupported schedule state version')
        config = ScheduleCfg(**saved['config']) if version == 2 else ScheduleCfg()
        result = cls(config=config, start_iteration=saved['iteration'], worlds=worlds,
                     start_transitions=saved['transitions'])
        if version == 2:
            anchor = saved['warmup_start_iteration']
            accumulator = saved['accumulator']
            if not isinstance(anchor, int) or not 0 <= anchor <= result.iteration:
                raise ValueError('Invalid warmup start iteration')
            if not math.isfinite(accumulator) or not 0 <= accumulator < 1:
                raise ValueError('Invalid schedule accumulator')
            result.start_iteration = anchor
            result.accumulator = float(accumulator)
            result.resume_provenance = saved.get('resume_provenance', 'saved_v2')
        elif replay_restored:
            # Old files lost the prior warmup anchor and fractional update phase.
            # Do not invent an exact phase, especially after earlier resumes.
            warnings.warn('Legacy schedule lacks warmup/accumulator/config state: using '
                          'source defaults, original warmup origin 0 and fractional phase 0. '
                          'This is a documented migration, not exact cadence recovery.',
                          RuntimeWarning, stacklevel=2)
            result.start_iteration = 0
            result.resume_provenance = 'legacy_defaults_origin0_phase0'
        if not replay_restored:
            # A deliberately empty replay needs data before any sampled updates.
            result.start_iteration = result.iteration
            result.accumulator = 0.
            result.resume_provenance = 'empty_replay_refill'
        return result

    @property
    def finished(self):
        return self.iteration >= self.config.total_iterations

    def after_environment_step(self):
        self.iteration += 1
        self.transitions += self.worlds
        updates = 0
        self.checkpoint_due = self.iteration % self.config.checkpoint_interval == 0
        self.update_iterations = []
        if self.iteration - self.start_iteration >= self.config.warmup_iterations:
            self.accumulator += self.config.updates_per_env_step
            while self.accumulator >= 1.:
                self.accumulator -= 1.
                updates += 1
                self.update_iterations.append(self.iteration)
        return updates


class TransitionSchedule(OnlineSchedule):
    """Opt-in collection-scaled cadence. Source OnlineSchedule stays unchanged.

    Updates are earned by new physical transitions AFTER the warmup boundary.
    Integer credits make restart and vector-width changes in separate fresh runs
    independent of floating-point accumulation. LR uses 256-transition units.
    """
    def __init__(self, worlds, warmup_transitions=10240, transitions_per_update=640,
                 lr_reference_worlds=256, config=None):
        for name, value in (("warmup_transitions", warmup_transitions),
                            ("transitions_per_update", transitions_per_update),
                            ("lr_reference_worlds", lr_reference_worlds)):
            if type(value) is not int or value < (0 if name == "warmup_transitions" else 1):
                raise ValueError(name + " must be an integer in its valid range")
        super().__init__(config=config, worlds=worlds)
        self.warmup_transitions = warmup_transitions
        self.transitions_per_update = transitions_per_update
        self.lr_reference_worlds = lr_reference_worlds
        self.warmup_start_transitions = 0
        self.transition_credit = 0

    def state_dict(self):
        result = super().state_dict()
        result.update(unit="physical_transitions_v2", state_version=3,
                      warmup_transitions=self.warmup_transitions,
                      transitions_per_update=self.transitions_per_update,
                      lr_reference_worlds=self.lr_reference_worlds,
                      warmup_start_transitions=self.warmup_start_transitions,
                      transition_credit=self.transition_credit)
        return result

    @classmethod
    def resume(cls, saved, worlds, *, replay_restored=True):
        if saved.get("unit") != "physical_transitions_v2" or saved.get("state_version") != 3:
            raise ValueError("Checkpoint lacks the physical-transitions schedule")
        if saved["worlds"] != worlds:
            raise ValueError("Checkpoint environment count differs")
        result = cls(worlds=worlds, config=ScheduleCfg(**saved["config"]),
                     **{k:saved[k] for k in ("warmup_transitions", "transitions_per_update",
                                             "lr_reference_worlds")})
        for key in ("iteration", "transitions", "warmup_start_transitions", "transition_credit"):
            value = saved[key]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid saved " + key)
            setattr(result, key, value)
        if result.warmup_start_transitions > result.transitions:
            raise ValueError("Warmup origin exceeds collected transitions")
        if result.transition_credit >= result.transitions_per_update:
            raise ValueError("Invalid saved transition_credit")
        result.start_iteration = saved["warmup_start_iteration"]
        result.resume_provenance = saved.get("resume_provenance", "saved_transitions_v2")
        if not replay_restored:
            result.warmup_start_transitions = result.transitions
            result.transition_credit = 0
            result.start_iteration = result.iteration
            result.resume_provenance = "empty_replay_refill"
        return result

    @property
    def finished(self):
        return self.transitions >= self.config.total_iterations * self.lr_reference_worlds

    def after_environment_step(self):
        previous = self.transitions
        self.iteration += 1
        self.transitions += self.worlds
        self.checkpoint_due = self.iteration % self.config.checkpoint_interval == 0
        boundary = self.warmup_start_transitions + self.warmup_transitions
        earned = max(0, self.transitions - max(previous, boundary))
        old_credit = self.transition_credit
        updates, self.transition_credit = divmod(old_credit + earned, self.transitions_per_update)
        # Attribute each update to the transition at which it was earned, not
        # the end of a large vector batch; the LR clock is collection-aligned.
        first = max(previous, boundary) + self.transitions_per_update - old_credit
        self.update_iterations = [(first + i*self.transitions_per_update)/self.lr_reference_worlds
                                  for i in range(updates)]
        return updates
