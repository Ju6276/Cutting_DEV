"""Per-control-step material force, work and safety accumulators.

The coupling applies DiSECt's summed contact-force magnitude to the knife along
+y of the DiSECt frame (``SliceItCoupling.feedback``), i.e. upward, opposing
the pressing direction (DiSECt y is up; the scene reference presses at
v_y = -0.05 m/s). This is the historical scalar branch. The net branch uses
the full wrench/twist power pairing at the knife origin, including sawing and rotation.
The scalar power the knife delivers to the material is therefore
``-F_knife . v = force_magnitude * (-v_y)``: positive while pressing down.
"""
import torch


class StepMetrics:
    """Integrals over the 2 ms coupling substeps of one control step."""

    def __init__(self, worlds, device, dtype=torch.float64):
        self.material_impulse = torch.zeros(worlds, device=device, dtype=dtype)
        self.knife_work = torch.zeros_like(self.material_impulse)
        self.knife_gross_work = torch.zeros_like(self.material_impulse)
        self.safety_margin = torch.zeros_like(self.material_impulse)

    def zero_(self):
        for value in (self.material_impulse, self.knife_work, self.knife_gross_work, self.safety_margin):
            value.zero_()

    def accumulate(self, force_magnitude, linear_velocity_disect, active, dt, *,
                   net_wrench=None, knife_position=None, angular_velocity_disect=None):
        """Add one substep: the material force produced while the knife moved
        with ``linear_velocity_disect`` (the twist sent to DiSECt) for ``dt``."""
        step = active.to(self.material_impulse.dtype) * dt
        force = force_magnitude.to(self.material_impulse.dtype) * step
        downward = -linear_velocity_disect[..., 1].to(self.material_impulse.dtype)
        self.material_impulse += force
        if net_wrench is None:
            self.knife_work += force * downward
            self.knife_gross_work += force * downward.abs()
        else:
            # Wrench is about the DiSECt origin; velocity is at the knife origin.
            # Shift its moment before pairing it with angular velocity.
            wrench = net_wrench.to(self.material_impulse)
            position = knife_position.to(wrench)
            moment = wrench[..., 3:] - torch.linalg.cross(position, wrench[..., :3])
            power = -(wrench[..., :3] * linear_velocity_disect.to(wrench)).sum(-1)
            power -= (moment * angular_velocity_disect.to(wrench)).sum(-1)
            self.knife_work += power * step
            self.knife_gross_work += power.abs() * step

    def check_wrench(self, observed_abs, force_limits, include):
        """The collision check's ratio |observation wrench| / limit, maximized."""
        ratio = (observed_abs / force_limits).amax(-1).to(self.safety_margin.dtype)
        self.safety_margin = torch.where(include, torch.maximum(self.safety_margin, ratio), self.safety_margin)

    def as_dict(self):
        return {'material_impulse': self.material_impulse.clone(),
                'knife_work': self.knife_work.clone(),
                'knife_gross_work': self.knife_gross_work.clone(),
                'safety_margin': self.safety_margin.clone()}
