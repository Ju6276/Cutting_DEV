"""Headless stepping of exported DiSECt Model/State objects.

Motion follows DiSECt cutting/motion.py DynamicMotion; xyzw orientation,
meters, seconds. The material integrator is the vendored DFlex implementation.
"""
from pathlib import Path
import sys

import torch

from .controller.fdcc import quat_mul


class DiSECt:
    def __init__(self, scene_path, sim_dt=None):
        """``sim_dt`` (s) overrides the scene's material substep, e.g. a smaller step for a
        mesh whose explicit-integration stability margin is too small for its material range."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/disect'))
        import disect.dflex as df

        df.config.no_grad = True
        scene = torch.load(scene_path, weights_only=False)
        self.model, self.state = scene['model'], scene['state']
        self.position = scene['initial_pos'].clone()
        self.rotation = scene['initial_rot'].clone()
        self.linear_velocity = scene['linear_velocity'].clone()
        self.angular_velocity = scene['angular_velocity'].clone()
        self.reference_linear_velocity = self.linear_velocity.clone()
        self.reference_duration = scene['settings']['sim_duration']
        self.dt = scene['settings']['sim_dt'] if sim_dt is None else float(sim_dt)
        if not self.dt > 0:
            raise ValueError('sim_dt must be positive')
        self.spine_height = scene['spine_height']
        self.integrator = df.SemiImplicitIntegrator()
        self.steps = 0

    def set_motion(self, twist, pose=None):
        self.linear_velocity.copy_(twist[:3])
        self.angular_velocity.copy_(twist[3:])
        if pose is not None:
            self.position.copy_(pose[:3])
            self.rotation.copy_(pose[3:])

    @torch.no_grad()
    def advance(self, duration):
        count = round(duration / self.dt)
        if abs(count * self.dt - duration) > 1e-12:
            raise ValueError('Coupling duration must contain whole DiSECt substeps')
        # Velocity stays fixed within the original 500 Hz exchange interval.
        angular_norm = torch.linalg.vector_norm(self.angular_velocity)
        rotating = bool(angular_norm != 0)
        if rotating:
            increment = torch.cat((torch.sin(angular_norm*self.dt/2)
                                   * self.angular_velocity/angular_norm,
                                   torch.cos(angular_norm*self.dt/2).reshape(1)))
        for _ in range(count):
            self.position += self.dt * self.linear_velocity
            if rotating:
                self.rotation = quat_mul(increment, self.rotation)
            self.state.joint_q[:3] = self.position
            self.state.joint_q[3:7] = self.rotation
            self.state.joint_qd[:3] = self.linear_velocity
            self.state.joint_qd[3:6] = self.angular_velocity
            self.state = self.integrator.forward(self.model, self.state, self.dt,
                                                 update_mass_matrix=False)
        self.steps += count
        if not torch.isfinite(self.state.particle_q).all() or not torch.isfinite(self.state.cut_spring_ke).all():
            raise FloatingPointError('Nonfinite DiSECt material state')
        return torch.linalg.vector_norm(self.state.knife_f, dim=1).sum()
