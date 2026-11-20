"""Headless stepping of exported DiSECt Model/State objects.

Motion follows DiSECt cutting/motion.py DynamicMotion; xyzw orientation,
meters, seconds. The material integrator is the vendored DFlex implementation.
"""
from pathlib import Path
import sys

import torch
import numpy as np

from .controller.fdcc import quat_mul


_SDF_CACHE = {}


def validate_sdf_payload(grid, expected):
    """Check the packed float32 grid against its source .npy SHA once on load.

    np.save reproduces the source header and data without trusting a path that
    may no longer exist on a deployment host. Stream into the digest to avoid
    retaining a second complete serialized grid in memory.
    """
    import hashlib
    digest = hashlib.sha256()
    class HashWriter:
        def write(self, data):
            digest.update(data)
            return len(data)
    array = grid.detach().cpu().numpy()
    if array.dtype != np.dtype("float32") or array.ndim != 1:
        raise ValueError("Knife SDF must be a packed one-dimensional float32 grid")
    np.save(HashWriter(), array, allow_pickle=False)
    if digest.hexdigest() != expected:
        raise ValueError("Knife SDF payload checksum mismatch")



class DiSECt:
    # Changes physical contact interpolation, independently of scalar/net feedback.
    contact_discretization_revision = 'edge_coordinate_consistent_v2'

    def __init__(self, scene_path, sim_dt=None):
        """``sim_dt`` (s) overrides the scene's material substep, e.g. a smaller step for a
        mesh whose explicit-integration stability margin is too small for its material range."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/disect'))
        import disect.dflex as df

        df.config.no_grad = True
        scene = torch.load(scene_path, weights_only=False)
        self.model, self.state = scene['model'], scene['state']
        self.tool_geometry = scene.get('tool_geometry')
        if getattr(self.model, 'knife_sdf_grid', None) is not None:
            if not self.tool_geometry:
                raise ValueError('Mesh SDF scene lacks geometry provenance')
            key=(self.tool_geometry['sdf_validation']['sha256'],str(self.state.particle_q.device))
            if key not in _SDF_CACHE:
                validate_sdf_payload(self.model.knife_sdf_grid, key[0])
                _SDF_CACHE[key]=self.model.knife_sdf_grid.to(self.state.particle_q.device).detach()
            self.model.knife_sdf_grid=_SDF_CACHE[key]
            self.contact_discretization_revision='mesh_sdf_edge_coordinate_v1'
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
