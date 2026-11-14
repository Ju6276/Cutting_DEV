"""UR5e GPU stepping with the source position-command execution order."""
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


def configure_position_hardware(model):
    model.opt.timestep = 0.001
    model.opt.gravity[:] = [0., 0., -9.8]
    # Gazebo POSITION mode also enables the ODE zero-speed motor with
    # fmax = the URDF effort limit. Its bounded braking impulse maps to
    # MuJoCo friction constraints with a_ref = -qvel / dt.
    model.dof_frictionloss[:] = model.actuator_ctrlrange[:, 1]
    model.dof_solref[:] = [0., -1./model.opt.timestep]
    model.dof_solimp[:, :2] = 1.


class Robot:
    def __init__(self, model_path, q, qd, worlds=1):
        wp.init()
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        configure_position_hardware(self.model)
        data = mujoco.MjData(self.model)
        data.qpos[:] = np.asarray(q)
        data.qvel[:] = np.asarray(qd)
        mujoco.mj_forward(self.model, data)
        self.stream = wp.stream_from_torch(torch.cuda.current_stream())
        with wp.ScopedStream(self.stream):
            self.gpu_model = mjw.put_model(self.model)
            self.data = mjw.put_data(self.model, data, nworld=worlds, nconmax=64, njmax=128)
            mjw.forward(self.gpu_model, self.data)
        self.q = wp.to_torch(self.data.qpos)
        self.qd = wp.to_torch(self.data.qvel)

    @torch.no_grad()
    def set_joint_positions(self, q, mask=None):
        """Place selected worlds at rest; preserve all other q/qd before forward."""
        with wp.ScopedStream(self.stream):
            q=torch.as_tensor(q, device=self.q.device, dtype=self.q.dtype)
            if mask is None:
                self.q.copy_(q)
                self.qd.zero_()
            else:
                mask=torch.as_tensor(mask,device=self.q.device,dtype=torch.bool)
                if mask.shape != self.q.shape[:1]:
                    raise ValueError('Joint-placement mask must select worlds')
                self.q[mask]=q[mask]
                self.qd[mask]=0
            mjw.forward(self.gpu_model, self.data)

    def save_reset_state(self):
        self.initial = {name: wp.to_torch(getattr(self.data, name)).clone()
                        for name in ('qpos', 'qvel', 'qacc_warmstart', 'time')}

    def reset(self, mask):
        with wp.ScopedStream(self.stream):
            mjw.reset_data(self.gpu_model, self.data, wp.from_torch(mask))
            for name, value in self.initial.items():
                wp.to_torch(getattr(self.data, name))[mask] = value[mask]
            mjw.forward(self.gpu_model, self.data)

    def sensor_wrench(self, body_name):
        """MuJoCo native force/torque sensor at the named body's origin."""
        body = self.model.body(body_name).id
        root = int(self.model.body_rootid[body])
        with wp.ScopedStream(self.stream):
            mjw.rne_postconstraint(self.gpu_model, self.data)
        internal = wp.to_torch(self.data.cfrc_int)[:, body]
        rotation = wp.to_torch(self.data.xmat)[:, body]
        origin = wp.to_torch(self.data.xpos)[:, body]
        com = wp.to_torch(self.data.subtree_com)[:, root]
        force = internal[:, 3:]
        torque = internal[:, :3] - torch.linalg.cross(origin-com, force)
        force = (rotation.transpose(-1, -2) @ force[..., None]).squeeze(-1)
        torque = (rotation.transpose(-1, -2) @ torque[..., None]).squeeze(-1)
        return torch.cat((force, torque), -1)

    @torch.no_grad()
    def step(self, position_command, active=None):
        # Source Gazebo uses SetPosition(..., preserveWorldVelocity=true).
        # Here the generalized joint velocity is retained across assignment.
        paused = None
        if active is not None and not bool(active.all()):
            paused = {name: wp.to_torch(getattr(self.data, name))[~active].clone()
                      for name in ('qpos', 'qvel', 'qacc_warmstart', 'time')}
        self.q.copy_(position_command)
        with wp.ScopedStream(self.stream):
            mjw.step(self.gpu_model, self.data)
        if paused is not None:
            for name, value in paused.items():
                wp.to_torch(getattr(self.data, name))[~active] = value
        return self.q, self.qd
