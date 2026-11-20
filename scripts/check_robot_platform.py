"""Bounded robot-only checks for either the legacy UR5e or a compiled UR10e bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from cutting_world_model.robot_platform import RobotPlatform
from cutting_gpu_backend.controller.fdcc import quat_matrix
from cutting_world_model.kinematics import SerialKinematics


def wrench(model, data, body):
    mujoco.mj_rnePostConstraint(model, data)
    internal = data.cfrc_int[body]
    origin = data.xpos[body]
    com = data.subtree_com[model.body_rootid[body]]
    rotation = data.xmat[body].reshape(3, 3)
    force = internal[3:]
    torque = internal[:3] - np.cross(origin-com, force)
    return np.r_[rotation.T @ force, rotation.T @ torque]


def stationary(model, data, q):
    data.qpos[:] = q
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_inverse(model, data)
    applied = np.zeros(model.nv)
    for body in range(model.nbody):
        force, torque = data.xfrc_applied[body, :3], data.xfrc_applied[body, 3:]
        if np.any(force) or np.any(torque):
            mujoco.mj_applyFT(model, data, force, torque, data.xipos[body], body, applied)
    data.ctrl[:] = data.qfrc_inverse - applied
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qacc, 0, atol=1e-8)


def check(root, output, device):
    bundle = RobotPlatform.load(root)
    chain, controller = bundle.controller()
    q = torch.tensor(bundle.q, dtype=torch.double)[None]
    controller.reset(q)
    controller.tare(q)
    frames, jacobian, _ = chain(q)
    model = mujoco.MjModel.from_binary_path(str(bundle.model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = bundle.q
    mujoco.mj_forward(model, data)
    tip = model.body(bundle.metadata["robot_tip"]).id
    sensor = model.body(controller.config.sensor_ref_link).id
    np.testing.assert_allclose(frames[chain.tip][0, :3].detach(), data.xpos[tip], atol=1e-9)
    np.testing.assert_allclose(quat_matrix(frames[chain.tip][0, 3:]).detach(),
                               data.xmat[tip].reshape(3, 3), atol=1e-9)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jp, jr, tip)
    np.testing.assert_allclose(jacobian[0].detach(), np.r_[jp, jr], atol=1e-9)
    fk = SerialKinematics(bundle.metadata["robot_xml"], chain.base, chain.tip, list(chain.joint_names)).double()
    np.testing.assert_allclose(fk(q)[0,:3].detach(), data.xpos[tip], atol=1e-9)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    assert np.linalg.eigvalsh(mass).min() > 0
    contacts = [(model.geom(int(c.geom[0])).name,model.geom(int(c.geom[1])).name,float(c.dist))
                for c in data.contact if c.dist < -1e-6]
    results = dict(status="passed", platform=str(Path(root).resolve()),
                   robot_base=chain.base, robot_tip=chain.tip,
                   robot_mass_kg=float(model.body_mass.sum()), initial_penetrations=contacts,
                   fk_and_jacobian="agree with MuJoCo and WM FK",
                   mass_matrix_min_eigenvalue=float(np.linalg.eigvalsh(mass).min()))
    results["robot_xml_sha256"]=hashlib.sha256(bundle.metadata["robot_xml"].encode()).hexdigest()
    results["initial_visual_poses"]={model.body(i).name: dict(xyz=data.xpos[i].tolist(),rotation=data.xmat[i].reshape(3,3).tolist())
                                    for i in range(model.nbody) if model.body(i).name}
    if "robot_platform" in bundle.metadata:
        assert not contacts, contacts
        stationary(model, data, bundle.q)
        actual = -wrench(model, data, sensor)
        expected = controller._gravity_sensor_wrench(frames)[0].detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        results["static_sensor_wrench_N_Nm"] = actual.tolist()
        results["expected_distal_gravity_N_Nm"] = expected.tolist()
        # Known spatial force at the physical knife's COM, using inverse dynamics
        # to hold all joints fixed; sensor must see force and its lever-arm moment.
        model.opt.gravity[:] = 0
        body = model.body("ikea_knife").id
        force, torque = np.array([3.,-4.,5.]), np.array([.1,.2,-.3])
        data.xfrc_applied[body] = np.r_[force, torque]
        stationary(model, data, bundle.q)
        actual = -wrench(model, data, sensor)
        rot = data.xmat[sensor].reshape(3,3)
        expected = np.r_[rot.T @ force,
                         rot.T @ (torque + np.cross(data.xipos[body]-data.xpos[sensor],force))]
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        results["known_external_wrench_max_error"] = float(np.abs(actual-expected).max())
    # A tared moving sensor must cancel only gravity changes, not contact.
    moved = q + q.new_tensor([[.01,-.02,.01,.02,-.01,.01]])
    moved_frames = chain(moved, mass_matrix=False)[0]
    reading = controller._gravity_sensor_wrench(moved_frames)-controller.tare_compensation
    error = controller.force_error(moved_frames, reading, torch.zeros_like(reading))
    assert error.abs().max() < 1e-10
    # Every force/torque channel must affect FDCC when its selection is enabled.
    changes = []
    for channel in range(6):
        controller.reset(q)
        baseline = torch.zeros((1,6), dtype=q.dtype)
        target = frames[chain.tip].detach()
        commands = []
        for value in (0., 1.):
            controller.reset(q);controller.tare(q)
            input_wrench = baseline.clone();input_wrench[:,channel] = value
            for _ in range(3):
                command = controller.update(q, target, input_wrench, selection=torch.zeros_like(baseline))
            commands.append(command.detach().clone())
        changes.append(float((commands[1]-commands[0]).norm()))
    assert min(changes) > 1e-12, changes
    results["six_wrench_channels_fdcc_joint_response_rad"] = changes
    if device == "cuda":
        robot = bundle.robot(2)
        chain_gpu, ctrl_gpu = bundle.controller("cuda")
        qg = robot.q.double().clone()
        ctrl_gpu.reset(qg);ctrl_gpu.tare(qg)
        target = chain_gpu(qg, mass_matrix=False)[0][chain_gpu.tip].clone()
        target[:,:3] += target.new_tensor([.001,0,0])
        for _ in range(50):
            command = ctrl_gpu.update(robot.q.double(), target, torch.zeros((2,6),device="cuda",dtype=torch.double))
            command = ctrl_gpu.enforce_position_limits(command, .002)
            robot.step(command)
            assert torch.isfinite(robot.q).all() and torch.isfinite(robot.qd).all()
        assert torch.isfinite(robot.sensor_wrench(ctrl_gpu.config.sensor_ref_link)).all()
        end=chain_gpu(robot.q.double(),mass_matrix=False)[0][chain_gpu.tip]
        position_error=(end[:,:3]-target[:,:3]).norm(dim=-1)
        assert (position_error < .001).all(), position_error
        results["gpu_after_50_steps_target_position_error_m"]=position_error.tolist()

        previous = robot.q[1].clone()
        robot.set_joint_positions(qg, mask=torch.tensor([True,False],device="cuda"))
        assert torch.equal(previous, robot.q[1])
        results["gpu_robot_fdcc"] = "50 steps, 2 worlds, finite sensor, partial reset isolated"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args=p.parse_args()
    check(args.platform,args.output,args.device)
