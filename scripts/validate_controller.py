"""Compare the tensor controller's generic dynamics against native KDL."""
import argparse
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from cutting_gpu_backend.controller import ControllerConfig, FDCCController, SerialChain
from cutting_gpu_backend.controller.fdcc import quat_matrix


def validate(task_path, episode_path, parameters_path, reference_binary, device='cpu'):
    task = json.loads(Path(task_path).read_text())
    parameters = json.loads(Path(parameters_path).read_text())
    q = np.load(episode_path)['joint_q']
    xml_path = Path(task_path).with_name('controller_robot.urdf')
    xml_path.write_text(task['robot_xml'])
    arguments = [str(reference_binary), str(xml_path), task['robot_base'], task['robot_tip'],
                 str(parameters['solver']['forward_dynamics']['link_mass'])]
    samples = io.StringIO()
    np.savetxt(samples, q)
    output = subprocess.check_output(arguments, input=samples.getvalue(), text=True)
    reference = np.loadtxt(io.StringIO(output))
    chain = SerialChain(task['robot_xml'], task['robot_base'], task['robot_tip'], task['joint_names']).to(device)
    qt = torch.tensor(q, device=device, dtype=torch.float64)
    frames, jacobian, mass = chain(qt)
    pose = frames[task['robot_tip']]
    rotation = quat_matrix(pose[:, 3:]).cpu().numpy()
    ref_rotation = quat_matrix(torch.tensor(reference[:, 75:79])).numpy()
    report = {'samples': len(q), 'device': device,
              'mass_matrix_max_abs_error': float(np.max(np.abs(mass.cpu().numpy()-reference[:, :36].reshape(-1,6,6)))),
              'jacobian_max_abs_error': float(np.max(np.abs(jacobian.cpu().numpy()-reference[:, 36:72].reshape(-1,6,6)))),
              'position_max_abs_error': float(np.max(np.abs(pose[:, :3].cpu().numpy()-reference[:, 72:75]))),
              'rotation_matrix_max_abs_error': float(np.max(np.abs(rotation-ref_rotation)))}
    config = ControllerConfig.from_parameters(parameters)
    controller = FDCCController(chain, config)
    controller.reset(qt)
    target = pose.clone()
    target[:, 0] += 0.01
    command = controller.update(qt, target, qt.new_zeros((len(q),6)))
    report['first_command_position_max_error'] = float(torch.max(torch.abs(command-qt)).item())
    report['max_internal_velocity'] = float(controller.velocity.abs().max().item())
    report['all_finite'] = bool(torch.isfinite(controller.velocity).all().item())
    controller.reset(qt[:1])
    rows, observed = [], []
    for i in range(len(q)):
        desired = pose[i:i+1].clone()
        desired[:, 0] += 0.005
        measured_force = qt.new_tensor([[0.1, -0.2, 0.3, 0.04, 0.05, -0.01]])
        gains_p = qt.new_tensor(config.p_gains).reshape(1,6)
        gains_d = qt.new_tensor([0.001,0.002,0.003,0.004,0.005,0.006]).reshape(1,6)
        stiffness = qt.new_tensor(config.stiffness).reshape(1,6)
        config.iterations = 1 if i % 2 == 0 else 3
        command = controller.update(qt[i:i+1], desired, measured_force,
                                    stiffness=stiffness, p_gains=gains_p, d_gains=gains_d,
                                    error_scale=0.02)
        observed.append(torch.cat((command, controller.velocity, controller.previous_error), -1).cpu().numpy()[0])
        rows.append(np.concatenate((q[i], desired.cpu().numpy()[0], measured_force.cpu().numpy()[0],
                                    stiffness.cpu().numpy()[0], gains_p.cpu().numpy()[0], gains_d.cpu().numpy()[0],
                                    [0.02, config.iterations])))
    samples = io.StringIO()
    np.savetxt(samples, rows, fmt=['%.17g']*38+['%d'])
    output = subprocess.check_output(arguments + [config.compliance_ref_link, config.sensor_ref_link],
                                     input=samples.getvalue(), text=True)
    reference_rollout = np.loadtxt(io.StringIO(output))
    observed = np.array(observed)
    report['rollout_command_max_abs_error'] = float(np.max(np.abs(observed[:, :6]-reference_rollout[:, :6])))
    report['rollout_velocity_max_abs_error'] = float(np.max(np.abs(observed[:, 6:12]-reference_rollout[:, 6:12])))
    report['rollout_error_history_max_abs_error'] = float(np.max(np.abs(observed[:, 12:]-reference_rollout[:, 12:])))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='runs/source/task.json')
    parser.add_argument('--episode', default='runs/source/episode_000.npz')
    parser.add_argument('--parameters', default='runs/controller_source/controller_params.json')
    parser.add_argument('--reference-binary', default='runs/controller_source/kdl_reference')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = validate(args.task, args.episode, args.parameters, args.reference_binary, args.device)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    checked = ('mass_matrix_max_abs_error', 'jacobian_max_abs_error',
               'position_max_abs_error', 'rotation_matrix_max_abs_error',
               'rollout_command_max_abs_error', 'rollout_velocity_max_abs_error')
    if not result['all_finite'] or any(result[key] > 1e-10 for key in checked):
        raise RuntimeError('Controller KDL comparison failed')
