"""Replay SliceIt actions through the ROS-free UR5e/controller/DiSECt loop."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_gpu_backend.controller.fdcc import ControllerConfig, FDCCController, SerialChain, SlicingActionMapper
from cutting_gpu_backend.coupling import SliceItCoupling
from cutting_gpu_backend.disect import DiSECt
from cutting_gpu_backend.robot import Robot
from provenance import git_revision


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def task_error(target, actual):
    desired, current = target[:, 3:], actual[:, 3:]
    orientation = current[:, 3:]*desired[:, :3]-desired[:, 3:]*current[:, :3]
    orientation += torch.linalg.cross(current[:, :3], desired[:, :3])
    orientation *= torch.sign((current*desired).sum(-1, keepdim=True))
    return torch.cat((target[:, :3]-actual[:, :3], orientation), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=Path, default=Path('runs/source/task.json'))
    parser.add_argument('--episode', type=Path, default=Path('runs/source/episode_000.npz'))
    parser.add_argument('--actions-file', type=Path)
    parser.add_argument('--model', type=Path, default=Path('runs/model_coupled/robot.mjb'))
    parser.add_argument('--controller', type=Path, default=Path('runs/controller_source/controller_params.json'))
    parser.add_argument('--scene', type=Path, default=Path('runs/disect_source_v4/scene.pt'))
    parser.add_argument('--max-actions', type=int)
    parser.add_argument('--hold-last-action', action='store_true')
    parser.add_argument('--wall-seconds', type=float, default=600.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    revision = git_revision()
    args.output.mkdir(parents=True, exist_ok=False)
    task = json.loads(args.task.read_text())
    episode = np.load(args.episode)
    actions = np.load(args.actions_file)['actions'] if args.actions_file else episode['actions']
    if args.hold_last_action:
        remaining = max(0, task['environment']['steps_per_episode'] - len(actions))
        actions = np.concatenate((actions, np.repeat(actions[-1:], remaining, axis=0)))
    actions = actions[:args.max_actions]
    write_json(args.output/'manifest.json', {
        'git_revision': revision, 'command': sys.argv,
        'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'actions': len(actions), 'robot_dt': .001, 'coupling_dt': .002,
        'agent_dt': task['environment']['agent_control_dt'],
        'endpoint': 'action sequence exhausted or original goal/force/workspace termination',
        'timeout': 'save trajectory and partial report, then exit', 'pid': os.getpid(),
    })
    wp.config.kernel_cache_dir = '/dataHDD/dong/.cache/warp/cutting_gpu_backend'
    torch.set_num_threads(2)
    device, dtype = 'cuda:0', torch.float64
    start = time.perf_counter()
    sim = DiSECt(args.scene)
    robot = Robot(args.model, episode['joint_q'][0], episode['joint_qd'][0])
    cfg = ControllerConfig.from_parameters(json.loads(args.controller.read_text()))
    chain = SerialChain(task['robot_xml'], task['robot_base'], task['robot_tip'], task['joint_names']).to(device, dtype)
    knife_chain = SerialChain(task['robot_xml'], task['robot_base'], 'b_bot_knife_sim', task['joint_names']).to(device, dtype)
    controller = FDCCController(chain, cfg)
    mapper = SlicingActionMapper(max_twist=task['controller']['max_twist'])
    mapper.reset(1, device, dtype)
    coupling = SliceItCoupling.from_urdf(task['robot_xml'], sim.spine_height, device=device, dtype=dtype)
    # Gazebo's original FT plugin reports child-to-parent wrench; MuJoCo's
    # native force sensor reports the opposite interaction direction.
    raw = -robot.sensor_wrench(cfg.sensor_ref_link).to(dtype)
    coupling.reset_filter(raw)
    hold = robot.q.clone()
    for _ in range(coupling.wrench_filter.window):
        robot.step(hold)
        robot.step(hold)
        raw = -robot.sensor_wrench(cfg.sensor_ref_link).to(dtype)
        coupling.wrench_filter.update(raw)
    coupling.wrench_filter.tare()
    controller.reset(robot.q.to(dtype), robot.qd.to(dtype))
    frames = chain(robot.q.to(dtype), mass_matrix=False)[0]
    knife_pose = knife_chain(robot.q.to(dtype), mass_matrix=False)[0]['b_bot_knife_sim']
    sim.set_motion(torch.zeros(6, device=device), coupling.reset_pose(knife_pose)[0].float())
    sim.advance(.002)
    robot_only_filter = copy.deepcopy(coupling.wrench_filter)
    rigid_only_sensor = robot_only_filter.update(raw)
    feedback = coupling.feedback(sim.state.knife_f[None].to(dtype),
                                 frames[cfg.sensor_ref_link], frames[chain.tip], raw)
    target = torch.as_tensor(episode['target_pose'][0], device=device, dtype=dtype)[None]
    force_limits = torch.tensor(task['controller']['max_force_torque'], device=device, dtype=dtype)
    desired_wrench = torch.tensor(task['controller']['desired_force_torque'], device=device, dtype=dtype)[None]
    env = task['environment']
    intervals = round(env['agent_control_dt']/.002)
    successes = 0
    maximum_force = 0.
    force_response = 0.
    records, particles = [], []
    reason = 'actions_exhausted'
    physics_time = 0.
    collision = False
    with torch.no_grad():
        for index, source_action in enumerate(actions):
            if time.perf_counter()-start >= args.wall_seconds:
                reason = 'wall_budget'
                break
            action = torch.tensor(source_action, device=device, dtype=dtype)[None]
            action_args = mapper(action, target, frames[chain.tip])
            for coupling_index in range(intervals):
                collision = bool((feedback['observation_wrench'].abs() > force_limits).any())
                if collision:
                    reason = 'force_limit'
                    break
                for robot_index in range(2):
                    q = robot.q.to(dtype)
                    if coupling_index == 0 and robot_index == 0:
                        shadow = copy.copy(controller)
                        shadow.update(q, sensor_wrench=rigid_only_sensor,
                                      target_wrench=desired_wrench, **action_args)
                    else:
                        shadow = None
                    command = controller.update(q, sensor_wrench=feedback['sensor_wrench'],
                                                target_wrench=desired_wrench, **action_args)
                    if shadow is not None:
                        force_response = max(force_response, float((controller.velocity-shadow.velocity).abs().max()))
                    robot.step(controller.enforce_position_limits(command, .001))
                frames = chain(robot.q.to(dtype), mass_matrix=False)[0]
                knife_pose = knife_chain(robot.q.to(dtype), mass_matrix=False)[0]['b_bot_knife_sim']
                motion = coupling.motion(knife_pose)
                sim.set_motion(torch.cat((motion['linear_velocity'][0], motion['angular_velocity'][0])).float())
                sim.advance(.002)
                raw = -robot.sensor_wrench(cfg.sensor_ref_link).to(dtype)
                rigid_only_sensor = robot_only_filter.update(raw)
                feedback = coupling.feedback(sim.state.knife_f[None].to(dtype),
                                             frames[cfg.sensor_ref_link], frames[chain.tip], raw)
                maximum_force = max(maximum_force, float(feedback['force_magnitude'][0]))
                physics_time += .002
            error = task_error(target, frames[chain.tip])[0, env['target_dims']].abs()
            reached = bool((error < error.new_tensor(env['goal_threshold'])).all())
            successes += int(reached)
            outside = bool((error > error.new_tensor(env['workspace_limit'])).any())
            row = {'action': index, 'sim_seconds': physics_time, 'wall_seconds': time.perf_counter()-start,
                   'force': float(feedback['force_magnitude'][0]), 'peak_force': maximum_force,
                   'released_springs': int((sim.state.cut_spring_ke == 0).sum()),
                   'knife_pose': torch.cat((sim.position, sim.rotation)).cpu().tolist(),
                   'ee_pose': frames[chain.tip][0].cpu().tolist(), 'task_error': error.cpu().tolist(),
                   'joint_q': robot.q[0].cpu().tolist(),
                   'sensor_wrench': feedback['sensor_wrench'][0].cpu().tolist(),
                   'observation_wrench': feedback['observation_wrench'][0].cpu().tolist(),
                   'force_response_velocity_delta': force_response}
            records.append(row)
            particles.append(sim.state.particle_q.cpu().numpy().copy())
            write_json(args.output/'progress.json', row)
            print(json.dumps(row), flush=True)
            if collision or outside or (reached and successes > env['successes_threshold']):
                reason = 'force_limit' if collision else ('workspace' if outside else 'goal')
                break
    write_json(args.output/'report.json', {'git_revision': revision, 'reason': reason,
               'actions': len(records), 'sim_seconds': physics_time, 'wall_seconds': time.perf_counter()-start,
               'max_force': maximum_force, 'force_response_velocity_delta': force_response,
               'released_springs': int((sim.state.cut_spring_ke == 0).sum()),
               'spring_count': len(sim.state.cut_spring_ke), 'records': records})
    np.savez_compressed(args.output/'trajectory.npz', positions=np.asarray(particles),
                        spring_final=sim.state.cut_spring_ke.cpu().numpy())


if __name__ == '__main__':
    main()
