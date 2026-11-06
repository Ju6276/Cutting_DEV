"""Prepare UR5e initial joints and a goal from an exported DiSECt knife motion."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_gpu_backend.controller.fdcc import SerialChain, quat_apply, quat_matrix
from cutting_gpu_backend.controller.kdl_math import rotation_error
from cutting_gpu_backend.coupling import SliceItCoupling, compose_pose, inverse_pose
from provenance import git_revision


def inverse_kinematics(chain, seed, target):
    joints = seed.clone()
    for _ in range(64):
        frames, jacobian, _ = chain(joints, mass_matrix=False)
        actual = frames[chain.tip]
        translation = target[:, :3] - actual[:, :3]
        rotation = rotation_error(quat_matrix(target[:, 3:]) @ quat_matrix(actual[:, 3:]).transpose(-1, -2))
        error = torch.cat((translation, rotation), -1)
        if error.abs().max() < 1e-7:
            if ((joints < chain.lower) | (joints > chain.upper)).any():
                raise ValueError('Initial knife pose requires joints outside the URDF limits')
            return joints, actual
        joints += torch.linalg.lstsq(jacobian, error[..., None]).solution.squeeze(-1)
    raise RuntimeError(f'Initial knife IK failed: residual {error.tolist()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/disect'))
    scene = torch.load(args.scene, map_location='cpu', weights_only=False)
    if torch.count_nonzero(scene['angular_velocity']):
        raise ValueError('The reference controller preserves knife orientation; angular motion needs a trajectory adapter')
    task = json.loads((args.capture / 'task.json').read_text())
    with np.load(args.capture / 'episode_000.npz') as episode:
        seed = torch.as_tensor(episode['joint_q'][0], dtype=torch.float64)[None]
    knife_chain = SerialChain(task['robot_xml'], task['robot_base'], 'b_bot_knife_sim', task['joint_names'])
    tip_chain = SerialChain(task['robot_xml'], task['robot_base'], task['robot_tip'], task['joint_names'])
    coupling = SliceItCoupling.from_urdf(task['robot_xml'], scene['spine_height'], base_link=task['robot_base'])
    start = torch.cat((scene['initial_pos'], scene['initial_rot'])).double()[None]
    robot_start = coupling.robot_pose_for_material_pose(start)
    # The source coupling projects onto one cut plane and knife-roll axis.
    if not torch.allclose(coupling.reset_pose(robot_start), start, atol=1e-7, rtol=0):
        raise ValueError('Exported knife pose is outside the SliceIt planar coupling')
    joints, actual = inverse_kinematics(knife_chain, seed, robot_start)
    tip = tip_chain(joints, mass_matrix=False)[0][task['robot_tip']]
    tip_from_knife = compose_pose(inverse_pose(tip), actual)
    duration = float(scene['settings']['sim_duration'])
    endpoint = start.clone()
    endpoint[:, :3] += scene['linear_velocity'].double() * duration
    goal_knife = coupling.robot_pose_for_material_pose(endpoint)
    goal_tip = compose_pose(goal_knife, inverse_pose(tip_from_knife))
    local_vertices = scene['model'].knife_tri_vertices.double()
    vertices = quat_apply(start[:, 3:].expand(len(local_vertices), -1), local_vertices) + start[:, :3]
    clearance = float(vertices[:, 1].min() - scene['state'].particle_q[:, 1].max())
    if clearance < 0:
        raise ValueError(f'Exported initial blade is below the material top by {-clearance} m')
    args.output.mkdir(parents=True)
    (args.output / 'task.json').write_text(json.dumps(task, indent=2) + '\n')
    np.savez(args.output / 'episode_000.npz', joint_q=joints.numpy(), joint_qd=np.zeros_like(joints.numpy()),
             target_pose=goal_tip.numpy())
    report = {'git': git_revision(), 'command': sys.argv, 'material_scene': str(args.scene.resolve()),
              'material_dt': scene['settings']['sim_dt'], 'reference_duration': duration,
              'initial_material_knife_pose': start[0].tolist(), 'initial_robot_knife_pose': actual[0].tolist(),
              'initial_tip_pose': tip[0].tolist(), 'goal_tip_pose': goal_tip[0].tolist(),
              'initial_blade_clearance_m': clearance,
              'ik_position_error_m': float((actual[:, :3] - robot_start[:, :3]).abs().max()),
              'ik_rotation_error_rad': float(rotation_error(
                  quat_matrix(robot_start[:, 3:]) @ quat_matrix(actual[:, 3:]).transpose(-1, -2)
              ).abs().max())}
    (args.output / 'reset.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
