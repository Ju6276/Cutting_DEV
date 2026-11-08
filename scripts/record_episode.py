#!/usr/bin/env python3
"""Record benchmark control boundaries and geometry in the MuJoCo world frame.

``episode_000.npz`` contains measured states/actions. ``episode.npz`` contains
geometry for offline rendering; interpolation between measured boundaries is
for display, with force peaks and release counts held until the next sample.
"""
import json
from pathlib import Path
import signal
import sys
import time

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from scripts.evaluate_benchmark import argument_parser, make_controller
from cutting_world_model.benchmark import (
    benchmark_scene, check_contract, evaluate, physical_contract, write_json,
)
from cutting_world_model.perturbed_eval import CaseNoise, make_cases
from cutting_world_model.provenance import git_revision
from cutting_gpu_backend.controller.fdcc import quat_apply


def interpolate_steps(steps, subdivisions):
    if subdivisions < 1 or not steps:
        raise ValueError('Recording needs states and positive display subdivisions')
    display = []
    for a, b in zip(steps[:-1], steps[1:]):
        for k in range(subdivisions):
            w = k / subdivisions
            display.append({
                key: ((1 - w) * a[key] + w * b[key]
                      if key in ('joint_q', 'particles', 'knife', 'sim_time') else a[key])
                for key in a})
    display.append(steps[-1])
    return display


class GeometryRecorder:
    def __init__(self, env, backend):
        self.env = env
        self.material = env.physics.material
        self.base_from_disect = env.physics.coupling.base_from_disect.to(env.device, torch.float64)
        self.mj = mujoco.MjModel.from_binary_path(str(backend / 'runs/model_coupled/robot.mjb'))
        self.data = mujoco.MjData(self.mj)
        mujoco.mj_forward(self.mj, self.data)
        base = self.mj.body(env.metadata['robot_base']).id
        self.base_pos = self.data.xpos[base].copy()
        self.base_mat = self.data.xmat[base].reshape(3, 3).copy()
        self.qpos_indices = np.array([self.mj.jnt_qposadr[self.mj.joint(name).id]
                                      for name in env.metadata['joint_names']])
        self.steps = []

    def to_world(self, points):
        p = torch.as_tensor(points, device=self.env.device, dtype=torch.float64)
        q = self.base_from_disect[3:].expand(len(p), -1)
        p_base = quat_apply(q, p) + self.base_from_disect[:3]
        return p_base.cpu().numpy() @ self.base_mat.T + self.base_pos

    def __call__(self, frame, info):
        model = self.material.model
        pose = self.material.state_field('body_X_sc')[0, model.knife_link_index].to(torch.float64)
        vertices = model.knife_tri_vertices.reshape(-1, 3).to(self.env.device, torch.float64)
        knife = quat_apply(pose[3:].expand(len(vertices), -1), vertices) + pose[:3]
        self.steps.append({
            'joint_q': frame['joint_q'][0].cpu().numpy().astype(np.float64),
            'particles': self.to_world(frame['particle_q'][0]),
            'knife': self.to_world(knife),
            'released': int((frame['cut_spring_ke'][0] == 0).sum()),
            'sim_time': float(frame['sim_time'][0, 0]),
            'force': 0. if info is None else float(info['peak_force'][0]),
            'sensor': float(frame['peak_wrench'][0, :3].abs().max()),
        })

    def save(self, output, subdivisions):
        display = interpolate_steps(self.steps, subdivisions)
        mj = self.mj
        kinds = {int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX),
                 int(mujoco.mjtGeom.mjGEOM_CYLINDER)}
        geoms = [g for g in range(mj.ngeom) if int(mj.geom_type[g]) in kinds and mj.geom_rgba[g, 3] > 0]
        if not geoms:
            raise RuntimeError('No visible robot geoms selected for rendering')
        positions, rotations, body_positions, body_rotations = [], [], [], []
        for row in display:
            self.data.qpos[self.qpos_indices] = row['joint_q']
            mujoco.mj_forward(mj, self.data)
            positions.append(self.data.geom_xpos[geoms].copy())
            rotations.append(self.data.geom_xmat[geoms].copy())
            body_positions.append(self.data.xpos[mj.geom_bodyid[geoms]].copy())
            body_rotations.append(self.data.xmat[mj.geom_bodyid[geoms]].copy())
        meshes, names = {}, []
        for g in geoms:
            name = ''
            if int(mj.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                m = mj.geom_dataid[g]
                name = mj.mesh(m).name
                v0, nv = mj.mesh_vertadr[m], mj.mesh_vertnum[m]
                f0, nf = mj.mesh_faceadr[m], mj.mesh_facenum[m]
                meshes[f'mesh_vert_{g}'] = mj.mesh_vert[v0:v0 + nv].copy()
                meshes[f'mesh_face_{g}'] = mj.mesh_face[f0:f0 + nf].copy()
            names.append(name)
        ground = self.to_world(np.array([[-.12, 0., -.08], [.12, 0., -.08],
                                         [.12, 0., .08], [-.12, 0., .08]]))
        model = self.material.model
        arrays = {key: np.stack([row[key] for row in display]) for key in display[0]}
        controls = {f'control_{key}': np.stack([row[key] for row in self.steps])
                    for key in ('sim_time', 'force', 'sensor', 'released', 'joint_q')}
        np.savez_compressed(
            output / 'episode.npz', **arrays, **controls,
            tri=model.tri_indices.cpu().numpy().reshape(-1, 3),
            cut_tri=model.cut_tri_indices.cpu().numpy().reshape(-1, 3),
            cut_edge_indices=model.cut_edge_indices.cpu().numpy().reshape(-1, 2),
            cut_edge_coords=model.cut_edge_coords.cpu().numpy(),
            knife_tri=model.knife_tri_indices.cpu().numpy().reshape(-1, 3), ground=ground,
            geoms=np.array(geoms), geom_type=mj.geom_type[geoms], geom_size=mj.geom_size[geoms],
            geom_rgba=mj.geom_rgba[geoms], geom_pos=np.stack(positions), geom_mat=np.stack(rotations),
            body_pos=np.stack(body_positions), body_mat=np.stack(body_rotations),
            geom_body=np.array([mj.body(mj.geom_bodyid[g]).name for g in geoms]),
            geom_mesh=np.array(names), total_springs=int(self.env.current['cut_spring_ke'].shape[-1]),
            **meshes)
        return len(display)


class PerturbedPolicy:
    def __init__(self, policy, case):
        self.policy, self.case = policy, case

    def reset(self, frame):
        self.policy.reset(frame)
        self.noise = CaseNoise(self.case)
        self.step = 0

    def act(self, frame):
        p = self.policy
        action = p.policy(p.history, frame['target_pose'].float())
        action = action + self.noise(self.step)[None].to(action.device)
        self.step += 1
        return p.spec.execute(action)

    def observe(self, frame):
        self.policy.observe(frame)

    def unchanged(self):
        return self.policy.unchanged()


def main():
    parser = argument_parser(default_method='supermimic')
    parser.description = __doc__
    parser.add_argument('--label', required=True)
    parser.add_argument('--case', type=int, help='predefined policy perturbation case')
    parser.add_argument('--seed-base', type=int, default=3_000_000)
    parser.add_argument('--substeps', type=int, default=12, help='display samples per control interval')
    args = parser.parse_args()
    if args.episodes != 1 or args.substeps < 1 or not 0 < args.wall_seconds < float('inf'):
        parser.error('Recording uses one episode, positive subdivisions and a finite wall budget')
    if args.method not in ('constant-zero', 'reference') and args.checkpoint is None:
        parser.error('Learned controllers require --checkpoint')
    if args.case is not None and (args.method != 'supermimic' or args.case < 0):
        parser.error('--case requires supermimic and a nonnegative case index')
    if args.trajectory is not None and args.method != 'reference':
        parser.error('--trajectory applies to --method reference')
    if args.method == 'sac' and any(value is None for value in
                                  (args.sac_python, args.sliceit_root, args.sac_config)):
        parser.error('SAC requires its Python environment, source root and config')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    def terminate(signum, stack):
        raise InterruptedError('Recording received termination signal')
    signal.signal(signal.SIGTERM, terminate)
    sys.path.insert(0, str(args.backend_root.resolve()))
    from cutting_world_model.gpu_env import GPUEnvironment
    contract = json.loads(args.contract.read_text())
    case = make_cases(args.case + 1, args.seed_base)[args.case] if args.case is not None else None
    manifest = {'git': git_revision(ROOT), 'command': sys.argv, 'method': args.method,
                'contract': contract, 'checkpoint': str(args.checkpoint) if args.checkpoint else None,
                'seed': args.seed, 'case': case, 'label': args.label,
                'wall_budget_seconds': args.wall_seconds, 'endpoint': 'One physical task episode'}
    write_json(args.output / 'manifest.json', manifest)
    policy = recorder = None
    try:
        env = GPUEnvironment(args.backend_root, worlds=1, scene=benchmark_scene(contract))
        check_contract(contract, physical_contract(args.backend_root, env))
        write_json(args.output / 'task.json', env.metadata)
        policy = make_controller(args, env, manifest)
        if case:
            policy = PerturbedPolicy(policy, case)
        recorder = GeometryRecorder(env, args.backend_root)
        write_json(args.output / 'manifest.json', manifest)
        report = evaluate(env, policy, args.output, 1, started + args.wall_seconds, on_frame=recorder)
        if report['state'] != 'completed':
            raise TimeoutError('Recording reached its wall budget')
    finally:
        if policy is not None and hasattr(policy, 'plans'):
            write_json(args.output / 'planning.json', {'planner': policy.manifest, 'steps': policy.plans})
        report_path = args.output / 'report.json'
        if report_path.exists() and recorder is not None and recorder.steps:
            report = json.loads(report_path.read_text())
            row = report['episodes'][0]
            ending = ('success' if row['success'] else 'collision' if row['collision'] else
                      'outside' if row['outside'] else 'truncated' if row['truncated'] else report['state'])
            report.update(label=args.label, method=args.method, case=case, outcome=ending,
                          checkpoint=manifest['checkpoint'], git=manifest['git'],
                          control_steps=row['steps'],
                          simulated_seconds=row['simulation_seconds'],
                          max_material_force_n=row['material_peak_force_n'],
                          display_frames=recorder.save(args.output, args.substeps),
                          substeps_per_control_step=args.substeps,
                          display_interpolation='linear positions and joint angles; held measured scalars',
                          total_wall_seconds=time.monotonic() - started)
            report['return'] = row['return']
            write_json(report_path, report)
            print(json.dumps(report), flush=True)
        if policy is not None and hasattr(policy, 'close'):
            policy.close()


if __name__ == '__main__':
    main()
