#!/usr/bin/env python3
"""Record one frozen-policy episode for video: robot geom poses and material/knife geometry.

All geometry is written in the MuJoCo world frame of `runs/model_coupled/robot.mjb`
(material and knife: DiSECt frame -> robot base via the coupling's `base_from_disect`,
then the base body's world pose). Between control steps, joint angles and particles are
interpolated linearly for display only; recorded control-step values are kept separately.
Optional: one predefined perturbation case of `scripts/evaluate_perturbed.py`.
Render with `scripts/render_episode.py` (numpy + VTK + Pillow).
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.gpu_env import GPUEnvironment
from cutting_world_model.perturbed_eval import CaseNoise, make_cases
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_gpu_backend.controller.fdcc import quat_apply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--case', type=int, help='perturbation case index (default: unperturbed)')
    parser.add_argument('--seed-base', type=int, default=3_000_000)
    parser.add_argument('--substeps', type=int, default=12, help='display frames per control step')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    backend = ROOT / 'backends/mujoco'
    env = GPUEnvironment(str(backend.resolve()), 1)
    frame = env.current
    spec = ControlStateSpec(ControlStateSpec.metadata_from_frame(
        env.metadata, {k: v[0].cpu().numpy() for k, v in frame.items()})).to(env.device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    policy = SliceItPolicy().to(env.device).eval()
    policy.load_state_dict(saved['learner']['policy'])
    case = make_cases(args.case + 1, args.seed_base)[args.case] if args.case is not None else None
    noise = CaseNoise(case) if case else None

    physics = env.physics
    material = physics.material
    model = material.model
    base_from_disect = physics.coupling.base_from_disect.to(env.device, torch.float64)
    mj = mujoco.MjModel.from_binary_path(str(backend / 'runs/model_coupled/robot.mjb'))
    data = mujoco.MjData(mj)
    mujoco.mj_forward(mj, data)
    base = mj.body('b_bot_base_link')
    base_pos, base_mat = data.xpos[base.id].copy(), data.xmat[base.id].reshape(3, 3).copy()

    def to_world(points_disect):
        p = torch.as_tensor(points_disect, device=env.device, dtype=torch.float64)
        q = base_from_disect[3:].expand(len(p), -1)
        p_base = quat_apply(q, p) + base_from_disect[:3]
        return p_base.cpu().numpy() @ base_mat.T + base_pos

    def knife_disect():
        pose = material.state_field('body_X_sc')[0, model.knife_link_index].to(torch.float64)
        vertices = model.knife_tri_vertices.reshape(-1, 3).to(env.device, torch.float64)
        return (quat_apply(pose[3:].expand(len(vertices), -1), vertices) + pose[:3]).cpu().numpy()

    steps = []

    def capture(info=None):
        steps.append({'joint_q': frame['joint_q'][0].cpu().numpy().astype(np.float64),
                      'particles': to_world(frame['particle_q'][0].cpu().numpy()),
                      'knife': to_world(knife_disect()),
                      'released': int((frame['cut_spring_ke'][0] == 0).sum()),
                      'sim_time': float(frame['sim_time'][0, 0]),
                      'force': 0. if info is None else float(info['peak_force'][0]),
                      'sensor': float(frame['peak_wrench'][0, :3].abs().max())})

    history = frame['observation'].float()[None].repeat(policy.history, 1, 1)
    capture()
    outcome, ret, t = None, 0., 0
    with torch.no_grad():
        while outcome is None:
            action = policy(history, frame['target_pose'].float())
            if noise:
                action = action + noise(t)[None].to(env.device)
            frame, reward, term, trunc, info = env.step(spec.execute(action))
            history = torch.cat((history[1:], frame['observation'].float()[None]), 0)
            ret += float(reward[0])
            t += 1
            capture(info)
            if bool(term[0] | trunc[0]):
                outcome = ('success' if bool(info['success'][0]) else 'collision' if bool(info['collision'][0])
                           else 'outside' if bool(info['outside'][0]) else 'truncated')

    # Display frames: linear interpolation between control boundaries.
    n = args.substeps
    display = []
    for a, b in zip(steps[:-1], steps[1:]):
        for k in range(n):
            w = k / n
            display.append({key: (1 - w) * a[key] + w * b[key] if isinstance(a[key], np.ndarray) else
                            (b[key] if key in ('released', 'force', 'sensor') else (1 - w) * a[key] + w * b[key])
                            for key in a})
    display.append(steps[-1])
    geoms = [g for g in range(mj.ngeom) if mj.geom_type[g] in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX,
                                                               mujoco.mjtGeom.mjGEOM_CYLINDER)]
    poses_pos, poses_mat = [], []
    for row in display:
        data.qpos[:len(row['joint_q'])] = row['joint_q']
        mujoco.mj_forward(mj, data)
        poses_pos.append(data.geom_xpos[geoms].copy())
        poses_mat.append(data.geom_xmat[geoms].copy())
    meshes = {}
    for g in geoms:
        if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m = mj.geom_dataid[g]
            v0, nv, f0, nf = mj.mesh_vertadr[m], mj.mesh_vertnum[m], mj.mesh_faceadr[m], mj.mesh_facenum[m]
            meshes[f'mesh_vert_{g}'] = mj.mesh_vert[v0:v0 + nv].copy()
            meshes[f'mesh_face_{g}'] = mj.mesh_face[f0:f0 + nf].copy()
    tri = model.tri_indices.cpu().numpy().reshape(-1, 3)
    ground = to_world(np.array([[-0.12, 0., -0.08], [0.12, 0., -0.08], [0.12, 0., 0.08], [-0.12, 0., 0.08]]))
    np.savez_compressed(
        args.output / 'episode.npz', particles=np.stack([r['particles'] for r in display]),
        knife=np.stack([r['knife'] for r in display]), released=np.array([r['released'] for r in display]),
        sim_time=np.array([r['sim_time'] for r in display]), force=np.array([r['force'] for r in display]),
        sensor=np.array([r['sensor'] for r in display]), tri=tri,
        knife_tri=model.knife_tri_indices.cpu().numpy().reshape(-1, 3), ground=ground,
        geoms=np.array(geoms), geom_type=mj.geom_type[geoms], geom_size=mj.geom_size[geoms],
        geom_rgba=mj.geom_rgba[geoms], geom_pos=np.stack(poses_pos), geom_mat=np.stack(poses_mat),
        total_springs=int(frame['cut_spring_ke'].shape[-1]), **meshes)
    report = {'git': git_revision(ROOT), 'checkpoint': str(args.checkpoint), 'label': args.label,
              'policy_updates': saved['learner'].get('updates'), 'case': case, 'outcome': outcome,
              'control_steps': t, 'return': ret, 'simulated_seconds': steps[-1]['sim_time'] - steps[0]['sim_time'],
              'max_material_force_n': max(r['force'] for r in steps), 'display_frames': len(display),
              'substeps_per_control_step': n}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
