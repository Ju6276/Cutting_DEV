import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from ur5e_model import ROOT, load_model


def validate(episode):
    metadata = json.loads((ROOT / 'runs/model/robot_metadata.json').read_text())
    model, data = load_model()
    recorded = np.load(episode)
    position_errors, rotation_errors = [], []
    for joints, pose in zip(recorded['joint_q'], recorded['ee_pose']):
        for name, q in zip(metadata['joint_names'], joints):
            data.qpos[model.joint(name).qposadr[0]] = q
        mujoco.mj_forward(model, data)
        base, tip = data.body(metadata['robot_base']), data.body(metadata['robot_tip'])
        rotation = base.xmat.reshape(3, 3).T
        position = rotation @ (tip.xpos - base.xpos)
        tip_rotation = rotation @ tip.xmat.reshape(3, 3)
        reference_rotation = np.empty(9)
        mujoco.mju_quat2Mat(reference_rotation, pose[[6, 3, 4, 5]])
        position_errors.append(float(np.linalg.norm(position - pose[:3])))
        rotation_errors.append(float(np.max(np.abs(tip_rotation - reference_rotation.reshape(3, 3)))))
    source_mass = sum(group['mass'] for group in metadata['inertia_groups'])
    return {'samples': len(position_errors), 'position_max_error_m': max(position_errors),
            'rotation_matrix_max_abs_error': max(rotation_errors),
            'source_mass_kg': source_mass, 'imported_mass_kg': float(model.body_mass.sum()),
            'joint_names': [model.joint(i).name for i in range(model.njnt)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = validate(args.episode)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report['position_max_error_m'] > 1e-10 or report['rotation_matrix_max_abs_error'] > 1e-10:
        raise RuntimeError('Imported kinematics disagree with recorded robot poses')
    if abs(report['source_mass_kg'] - report['imported_mass_kg']) > 1e-10:
        raise RuntimeError('Imported mass differs from source')
