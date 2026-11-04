from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'runs/model/robot.mjb'


def load_model():
    model = mujoco.MjModel.from_binary_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[:] = [0, -1.57, 1.57, -1.57, -1.57, 0]
    mujoco.mj_forward(model, data)
    return model, data


def action_mapping(model):
    return [{'index': i, 'actuator': model.actuator(i).name,
             'joint': model.joint(int(model.actuator_trnid[i, 0])).name,
             'ctrlrange_nm': model.actuator_ctrlrange[i].tolist()}
            for i in range(model.nu)]


def targets(model, time, world):
    return 0.1 * np.sin(np.pi * time + world * 0.1 + np.arange(model.nu) * 0.2)
