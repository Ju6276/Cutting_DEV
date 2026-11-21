#!/usr/bin/env python3
"""Offline analytic MPC diagnostic. Contains no ROS or robot command interface.

Measured UR10e joints initialize a hypothetical tool0 point and zero-wrench scene.
This is not a physical cutting simulation or validation of the live FZI plugin.
"""
import argparse
import copy
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
p.add_argument('--snapshot', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--steps', type=int, default=30)
a = p.parse_args()
if a.steps < 1:
    p.error('--steps must be positive')
if a.output.exists():
    p.error('--output must be a new directory to preserve previous evidence')
a.output.mkdir(parents=True)
sys.path[:0] = [str(a.repo), str(a.repo/'backends/mujoco')]
from cutting_world_model.shared_baselines import AnalyticFDCCModel, SharedMPPIPolicy
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.mpc import MPPIConfig
from cutting_world_model.kinematics import SerialKinematics
from cutting_world_model.task_heads import configure_force_observation, SliceItHeads, pose_error
from cutting_world_model.pose_depth import PROGRESS_SEMANTICS
from cutting_gpu_backend.controller.fdcc import FDCCController, SerialChain, ControllerConfig

torch.set_num_threads(1)
torch.manual_seed(0)
snapshot = json.loads((a.snapshot/'feedback.json').read_text())
common = json.loads((a.repo/'configs/shared/ur10e_ati_ikea_v1.json').read_text())['common']
# Reuse the source observation/action interface only, replacing all robot geometry.
meta = copy.deepcopy(json.loads((a.repo/'configs/public_prism/benchmark.json').read_text())['task'])
names = snapshot['fdcc_parameters']['joints']
order = [snapshot['last']['names'].index(n) for n in names]
q = torch.tensor([[snapshot['last']['q'][i] for i in order]], dtype=torch.float64)
qd = torch.tensor([[snapshot['last']['qd'][i] for i in order]], dtype=torch.float64)
xml = (a.snapshot/'robot_description.urdf').read_text()
fk = SerialKinematics(xml, 'base_link', 'tool0', names).double()
pose = fk(q)
target = pose.clone()
target[:, 2] -= .005
task = copy.deepcopy(common['task'])
task['workspace']['lower'] = (pose[0,:3]-.1).tolist()
task['workspace']['upper'] = (pose[0,:3]+.1).tolist()
virtual_board_z = float(target[0,2]-task['board']['stop_gap_m'])
meta.update(robot_xml=xml, robot_base='base_link', robot_tip='tool0', joint_names=names,
    free_cut_task=task, goal_dim=8, tool_revision=task['tool_revision'],
    # The existing state spec requires this schema identity. No backend is run.
    physical_backend='mujoco_warp_disect', progress_semantics=PROGRESS_SEMANTICS,
    diagnostic_actual_backend='analytic_fdcc_only_no_material_simulator',
    diagnostic_actual_tool='hypothetical_tool0_point_diagnostic_v1',
    board_pose_geometry=dict(vertices_in_ee=[[0.,0.,0.]],
        board_origin_base=[0.,0.,virtual_board_z],
        board_normal_base=[0.,0.,1.], stop_gap_m=task['board']['stop_gap_m']))
meta['controller']['max_force_torque'] = common['force_limits']
meta = configure_force_observation(meta, common['force_observation'])
heads = SliceItHeads(meta['environment'], meta['controller'])
zero = torch.zeros(1,6,dtype=torch.float64)
obs = heads.observation(pose_error(target,pose), zero, zero, zero,
                        torch.zeros(1,heads.history,6,dtype=torch.float64))
frame = dict(joint_q=q[0].numpy(), joint_qd=qd[0].numpy(),
    wrench_history=np.zeros((heads.history,6)), peak_wrench=np.zeros(6), last_wrench=np.zeros(6),
    motion_command=np.zeros(6), previous_ee_pose=pose[0].numpy(),
    previous_velocity=np.zeros(6), previous_acceleration=np.zeros(6),
    observation=np.concatenate([obs[0].numpy(),[0.]]), success_counter=np.zeros(1),
    step_count=np.zeros(1), cut_progress=np.zeros(1), depth_span=np.array([.005]),
    board_contact=np.zeros(1), completion_xy_error=np.zeros(1))
meta = ControlStateSpec.metadata_from_frame(meta, frame)
params = json.loads((a.repo/'configs/public_prism/controller.json').read_text())
params.update(compliance_ref_link='tool0', ft_sensor_ref_link='tool0',
              robot_base_link='base_link', end_effector_link='tool0', joints=names)
batch = {k:torch.tensor(v,dtype=torch.float32)[None] for k,v in frame.items()}
batch.update(target_pose=target.float(), target_speed=torch.tensor([[.01]]))
env = SimpleNamespace(worlds=1, device=torch.device('cpu'), metadata=meta, current=batch)
model = AnalyticFDCCModel(meta, params, substeps=5)
chain = SerialChain(xml, 'base_link', 'tool0', names).double()
fdcc = FDCCController(chain, ControllerConfig.from_parameters(params))
fdcc.reset(q,torch.zeros_like(q))
hold = fdcc.update(q,target_pose=pose,sensor_wrench=zero)
fdcc.reset(q,torch.zeros_like(q))
small = fdcc.update(q,target_pose=target,sensor_wrench=zero)
# Source explicit Euler integrates old velocity, so the first position is unchanged.
small = fdcc.update(small,target_pose=target,sensor_wrench=zero)
assert torch.isfinite(small).all()
assert (hold-q).abs().max() < 1e-7
assert (small-q).abs().max() > 1e-10
tf_pose = torch.tensor(snapshot['transforms']['base_link <- tool0']['xyz'])
goal = torch.cat([target.float(),torch.tensor([[.01]])],dim=-1)
report = dict(git_sha=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),
    git_dirty=bool(subprocess.check_output(['git','-C',str(a.repo),'status','--porcelain'],text=True).strip()),
    diagnostic_revision='ur10e_tool0_assumed_mpc_v1',
    created_unix=time.time(), python=platform.python_version(),
    dependencies={n:importlib.metadata.version(n) for n in ['torch','numpy','pytorch-mppi','arm-pytorch-utilities']},
    motion_commands_sent=0, actual_backend='analytic_fdcc_only_no_material_simulator',
    assumptions=['tool0 point is control reference, not calibrated blade TCP',
        'zero external wrench is synthetic, not ATI or UR force measurement',
        'virtual target 5 mm below measured tool0, virtual board and +/-100 mm workspace',
        'initial controller/observation history reset and padded by existing state spec',
        'source FDCC surrogate parameters, not live FZI controller equivalence',
        'mujoco_warp_disect metadata tag is required by state schema only, no material backend run',
        'ikea_cad_mesh_v1 task tag retained for schema only, geometry explicitly replaced by tool0 point',
        'material height_m=0 schema field unused, virtual plane defined by board_pose_geometry',
        'ideal joint tracking and held wrench, no payload or contact force validation'],
    feedback=dict(rate_hz=snapshot['rate_hz'], max_stamp_age_s=snapshot['max_stamp_age_s'],
                  captured_unix=snapshot['captured_unix']),
    controllers=snapshot['controllers'],
    coordinate_check=dict(base='base_link', tip='tool0', joint_names=names,
        initial_pose_xyzw=pose[0].tolist(), target_pose_xyzw=target[0].tolist(),
        fk_vs_ros_tf_position_m=float(torch.linalg.vector_norm(pose[0,:3]-tf_pose)),
        tool0_vs_controller=snapshot['transforms'].get('tool0 <- tool0_controller')),
    controller_proxy=dict(hold_max_joint_delta_rad=float((hold-q).abs().max()),
        target5mm_max_joint_delta_rad=float((small-q).abs().max()), finite=True), timing=[])
(a.output/'configuration.json').write_text(json.dumps(dict(metadata=meta,controller_parameters=params,
    live_ros_controller_parameters=snapshot['fdcc_parameters']),indent=2))
for samples in [8,512]:
    planner = SharedMPPIPolicy(env, MPPIConfig(samples=samples,horizon=8,discount=1.),
                               controller_parameters=params,analytic_substeps=5)
    for _ in range(4):
        action = planner.act(batch)
        assert torch.isfinite(action).all()
        # Planner keeps a six-dimensional action. Disabled pitch is masked in
        # the action mapper, not necessarily zero in the raw optimizer output.
    seconds = [x['planning_seconds'] for x in planner.plans]
    result = dict(samples=samples,horizon=8,substeps=5,cold_seconds=seconds[0],
        warm_median_seconds=float(np.median(seconds[1:])),max_seconds=max(seconds),
        warm_count=3,period_seconds=.05,within_50ms=all(t<=.05 for t in seconds),plans=planner.plans)
    report['timing'].append(result)
    print(json.dumps({k:v for k,v in result.items() if k!='plans'}),flush=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
# Receding horizon against the same analytic model is a self-consistency check.
planner = SharedMPPIPolicy(env, MPPIConfig(samples=512,horizon=8,discount=1.),
                           controller_parameters=params,analytic_substeps=5)
state = planner.state.clone()
trajectory = []
reason = 'diagnostic_step_budget'
for step in range(a.steps):
    action = planner.act(batch)
    state,reward,logits = model.predict(state,action,goal)
    assert torch.isfinite(state).all() and torch.isfinite(reward).all()
    assert model.spec.field(state,'motion_command')[0,4] == 0
    planner.state = state.clone()
    sp = model.spec
    row = dict(step=step+1, action=action[0].tolist(), reward=float(reward[0]),
        q=sp.field(state,'joint_q')[0].tolist(), qd=sp.field(state,'joint_qd')[0].tolist(),
        pose=sp.fk(sp.field(state,'joint_q'))[0].tolist(),
        progress=float(sp.field(state,'cut_progress')[0,0]),
        arrival_count=float(sp.field(state,'success_counter')[0]),
        planning_seconds=planner.plans[-1]['planning_seconds'],
        target_error_m=float(torch.linalg.vector_norm(sp.fk(sp.field(state,'joint_q'))[0,:3]-target[0,:3])))
    trajectory.append(row)
    print(json.dumps(row),flush=True)
    if bool((logits>0).any()):
        reason = 'hypothetical_pose_success' if bool(sp.pose_success(state)[0]) else 'hypothetical_failure_or_timeout'
        break
report['closed_loop'] = dict(reason=reason,steps=len(trajectory),same_model_self_consistency_only=True,
    initial_target_error_m=.005,final_target_error_m=trajectory[-1]['target_error_m'],
    final_arrival_count=trajectory[-1]['arrival_count'],trajectory=trajectory)
times = np.array([row['planning_seconds'] for row in trajectory])
xyz_error = np.array(trajectory[-1]['pose'][:3])-target[0,:3].numpy()
report['closed_loop'].update(planning_median_seconds=float(np.median(times)),
    planning_max_seconds=float(times.max()),wall_planning_seconds=float(times.sum()),
    simulated_seconds=.05*len(trajectory),final_xyz_error_m=xyz_error.tolist(),
    final_xy_error_m=float(np.linalg.norm(xyz_error[:2])))
(a.output/'report.json').write_text(json.dumps(report,indent=2))
print('Saved '+str(a.output/'report.json'),flush=True)
