"""Build the bounded native-sensor air-planning task from measured state."""
from settings import REPOSITORY, KNIFE_MESH

def build(r):
    import sys,json,copy,struct,time,math
    from pathlib import Path
    from types import SimpleNamespace
    src=REPOSITORY
    sys.path[:0]=[str(src),str(src/"backends/mujoco")]
    import numpy as np,torch
    torch.set_num_threads(1);torch.manual_seed(0)
    from cutting_world_model.shared_baselines import SharedMPPIPolicy
    from cutting_world_model.control_state import ControlStateSpec
    from cutting_world_model.mpc import MPPIConfig
    from cutting_world_model.kinematics import SerialKinematics
    from cutting_world_model.task_heads import configure_force_observation,SliceItHeads,pose_error
    from cutting_world_model.pose_depth import PROGRESS_SEMANTICS
    from cutting_gpu_backend.controller.fdcc import SlicingActionMapper
    live=json.loads((r/"feedback.json").read_text());params=live["controller_parameters"];names=params["joints"]
    assert params["tool"]["mass"]==0,"Native UR input must not have additional payload compensation"
    xml=(r/"robot.urdf").read_text();x=live["rows"][-1]
    q=torch.tensor([x["q"]],dtype=torch.float64);qd=torch.tensor([x["qd"]],dtype=torch.float64)
    fk=SerialKinematics(xml,"base_link","knife_control",names).double();pose=fk(q)
    target=pose.clone();target[:,2]-=.010
    if "commissioning_goal" in live: target=torch.tensor([live["commissioning_goal"]],dtype=torch.float64)
    # CAD blade vertices, expressed in the current actual control reference.
    data=KNIFE_MESH.read_bytes()
    n=struct.unpack_from("<I",data,80)[0];assert len(data)==84+50*n
    vertices=np.asarray([struct.unpack_from("<12fH",data,84+50*i)[3:12] for i in range(n)]).reshape(-1,3)
    vertices=np.unique(vertices,axis=0)
    def rot(q):
     x,y,z,w=q
     return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    t=live["control_from_mount"];vertices=vertices@rot(t[3:]).T+np.asarray(t[:3])
    world_vertices=vertices@rot(pose[0,3:].tolist()).T+pose[0,:3].numpy()
    board_z=float(world_vertices[:,2].min()-.03) # schema-only boundary 30 mm below; no physical board present
    meta=copy.deepcopy(json.loads((src/"configs/public_prism/benchmark.json").read_text())["task"])
    common=json.loads((src/"configs/shared/ur10e_five_food_v1.json").read_text())["common"]
    task=copy.deepcopy(common["task"]);task["workspace"]["lower"]=(pose[0,:3]-.1).tolist();task["workspace"]["upper"]=(pose[0,:3]+.1).tolist()
    meta.update(robot_xml=xml,robot_base="base_link",robot_tip="knife_control",joint_names=names,free_cut_task=task,goal_dim=8,
     physical_backend="mujoco_warp_disect",progress_semantics=PROGRESS_SEMANTICS,tool_revision=task["tool_revision"],
     diagnostic_actual_backend="native_UR_readonly_no_material_or_board",
     board_pose_geometry=dict(vertices_in_ee=vertices.tolist(),board_origin_base=[0.,0.,board_z],board_normal_base=[0.,0.,1.],stop_gap_m=task["board"]["stop_gap_m"]))
    meta["controller"]["max_force_torque"]=common["force_limits"];meta=configure_force_observation(meta,common["force_observation"])
    heads=SliceItHeads(meta["environment"],meta["controller"]);zero=torch.zeros(1,6,dtype=torch.float64)
    wh=torch.tensor([[z["wrench"] for z in live["rows"][-heads.history:]]],dtype=torch.float64)
    obs=heads.observation(pose_error(target,pose),zero,zero,zero,wh)
    frame=dict(joint_q=q[0].numpy(),joint_qd=qd[0].numpy(),wrench_history=wh[0].numpy(),peak_wrench=wh.abs().amax(1)[0].numpy(),
     last_wrench=np.asarray(x["wrench"]),motion_command=np.zeros(6),previous_ee_pose=pose[0].numpy(),previous_velocity=np.zeros(6),
     previous_acceleration=np.zeros(6),observation=np.concatenate([obs[0].numpy(),[0.]]),success_counter=np.zeros(1),
     step_count=np.zeros(1),cut_progress=np.zeros(1),depth_span=np.array([.03-task["board"]["stop_gap_m"]]),board_contact=np.zeros(1),completion_xy_error=np.zeros(1))
    meta=ControlStateSpec.metadata_from_frame(meta,frame)
    batch={k:torch.tensor(v,dtype=torch.float32)[None] for k,v in frame.items()}
    batch.update(target_pose=target.float(),target_speed=torch.tensor([[.005]]),controller_tare_compensation=torch.zeros(1,6))
    batch={k:v.cuda() for k,v in batch.items()}
    env=SimpleNamespace(worlds=1,device=torch.device("cuda"),metadata=meta,current=batch)
    (r/"configuration_cuda.json").write_text(json.dumps(dict(metadata=meta,controller_parameters=params),indent=2))
    planner=SharedMPPIPolicy(env,MPPIConfig(samples=64,horizon=4,discount=task["gamma"]),controller_parameters=params,analytic_substeps=5)
    mapper=SlicingActionMapper(max_twist=meta["controller"]["max_twist"],robot_control_dt=meta["environment"]["robot_control_dt"],pitch_control=task.get("pitch_control"),agent_control_dt=.05)

    return dict(meta=meta,params=params,planner=planner,batch=batch,target=target,mapper=mapper,live=live)
