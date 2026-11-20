"""One frozen-controller cutting episode following a depth-selected mid-section.

Bounded development demonstration: optional explicit cross-backend policy transfer,
simulation reset to pre-cut start; optional frozen learned perception, no policy optimization.
"""
import argparse,copy,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from cutting_world_model.blademaster_backend import BladeMasterEnvironment
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.training_snapshot import capture_training_environment
from cutting_world_model.visual_execution import place_before_episode
from cutting_world_model.visual_cut import front_camera
from cutting_world_model.visual_center import center_plan,load_center
from cutting_world_model.episode_log import episode_row

ROOT=Path(__file__).resolve().parents[1]
def save_results_and_close(env, planner, traces, rows, status, output, write):
    """Keep physics cleanup unconditional, including disk/serialization failures."""
    try:
        if planner is not None:
            write("planning.json", planner.plans)
        torch.save(traces, output / "trajectories.pt")
        write("summary.json", dict(status=status, episodes=rows))
    except BaseException as error:
        status.update(state="failed", phase="persist_results", error=repr(error))
        # Preserve the original error even when the filesystem is unavailable.
        try:
            write("progress.json", status)
        except Exception:
            pass
        raise
    finally:
        if env is not None:
            env.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",type=Path,required=True);p.add_argument("--backend-checkpoint",type=Path,required=True)
    p.add_argument("--render-python",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--allow-backend-transfer",action="store_true")
    p.add_argument("--selection",choices=["visible-midpoint","fixed-midpoint","learned-midpoint"],default="visible-midpoint")
    p.add_argument("--center-checkpoint",type=Path)
    p.add_argument("--wall-seconds",type=float,default=1500)
    p.add_argument("--method",choices=["wm_policy","wm_mppi","analytic_mpc"],default="wm_policy")
    p.add_argument("--model-checkpoint",type=Path)
    p.add_argument("--planner-horizon",type=int,default=8)
    p.add_argument("--planner-samples",type=int,default=512)
    p.add_argument("--planner-refinements",type=int,default=1)
    p.add_argument("--analytic-substeps",type=int,default=5)
    a=p.parse_args()
    if (a.method=="wm_mppi") != (a.model_checkpoint is not None):
        p.error("--model-checkpoint is required only for wm_mppi")
    if (a.selection=="learned-midpoint") != (a.center_checkpoint is not None):
        p.error("--center-checkpoint is required only for learned-midpoint")
    center_model,center_data=load_center(a.center_checkpoint) if a.center_checkpoint else (None,None)
    a.output.mkdir(exist_ok=False)
    start=time.monotonic();env=None;planner=None;traces=[];rows=[];status={"state":"initializing"}
    def write(n,d): (a.output/n).write_text(json.dumps(d,indent=2,allow_nan=False))
    def progress(**kw):
        status.update(kw,elapsed_seconds=time.monotonic()-start);write("progress.json",status)
    def run(argv):
        remaining=a.wall_seconds-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError("Visual evaluation wall budget")
        subprocess.run([str(x) for x in argv],check=True,timeout=remaining,
                       env={**os.environ,"PYTHONPATH":str(ROOT)+":"+str(ROOT/"scripts")})
    try:
        torch.set_num_threads(2);torch.manual_seed(20261117)
        data=torch.load(a.policy,map_location="cpu",weights_only=False)
        backend=torch.load(a.backend_checkpoint,map_location="cpu",weights_only=False)
        config=copy.deepcopy(backend["metadata"]["backend_config"]);del backend
        config["worlds"]=1;config["groups"]=[config["groups"][0]];config["groups"][0]["worlds"]=1
        config["start_clearance_m"]=[.02];config["case_seed"]=20261117
        config["episode_conditions"].update(seed=20261117,youngs_modulus=[1.,1.],density=[1.,1.],
                                              cut_strength=[1.,1.],target_speed=[.03,.03])
        if data["metadata"]["physical_backend"]!="mujoco_warp_blademaster" and not a.allow_backend_transfer:
            raise ValueError("Cross-backend inference requires explicit flag")
        if data["metadata"]["free_cut_task"] != config["task"]:
            raise ValueError("Policy task/reward differs from evaluation")
        if data["metadata"]["goal_dim"]!=8 or data["metadata"]["state_shapes"]["observation"]!=[34]:
            raise ValueError("Current goal/observation interface required")
        manifest=dict(method=a.method,policy_identity=data["identity"] if a.method!="analytic_mpc" else None,
            goal_convention_reference=data["identity"],config=config,visual_selection=a.selection,
            center_model=None if not a.center_checkpoint else dict(path=str(a.center_checkpoint),
                sha256=hashlib.sha256(a.center_checkpoint.read_bytes()).hexdigest(),best_epoch=center_data["best_epoch"],
                architecture=center_data.get("architecture","local_v1")),
            cross_backend=data["metadata"]["physical_backend"]!="mujoco_warp_blademaster",
            budget_seconds=a.wall_seconds,episodes=1,
            scope="Pre-cut depth selection then force-feedback policy; not continuously replanned vision",
            approach="IK/reset placement, not a collision-checked approach trajectory")
        manifest["source_sha256"]={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ["scripts/evaluate_visual_cut.py","scripts/capture_visual_studio.py","cutting_world_model/visual_cut.py","cutting_world_model/visual_execution.py","cutting_world_model/visual_center.py","cutting_world_model/visual_center_features.py","cutting_world_model/vendor/dp3/pointnet.py","cutting_world_model/vendor/dp3/SOURCE.json","cutting_world_model/shared_baselines.py","cutting_world_model/mpc.py"]}
        write("manifest.json",manifest);progress(phase="physics_initialize")
        env=BladeMasterEnvironment(ROOT/"backends/mujoco",config)
        if env.metadata["controller"]!=data["metadata"]["controller"]:
            raise ValueError("Controller/normalization mismatch")
        policy=None;frozen=None
        if a.method=="wm_policy":
            policy=SliceItPolicy(8,34).to(env.device).eval()
            policy.load_state_dict(data["policy"]);policy.requires_grad_(False)
            frozen={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        if a.model_checkpoint:
            from cutting_world_model.visual_execution import validate_paired_visual_checkpoint
            paired=torch.load(a.model_checkpoint,map_location="cpu",weights_only=False)
            validate_paired_visual_checkpoint(data,paired)
            manifest["model_checkpoint"]=dict(path=str(a.model_checkpoint),
                sha256=hashlib.sha256(a.model_checkpoint.read_bytes()).hexdigest())
            del paired
        # Move the reset pose clear of the camera's view; no trajectory claimed.
        park=env.current["ee_pose"][0].cpu().numpy().copy();park[0]=.043;park[2]=.245
        frame=place_before_episode(env,park)
        progress(phase="camera_capture")
        pre=a.output/"perception_source"
        capture_training_environment(env,ROOT/"backends/mujoco",pre/"geometry/0000",
             dict(iteration=0,policy_updates=data["identity"]["policy_updates"] if policy is not None else None,reference_policy_updates=data["identity"]["policy_updates"],training_stage="visual_pre_cut",ee_pose_base=frame["ee_pose"].cpu().tolist()))
        geometry=env.metadata["board_pose_geometry"]
        camera=front_camera(geometry["board_origin_base"])
        write("camera.json",camera.to_dict())
        visual=json.loads((ROOT/"configs/vision/visual_midpoint_v1.json").read_text())
        visual.update({k:geometry[k] for k in ["vertices_in_ee","board_origin_base","board_normal_base","stop_gap_m"]})
        visual["tool_orientation_xyzw"]=env.physics.target[0,3:].cpu().tolist()
        reference=(data["metadata"]["environment"]["target_pose"][3:]
                   if data["metadata"]["physical_backend"]=="mujoco_warp_disect"
                   else env.physics.target[0,3:].cpu().tolist())
        visual["policy_quaternion_reference_xyzw"]=reference
        manifest["policy_quaternion_reference_xyzw"]=reference
        visual["workspace"]=config["task"]["workspace"]
        write("visual_config.json",visual)
        roots=["--source-root","/opt/ros/noetic/share/ur_description",
               "--source-root","/dataHDD/dong/Projects/SliceIt/catkin_ws/src"]
        run([a.render_python,ROOT/"scripts/export_evaluation_scene.py","--evaluation",pre,"--group","0",
             "--snapshot","--backend","BladeMaster",*roots,"--output",pre/"scene.npz"])
        capture_args=[a.render_python,ROOT/"scripts/capture_visual_studio.py","--scene",pre/"scene.npz",
             "--camera",a.output/"camera.json","--config",a.output/"visual_config.json",
             "--output",a.output/"perception"]
        if a.selection!="visible-midpoint":capture_args.append("--capture-only")
        run(capture_args)
        if a.selection=="visible-midpoint":
            plan=json.loads((a.output/"perception/plan.json").read_text())
        else:
            depth=np.load(a.output/"perception/depth.npz",allow_pickle=False)
            plan=center_plan(depth["depth_m"],camera,visual,depth["robot_mask"],model=center_model)
            mask=plan.pop("object_mask")
            np.savez_compressed(a.output/"perception/points.npz",object_points_base=camera.points(depth["depth_m"])[mask])
            write("perception/plan.json",plan)
            from PIL import Image,ImageDraw
            preview=Image.open(a.output/"perception/rgb.png").convert("RGB")
            centre=np.asarray(plan["selected_plane_point_base"]);along=np.cross(plan["selected_plane_normal_base"],[0,0,1])
            line=camera.project(np.array([centre-.075*along,centre+.075*along]))
            ImageDraw.Draw(preview).line([tuple(v) for v in line],fill=(245,185,45),width=3)
            preview.save(a.output/"perception/perception.png")
        if plan["prescribed_cut_pose_used"]:raise AssertionError("Preset pose leakage")
        old_goal=env.physics.target[0].cpu().numpy().copy()
        frame=place_before_episode(env,plan["start_ee_pose"],plan["policy_goal"])
        if a.method!="wm_policy":
            from cutting_world_model.mpc import MPPIConfig
            from cutting_world_model.shared_baselines import SharedMPPIPolicy
            from cutting_world_model.controller_profile import controller_parameters
            params=controller_parameters(env.metadata,ROOT/"backends/mujoco")
            planner=SharedMPPIPolicy(env,MPPIConfig(horizon=a.planner_horizon,
                samples=a.planner_samples,refinements=a.planner_refinements,
                discount=config["task"]["gamma"],terminal_value="none"),
                checkpoint=a.model_checkpoint,controller_parameters=params,
                analytic_substeps=a.analytic_substeps,
                allow_backend_transfer=a.allow_backend_transfer)
            manifest["planner"]=planner.manifest
            torch.manual_seed(20261117)
        manifest.update(visual_plan=plan,old_goal_for_audit_only=old_goal.tolist(),
                        xy_goal_shift_mm=((np.array(plan["goal_ee_pose"])[:2]-old_goal[:2])*1000).tolist(),
                        actual_metadata=env.metadata)
        write("manifest.json",manifest)
        conditions=env.conditions([0]);write("cases.json",[dict(case=0,group=env.object_names[0],condition=conditions[0])])
        def capture(step):
            capture_training_environment(env,ROOT/"backends/mujoco",a.output/"geometry"/("%04d"%step),
                dict(iteration=step,policy_updates=data["identity"]["policy_updates"] if policy is not None else None,reference_policy_updates=data["identity"]["policy_updates"],training_stage="frozen_visual_evaluation",ee_pose_base=frame["ee_pose"].cpu().tolist()))
        history=frame["observation"].float()[None].repeat(10,1,1)
        acc={k:torch.zeros(1,device=env.device) for k in ["return","steps","force","ratio","impulse","speed"]}
        capture(0);progress(state="running",phase="policy",step=0)
        with torch.no_grad(),(a.output/"episodes.jsonl").open("w") as log:
            for step in range(1,config["task"]["horizon_steps"]+1):
                if time.monotonic()-start>a.wall_seconds:raise TimeoutError("Visual trial wall budget")
                goal=env.goal(frame)
                clean=planner.act(frame) if planner is not None else policy(history,goal)
                action=clean.clamp(-1,1)
                if not bool(torch.isfinite(action).all()):raise FloatingPointError("Nonfinite visual control action")
                before={k:frame[k].cpu().clone() for k in ["observation","joint_q","ee_pose","step_count","cut_progress","peak_wrench"]}
                frame,reward,ended,truncated,info=env.step(action)
                if planner is not None:planner.observe(frame)
                capture(step)
                acc["return"]+=reward;acc["steps"]+=1
                acc["force"]=torch.maximum(acc["force"],info["peak_force"].float())
                acc["ratio"]=torch.maximum(acc["ratio"],info["safety_margin"].float())
                acc["impulse"]+=info["material_impulse"]
                acc["speed"]+=(env.heads.cutting_speed(frame["observation"])-frame["target_speed"]).abs()
                traces.append(dict(before=before,active=torch.ones(1,dtype=torch.bool),action=action.cpu(),clean_action=clean.cpu(),
                    goal=goal.cpu(),reward=reward.cpu(),terminated=ended.cpu(),truncated=truncated.cpu(),
                    next_observation=frame["observation"].cpu().clone(),next_pose=frame["ee_pose"].cpu().clone(),
                    next_step_count=frame["step_count"].cpu().clone(),
                    info={k:v.cpu().clone() for k,v in info.items() if isinstance(v,torch.Tensor)}))
                if step%10==0 or bool(ended.any()):progress(step=step,return_value=float(acc["return"][0]))
                if bool(ended.any()):
                    row=episode_row(0,0,step,acc["return"],acc["steps"],info,truncated,acc["force"],acc["ratio"],conditions[0])
                    row.update(case=0,group=env.object_names[0],material_impulse_ns=float(acc["impulse"][0]),
                               speed_error_mps=float(acc["speed"][0]/acc["steps"][0]))
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+"\n");break
                history=torch.cat([history[1:],frame["observation"].float()[None]],0)
        if not rows:raise AssertionError("Episode failed to terminate")
        if policy is not None and any(not torch.equal(v.cpu(),frozen[k]) for k,v in policy.state_dict().items()):
            raise AssertionError("Frozen policy changed")
        if planner is not None and not planner.unchanged():raise AssertionError("Frozen planner weights changed")
        progress(state="completed")
    except BaseException as error:
        progress(state="failed",error=repr(error));raise
    finally:
        save_results_and_close(env, planner, traces, rows, status, a.output, write)
if __name__=="__main__":main()
