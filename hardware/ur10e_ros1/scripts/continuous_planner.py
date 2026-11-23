"""Air commissioning: GPU MPC with real feedback; ROS runs independently."""
import sys,json,time,traceback
from pathlib import Path
import torch
from planner_setup import build
ipc=Path(sys.argv[1])
context=build(ipc)
meta,params,planner,batch,target,mapper,live=[context[k] for k in ("meta","params","planner","batch","target","mapper","live")]
from fast_native_model import FastNativeModel
from cutting_world_model.pose_depth import depth_progress
fast=FastNativeModel(meta,params,5,"cuda",native_gain=.2)
planner.model=fast
for rollout in planner.rollouts:rollout.wm=fast
# Authorized vertical air commissioning: no sawing or orientation action.
for p in planner.planners:p.u_min[4:]=0;p.u_max[4:]=0;p.U[:,4:]=0
if "commissioning_stiffness_bounds" in live:
 low,high=live["commissioning_stiffness_bounds"]
 assert (low,high)==(3000,5000)
 for p in planner.planners:
  p.u_min[0]=(low/2-500)/4500+1e-6
  p.u_max[0]=(high/2-500)/4500
spec=planner.spec
goal=torch.cat((batch["target_pose"],batch["target_speed"].reshape(1,1)),-1)
def write(path,data):
 tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
# Prewarm without commanding the robot.
warm=[]
for _ in range(2):
 t=time.monotonic();a=planner.act(batch);warm.append(time.monotonic()-t)
for p in planner.planners:p.U.copy_(torch.maximum(torch.minimum(torch.zeros_like(p.U),p.u_max),p.u_min))
planner.steps=[0]
write(ipc/"ready.json",dict(device="cuda",samples=64,horizon=4,gain=.2,warm_seconds=warm,
 mode="continuous_fdcc_asynchronous_mpc_air_only",goal=target[0].tolist(),motion_commands_sent=0))
from graph_feedback import GraphFeedback
updater=GraphFeedback(spec,goal)
# Validate actual-state derivation before accepting any live execution.
args0=[planner.state]+[spec.field(planner.state,n).clone() for n in ("joint_q","joint_qd","wrench_history","peak_wrench","last_wrench")]+[torch.zeros(1,6,device="cuda")]
expected=updater.raw(*args0);actual=updater(*args0)
assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(expected,actual)), "Feedback graph mismatch"
write(ipc/"feedback_graph_validation.json",dict(max_state_error=float((expected[0]-actual[0]).abs().max()),max_reward_error=float((expected[1]-actual[1]).abs().max()),termination_equal=bool(torch.equal(expected[2],actual[2])),motion_commands_sent=0))
deadline=time.monotonic()+160;last_seq=-1;last_frame=-1
while time.monotonic()<deadline and not (ipc/"stop").exists():
 requests=sorted(ipc.glob("request_*.json"))
 for p in requests:
  req=json.loads(p.read_text())
  if req["sequence"]<=last_seq:continue
  out=ipc/p.name.replace("request_","response_")
  try:
   frames=req["frames"];process_start=time.monotonic()
   for frame in frames:
    assert frame["sequence"]==last_frame+1,"Feedback sequence gap"
    inputs=[planner.state]+[torch.tensor(frame[name],dtype=torch.float32,device="cuda")[None] for name in ("joint_q","joint_qd","wrench_history","peak_wrench","last_wrench")]+[torch.tensor(frame["action"],dtype=torch.float32,device="cuda")[None]]
    planner.state,_,done=updater(*inputs)
    # Air commissioning has its own stricter 3-D position stop; no paper success claim.
    last_frame=frame["sequence"]
   frame_seconds=time.monotonic()-process_start
   # U belongs to the previous plan; do not pretend only 50 ms elapsed.
   for ppi in planner.planners:ppi.U.copy_(torch.maximum(torch.minimum(torch.zeros_like(ppi.U),ppi.u_max),ppi.u_min))
   planner.steps=[0]
   t=time.monotonic();a=planner.act(batch);seconds=time.monotonic()-t
   assert a[0,4:].abs().max()==0
   actual_pose=fast.chain(spec.field(planner.state,"joint_q").double(),mass_matrix=False)[0][fast.chain.tip]
   mapper.reset(1);mapper.added_motion_command=spec.field(planner.state,"motion_command").cpu().double().clone()
   mapped=mapper(a.cpu().double(),target,actual_pose.cpu());mapped["error_scale"].fill_(.2)
   write(out,dict(sequence=req["sequence"],input_frame=last_frame,input_monotonic=req["snapshot_monotonic"],
    frame_seconds=frame_seconds,planning_seconds=seconds,action=a[0].tolist(),
    mapped={k:v[0].tolist() for k,v in mapped.items()},goal=target[0].tolist(),
    request_q=spec.field(planner.state,"joint_q")[0].tolist(),device="cuda",samples=64,horizon=4))
   last_seq=req["sequence"]
  except Exception as e:
   write(out,dict(sequence=req["sequence"],error=str(e),traceback=traceback.format_exc()));raise
 time.sleep(.01)
