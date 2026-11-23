#!/usr/bin/env python3
"""Replay captured feedback on a GPU; no ROS imports or robot connection."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from planner_setup import build

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--record-dir",type=Path,required=True,help="Saved feedback.json and robot.urdf")
a=ap.parse_args()
ctx=build(a.record_dir)
import torch
from fast_native_model import NativeGainModel,FastNativeModel
from graph_feedback import GraphFeedback
reference=ctx["planner"].model
reference.__class__=NativeGainModel
reference.native_gain=.2
fast=FastNativeModel(ctx["meta"],ctx["params"],5,"cuda",native_gain=.2)
fast.set_tare_compensation(torch.zeros(1,6,device="cuda"))
planner=ctx["planner"];spec=planner.spec
goal=torch.cat((ctx["batch"]["target_pose"],ctx["batch"]["target_speed"].reshape(1,1)),-1)
results=[]
for count in (1,32,64):
    state=planner.state.repeat(count,1)
    actions=torch.rand(count,6,device="cuda")*2-1
    goals=goal.repeat(count,1)
    expected=reference.predict(state,actions,goals)
    actual=fast.predict(state,actions,goals)
    errors=[float((x-y).abs().max()) for x,y in zip(expected[:2],actual[:2])]
    if not torch.equal(expected[2],actual[2]):raise AssertionError("Termination mismatch")
    for x,y in zip(expected[:2],actual[:2]):
        torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    results.append(dict(batch=count,max_state_reward_errors=errors))
updater=GraphFeedback(spec,goal)
inputs=[planner.state]+[spec.field(planner.state,n).clone() for n in ("joint_q","joint_qd","wrench_history","peak_wrench","last_wrench")]+[torch.zeros(1,6,device="cuda")]
expected=updater.raw(*inputs);actual=updater(*inputs)
for x,y in zip(expected,actual):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
planner.model=fast
for rollout in planner.rollouts:rollout.wm=fast
for p in planner.planners:p.u_min[4:]=0;p.u_max[4:]=0;p.U[:,4:]=0
start=time.monotonic()
action=planner.act(ctx["batch"])
assert torch.isfinite(action).all() and action[:,4:].abs().max()==0
report=dict(status="passed",device=torch.cuda.get_device_name(),equivalence=results,
            actual_feedback_graph="passed",plan_seconds=time.monotonic()-start,
            action=action[0].tolist(),robot_connection=False,motion_commands_sent=0)
(a.record_dir/"module_validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
