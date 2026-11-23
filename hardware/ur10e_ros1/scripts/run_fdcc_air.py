#!/usr/bin/env python3
"""Run the bounded FDCC air test with temporary gain; restore it on exit."""
import argparse
import json
import subprocess
import sys
import time
from settings import MODULE, record_directory

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--execute",action="store_true")
args=ap.parse_args()
command=[sys.executable,str(MODULE/"scripts/fdcc_air.py")]
if not args.execute:
    sys.exit(subprocess.call(command))
import rospy
from dynamic_reconfigure.client import Client
from controller_manager_msgs.srv import ListControllers,SwitchController,SwitchControllerRequest
rospy.init_node("fdcc_air_supervisor",anonymous=True)
states=rospy.ServiceProxy("/controller_manager/list_controllers",ListControllers)().controller
if any(c.state=="running" and any(r.resources for r in c.claimed_resources) for c in states):
    raise RuntimeError("Motion controller already running")
if rospy.get_param("/fdcc_shared/tool/mass") != 0:
    raise RuntimeError("This entry requires native compensated wrench / zero extra payload")
client=Client("/fdcc_shared/solver",timeout=3)
original=client.get_configuration()["error_scale"]
report=dict(original_gain=original,test_gain=.2)
try:
    got=client.update_configuration(dict(error_scale=.2))
    if got["error_scale"] != .2:
        raise RuntimeError("Gain readback mismatch")
    report["exit_code"]=subprocess.call(command+["--execute"])
finally:
    req=SwitchControllerRequest()
    req.stop_controllers=["fdcc_shared"];req.strictness=1;req.timeout=1.
    try:
        report["stop_ok"]=rospy.ServiceProxy("/controller_manager/switch_controller",SwitchController)(req).ok
    finally:
        restored=client.update_configuration(dict(error_scale=original))["error_scale"]
        report["restored_gain"]=restored
        if restored != original:raise RuntimeError("Gain restoration failed")
        path=record_directory("fdcc_air")/("supervisor_"+time.strftime("%Y%m%d_%H%M%S")+".json")
        path.write_text(json.dumps(report,indent=2))
sys.exit(report.get("exit_code",1) if report.get("stop_ok") else 1)
