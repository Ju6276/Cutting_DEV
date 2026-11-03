#!/usr/bin/env python3
"""Run the public-data scene with the repository's default SAC settings."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ORCHESTRATOR = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--sliceit-root", required=True)
args = parser.parse_args()
ROOT = Path(args.sliceit_root).resolve()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
children = {}
logs = []
status = {"state": "starting", "pids": {}, "commands": {}, "wall_budget_seconds":259200}
started = time.monotonic()
import socket
for port in (11321,11355):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",port))


def write_status():
    status["updated_unix"] = time.time()
    path = output / "status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(path)


def shell(command):
    return ["bash", "-lc", "source " + shlex.quote(str(ROOT / "scripts/public_env.sh")) + "; export SAC_RUN_OUTPUT=" + shlex.quote(str(output)) + " ROS_HOME=" + shlex.quote(str(output / "ros_home")) + " ROS_LOG_DIR=" + shlex.quote(str(output / "ros_logs")) + "; " + command]


def start(name, command):
    log = open(output / (name + ".log"), "w")
    logs.append(log)
    process = subprocess.Popen(shell("exec " + command), stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    children[name] = process
    status["pids"][name] = process.pid
    status["commands"][name] = command
    write_status()


def check_children():
    for name, process in children.items():
        code = process.poll()
        if code is not None:
            raise RuntimeError(name + " exited with " + str(code))


def wait_service(name):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        check_children()
        result = subprocess.run(shell("rosservice list"), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, timeout=10)
        if name in result.stdout.splitlines():
            return
        time.sleep(1)
    raise RuntimeError("Service did not start: " + name)


try:
    import yaml
    original_path = ROOT / "underlay_ws/src/ur_python_utilities/ur3e_rl/config/simulation/slicing_3d_disect.yaml"
    original = yaml.safe_load(original_path.read_text())
    public_path = ROOT / "configs/public_prism_ros.yaml"
    public = yaml.safe_load(public_path.read_text())
    original["ur3e_gym"]["reset_motion"][2] = public["ur3e_gym"]["reset_motion"][2]
    if original != public:
        raise RuntimeError("Unexpected environment configuration changes.")
    (output / "environment.yaml").write_text(public_path.read_text())
    (output / "cutting_scene.json").write_text((ROOT / "configs/public_prism.json").read_text())
    revisions = {}
    for rel in [".", "disect", "underlay_ws/src/ur_python_utilities"]:
        repo = ROOT / rel
        revisions[rel] = {
            "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()),
        }
    revisions['orchestrator'] = {
        'commit': subprocess.check_output(['git', '-C', str(ORCHESTRATOR), 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', '-C', str(ORCHESTRATOR), 'status', '--porcelain'], text=True).strip()),
    }
    (output / "manifest.json").write_text(json.dumps({"git": revisions, "config": str(public_path),
        'command': sys.argv}, indent=2) + "\n")
    def request_stop(signum, frame):
        raise InterruptedError("Supervisor stop requested")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start("gazebo", "roslaunch -p 11321 o2ac_gazebo b_bot_gazebo.launch gui:=false")
    wait_service("/controller_manager/list_controllers")
    start("bridge", "xvfb-run -a python -u " + shlex.quote(str(ROOT / "scripts/public_ros_bridge.py"))
          + " --config " + shlex.quote(str(ROOT / "configs/public_prism.json"))
          + " --output " + shlex.quote(str(output / "episodes")))
    wait_service("/disect/load")
    # Give the controller spawners a chance to complete without advancing the RL task.
    subprocess.run(shell("rosservice call /controller_manager/list_controllers"), check=True,
                   stdout=open(output / "controllers.txt", "w"))
    config = json.loads((output / "run_config.json").read_text())
    command = ("env MPLBACKEND=Agg python -u " + shlex.quote(str(ORCHESTRATOR / "scripts/train_sac.py"))
               + " --sliceit-root " + shlex.quote(str(ROOT))
               + " --output " + shlex.quote(str(output))
               + " -e 3 --config " + shlex.quote(str(public_path))
               + " --logdir " + shlex.quote(str(output / "learner")))
    if config.get('resume'):
        command += " --model-dir " + shlex.quote(config['resume'])
    start("sac", command)
    status["state"] = "running"
    write_status()
    while children["sac"].poll() is None:
        if time.monotonic() - started > 259200:
            status["stop_reason"] = "72-hour wall budget"
            os.kill(children["sac"].pid, signal.SIGINT)
            children["sac"].wait(timeout=120)
            break
        for name in ("gazebo", "bridge"):
            if children[name].poll() is not None:
                raise RuntimeError(name + " exited during training")
        write_status()
        time.sleep(2)
    status["exit_code"] = children["sac"].returncode
    status["state"] = "completed" if status["exit_code"] == 0 else "failed"
except InterruptedError as exc:
    status["state"] = "interrupted"
    status["stop_reason"] = str(exc)
except BaseException as exc:
    status["state"] = "failed"
    status["error"] = str(exc)
finally:
    if "sac" in children and children["sac"].poll() is None:
        os.kill(children["sac"].pid, signal.SIGINT)
        try:
            children["sac"].wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(children["sac"].pid, signal.SIGTERM)
            try:
                children["sac"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(children["sac"].pid, signal.SIGKILL)
                children["sac"].wait()
    for name in ("bridge", "gazebo"):
        process = children.get(name)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    status["wall_seconds"] = time.monotonic() - started
    write_status()
    for log in logs:
        log.close()

if status["state"] == "failed":
    raise SystemExit(1)
