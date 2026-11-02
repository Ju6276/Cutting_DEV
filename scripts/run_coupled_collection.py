#!/usr/bin/env python3
"""Launch an isolated headless Gazebo/DiSECt collection job with a wall-time budget."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sliceit-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--wall-seconds", type=int, default=900)
    parser.add_argument("--mode", choices=("collect", "train", "evaluate"), default="collect")
    parser.add_argument("--max-hours", type=float, default=6.)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--resume")
    parser.add_argument("--model-state", choices=("native", "control-history"), default="native")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sliceit = Path(args.sliceit_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    children, logs = {}, []
    status = {"state": "starting", "pids": {}, "commands": {}, "arguments": vars(args)}
    started = time.monotonic()
    prefix = "source " + shlex.quote(str(sliceit / "scripts/public_env.sh")) + "; "
    overrides = {
        "TORCHINDUCTOR_CACHE_DIR": "/dataHDD/dong/.cache/torchinductor",
        "TRITON_CACHE_DIR": "/dataHDD/dong/.cache/triton",
        "ROS_MASTER_URI": "http://127.0.0.1:11322",
        "GAZEBO_MASTER_URI": "http://127.0.0.1:11356",
        "ROS_HOME": str(output / "ros_home"), "ROS_LOG_DIR": str(output / "ros_logs"),
    }
    prefix += "export " + " ".join(key + "=" + shlex.quote(value) for key, value in overrides.items()) + "; "

    def shell(command):
        return ["bash", "-lc", prefix + command]

    def write_status():
        status["wall_seconds"] = time.monotonic() - started
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    def check():
        if time.monotonic() - started > args.wall_seconds:
            raise TimeoutError("Collection wall-time budget exhausted")
        for name, process in children.items():
            if name != "collector" and process.poll() is not None:
                raise RuntimeError("%s exited: %s" % (name, process.returncode))

    def start(name, argv):
        command = shlex.join(argv)
        log = open(output / (name + ".log"), "w")
        logs.append(log)
        process = subprocess.Popen(shell("exec " + command), stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        children[name] = process
        status["pids"][name] = process.pid
        status["commands"][name] = command
        write_status()

    def wait_service(service):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            check()
            result = subprocess.run(shell("rosservice list"), capture_output=True, text=True, timeout=10)
            if service in result.stdout.splitlines():
                return
            time.sleep(1)
        raise TimeoutError("Service did not start: " + service)

    def stop(signum, frame):
        raise InterruptedError("Collection interrupted")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        import socket
        for port in (11322, 11356):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        start("gazebo", ["roslaunch", "-p", "11322", "o2ac_gazebo", "b_bot_gazebo.launch", "gui:=false"])
        wait_service("/controller_manager/list_controllers")
        start("bridge", ["xvfb-run", "-a", "python", "-u", str(root / "scripts/coupled_bridge.py"),
                         "--sliceit-root", str(sliceit), "--config", str(sliceit / "configs/public_prism.json"),
                         "--output", str(output / "bridge")])
        wait_service("/world_model/snapshot")
        if args.mode == "collect":
            worker = ["python", "-u", str(root / "scripts/collect_coupled.py"),
                      "--episodes", str(args.episodes)]
        else:
            worker = ["python", "-u", str(root / "scripts/train_coupled.py"),
                      "--max-hours", str(args.max_hours), "--microbatch", str(args.microbatch)]
            if args.max_iterations is not None:
                worker += ["--max-iterations", str(args.max_iterations)]
            worker += ["--model-state", args.model_state]
            if args.resume:
                worker += ["--resume", str(Path(args.resume).resolve())]
            if args.mode == "evaluate":
                worker += ["--evaluate", "--eval-episodes", str(args.episodes)]
        worker += ["--sliceit-root", str(sliceit), "--config", str(sliceit / "configs/public_prism_ros.yaml"),
                   "--output", str(output / "data")]
        start("collector", worker)
        status["state"] = "running"
        while children["collector"].poll() is None:
            check()
            write_status()
            time.sleep(2)
        status["exit_code"] = children["collector"].returncode
        if status["exit_code"]:
            raise RuntimeError("Collector failed: %d" % status["exit_code"])
        status["state"] = "completed"
    except BaseException as error:
        status["state"] = "failed"
        status["error"] = str(error)
    finally:
        for process in reversed(list(children.values())):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        write_status()
        for log in logs:
            log.close()
    print(json.dumps(status), flush=True)
    if status["state"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
