"""Bounded baseline adapter checks on recorded UR data or a live shared environment.

Recorded mode checks planning and SAC updates without physics; it is never reported
as a closed-loop result. Live mode additionally executes each controller's actions.
"""
import argparse, hashlib, json, signal, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backends/mujoco")]
import numpy as np
import torch
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.shared_baselines import SharedMPPIPolicy
from cutting_world_model.controller_profile import controller_parameters
from cutting_world_model.mpc import MPPIConfig
from cutting_world_model.sac_vector import CuttingVecEnv, TransitionSAC
from stable_baselines3.common.logger import configure
from gymnasium import Env, spaces

class SpacesOnly(Env):
    observation_space = spaces.Box(-np.inf, np.inf, (348,), np.float32)
    action_space = spaces.Box(-1., 1., (6,), np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(348, np.float32), {}
    def step(self, action):
        raise RuntimeError("Recorded check must not invent physical transitions")

def recorded_frame(spec, state, goal):
    frame = {name: spec.field(state, name).clone() for name in spec.base_fields}
    for name, shape in spec.metadata["state_shapes"].items():
        if name in frame:
            frame[name] = frame[name].reshape(len(state), *shape)
    frame["target_pose"] = goal[:, :7]
    frame["target_speed"] = goal[:, 7:]
    return frame

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--live", choices=["disect", "blademaster"])
    p.add_argument("--seconds", type=int, default=900)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "mode": a.live or "recorded_cpu", "checks": {},
        "evidence": str(a.evidence), "source_commit": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git","status","--porcelain"], text=True),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "interface/update verification; eight planner candidates, not performance evaluation"}
    def timeout(*_): raise TimeoutError("Baseline adapter budget")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(a.seconds)
    env = vec = None
    try:
        torch.set_num_threads(2); torch.manual_seed(0)
        metadata = json.loads((a.evidence / "metadata.json").read_text())
        learner = torch.load(a.evidence / "learner.pt", map_location="cpu", weights_only=False)
        checkpoint = a.output / "paired_wm.pt"
        torch.save({"metadata": metadata, "learner": learner}, checkpoint)
        spec = CuttingWorldModel(metadata).spec
        data = torch.load(a.evidence / "replay.pt", map_location="cpu", weights_only=False)["buffer"]["tensors"]
        if a.live:
            cfg = json.loads((a.evidence / "config.json").read_text())
            if a.live == "disect":
                from cutting_world_model.multi_env import MultiConditionEnvironment
                env = MultiConditionEnvironment.from_config(ROOT / "backends/mujoco", cfg)
            else:
                from cutting_world_model.blademaster_backend import BladeMasterEnvironment
                env = BladeMasterEnvironment(ROOT / "backends/mujoco", cfg)
        else:
            # One actual recorded state per object, not fabricated geometry.
            indexes = torch.tensor([0, 3, 6])
            state, goal = data["s"][indexes, 10], data["xi"][indexes, 10]
            env = SimpleNamespace(worlds=3, device=torch.device("cpu"), metadata=metadata,
                current=recorded_frame(spec, state, goal))
        cfg = MPPIConfig(horizon=8, samples=8, discount=metadata["free_cut_task"]["gamma"])
        for name, weights in (("wm_mppi", checkpoint), ("analytic_mpc", None)):
            begin = time.monotonic()
            planner = SharedMPPIPolicy(env, cfg, weights,
                controller_parameters(metadata, ROOT / "backends/mujoco"))
            actions = planner.act(env.current)
            assert actions.shape == (env.worlds, 6) and torch.isfinite(actions).all()
            assert (actions.abs() <= 1).all()
            steps = 0
            if a.live:
                frame, reward, done, truncated, _ = env.step(actions)
                assert torch.isfinite(reward).all() and not truncated.any()
                planner.observe(frame)
                if done.any():
                    frame = env.reset(done)
                    planner.reset(frame, done)
                actions = planner.act(frame)
                frame, reward, done, truncated, _ = env.step(actions)
                assert torch.isfinite(reward).all() and not truncated.any()
                if done.any(): env.reset(done)
                steps = 2
            assert planner.unchanged()
            report["checks"][name] = {"passed": True, "physical_steps": steps,
                "seconds": time.monotonic()-begin, "planning": planner.plans}
            print(name, "passed", flush=True)
        if a.live:
            vec = CuttingVecEnv(env)
            model = TransitionSAC("MlpPolicy", vec, device=env.device, learning_starts=0,
                buffer_size=1024, batch_size=8, train_freq=1, gradient_steps=1,
                transitions_per_gradient=env.worlds, gamma=metadata["free_cut_task"]["gamma"],
                policy_kwargs={"net_arch": [256, 256]}, seed=0, verbose=0)
            model.learn(total_timesteps=env.worlds*8)
        else:
            model = TransitionSAC("MlpPolicy", SpacesOnly(), device="cpu", learning_starts=0,
                buffer_size=1024, batch_size=8, gamma=metadata["free_cut_task"]["gamma"],
                policy_kwargs={"net_arch": [256, 256]}, seed=0, verbose=0)
            model.set_logger(configure(str(a.output / "sac_log"), []))
            for world in (0, 3, 6):
                for t in range(10, 30):
                    assert data["valid"][world,t] and data["valid"][world,t+1]
                    assert not data["terminal"][world,t] and data["ep_id"][world,t] == data["ep_id"][world,t+1]
                    s, ns = data["s"][world,t:t+1], data["s"][world,t+1:t+2]
                    h = spec.field(s,"observation")[:,None].repeat(1,10,1)
                    nh = spec.field(ns,"observation")[:,None].repeat(1,10,1)
                    # Actual ten-frame policy history is stored in packed state.
                    for j in range(10):
                        h[:,9-j] = spec.field(data["s"][world,t-j:t-j+1],"observation")
                        nh[:,9-j] = spec.field(data["s"][world,t+1-j:t+2-j],"observation")
                    obs = torch.cat((h.flatten(1),data["xi"][world,t:t+1]),-1).numpy()
                    nxt = torch.cat((nh.flatten(1),data["xi"][world,t+1:t+2]),-1).numpy()
                    model.replay_buffer.add(obs,nxt,data["a_exec"][world,t:t+1].numpy(),
                        data["r"][world,t:t+1].numpy(),data["done"][world,t+1:t+2].numpy(),[{}])
            model.train(gradient_steps=2, batch_size=8)
        assert model._n_updates > 0 and all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
        model.save(a.output / "sac")
        restored = TransitionSAC.load(a.output / "sac", device="cpu")
        assert all(torch.equal(v.cpu(),restored.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
        report["checks"]["sac"] = {"passed": True, "updates": model._n_updates,
            "replay_rows": model.replay_buffer.size(), "physical_steps": 8 if a.live else 0,
            "save_load_weights_exact": True}
        report["status"] = "passed"
        print("sac passed", flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error)); raise
    finally:
        signal.alarm(0)
        try:
            if vec is not None: vec.close()
            elif a.live and env is not None: env.close()
        except BaseException as error:
            report.update(status="failed", cleanup_error=repr(error))
            raise
        finally:
            report["wall_seconds"] = time.monotonic()-started
            (a.output / "report.json").write_text(json.dumps(report, indent=2))
        report["wall_seconds"] = time.monotonic()-started
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
if __name__ == "__main__": main()
