"""Bounded full-reference probe for shared knife geometry, rigid mount and board stop."""

import argparse, json, signal, time
from pathlib import Path
import torch
import numpy as np
from cutting_gpu_backend.coupling import compose_pose, inverse_pose
from cutting_gpu_backend.controller.fdcc import quat_apply
from cutting_world_model.shared_knife import (
    canonical_mount,
    vertices_disect,
    vertices_blade,
    REVISION,
)


def run(a):
    def expire(*_):
        raise TimeoutError("Shared-tool probe budget")

    signal.signal(signal.SIGALRM, expire)
    signal.alarm(a.seconds)
    if a.output.exists():
        raise FileExistsError(a.output)
    cfg = json.loads(a.config.read_text())
    env = None
    report = {"status": "running", "config": cfg}
    started = time.monotonic()
    torch.set_num_threads(2)
    try:
        if a.backend == "disect":
            from cutting_world_model.multi_env import MultiConditionEnvironment

            env = MultiConditionEnvironment.from_config("backends/mujoco", cfg)
            physical = [m.env.physics for m in env.members]
        else:
            from cutting_world_model.blademaster_backend import BladeMasterEnvironment

            env = BladeMasterEnvironment("backends/mujoco", cfg)
            physical = [env.physics]
        n = env.worlds
        first = {}
        rows = []
        worst = 0.0
        min_gap = 1.0
        for step in range(a.steps):
            action = torch.zeros((n, 6), device=env.device, dtype=torch.double)
            frame, reward, done, truncated, info = env.step(action, reference=True)
            for physics in physical:
                ee = physics.frames[physics.chain.tip]
                if a.backend == "disect":
                    mount = physics.rigid_tool_mount
                    actual = compose_pose(
                        physics.coupling.base_from_disect,
                        torch.cat(
                            (physics.material.position, physics.material.rotation), -1
                        ).double(),
                    )
                    vertices = ee.new_tensor(vertices_disect())
                else:
                    mount = physics.ee_from_blade
                    physical_pose = ee.new_tensor(
                        np.stack(
                            [
                                np.r_[
                                    s.tool_pos.numpy().reshape(-1, 3)[0],
                                    s.tool_rot.numpy().reshape(-1, 4)[0],
                                ]
                                for s in physics.solvers
                            ]
                        )
                    )
                    actual = compose_pose(physics.base_from_material, physical_pose)
                    vertices = ee.new_tensor(vertices_blade())
                expected = compose_pose(ee, mount.expand(len(ee), -1))
                av = (
                    quat_apply(
                        actual[:, None, 3:].expand(-1, len(vertices), -1),
                        vertices.expand(len(ee), -1, -1),
                    )
                    + actual[:, None, :3]
                )
                ev = (
                    quat_apply(
                        expected[:, None, 3:].expand(-1, len(vertices), -1),
                        vertices.expand(len(ee), -1, -1),
                    )
                    + expected[:, None, :3]
                )
                error = float((av - ev).norm(dim=-1).max())
                worst = max(worst, error)
                assert error < 5e-6, ("Rigid mount mismatch", error)
            gap = frame["knife_board_gap"].flatten()
            min_gap = min(min_gap, float(gap.min()))
            assert float(gap.min()) > -2e-6, ("Board penetration", gap.tolist())
            assert torch.isfinite(frame["peak_wrench"]).all()
            rows.append(
                dict(
                    step=step + 1,
                    gap_m=gap.tolist(),
                    progress=frame["cut_progress"].flatten().tolist(),
                    reward=reward.tolist(),
                    done=done.tolist(),
                    peak_wrench=frame["peak_wrench"].tolist(),
                )
            )
            for i in done.nonzero().flatten().tolist():
                if i not in first:
                    first[i] = dict(
                        step=step + 1,
                        info={
                            k: v[i].tolist()
                            for k, v in info.items()
                            if torch.is_tensor(v) and v.ndim and len(v) == n
                        },
                    )
            print(
                a.backend,
                step + 1,
                "gap",
                gap.tolist(),
                "first ended",
                len(first),
                flush=True,
            )
            if len(first) == n:
                break
            if done.any():
                env.reset(done)
        report.update(
            status="passed",
            tool_revision=REVISION,
            max_mount_vertex_error_m=worst,
            minimum_gap_m=min_gap,
            first_episodes=first,
            steps=rows,
            limitation="Reference-control engineering probe; not policy performance or material-separation validation",
        )
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        if env is not None:
            env.close()
        report["wall_seconds"] = time.monotonic() - started
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2))
        signal.alarm(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["disect", "blademaster"], required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=90)
    p.add_argument("--seconds", type=int, default=1200)
    run(p.parse_args())
