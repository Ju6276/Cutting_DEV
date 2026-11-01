#!/usr/bin/env python3
"""Fit coupled dynamics using SuperMimic's 8-step supervised objective."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.provenance import git_revision
from cutting_world_model.training import EpisodeWindows, CoupledWorldModel, evaluate, evaluate_episode_rollouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--wall-seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()
    train = EpisodeWindows(args.train)
    validation = EpisodeWindows(args.validation)
    if set(train.paths) & set(validation.paths):
        raise ValueError("Training and validation must use separate episodes")
    if train.layout != validation.layout:
        raise ValueError("Training and validation layouts differ")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(2)
    device = "cuda"
    statistics = train.statistics()
    model = CoupledWorldModel(args.upstream_root, *statistics).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    manifest = {"git": git_revision(Path(__file__).resolve().parents[1]),
                "upstream_git": git_revision(args.upstream_root), "arguments": vars(args),
                "command": sys.argv, "window": 8, "batch_size": 2048,
                "batches_per_update": 10, "learning_rate": 2e-4, "gradient_clip_norm": 1.,
                "train_episodes": [str(path) for path in train.paths],
                "validation_episodes": [str(path) for path in validation.paths],
                "layout": train.layout, "parameters": sum(p.numel() for p in model.parameters())}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    initial = evaluate(model, validation, device)
    (output / "initial_validation.json").write_text(json.dumps(initial, indent=2) + "\n")
    completed = 0
    state = "running"
    try:
        for update in range(args.updates):
            losses = []
            for _ in range(10):
                if time.monotonic() - started > args.wall_seconds:
                    raise TimeoutError("World-model training budget exhausted")
                batch = train.batch(rng.integers(len(train.indices), size=2048), device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.window_loss(*batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                completed += 1
                losses.append(float(loss.detach()))
                progress = {"update": update + 1, "optimizer_steps": completed,
                            "loss": losses[-1], "grad_norm_before_clip": float(grad_norm),
                            "wall_seconds": time.monotonic() - started}
                (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
        state = "completed"
    finally:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "layout": train.layout, "optimizer_steps": completed, "manifest": manifest},
                   output / "checkpoint.pt")
        (output / "status.json").write_text(json.dumps({"state": state if state == "completed" else "interrupted",
            "optimizer_steps": completed, "wall_seconds": time.monotonic() - started}, indent=2) + "\n")
    result = {"validation": evaluate(model, validation, device),
              "validation_episode_rollouts": evaluate_episode_rollouts(model, validation, device),
              "training": evaluate(model, train, device), "optimizer_steps": completed,
              "wall_seconds": time.monotonic() - started,
              "peak_torch_gpu_mib": torch.cuda.max_memory_allocated() / 2**20}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
