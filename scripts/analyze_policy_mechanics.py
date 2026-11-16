"""Frozen trace mechanics and decoder features; no physical simulation or training."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.task_heads import SliceItHeads
from cutting_world_model.pose_cut import gap_from_pose


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mapped_controls(action, max_twist):
    """Actual FDCC gains and per-control-step target increments; no accumulated state."""
    from cutting_gpu_backend.controller.fdcc import SlicingActionMapper

    a = action.double()
    mapper = SlicingActionMapper(max_twist=max_twist)
    mapper.reset(len(a), a.device, a.dtype)
    pose = a.new_zeros((len(a), 7))
    pose[:, 6] = 1
    out = mapper(a, pose, pose)
    return torch.stack(
        (
            out["stiffness"][:, 0],
            out["stiffness"][:, 4],
            out["p_gains"][:, 2],
            out["p_gains"][:, 4],
            mapper.added_motion_command[:, 0],
            mapper.added_motion_command[:, 4],
        ),
        -1,
    )


def run(evaluation, checkpoint, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((evaluation / "manifest.json").read_text())
    cases = json.loads((evaluation / "cases.json").read_text())
    endings = [
        json.loads(x) for x in (evaluation / "episodes.jsonl").read_text().splitlines()
    ]
    assert (
        digest(checkpoint) == manifest["checkpoint_sha256"]
    ), "Checkpoint identity differs from recorded policy"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = saved["metadata"]
    policy = SliceItPolicy(
        meta["goal_dim"], meta["state_shapes"]["observation"][0]
    ).eval()
    policy.load_state_dict(saved["learner"]["policy"])
    del saved
    heads = SliceItHeads(meta["environment"], meta["controller"])
    traces = torch.load(
        evaluation / "trajectories.pt", map_location="cpu", weights_only=False
    )
    history = traces[0]["before"]["observation"].float()[None].repeat(10, 1, 1)
    histories = []
    goals = []
    indices = []
    clean = []
    times = []
    raw = {}
    keys = (
        "force",
        "torque",
        "position",
        "speed",
        "peak_force",
        "ratio",
        "progress",
        "board_gap",
        "x_error",
        "reward",
        "separated",
    )
    raw = {k: [] for k in keys}
    for t, row in enumerate(traces):
        obs = row["before"]["observation"].float()
        if t:
            history = torch.cat((history[1:], obs[None]), 0)
        active = row["active"].bool()
        ids = active.nonzero().flatten()
        histories.append(history[:, ids])
        goals.append(row["goal"][ids].float())
        indices.extend(ids.tolist())
        times.extend([t] * len(ids))
        clean.append(row["clean_action"][ids].float())
        w = obs[:, heads.force_slice].reshape(
            len(obs), len(heads.force_dims), heads.history
        )[:, :, -1] * torch.tensor(heads.force_limits)
        raw["force"].append(w[ids, :3])
        raw["torque"].append(w[ids, 3:])
        pose = row["before"]["ee_pose"]
        raw["position"].append(pose[ids, :3])
        raw["speed"].append(heads.cutting_speed(obs)[ids])
        raw["peak_force"].append(row["info"]["peak_force"][ids])
        raw["ratio"].append(row["info"]["safety_margin"][ids])
        raw["progress"].append(row["before"]["cut_progress"][ids, 0])
        raw["board_gap"].append(gap_from_pose(pose, meta["board_pose_geometry"])[ids])
        raw["x_error"].append((row["goal"][:, 0] - pose[:, 0])[ids])
        raw["reward"].append(row["reward"][ids])
        separated = row["info"].get("separated", torch.full((len(obs),), float("nan")))
        raw["separated"].append(separated.float()[ids])
    H = torch.cat(histories, 1)
    G = torch.cat(goals)
    A = torch.cat(clean)
    case = np.array(indices)
    step = np.array(times)
    features = []
    goalf = []
    actions = []
    ablated = []

    def latent_hook(module, args):
        features.append(args[0].detach().numpy().copy())

    hook = policy.decoder.net[-1].register_forward_pre_hook(latent_hook)
    with torch.no_grad():
        for start in range(0, len(G), 128):
            h = H[:, start : start + 128]
            g = G[start : start + 128]
            actions.append(policy(h, g))
            goalf.append(torch.tanh(policy.encoder(g)))
    hook.remove()
    with torch.no_grad():
        for start in range(0, len(G), 128):
            h = H[:, start : start + 128].clone()
            g = G[start : start + 128]
            h[:-1, :, heads.force_slice] = h[-1:, :, heads.force_slice]
            ablated.append(policy(h, g))
    prediction = torch.cat(actions)
    ablated = torch.cat(ablated)
    raw_ablation = ablated.numpy().copy()
    error = (prediction - A).abs().max().item()
    if error > 1e-3:
        raise AssertionError(f"History reconstruction / policy mismatch: {error}")
    latent = np.concatenate(features)
    goal_latent = torch.cat(goalf).numpy()
    clipped = prediction.clamp(-1, 1).numpy()
    ablated = ablated.clamp(-1, 1).numpy()
    values = {k: torch.cat(v).numpy() for k, v in raw.items()}
    # Each plotted feature sample is one episode at a named phase, never all
    # highly correlated frames treated as independent materials.
    phases = []
    onsets = {}
    missing = []
    for i in range(len(cases)):
        selected = np.flatnonzero(case == i)
        peaks = values["peak_force"][selected]
        hits = np.flatnonzero(peaks > 0.5)
        if not len(hits):
            missing.append(i)
            continue
        onset = min(
            int(hits[0]) + 1, len(selected) - 1
        )  # reaction available at next control input
        onsets[i] = int(step[selected[onset]])
        phases.append((i, "reset", int(selected[0])))
        for phase, offset in [("pre-proxy", -1), ("contact+3", 3), ("contact+5", 5)]:
            local = onset + offset
            if 0 <= local < len(selected):
                phases.append((i, phase, int(selected[local])))
    ids = np.array([x[2] for x in phases])
    X = latent[ids]
    mean = X.mean(0)
    U, S, V = np.linalg.svd(X - mean, full_matrices=False)
    coords = (X - mean) @ V[:2].T
    variance = (S[:2] ** 2 / (S**2).sum()).tolist()
    # Input/goal controls make pre-existing geometric clustering visible.
    goalX = goal_latent[ids].astype(np.float64)
    _, gs, gv = np.linalg.svd(goalX - goalX.mean(0), full_matrices=False)
    goal_coords = (goalX - goalX.mean(0)) @ gv[:2].T
    ending_by_case = {row["case"]: row for row in endings}
    assert len(ending_by_case) == len(cases)
    control = mapped_controls(
        torch.from_numpy(clipped), meta["controller"]["max_twist"]
    ).numpy()
    ablated_control = mapped_controls(
        torch.from_numpy(ablated), meta["controller"]["max_twist"]
    ).numpy()
    stats = []
    for i, c in enumerate(cases):
        pick = np.flatnonzero(case == i)
        delta = clipped[pick] - ablated[pick]
        stats.append(
            dict(
                case=i,
                group=c["group"],
                ending=ending_by_case[i]["ending"],
                force_history_ablation_action_mae=float(np.abs(delta).mean()),
                action_mae_by_dimension=np.abs(delta).mean(0).tolist(),
                raw_action_change=float(
                    np.abs(prediction.numpy()[pick] - raw_ablation[pick]).mean()
                ),
                saturated_fraction=float((np.abs(clipped[pick]) >= 1 - 1e-7).mean()),
                observed_steps=len(pick),
            )
        )
    summary = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        source=str(evaluation),
        identity=manifest["checkpoint_identity"],
        cases=len(cases),
        observed_steps=len(G),
        recorded_separation_available=bool(np.isfinite(values["separated"]).any()),
        recorded_separation_episodes=(
            int(
                sum(
                    np.any(values["separated"][case == i] > 0.5)
                    for i in range(len(cases))
                )
            )
            if np.isfinite(values["separated"]).any()
            else None
        ),
        observation_dim=policy.observation_dim,
        history=10,
        decoder_feature_dim=latent.shape[1],
        reconstructed_action_max_error=error,
        force_history_ablation="Replace older nine wrench blocks with the current wrench block; keep all other history and goal unchanged. Open-loop network sensitivity.",
        contact_onset="First interval material force norm >0.5 N; next control input is contact index 0.",
        missing_contact_cases=missing,
        pca_variance_fraction=variance,
        source_task=meta["free_cut_task"]["version"],
        backend=meta["physical_backend"],
        case_metrics=stats,
        max_twist=meta["controller"]["max_twist"],
        control_channels=[
            "Kx_N_per_m",
            "Kpitch_Nm_per_rad",
            "Pz",
            "Ppitch",
            "dx_target_m_per_step",
            "dpitch_target_rad_per_step",
        ],
        selection="All recorded first episodes; feature phases reset/pre-proxy/contact+3/contact+5.",
        paired_material_scan=False,
    )
    np.savez_compressed(
        output / "analysis.npz",
        case=case,
        step=step,
        action=clipped,
        raw_action=prediction.numpy(),
        ablated_raw_action=raw_ablation,
        ablated_action=ablated,
        goal=G.numpy(),
        control=control,
        ablated_control=ablated_control,
        latent=latent,
        goal_latent=goal_latent,
        phase_case=np.array([x[0] for x in phases]),
        phase_name=np.array([x[1] for x in phases]),
        phase_index=ids,
        pca=coords,
        goal_pca=goal_coords,
        contact_cases=np.array(list(onsets)),
        contact_steps=np.array(list(onsets.values())),
        **values,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "cases.json").write_text(json.dumps(cases, indent=2))
    (output / "episodes.json").write_text(json.dumps(endings, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "cases",
                    "observed_steps",
                    "reconstructed_action_max_error",
                    "pca_variance_fraction",
                )
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    run(a.evaluation, a.checkpoint, a.output)


if __name__ == "__main__":
    main()
