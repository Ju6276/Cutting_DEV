"""Publication-ready descriptive plots from verified frozen policy traces."""

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2274A5", "#E0A12D", "#B34252"]


def save(fig, path):
    fig.savefig(str(path) + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def material_timelines(d, cases, groups, out):
    """Same geometry, low/high E cases; all other randomized conditions disclosed."""
    fig, axs = plt.subplots(4, 3, figsize=(16, 11), sharex="col")
    onset = dict(zip(d["contact_cases"].tolist(), d["contact_steps"].tolist()))
    selected = []
    for col, group in enumerate(groups):
        ids = [c["case"] for c in cases if c["group"] == group]
        ids.sort(key=lambda i: (cases[i]["condition"]["youngs_modulus"], i))
        pair = [ids[0], ids[-1]]
        selected.append(pair)
        for i, color in zip(pair, (COLORS[0], COLORS[2])):
            c = cases[i]["condition"]
            ix = d["case"] == i
            time = (d["step"][ix] - onset.get(i, 0)) * 0.05
            description = (
                f"case {i}: E x{c['youngs_modulus']:.2f}, spring x{c['cut_spring_stiffness']:.2f}, "
                f"density x{c['density']:.2f}\n"
                f"command {c['target_speed_mps']*1000:.1f} mm/s, start lift {c['start_lift_m']*1000:.1f} mm"
            )
            axs[0, col].plot(time, d["force"][ix, 2], color=color, label=description)
            axs[1, col].plot(time, d["control"][ix, 2], color=color)
            axs[2, col].plot(time, d["speed"][ix] * 1000, color=color)
            axs[2, col].plot(time, d["goal"][ix, -1] * 1000, color=color, ls="--")
            axs[3, col].plot(time, d["control"][ix, 4] * 1000, color=color)
        axs[0, col].set_title(group.replace("free_", ""), pad=65)
        axs[0, col].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.02))
        for row in range(4):
            axs[row, col].axvline(0, color="grey", ls=":")
            axs[row, col].grid(alpha=0.2)
        axs[-1, col].set_xlabel("Time from force-contact proxy (s)")
    for row, text in enumerate(
        (
            "Sensor Fz (N)",
            "Vertical P gain",
            "Feed speed (mm/s)",
            "X target increment (mm/step)",
        )
    ):
        axs[row, 0].set_ylabel(text)
    fig.suptitle(
        "Main frozen policy: low/high Young modulus within each geometry\n"
        "Observed randomized cases; other parameters differ, so this is not an isolated stiffness effect"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out / "material_force_action_timeline")
    return selected


def run(root):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    allstats = {}
    for backend, label in [
        ("main", "Main policy / DiSECt: 1109 actor updates"),
        ("blademaster", "BladeMaster policy: 866 actor updates"),
    ]:
        p = root / backend
        d = np.load(p / "analysis.npz")
        cases = json.loads((p / "cases.json").read_text())
        episodes = {r["case"]: r for r in json.loads((p / "episodes.json").read_text())}
        groups = list(dict.fromkeys(c["group"] for c in cases))
        out = p / "figures"
        out.mkdir(exist_ok=True)
        selected = []
        for g in groups:
            ids = [c["case"] for c in cases if c["group"] == g]
            ids.sort(key=lambda i: (episodes[i]["return"], i))
            selected.append(ids[(len(ids) - 1) // 2])
        fig, axs = plt.subplots(6, 3, figsize=(15, 14), sharex="col")
        for col, i in enumerate(selected):
            ix = d["case"] == i
            t = d["step"][ix] * 0.05
            title = cases[i]["group"].replace("free_", "") + " | case " + str(i)
            axs[0, col].set_title(title + " | " + episodes[i]["ending"])
            for k, n in enumerate(["Fx", "Fy", "Fz"]):
                axs[0, col].plot(t, d["force"][ix, k], label=n, lw=1)
            for k, n in enumerate(["Mx", "My", "Mz"]):
                axs[1, col].plot(t, d["torque"][ix, k], label=n, lw=1)
            for k in range(6):
                axs[2, col].plot(t, d["action"][ix, k], label="a" + str(k), lw=1)
            axs[3, col].plot(t, d["speed"][ix] * 1000, label="Actual feed")
            axs[3, col].plot(t, d["goal"][ix, -1] * 1000, "k--", label="Command")
            axs[4, col].plot(t, d["board_gap"][ix] * 1000, label="Blade / board gap")
            axs[4, col].plot(t, d["x_error"][ix] * 1000, label="Lateral target error")
            hit = np.flatnonzero(d["board_gap"][ix] <= 1e-5)
            if len(hit):
                for row in range(6):
                    axs[row, col].axvline(t[hit[0]], color="#a84632", ls="--", lw=1)

            on = dict(zip(d["contact_cases"].tolist(), d["contact_steps"].tolist()))
            for row in range(6):
                axs[row, col].grid(alpha=0.2)
                if i in on:
                    axs[row, col].axvline(on[i] * 0.05, color="grey", ls=":", lw=1)
            axs[5, col].plot(
                t, d["ratio"][ix], color="#a84632", label="Peak six-axis utilization"
            )
            axs[5, col].axhline(1, color="black", ls="--", label="Task limit")
            axs[5, col].set_ylim(0, 1.08)
            axs[5, col].set_xlabel("Time (s)")
        for row, y in enumerate(
            [
                "Sensor force (N)",
                "Sensor moment (N m)",
                "Clipped policy output [-1, 1]",
                "Feed speed (mm/s)",
                "Geometry (mm)",
                "Force / moment limit utilization",
            ]
        ):
            axs[row, 0].set_ylabel(y)
            axs[row, 2].legend(fontsize=8, ncol=3)
        fig.suptitle(
            label
            + "\nRepresentative = lower median return within each object; dotted: contact proxy; red dashed: board interlock proximity (not learned braking)",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save(fig, out / "force_action_timeline")
        fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharex="row", sharey="row")
        phases = ["reset", "pre-proxy", "contact+3", "contact+5"]
        for col, phase in enumerate(phases):
            for j, g in enumerate(groups):
                mask = np.array(
                    [cases[int(i)]["group"] == g for i in d["phase_case"]]
                ) & (d["phase_name"] == phase)
                for row, key in enumerate(["pca", "goal_pca"]):
                    axs[row, col].scatter(
                        d[key][mask, 0],
                        d[key][mask, 1],
                        s=24,
                        color=COLORS[j],
                        label=g.replace("free_", ""),
                        alpha=0.7,
                    )
            axs[0, col].set_title(phase)
            axs[1, col].set_xlabel("PC1")
            for row in range(2):
                axs[row, col].grid(alpha=0.2)
        axs[0, 0].set_ylabel("History + goal decoder / PC2")
        axs[1, 0].set_ylabel("Goal-only encoder / PC2")
        for ax in axs.flat:
            ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 3))
        axs[0, 2].legend(fontsize=8)
        fig.suptitle(
            label
            + "\nOne point per episode and phase; each row uses a shared PCA basis across phases",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save(fig, out / "features_pca_controls")
        fig, axs = plt.subplots(5, 3, figsize=(13, 10), sharex="col")
        names = [
            "X stiffness (N/m)",
            "Pitch stiffness (N m/rad)",
            "Vertical P gain",
            "Pitch P gain",
            "X target increment (mm/step)",
        ]
        for col, i in enumerate(selected):
            ix = d["case"] == i
            t = d["step"][ix] * 0.05
            for k in range(5):
                scale = 1000 if k == 4 else 1
                axs[k, col].plot(t, d["control"][ix, k] * scale, color=COLORS[col])
                axs[k, col].grid(alpha=0.2)
                if col == 0:
                    axs[k, col].set_ylabel(names[k])
            axs[0, col].set_title(cases[i]["group"].replace("free_", ""))
            axs[-1, col].set_xlabel("Time (s)")
        fig.suptitle(
            label
            + "\nMapped FDCC commands; pitch target increment is disabled (max_twist[1] = 0)",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save(fig, out / "mapped_controller_commands")
        stats = []
        fig, ax = plt.subplots(figsize=(8, 4))
        for j, g in enumerate(groups):
            ids = [c["case"] for c in cases if c["group"] == g]
            per = []
            sats = []
            for i in ids:
                ix = d["case"] == i
                per.append(np.abs(d["action"][ix] - d["ablated_action"][ix]).mean(0))
                sats.append((np.abs(d["action"][ix]) >= 1 - 1e-7).mean())
            per = np.array(per)
            ax.bar(
                np.arange(6) + (j - 1) * 0.25,
                per.mean(0),
                width=0.24,
                color=COLORS[j],
                label=g.replace("free_", ""),
            )
            stats.append(
                dict(
                    group=g,
                    cases=len(ids),
                    history_ablation_mean_action_change=float(per.mean()),
                    saturated_action_fraction=float(np.mean(sats)),
                )
            )
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(["a" + str(k) for k in range(6)])
        ax.set_ylabel("Mean absolute clipped-output change")
        ax.set_title(
            "Replace older wrench history with current wrench\nOpen-loop sensitivity; not a performance ablation"
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        save(fig, out / "force_history_sensitivity")
        allstats[backend] = dict(representative_cases=selected, groups=stats)
        if backend == "main":
            allstats[backend]["material_extreme_cases"] = material_timelines(
                d, cases, groups, out
            )
            fig, axs = plt.subplots(1, 3, figsize=(13, 4))
            for ax, g in zip(axs, groups):
                mask = np.array(
                    [cases[int(i)]["group"] == g for i in d["phase_case"]]
                ) & (d["phase_name"] == "contact+5")
                stiffness = np.array(
                    [
                        cases[int(i)]["condition"]["youngs_modulus"]
                        for i in d["phase_case"]
                    ]
                )
                im = ax.scatter(
                    d["pca"][mask, 0],
                    d["pca"][mask, 1],
                    c=stiffness[mask],
                    cmap="viridis",
                    vmin=0.5,
                    vmax=2,
                    s=45,
                )
                ax.set_title(g.replace("free_", ""))
                ax.set_xlabel("PC1")
                ax.grid(alpha=0.2)
            axs[0].set_ylabel("Decoder PC2")
            fig.colorbar(im, ax=axs, label="Young modulus multiplier", shrink=0.8)
            fig.suptitle(
                "Contact + 5: material colors are descriptive; density, cut stiffness, speed and start also vary"
            )
            save(fig, out / "material_feature_projection")
    (root / "plot_summary.json").write_text(json.dumps(allstats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root)
