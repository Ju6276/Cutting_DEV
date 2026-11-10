#!/usr/bin/env python3
"""Rebuild the archived 2026-11-10 evidence package; not the current paper evaluation.

Python >=3.8, numpy and matplotlib. Outputs are regenerated from immutable input hashes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = ("success", "collision", "outside", "truncated")
METRICS = ("max_material_force_n", "mean_material_force_n",
           "material_impulse_ns", "safety_margin")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_name(manifest):
    contract = manifest["contract"]
    return contract["name"] if isinstance(contract, dict) else contract


def load_cases(directory, inputs):
    def read(name):
        path = directory / name
        inputs[str(path.resolve())] = digest(path)
        return path.read_text()
    manifest = json.loads(read("manifest.json"))
    summary = json.loads(read("summary.json"))
    rows = [json.loads(line) for line in read("cases.jsonl").splitlines() if line.strip()]
    ids = [c["case"] for c in manifest["cases"]]
    if summary["unfinished_cases"] or len(rows) != len(ids):
        raise ValueError("Incomplete cases: " + str(directory))
    if sorted(r["id"] for r in rows) != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("Case IDs differ or repeat: " + str(directory))
    if manifest.get("perturbation_point") != "executed_action":
        raise ValueError("This evidence package requires executed_action perturbations")
    for row in rows:
        if row["outcome"] not in OUTCOMES:
            raise ValueError("Unknown outcome")
        for key in METRICS + ("steps", "simulation_seconds"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Invalid metric: " + key)
        if not math.isclose(row["mean_material_force_n"] * row["simulation_seconds"],
                            row["material_impulse_ns"], rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Impulse / duration / mean force mismatch")
    expected = summary["summary"]
    for name in OUTCOMES:
        actual = sum(r["outcome"] == name for r in rows) / len(rows)
        if not math.isclose(actual, expected[name + "_rate"], abs_tol=1e-10):
            raise ValueError("Summary outcome does not match cases")
    for key in METRICS:
        actual = sum(r[key] for r in rows) / len(rows)
        if not math.isclose(actual, expected[key + "_mean"], rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("Summary metric does not match cases: " + key)
    return manifest, sorted(rows, key=lambda r: r["id"]), expected


def require_same_cases(manifests):
    first = manifests[0]
    for other in manifests[1:]:
        if (other["cases"] != first["cases"]
                or contract_name(other) != contract_name(first)
                or other.get("perturbation_point") != first.get("perturbation_point")):
            raise ValueError("Mixed task, perturbation rule or case configuration")


def classify_episode(episode):
    if not episode["completed"]:
        return "censored"
    for field, outcome in (("success", "success"), ("collision", "collision"),
                           ("outside", "outside"), ("truncated", "truncated")):
        if episode.get(field):
            return outcome
    raise ValueError("Completed episode lacks an outcome")


def verify_pair(report, a, b):
    if report.get("perturbation_point") != "executed_action":
        raise ValueError("Mixed comparison rule")
    both = [i for i in range(len(a)) if a[i]["outcome"] == b[i]["outcome"] == "success"]
    if report["a_minus_b"]["both_success"]["cases"] != len(both):
        raise ValueError("Paired-success count mismatch")
    for group, ids in (("all_cases", range(len(a))), ("both_success", both)):
        for metric, result in report["a_minus_b"][group].items():
            if metric == "cases":
                continue
            def value(row):
                return float(row["outcome"] == metric) if metric in OUTCOMES else row[metric]
            diffs = [value(a[i]) - value(b[i]) for i in ids]
            if not diffs:
                continue
            if not math.isclose(sum(diffs) / len(diffs), result["mean_difference"],
                                rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError("Saved paired statistic does not match cases: " + metric)
            lo, hi = result["interval_95"]
            if not all(math.isfinite(x) for x in (lo, hi)) or lo > hi:
                raise ValueError("Invalid interval")


def csv_file(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)


def build(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cfg = json.loads(args.config.read_text())
    inputs = {str(args.config.resolve()): digest(args.config)}
    resolve = lambda s: Path(s.format(runs=args.runs_root, legacy=args.legacy_root))
    def read(path):
        inputs[str(path.resolve())] = digest(path)
        return json.loads(path.read_text())
    loaded = {}
    for item in cfg["perturbed"]:
        loaded[item["id"]] = load_cases(resolve(item["directory"]), inputs)
    require_same_cases([v[0] for v in loaded.values()])
    pairs = {}
    for item in cfg["comparisons"]:
        report = read(resolve(item["file"]))
        verify_pair(report, loaded[item["a"]][1], loaded[item["b"]][1])
        pairs[item["b"]] = report
    fixed, provenance = [], []
    for item in cfg["unperturbed"]:
        directory = resolve(item["directory"])
        report, manifest = read(directory / "report.json"), read(directory / "manifest.json")
        if contract_name(manifest) != contract_name(next(iter(loaded.values()))[0]):
            raise ValueError("Unperturbed task differs: " + item["id"])
        if report.get("optimizer_updates") != 0 or report.get("policy_unchanged") is not True:
            raise ValueError("Evaluation weights not verified frozen: " + item["id"])
        steps = sum(e["steps"] for e in report["episodes"])
        provenance.append({**item, "git": manifest["git"],
                           "checkpoint": manifest.get("checkpoint"),
                           "checkpoint_identity": manifest.get("checkpoint_identity"),
                           "planning_or_inference_ms_per_step": report["inference_seconds"] / steps * 1000,
                           "physics_ms_per_step": report["physics_seconds"] / steps * 1000,
                           "report": str((directory / "report.json").resolve()),
                           "recorded_episodes": len(report["episodes"])})
        for episode in report["episodes"]:
            fixed.append({"controller": item["id"], "label": item["label"],
                          "episode": episode["episode"], "outcome": classify_episode(episode),
                          "steps": episode["steps"], "simulation_seconds": episode["simulation_seconds"],
                          "material_peak_force_n": episode["material_peak_force_n"],
                          "material_impulse_ns": episode["material_impulse_ns"],
                          "sensor_limit_utilization": episode["safety_margin"],
                          "released_springs": episode["released_springs"],
                          "total_cut_springs": episode["total_cut_springs"]})

    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    palette = ["#526b91", "#147d70", "#c06a21"]
    items = cfg["perturbed"]
    captions = []
    def save(fig, name, caption):
        fig.savefig(args.output / (name + ".png"), dpi=170, bbox_inches="tight")
        fig.savefig(args.output / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
        captions.append((name, caption))

    # 1. Preserve all endings. Material and sensor quantities have separate axes.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    colors = ["#2c8b72", "#b64c4c", "#d6a64e", "#8d8d8d"]
    left = np.zeros(len(items))
    for outcome, color in zip(OUTCOMES, colors):
        counts = np.array([sum(r["outcome"] == outcome for r in loaded[i["id"]][1]) for i in items])
        fractions = counts / len(loaded[items[0]["id"]][1])
        ax.barh(np.arange(len(items)), fractions, left=left, color=color,
                label={"collision": "Force limit", "outside": "Workspace",
                       "truncated": "Task time limit", "success": "Success"}[outcome])
        for j, (fraction, count) in enumerate(zip(fractions, counts)):
            if count:
                ax.text(left[j] + fraction / 2, j, str(count), ha="center", va="center",
                        color="white" if outcome in ("success", "collision") else "#222", fontsize=9)
        left += fractions
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels([i["label"] for i in items])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of 256 fixed cases (numbers are counts)")
    ax.set_title("A  Task outcomes")
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .965),
               ncol=4, fontsize=9, frameon=False)
    for ax, key, title, xlabel in [
            (axes[0, 1], "max_material_force_n", "B  Material peak", "Episode peak material force [N]"),
            (axes[1, 0], "material_impulse_ns", "C  Material impulse", "Episode material force impulse [N s]"),
            (axes[1, 1], "safety_margin", "D  Sensor limit utilization", "Maximum component-wise sensor / limit ratio")]:
        for item, color in zip(items, palette):
            values = np.sort([r[key] for r in loaded[item["id"]][1]])
            ax.step(values, np.arange(1, len(values) + 1) / len(values),
                    where="post", color=color, label=item["label"], linewidth=1.7)
        if key == "safety_margin":
            ax.axvline(1, linestyle="--", color="#555", linewidth=1, label="Sensor limit")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Fraction of cases at or below x")
        ax.set_ylim(0, 1.02); ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Frozen-controller comparison under executed-action perturbations", fontsize=14, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, "01_perturbed_outcomes_and_force",
         "All 256 cases per checkpoint; identical executed-action disturbance cases. "
         "ECDFs include success and every failure. The threshold at 1 applies only to "
         "sensor utilization, never to the material-force panel. Curves describe selected "
         "checkpoints, not training-seed uncertainty or general algorithm superiority.")

    # 2. Use the already-recorded paired bootstrap, checked against raw-case mean differences.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
    specs = [("all_cases", "success", 100, "Success [percentage points]"),
             ("all_cases", "collision", 100, "Force-limit endings [percentage points]"),
             ("all_cases", "max_material_force_n", 1, "Material peak [N]"),
             ("all_cases", "material_impulse_ns", 1, "Material impulse [N s]"),
             ("both_success", "steps", 1, "Both-success duration [control steps]"),
             ("both_success", "max_material_force_n", 1, "Both-success material peak [N]")]
    for ax, (group, key, scale, title) in zip(axes.flat, specs):
        for y, other in enumerate(("policy62", "sac85k")):
            value = pairs[other]["a_minus_b"][group][key]
            mean = value["mean_difference"] * scale
            low, high = np.array(value["interval_95"]) * scale
            ax.plot([low, high], [y, y], color=palette[y * 2], linewidth=2)
            ax.scatter([mean], [y], color=palette[y * 2], s=35, zorder=3)
            ax.annotate("{:.2f} [{:.2f}, {:.2f}]".format(mean, low, high), (mean, y),
                        xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)
        ax.axvline(0, color="#666", linestyle="--", linewidth=1)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["vs policy62", "vs SAC85k"])
        ax.set_ylim(1.6, -.65); ax.margins(x=.30); ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=.18)
    fig.suptitle("Working policy minus comparator: paired differences and 95% intervals", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, "02_paired_checkpoint_effects",
         "Saved case-paired bootstrap intervals, with means and subset membership verified "
         "from raw cases. All-case n=256; both-success n=188 vs policy62 and n=202 vs SAC85k. "
         "Different successful subsets are not interchangeable. These intervals do not "
         "include variability across training seeds. Zero-crossing force-limit/impulse "
         "intervals do not establish equality or safety equivalence.")

    # 3. Each point is one recorded episode; do not turn a censored prefix into failure.
    fixed_ids = [i["id"] for i in cfg["unperturbed"]]
    labels = [i["label"] for i in cfg["unperturbed"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.2), sharey=True)
    marker = {"success": ("o", "#147d70"), "collision": ("x", "#b64c4c"),
              "outside": ("x", "#d6a64e"), "truncated": ("s", "#777"),
              "censored": (">", "#666")}
    for ax, key, label in zip(axes, ["simulation_seconds", "material_peak_force_n", "material_impulse_ns"],
                              ["Simulated elapsed time [s, log scale]", "Material peak [N]", "Material impulse [N s]"]):
        for y, identity in enumerate(fixed_ids):
            rows = [r for r in fixed if r["controller"] == identity]
            for row in rows:
                m, c = marker[row["outcome"]]
                ax.scatter(row[key], y, marker=m, color=c, s=45, alpha=.85)
            # Equal deterministic repeats overlap; show their recorded mean once.
            value = np.mean([r[key] for r in rows])
            text = "{:.2f}".format(value) + (" (prefix)" if rows[0]["outcome"] == "censored" else "")
            ax.annotate(text, (value, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel(label); ax.grid(axis="x", alpha=.2)
        if key == "simulation_seconds":
            ax.set_xscale("log")
            ax.set_xlim(min(r[key] for r in fixed) * .7, max(r[key] for r in fixed) * 2)
        else:
            ax.set_xlim(left=0, right=max(r[key] for r in fixed) * 1.35)
    axes[0].set_yticks(np.arange(len(labels))); axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.suptitle("Unperturbed executions: complete episodes and one budget-censored prefix", fontsize=13)
    handles = [Line2D([], [], marker=m, linestyle="", color=c, label=name)
               for name, (m, c) in [("Goal reached", marker["success"]),
                                    ("Force limit", marker["collision"]),
                                    ("Wall-budget prefix", marker["censored"])]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, .95))
    save(fig, "03_unperturbed_execution",
         "Each MPPI row is one planner seed, with critic settings separated. SAC has eight "
         "deterministic repeats (overlapping points), not eight training seeds. Short force-limit "
         "episodes are failures, not faster completions. Zero action ended at a wall budget "
         "after 323/500 steps; all of its numbers are prefix metrics. Reference trajectory "
         "uses its separate action branch. Goal completion does not test full material separation.")

    # 4. Report actual call time separate from physics; no SAC process/network timing conflation.
    timing_ids = ["policy62", "working", "sac85k"] + [x for x in fixed_ids if x.startswith("mppi")]
    timing = [next(p for p in provenance if p["id"] == identity) for identity in timing_ids]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True)
    for ax, key, title in [
            (axes[0], "planning_or_inference_ms_per_step", "Controller call"),
            (axes[1], "physics_ms_per_step", "Physical simulation step")]:
        values = [p[key] for p in timing]
        ax.barh(np.arange(len(values)), values, color=["#147d70"] * 2 + ["#c06a21"] + ["#526b91"] * 6)
        ax.set_xscale("log"); ax.set_xlim(.5, max(values) * 2.2)
        for y, value in enumerate(values):
            ax.text(value * 1.08, y, "{:.2f}".format(value), va="center", fontsize=9)
        ax.set_title(title); ax.set_xlabel("Recorded wall time per executed action [ms]")
        ax.grid(axis="x", alpha=.2)
    axes[0].axvline(50, color="#444", linestyle="--", linewidth=1, label="50 ms task period")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_yticks(np.arange(len(timing))); axes[0].set_yticklabels([p["label"] for p in timing])
    axes[0].invert_yaxis()
    fig.suptitle("Measured online call cost and physics cost are separate", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "04_recorded_online_cost",
         "Report totals divided by executed steps. SAC call time includes the Python-process "
         "bridge; WM policy and MPPI have different call implementations. Reports were collected "
         "at different times, not in a controlled throughput experiment. The 50 ms line is "
         "the simulated task period. Fast policy inference does not imply real-time coupled physics.")

    all_rows = [{"controller": item["id"], **row} for item in items for row in loaded[item["id"]][1]]
    csv_file(args.output / "perturbed_cases.csv", all_rows)
    csv_file(args.output / "unperturbed_episodes.csv", fixed)
    effect_rows = []
    for other, report in pairs.items():
        for group, values in report["a_minus_b"].items():
            for key, value in values.items():
                if key == "cases":
                    continue
                effect_rows.append({"comparator": other, "subset": group, "metric": key,
                                    "cases": value["episodes"], "working_minus_comparator": value["mean_difference"],
                                    "ci95_low": value["interval_95"][0], "ci95_high": value["interval_95"][1]})
    csv_file(args.output / "paired_effects.csv", effect_rows)
    summary = {item["id"]: loaded[item["id"]][2] for item in items}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output / "controllers.json").write_text(json.dumps(
        {"interface_and_training": cfg["controllers"], "frozen_report_provenance": provenance},
        indent=2, allow_nan=False) + "\n")
    index = ["# Cutting evidence package", "", "Saved data only; no new physical interaction or optimizer updates.", "",
             "Named frozen controllers on public-prism-ur5e-v1. Historical package; see docs/archive/EVIDENCE_20261110.md. Perturbation results are excluded from the paper.", ""]
    for name, caption in captions:
        index += ["## " + name, "", "![" + name + "](" + name + ".png)", "", caption, "",
                  "[Vector PDF](" + name + ".pdf)", ""]
    index += ["CSV: perturbed_cases.csv, unperturbed_episodes.csv, paired_effects.csv.",
              "Provenance: controllers.json and manifest.json.", ""]
    (args.output / "index.md").write_text("\n".join(index))
    manifest = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "script_sha256": digest(__file__), "inputs_sha256": inputs,
                "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "physical_steps": 0, "optimizer_updates": 0,
                "outputs_sha256": {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "perturbed_cases": len(all_rows),
                      "recorded_unperturbed_episodes": len(fixed), "figures": len(captions)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/archive/evidence_20261110.json")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--legacy-root", type=Path,
                        default=Path("/dataHDD/dong/Experiments/Cutting_DEV/legacy_metrics/metrics_eval_20261106"))
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
