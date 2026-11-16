"""Append synchronized sensor-wrench charts below an existing shared-studio video.

The original scene is never rerendered. Curves are signed, filtered sensor-frame
20 Hz endpoint samples, not 500 Hz peaks or material contact-resultant forces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sensor_series(traces, case, steps, task):
    if task.get("policy_observation_revision") != "sensor_wrench6_v1":
        raise ValueError("Full six-axis observation metadata is required")
    limits = np.asarray(task["controller"]["max_force_torque"], dtype=float)
    dt = float(task["environment"]["agent_control_dt"])
    if limits.shape != (6,) or not np.isfinite(limits).all() or (limits <= 0).any() or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid wrench limits or control interval")
    if not traces or steps < 1 or steps > len(traces) or case < 0:
        raise ValueError("Missing first-episode transitions")
    def array(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    obs = [array(traces[0]["before"]["observation"])[case]]
    for i, row in enumerate(traces[:steps]):
        if not bool(array(row["active"])[case]):
            raise ValueError("Case already ended; later episodes cannot be plotted")
        if not np.allclose(array(row["before"]["observation"])[case], obs[-1], rtol=1e-5, atol=1e-6):
            raise ValueError("Discontinuous observation history")
        if int(array(row["next_step_count"])[case].item()) != i + 1:
            raise ValueError("Trace time/reset mismatch")
        obs.append(array(row["next_observation"])[case])
    obs = np.asarray(obs)
    if obs.shape != (steps + 1, 34) or not np.isfinite(obs).all():
        raise ValueError("Expected finite 34-dimensional endpoint observations")
    # First15 motion/action terms; six axes x three normalized history samples;
    # last field is time. Preserve signed latest sample, not an axiswise peak.
    wrench = obs[:, 15:33].reshape(-1, 6, 3)[:, :, -1] * limits
    return np.arange(steps + 1) * dt, wrench, limits


def video_indices(times, fps, speed, expected_frames):
    if fps <= 0 or not np.isfinite(speed) or speed <= 0:
        raise ValueError("Invalid playback rate")
    times = np.asarray(times)
    if len(times) < 2 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Invalid recording timeline")
    # Exactly the shared studio renderer's frame sequence, including final hold.
    indices = np.searchsorted(times, np.arange(0, times[-1], speed / fps)).clip(0, len(times)-1)
    indices = np.r_[indices, np.full(fps, len(times)-1)]
    if len(indices) != expected_frames:
        raise ValueError("Video frame count differs from the recorded timeline")
    return indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--render", type=Path, required=True, help="Existing studio MP4 and JSON sidecar")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.resolve() == a.render.resolve():
        raise FileExistsError("Refuse to replace source or existing output")
    meta = json.loads(a.render.with_suffix(".json").read_text())
    if not meta.get("video") or meta.get("preview_only"):
        raise ValueError("A complete shared-studio video is required")
    recording = Path(meta["recording"])
    case = int(meta["case"]); steps = int(meta["episode"]["steps"])
    if int(meta["episode"]["case"]) != case:
        raise ValueError("Video case identity mismatch")
    initial = json.loads((recording/"geometry/0000/report.json").read_text())
    group = next(g for g in initial["groups"] if g["first_world"] <= case < g["first_world"] + g["worlds"])
    task = json.loads((recording/"geometry/0000"/group["path"]/"task.json").read_text())
    rows = [json.loads(line) for line in (recording/"episodes.jsonl").read_text().splitlines()]
    episode = next(row for row in rows if row["case"] == case)
    if episode["steps"] != steps:
        raise ValueError("Video and episode length differ")
    import torch
    traces = torch.load(recording/"trajectories.pt", map_location="cpu", weights_only=False)
    times, wrench, limits = sensor_series(traces, case, steps, task)
    fps = int(meta["fps"])
    indices = video_indices(times, fps, float(meta["speed"]), int(meta["frames"]))
    source_probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(a.render)], text=True))["streams"][0]
    width, height = int(source_probe["width"]), int(source_probe["height"])
    if int(source_probe["nb_read_frames"]) != len(indices):
        raise ValueError("Encoded movie differs from sidecar frame count")
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a.output.parent.mkdir(parents=True, exist_ok=True)
    panel_height = 300
    fig, axes = plt.subplots(1, 2, figsize=(width/100, panel_height/100), dpi=100)
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.74, wspace=.22)
    colors = ("#cf5146", "#40946a", "#397ca8")
    for j, ax in enumerate(axes):
        for k, c in enumerate(colors):
            ax.plot(times, wrench[:, 3*j+k], color=c, label=("F" if j == 0 else "M")+"xyz"[k], lw=1.7)
        ax.set(xlim=(0, times[-1]), xlabel="Time [s]", ylabel="Force [N]" if j == 0 else "Torque [N m]")
        ax.grid(alpha=.18); ax.legend(loc="upper right", ncol=3, fontsize=8)
    cursors = [ax.axvline(0, color="#333333", lw=1) for ax in axes]
    title = fig.suptitle("", fontsize=10)
    fig.text(.07, .025, "Filtered sensor-frame endpoints; not substep peaks. Dashed limits shown only within plot range.", fontsize=8)
    for j, ax in enumerate(axes):
        lo, hi = ax.get_ylim()
        for limit in set(limits[j*3:j*3+3]):
            for signed in (-limit, limit):
                if lo <= signed <= hi: ax.axhline(signed, color="#a44", ls="--", lw=.8)
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(a.render), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    encoder = subprocess.Popen(["ffmpeg", "-n", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height+panel_height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.output)], stdin=subprocess.PIPE)
    cache = {}; completed = False
    try:
        for frame_no, idx in enumerate(indices):
            data = decoder.stdout.read(width*height*3)
            if len(data) != width*height*3: raise ValueError("Truncated source video")
            idx = int(idx)
            if idx not in cache:
                for cursor in cursors: cursor.set_xdata([times[idx], times[idx]])
                title.set_text(f"Sensor wrench | t={times[idx]:.2f} s | {1/(times[1]-times[0]):g} Hz endpoints | case {case}")
                fig.canvas.draw()
                cache[idx] = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            image = Image.new("RGB", (width, height+panel_height), "white")
            image.paste(Image.frombytes("RGB", (width, height), data), (0, 0))
            image.paste(cache[idx], (0, height))
            encoder.stdin.write(image.tobytes())
            if frame_no == len(indices)//2: image.save(a.output.with_suffix(".preview.png"))
        encoder.stdin.close(); decoder.stdout.close()
        if encoder.wait() or decoder.wait(): raise RuntimeError("FFmpeg failed")
        completed = True
    finally:
        for process in (encoder, decoder):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        plt.close(fig)
        if not completed:
            a.output.unlink(missing_ok=True)
            a.output.with_suffix(".preview.png").unlink(missing_ok=True)
    np.savez_compressed(a.output.with_suffix(".wrench.npz"), time_s=times, sensor_wrench=wrench, force_limits=limits)
    result = dict(source_video=str(a.render.resolve()), source_sha256=hashlib.sha256(a.render.read_bytes()).hexdigest(),
                  recording=str(recording), case=case, frames=len(indices), fps=fps, speed=meta["speed"],
                  signals="signed filtered sensor-frame Fx Fy Fz Mx My Mz, control-step endpoints; not substep peaks",
                  scene="Source video preserved in upper panel; no geometry or physics changes")
    a.output.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
