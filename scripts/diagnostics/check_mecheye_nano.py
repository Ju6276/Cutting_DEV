"""Read Mech-Eye camera state and export one unmodified SDK RGB-D frame.

Requires the installed MechEyeAPI, NumPy and OpenCV packages. This never sets
camera parameters, selects a user set, moves a robot, or changes the point unit.
The output directory must be new and outside the Git checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def timestamp():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def plain(value):
    """Serialize inspected SWIG properties, retaining original field names."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_string"):
        return value.to_string()
    if isinstance(value, (list, tuple)) or hasattr(value, "__len__"):
        return [plain(value[index]) for index in range(len(value))]
    fields = {name: plain(getattr(value, name))
              for name, descriptor in vars(type(value)).items()
              if isinstance(descriptor, property) and name not in ("this", "thisown", "data")}
    if not fields:
        raise TypeError(f"Unexposed SDK value: {type(value).__name__}")
    return fields


def check(status):
    if not status.is_ok():
        raise RuntimeError(f"{status.error_code}: {status.error_description}")


def read_parameters(user_set, parameter_type):
    getters = {
        parameter_type._Int: "get_int_value", parameter_type._Float: "get_float_value",
        parameter_type._Bool: "get_bool_value", parameter_type._Enum: "get_enum_value_string",
        parameter_type._Roi: "get_roi_value", parameter_type._Range: "get_range_value",
        parameter_type._FloatArray: "get_float_array_value", parameter_type._RoiArray: "get_roi_array_value",
        parameter_type._ProfileRoi: "get_profile_roi_value",
    }
    result = {}
    for parameter in user_set.get_available_parameters():
        entry = {"type": int(parameter.type()), "description": parameter.description(),
                 "readable": parameter.is_readable(), "writable": parameter.is_writable()}
        method = getters.get(parameter.type())
        if method and entry["readable"]:
            try:
                status, value = getattr(user_set, method)(parameter.name())
                entry.update(error_code=status.error_code, error_description=status.error_description)
                if status.is_ok():
                    entry["value"] = plain(value)
            except (TypeError, RuntimeError) as error:
                entry["unavailable_reason"] = str(error)
        else:
            entry["unavailable_reason"] = "No matching exposed getter or not readable"
        result[parameter.name()] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", required=True, help="Actual scene; never relabel a desk as a board sample")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.is_relative_to(repo):
        parser.error("Raw data must be outside the Git checkout")
    output.mkdir(parents=True, exist_ok=False)

    # Delay SDK imports so --help also works on a computer without the camera SDK.
    import cv2
    import numpy as np
    import mecheye
    from mecheye.area_scan_3d_camera import Camera, CameraInfo, CameraIntrinsics, CameraResolutions, Frame2DAnd3D
    from mecheye.area_scan_3d_camera_utils import print_camera_intrinsics
    from mecheye.shared import CoordinateUnit_Meter, CoordinateUnit_Millimeter, FileFormat_PLY, Parameter, get_api_version_info

    script = Path(__file__).resolve()
    metadata = {
        "started_at_host": timestamp(), "timestamp_source": "host wall clock, Europe/Berlin; not sensor exposure time",
        "hostname": socket.gethostname(), "operator": "Codex on user's camera PC",
        "linux": dict(platform.freedesktop_os_release()), "python": sys.version,
        "python_executable": sys.executable, "api_native_version": get_api_version_info(),
        "package_versions": {p: importlib.metadata.version(p) for p in ("MechEyeAPI", "numpy", "opencv-python-headless")},
        "script_path": str(script), "script_sha256": sha256(script),
        "script_git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "script_tracked_diff": subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--", str(script)], text=True),
        "scene": args.scene, "device_configuration_changed_by_script": False,
        "postprocessing_by_script": "None: no crop, resize, undistortion, registration, filtering or downsampling",
        "capture_method": "Camera.capture_2d_and_3d(Frame2DAnd3D, 60000)",
        "synchronization": "one combined API call; exact RGB/depth exposure simultaneity not measured",
    }
    camera = Camera()
    try:
        check(camera.connect(args.ip, 10000))
        info = CameraInfo()
        check(camera.get_camera_info(info))
        metadata["camera_info"] = plain(info)
        manager = camera.user_set_manager()
        current = camera.current_user_set()
        status, name = current.get_name()
        check(status)
        metadata["active_user_set"] = name
        status, names = manager.get_all_user_set_names()
        check(status)
        metadata["all_user_set_names"] = names
        check(manager.save_to_file(str(output / "user_sets_before.json")))
        before = read_parameters(current, Parameter)
        write_json(output / "active_parameters_before.json", before)

        intrinsics = CameraIntrinsics()
        check(camera.get_camera_intrinsics(intrinsics))
        intrinsics_json = plain(intrinsics)
        write_json(output / "intrinsics.json", intrinsics_json)
        from contextlib import redirect_stdout
        with (output / "intrinsics_sdk_print.txt").open("w") as stream, redirect_stdout(stream):
            print_camera_intrinsics(intrinsics)
        resolutions = CameraResolutions()
        check(camera.get_camera_resolutions(resolutions))
        metadata["camera_resolutions_api"] = plain(resolutions)
        status, unit = camera.get_point_cloud_unit()
        check(status)
        metadata["point_cloud_unit_api"] = {"enum": int(unit), "name": {
            CoordinateUnit_Millimeter: "Millimeter", CoordinateUnit_Meter: "Meter"}.get(unit, "unrecognized")}

        frame = Frame2DAnd3D()
        metadata["capture_started_at_host"] = timestamp()
        start = time.perf_counter_ns()
        check(camera.capture_2d_and_3d(frame, 60000))
        metadata["capture_elapsed_seconds"] = (time.perf_counter_ns() - start) / 1e9
        metadata["capture_finished_at_host"] = timestamp()
        frame2d, frame3d = frame.frame_2d(), frame.frame_3d()
        metadata["sdk_frame_ids"] = {"rgb": int(frame2d.frame_id()), "depth": int(frame3d.frame_id())}
        bgr = np.array(frame2d.get_color_image().data(), copy=True)
        depth = np.array(frame3d.get_depth_map().data(), copy=True)
        xyz = np.array(frame3d.get_untextured_point_cloud().data(), copy=True)
        for name, array in (("rgb_bgr", bgr), ("depth", depth), ("xyz", xyz)):
            np.save(output / f"{name}.npy", array, allow_pickle=False)
        if not cv2.imwrite(str(output / "rgb.png"), bgr):
            raise RuntimeError("Failed to save lossless RGB PNG")
        check(frame3d.save_untextured_point_cloud(FileFormat_PLY, str(output / "xyz_organized.ply"), True))
        check(frame.save_textured_point_cloud(FileFormat_PLY, str(output / "xyzrgb_organized.ply"), True))
        metadata["arrays"] = {name: {"shape": list(array.shape), "dtype": str(array.dtype),
                                    "byte_order": array.dtype.byteorder}
                              for name, array in (("rgb_bgr", bgr), ("depth", depth), ("xyz", xyz))}
        valid = np.isfinite(depth) & (depth > 0)
        metadata["depth_statistics"] = {
            "valid_positive_finite": int(valid.sum()), "total_pixels": int(depth.size),
            "nan": int(np.isnan(depth).sum()), "inf": int(np.isinf(depth).sum()),
            "zero": int((depth == 0).sum()), "negative_finite": int((np.isfinite(depth) & (depth < 0)).sum()),
        }
        if valid.any():
            metadata["depth_statistics"].update(min=float(depth[valid].min()), max=float(depth[valid].max()))
        joint = valid & np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
        metadata["depth_xyz_z_max_abs_difference"] = float(np.max(np.abs(depth[joint] - xyz[..., 2][joint]))) if joint.any() else None
        # Check the SDK's output pixel geometry without modifying any saved data.
        rows, cols = np.nonzero(joint)
        if len(rows):
            selected = np.linspace(0, len(rows) - 1, min(10000, len(rows)), dtype=int)
            points = xyz[rows[selected], cols[selected]]
            matrix = intrinsics.depth.camera_matrix
            projected_u = matrix.fx * points[:, 0] / points[:, 2] + matrix.cx
            projected_v = matrix.fy * points[:, 1] / points[:, 2] + matrix.cy
            residual = np.hypot(projected_u - cols[selected], projected_v - rows[selected])
            metadata["sdk_output_pinhole_check"] = {"points": int(len(selected)),
                "max_residual_px": float(residual.max()), "median_residual_px": float(np.median(residual)),
                "scope": "SDK XYZ/depth-intrinsics internal consistency; not independent calibration or physical unit validation"}

        after = read_parameters(current, Parameter)
        write_json(output / "active_parameters_after.json", after)
        check(manager.save_to_file(str(output / "user_sets_after.json")))
        status, after_name = current.get_name()
        check(status)
        metadata["active_user_set_after"] = after_name
        metadata["parameters_before_after_equal"] = before == after
        metadata["native_user_sets_before_after_equal"] = json.loads((output / "user_sets_before.json").read_text()) == json.loads((output / "user_sets_after.json").read_text())
        metadata["finished_at_host"] = timestamp()
        print(json.dumps({k: metadata[k] for k in ("camera_info", "active_user_set", "point_cloud_unit_api", "arrays", "capture_elapsed_seconds", "depth_statistics", "sdk_output_pinhole_check", "parameters_before_after_equal", "native_user_sets_before_after_equal")}, indent=2), flush=True)
    except Exception as error:
        metadata["error"] = str(error)
        raise
    finally:
        camera.disconnect()
        write_json(output / "metadata.json", metadata)
        records = {str(path.relative_to(output)): {"sha256": sha256(path), "bytes": path.stat().st_size}
                   for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        write_json(output / "manifest.json", {"files": records})


if __name__ == "__main__":
    main()
