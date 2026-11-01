from pathlib import Path
import subprocess


def git_revision(path):
    path = str(Path(path).resolve())
    commit = subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", path, "status", "--porcelain"], text=True
    ).strip())
    return {"commit": commit, "dirty": dirty}
