"""Make the bundled backend importable independent of test collection order."""
from pathlib import Path
import sys

BACKEND = str(Path(__file__).resolve().parents[1] / 'backends' / 'mujoco')
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
