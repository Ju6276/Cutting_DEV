# Machines and access

Checked 2026-11-21. Secrets are never printed, committed, or copied between hosts;
this page names where credentials live, not their contents.

## Hosts

| Host | Address | Role | GPUs |
|---|---|---|---|
| tams61 (jump) | dong@100.70.97.92 | SSH jump host only; holds a read-only clone at ~/Projects/Cutting_DEV for reading docs. No training, data or weights. | — |
| GPU7 (tamsGPU7) | 134.100.13.217 | Main development and run host; the only host that pushes to GitHub | 1x RTX 4090 (GPU0) |
| A6000 (tamsGPU5) | 134.100.13.216 | Run host | 2x RTX A6000; check occupancy before use (other users run there) |
| GPU9 (tamsGPU9) | 134.100.13.219 | Run host | 1x RTX 4090 (GPU0) |

Hardware computers (separate from GPU training hosts):

- **tamsCage**: dong@134.100.13.143 via tams61; Ubuntu 20.04.6 / ROS Noetic,
  UR10e control PC. Repository /homeL/dong/Cutting_DEV; original FDCC workspace
  /homeL/dong/ur10e_fdcc_ws, offline aligned candidate /homeL/dong/ur10e_fdcc_alignment_ws.
  See [field record](hardware/TAMSCAGE.md); robot is powered off at the user's latest update.
- **ju-ROG**: Mech-Eye capture PC, Ubuntu 22.04.5; camera NIC enp108s0
  192.168.122.100/24 to camera 192.168.122.101. This camera subnet is not a
  verified remote SSH address. Camera environment /home/ju/Downloads/mechmind-camera/.venv,
  evidence /home/ju/Downloads/mechmind-camera/checklist-20261121-round1.
  See [Nano field record](hardware/MECH_EYE_NANO_CHECKLIST_ZH.md).

Each run host has the project at /dataHDD/dong/Projects/Cutting_DEV. Active runs
retain their launched source version; do not assume all live checkouts follow current main.
Confirm the commit with `git rev-parse HEAD` before a run. Runs: /dataHDD/dong/Experiments/Cutting_DEV/runs
(GPU7; local `runs/` directories on A6000/GPU9). Locations of data and frozen runs: [DATA](DATA.md).

## Connecting

- From the Mac: SSH aliases `cutting-gpu7` and `cutting-a6000` (Mac `~/.ssh/config`, both `ProxyJump dong@100.70.97.92`).
  GPU9 has no Mac alias: `ssh -J dong@100.70.97.92 dong@134.100.13.219`, or go through GPU7.
- From GPU7: aliases `a6000` and `gpu9` in `~/.ssh/config`; A6000 uses the dedicated key
  `~/.ssh/id_ed25519_a6000`. Copy files server-to-server from GPU7, never through the Mac.
- VS Code Remote SSH opens GPU7 for development.

## GitHub

- Repository: https://github.com/Ju6276/Cutting_DEV. Only GPU7 can push, with the deploy key
  `~/.ssh/id_ed25519_github_cutting` (alias `github.com` in GPU7 `~/.ssh/config`).
- A6000, GPU9 and tams61 have no GitHub key: they fetch over https and cannot push. To publish a
  commit made elsewhere, fetch it from GPU7 (`git fetch a6000:/dataHDD/dong/Projects/Cutting_DEV <ref>`) and push from GPU7.

## Weights & Biases

- Every long run uses `--wandb-mode online`, project `sliceit-supermimic`, entity
  `ju-dong6276-technical-university-of-munich`.
- Credentials: `/dataHDD/dong/.config/wandb/netrc` on GPU7, A6000 and GPU9; `scripts/env/cutting_env.sh`
  sets `NETRC` to it. Do not print it or copy it to other hosts.

## Host differences that matter

- Tracked environment builders/entry points are shared (`scripts/env/`):
  base `.venv`, SAC/planners `.venv-sac`, renderer `.venv-render` and BladeMaster
  extras. This does not prove installed packages/drivers are identical;
  each run records its runtime versions. See [RUNNING](RUNNING.md).
- ROS Noetic (UR meshes for the source-URDF test and studio export): GPU7 and GPU9 only; A6000 has none.
- Blender: 2.82 on GPU7 and GPU9 (the studio script's target); 3.4.1 on A6000 (not validated).
- `cutting_env.sh` defaults: A6000 GPU1 and CUDA arch 8.6, GPU7/GPU9 GPU0 and 8.9; `CUDA_VISIBLE_DEVICES` overrides.
  These defaults are not reservations; check `nvidia-smi` first.
