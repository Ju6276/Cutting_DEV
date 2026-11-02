# Results

## 2026-11-02

Collector `9e7273d`, SuperMimic-Reproduced `ba610534`, DiSECt `db7de0f`.
`collect_disect.py --config ../configs/public_prism.json --disect-root .
--output /dataHDD/dong/Projects/CuttingWorldModel/runs/prism_straight`:
1.3 s, 26 control intervals, 1187 vertices, 304 springs, 281 released at the
endpoint; peak force 10.3782 N, wall time 87.10 s while SAC was also running.
Full positions, velocities, remaining spring stiffness, knife pose/twist and
force are stored in `runs/prism_straight/trajectory.npz`.

`check_state_restart.py --dataset runs/prism_straight --frame 13`:
110 springs released at restart. One-step position RMS/max difference
3.78e-8/3.80e-7 m; velocity RMS/max 4.39e-5/6.17e-4 m/s; spring stiffness
RMS/max 0.00572/0.07373. Record: `restart_check.json`.

During additional physics collection, GPU utilization reached 98% and one
SAC episode took 96.22 s for 22 steps, versus about 1.76 s/step beforehand.
The collector exited after one trajectory. Further large physics collection
must be scheduled rather than continuously competing with the SAC baseline.

Task interface `ec8c49f`, source `ba610534`:
`run_coupled_collection.py --sliceit-root /dataHDD/dong/Projects/SliceIt
--output runs/interface_source_policy_aligned --episodes 1 --wall-seconds 600`:
41 transitions, 233.39 s total wall time (197.08 s collector); episode ended at
the original force limit, measured x force -52.318 N. Policy updates: 0.

`check_task_interface.py --dataset runs/interface_source_policy_aligned/data`:
observation maximum absolute error 0; reward maximum absolute error 5.55e-17;
termination and success mismatches 0; success-counter error 0. Upstream replay:
42 rows; eight-step window batch shape [9, 2048, 7535].
`python -m pytest tests -q`: 12 passed in 0.74 s.
Next: cutting StateSpec and connected online world-model/policy training.
