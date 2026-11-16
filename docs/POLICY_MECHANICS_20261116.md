# Frozen policy mechanics and representation analysis — 2026-11-16

This is a descriptive analysis of two existing policies on their **recorded original physics**.
It does not evaluate the new shared knife or the new `pose_depth_v2` reward.
Generated PNG/PDF figures and numerical arrays are on GPU7:
`runs/tool_analysis_20261116/{main,blademaster}/`.

## Data and identity

- Main policy: 1109 actor updates; 48 first episodes, 16 per object, 8640 transitions.
  These episodes all timed out under the recorded pose-success criterion.
- BladeMaster policy: 866 actor updates; 48 first episodes, 16 per object, 1632 transitions.
  All reached the recorded pose-success criterion. This is not a universal separation certificate.
- Objects: cucumber, half potato and prism. No tofu, banana or carrot trials are represented.
- Main episodes vary material, start height and target speed. BladeMaster cases repeat nominal
  object conditions. They are not a matched material sweep or a fair backend performance comparison.
- Checkpoint SHA256 is verified against each evaluation manifest. Reconstructed raw policy
  actions agree with the recorded actions within 6.68e-6 (main) / 1.67e-6 (BladeMaster).

## What the network actually receives

The 10 × 34 = 340 numbers are the **whole observation history**, not 340 force features.
Each observation contains pose error, motion/jerk, previous actions, six wrench components
with three within-observation samples, and remaining time. Wrench values are restored to
sensor-frame N and N m in the plots.

The goal encoder does not encode the force history. We extract the 256-dimensional last
hidden activation of the history-plus-goal decoder, before its final action linear layer.
The separate goal encoder is plotted as a control.
Both frozen checkpoints have max_twist=[1,0]. The pitch target-increment channel
is disabled. Action a5 can change numerically without changing the requested knife
angle. A force-dependent inclination claim is therefore not supported by these runs.

Actions a0/a1 select controller x/pitch stiffness, a2/a3 vertical/pitch P gains,
a4 accumulates an x-target increment, and a5 would accumulate pitch if enabled.
The first four are additionally clamped to [0,1] and quantized by the mapper.
The clipped [-1,1] policy output is consequently not a direct actuator command.
`mapped_controller_commands.{png,pdf}` plots the actual mapper gains and target
increments; it does not infer knife motion from raw a5.


## Figures and conclusions

`force_action_timeline.{png,pdf}` shows sensor force, sensor moment, the six clipped actions,
feed velocity versus command, finite-blade board clearance and lateral target error.
The representative case is the lower median-return episode of each object, selected by a
fixed rule. Dotted marks denote a contact proxy (>0.5 N interval material peak, available
to the next observation); red dashed marks denote proximity to the board interlock.

- Main policy: median material force peaks are 6.39 N (cucumber), 8.72 N (potato), 3.34 N
  (prism). Approximately 84% of action components over the episodes are at a clipping bound.
- BladeMaster policy: median material force peaks are 6.01, 9.63 and 3.44 N.
  The representative trajectories accelerate above the 30 mm/s command before slowing near
  the board. The board constraint is active there; this is **not evidence of learned
  post-separation braking**.
- Sensor-axis forces and material resultant peaks are different quantities. The curves do
  not demonstrate stiffness identification, causal peak-force reduction, or learned sawing.
  A large force alone does not identify material stiffness.

`features_pca_controls.{png,pdf}` uses one feature per episode at reset, pre-proxy,
contact+3 and contact+5, with one shared PCA basis per policy across phases.
The first two decoder components explain 96.48% (main) and 99.98% (BladeMaster) of this
selected feature variance. Goal-only projections are a control, not evidence of identification.
Tiny goal-PC2 scales are numerical residue; do not interpret their visual separation.

- BladeMaster reset features overlap, and object features separate later. Its goal-only
  features remain identical. This supports a descriptive statement about evolving
  representations, but the contact-aligned phases occur at different absolute times and
  robot positions. Pre-proxy means the frame before the >0.5 N marker, not guaranteed
  absence of all earlier contact. These plots cannot isolate material estimation from
  motion/geometry/history effects or prove identification after 3–5 force frames.
- Main features do not show a clean newly emerging material partition.
  `material_feature_projection.{png,pdf}` colors contact+5 points by Young-modulus
  multiplier within each object. Other material parameters, start height and commanded
  speed also vary, preventing a causal stiffness interpretation.
- PCA grouping is not proof of material identification. Geometry, motion, goal and time
  are potential alternative explanations; t-SNE would not remove those confounds.

The main-policy material_force_action_timeline figure additionally contrasts the
lowest/highest Young-modulus cases within each geometry, aligned at the force
contact proxy. Every panel discloses spring stiffness, density, start lift and
commanded speed: these quantities also differ, so it is an observed low/high-E
contrast rather than an isolated material-stiffness experiment.

## Force-history intervention

`force_history_sensitivity.{png,pdf}` replaces the older nine wrench blocks with the
current wrench block while leaving all other history and the goal fixed. It is an
open-loop sensitivity measurement on the recorded state, not a closed-loop performance
ablation and not deletion of all force information.

Mean absolute change in the clipped action passed to the mapper, over dimensions and time, averaged per episode:

| Object | Main | BladeMaster |
|---|---:|---:|
| Cucumber | 0.000531 | 0.004899 |
| Half potato | 0.000910 | 0.007027 |
| Prism | 0.000171 | 0.001288 |

The action range is [-1,1]. Main clipping can hide changes to raw network outputs; the
arrays and per-case summaries retain raw changes too. The overall mean raw-action
changes are 0.00927 (main) and 0.00831 (BladeMaster), so the much smaller main
clipped-action change does not show that its network ignores force history.
BladeMaster clipped outputs respond more
in this particular intervention, but the two policies and tasks differ. Neither number
establishes an algorithmic advantage. With the actual mapper applied, the main
intervention changes only the x-target increment (mean absolute change 6.45 micrometres
per control step); its four mapped gains stay unchanged on these traces. For
BladeMaster, mean changes are about 48.8 N/m in x stiffness, 0.472 N m/rad in
pitch stiffness, 0.000431 in vertical P gain, and 9.64 micrometres in x increment.
Pitch P gain and the disabled pitch target increment do not change.

## Supported paper statement

> The frozen policies exhibit object-dependent force and motion traces. Their decoder
> representations and force-history sensitivity can be inspected directly, but the
> present trajectories do not establish implicit material identification or learned
> protection against board impact.

A positive material-identification claim would need controlled same-geometry conditions,
matched starting states, and a held-out material probe or closed-loop history intervention.
Those experiments were not silently added to this analysis.

## Reproduce

Use `scripts/analyze_policy_mechanics.py` with the evaluation directory, exact checkpoint,
and a new output directory. It checks action reconstruction and checkpoint identity.
Then run `scripts/plot_policy_mechanics.py --root runs/tool_analysis_20261116`.
See each generated `summary.json` for the exact source paths and hashes.
No policy, world model or critic was trained in this analysis.
