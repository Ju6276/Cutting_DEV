# Safe Learning for Contact-Rich Robot Tasks: complete main-text reading

- Source: [arXiv:2512.11908v4](https://arxiv.org/abs/2512.11908v4), revised 2026-10-29, accepted npj Robotics; the arXiv entry describes 94 pages.
- **Complete main-text reading, with a separate bibliography boundary.** On 2026-11-10 I completed the full main text (PDF pp. 1–56), the disclosure statements on p. 57, and the comparative tables placed after the references (pp. 90–93). This includes the previously omitted method-rating rationale (15–17), task families (23–27), sensing/policy modalities (27–34), safety abstraction levels and foundation-model examples (43–47), and enforcement spaces (47–50). The source is advertised as 94 pages; the extracted PDF has 93 non-empty pages. The bibliography on pp. 57–89 was not followed paper by paper: reading this survey does not constitute reading its cited primary papers. No additional bibliography task was opened.
- References in the survey are discovery pointers, **not automatically papers we have read**.
- Remote source: `/dataHDD/dong/Research/cutting_literature_20261110/contact-survey.pdf`.
- Author [resource repository](https://github.com/jack-sherman01/Awesome-Learning4Safe-Contact-rich-tasks), commit `2cc5b031ccb3e5b2a9e1ada9d88f2a0d1d46ecba`. This is a literature collection and analysis repository, not a robot controller implementation.

## What the main text says

The survey defines contact-rich tasks through sustained, changing physical interaction with tightly coupled force and motion. Cutting fits that scope, but this does not require our project to add every contact-rich task.

Safety is separated into several non-equivalent claims: physical force/impulse/energy bounds; stability or passivity; invariant safe sets; expected or probabilistic risk; and semantic appropriateness of high-level plans. A correct plan, a stable controller and a low learned risk score do not imply the same guarantee.

Known compliant controllers, learned policies, uncertainty-aware MPC and safety filters are complementary parts of an architecture. Meaningful force constraints depend on sensing, latency, contact mode and modeling assumptions. Learning pose/force/impedance references is common and does not by itself constitute a new algorithm.

The formal-method discussion explicitly warns that Lyapunov stability does not automatically bound transient force, CBF validity depends on dynamics/model assumptions, multiple constraints can be infeasible, and contact constraints can have difficult relative degree. Reachability methods trade offline cost and model fidelity against fast online intervention.

## Data and evaluation lessons

The data discussion distinguishes broad simulation coverage from physically faithful interaction data. Failure/near-miss episodes and synchronized high-frequency wrench traces are especially valuable for learning safety-relevant dynamics; large vision-language corpora do not replace them.

The evaluation section recommends separate task, safety and efficiency statistics rather than only total reward. Useful quantities include completion, violation count/severity, peak/mean force, energy or impulse when relevant, data/compute cost and variability. It distinguishes aggregate results, distributions/training trends and descriptive single-episode examples.

The survey **does discuss observation/action disturbances and distribution shifts as valid tools**. Therefore the earlier blanket claim “other papers do not do perturbation evaluation” is not supported. That does not make our old ad hoc perturbation suite necessary or appropriate for the approved paper claim; its relevance must follow the task distribution and hypothesis. This note does not reactivate that suite.

The taxonomy's radar ratings are author-assigned ordinal summaries, not head-to-head benchmark measurements. Its statements about entire method families should be checked against the primary papers before importing them as hard guarantees or algorithm rankings.

## Additional findings from the completed sections

1. **Task generality must follow the interaction mechanism.** The task catalogue distinguishes tight geometric insertion/jamming, sustained surface-force regulation, moving/deformable object manipulation, and human interaction; these have different failure quantities and assumptions. Its surface-interaction discussion often assumes the workpiece stays fixed and the objective is sustained contact. Cutting introduces material removal or irreversible separation, so a safety mechanism that works for insertion or surface following cannot be treated as already applicable without checking those changed dynamics. This supports retaining cutting as our experimental scope while borrowing a clearly stated mechanism; the survey does not establish that our cutting-specific mechanism is novel.

2. **Sensing and prediction have different roles.** The modality sections distinguish global geometric anticipation from vision, local wrench feedback from F/T, and spatially distributed pressure/slip information from tactile sensing. Force histories and latent contact-state estimates are already common, so adding them is not a contribution. A known-geometry cutting task need not add camera inputs merely because it is contact-rich. Conversely, measuring a wrench, or observing a low filtered wrench, does not itself bound the next contact impulse: sensor alignment, bias/gravity compensation, sampling, latency and controller authority remain relevant. The table's wording about directly measured quantities must not be upgraded into an unconditional force guarantee.

3. **The location of a risk model matters for the claim.** The survey separates the abstraction level (planning, control, end-to-end) from the coordinate space of enforcement (task, joint, policy). A learned force-risk objective used during BPTT belongs to learning/planning; it is not automatically a runtime task-space safety filter. Bounded compliance actions, BC regularization or an empirically calibrated predictor do not supply the invariance/feasibility assumptions of a certified shield. The plan–parameterize–enforce discussion explicitly treats combinations of known modules as established architecture, not a new safety principle. Our method claim therefore still depends on a distinct, evidenced mechanism and improved closed-loop behavior.

The added reading **does not change the recommended project scope** and does not justify adding non-cutting tasks, cameras, VLA modules or new experiments. It sharpens the terminology and the boundary between predicting risk, optimizing with that prediction, and physically enforcing a constraint. The broad platform capability checkmarks and family-level guarantee summaries in Tables 3–8 are secondary-source syntheses; specific simulator capabilities or guarantees must be checked in their own primary sources before they are used to justify implementation choices.

## Resource/code audit

The repository contains a literature spreadsheet, bibliography, source TeX and plotting/curation scripts. I read:
- `scripts/bib_counts.py`: regex-based bibliography parsing/year counting and heuristic safety/RL keyword tagging.
- `scripts/plot_publications_trend.py`: a hard-coded overall yearly count series with optional bibliography-derived overlays.
- The repository file inventory and the paper's data/code-availability statements.

No neural model, policy gradient or physics gradient is implemented in these scripts. They do not reproduce the controllers surveyed. A path mismatch also remains in the checked counting script: it expects root `reference.bib`, whereas the listed bibliography is under `data/reference.bib`. I did not run or repair those external scripts.

The stated systematic search window is 2018–2025, last database search August 2025, even though the revised paper discusses some newer work. Consequently the survey cannot establish absence of a 2026 method.

## How it should influence our scope

Use the survey to organize assumptions and reporting, not to expand the project into a generic contact-rich benchmark or to claim an untouched research gap.

For the current cutting route, the relevant unresolved question is whether learned interaction predictions remain useful when the policy changes its action around contact and damage transitions. Calibrating errors on old trajectories is only one part of that problem. A prospective mechanism still needs to be distinguished from the primary prior work, and any safety claim must match the measured force/event definition and its assumptions.
