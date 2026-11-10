# Mitsioni / SafeDDC — accessible preprint, final cutting section unavailable

Reviewed 2026-11-10. Final article: [T-RO 2023, DOI 10.1109/TRO.2023.3266995](https://doi.org/10.1109/TRO.2023.3266995).
Retrieved [KTH preprint](https://www.diva-portal.org/smash/get/diva2%3A1620620/FULLTEXT01.pdf).
Read scope: preprint method (§3), prediction/safety evaluation, results and
limitations (§§4–5). This version contains the broken-pendulum experiment.
**The final article's cutting section has not been obtained/read.** Do not
attribute that experiment's detailed force rules or results to this read.

## Mechanism

RRT exploration provides broader state/input data for learned dynamics.
A beta-VAE embeds safe demonstration transitions; DBSCAN clusters and contours
approximate a safe region. Online random-shooting MPC penalizes distance outside
that region and uses a separately constructed RRT for long-term goal distance.
Only the first selected command executes before replanning.

Safety classification is empirical and dependent on the learned representation.
The paper reports false-positive safe encodings and oscillation between safe
sets; safety cost alone can reduce convergence. Combining goal foresight and
safety checks matters. A lower state prediction error is not sufficient evidence
of better closed-loop decisions.

## Author code inspected

[imitsioni/SafeDDC, 93c10a022d6a793850a68e4d938255e1e114328a](https://github.com/imitsioni/SafeDDC/tree/93c10a022d6a793850a68e4d938255e1e114328a).
The released entry points are under `inverted_pendulum/`, not a ready cutting
adapter. `pendulum_mpc.py::check_safety` embeds predicted trajectories and
detaches to NumPy for polygon lookup. `cost_function` penalizes negative
signed distance exponentially; the loop evaluates candidates and executes
the minimum-cost one. `safety/safeset.py` computes signed polygon distances.
This is sampling MPC, not BPTT through a differentiable safety layer.

## Consequence

Safe learned-model planning and trajectory-level reliability are strong prior
art. Our same-WM MPPI can be a comparator, but is not a reproduction of this
paper. The unavailable final cutting experiment remains an explicit literature
coverage gap, rather than being filled with claims from its abstract.
