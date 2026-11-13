#!/usr/bin/env python3
"""Platform check for the free-object single-cut task on one scene (no policy, no training).

Answers three questions with a prescribed knife motion (rest, press down, hold, retract):
1. Object-board contact: does the unheld object rest on the ground without sinking or drifting?
2. Knife contact: does the press produce a finite net contact wrench and move/cut the object?
3. Severed-state check: are the two sides of the cut plane connected only through cut springs
   (mesh connectivity), and are all of them released at the end?
Force-limit termination is disabled; this is a measurement, not a task episode. Forces are
sampled at every 2 ms coupling step (the rate of the limit check): the net contact force on
the knife and the six-component sensor wrench the limit applies to. Each world is settled
with its own material scales before the episode.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.provenance import git_revision  # noqa: E402
from cutting_gpu_backend.controller.fdcc import quat_apply  # noqa: E402


def transformed_knife_vertices(vertices, position, rotation):
    """Actual knife mesh in the DiSECt world frame; quaternions are xyzw."""
    vertices = vertices.to(position)
    return position[:, None, :] + quat_apply(rotation[:, None, :], vertices[None, :, :])


def spring_endpoints(particles, edges, coordinates, springs):
    """Material spring endpoints, exactly as eval_cut_springs interpolates them.

    These are NOT the moving knife contact-search points. Both spring sides use
    the stored edge direction and cut_edge_coords without reversing either side.
    Global batched particle/edge indices are supported.
    """
    edge = edges.long()[springs.long()]
    u = coordinates[springs.long()][..., None]
    return (1 - u) * particles[edge[..., 0]] + u * particles[edge[..., 1]]



def components(count, tets):
    """Connected components of the particles through shared tetrahedra."""
    parent = np.arange(count)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for tet in tets:
        root = find(tet[0])
        for other in tet[1:]:
            parent[find(other)] = root
    return np.array([find(i) for i in range(count)])


def knife_summary(r, press_end_time):
    """Height relative to the initial settled bottom, not to the ground plane."""
    press = [x for x in r if x['phase'] == 'press']
    final_cmd = press[-1]['commanded_edge_vs_initial_bottom_mm']
    # Arrival is entry into a symmetric tolerance band; deeper overshoot alone
    # is not arrival. Only the post-command hold is eligible.
    reach = next((x['sim_time_s'] for x in r
                  if x['phase'] in ('press', 'hold') and x['sim_time_s'] >= press_end_time
                  and abs(x['edge_vs_initial_bottom_mm'] - final_cmd) <= .5), None)
    lag = [x['edge_vs_initial_bottom_mm'] - x['commanded_edge_vs_initial_bottom_mm'] for x in press]
    return {'commanded_final_edge_vs_initial_bottom_mm': final_cmd,
            'edge_at_press_end_vs_initial_bottom_mm': press[-1]['edge_vs_initial_bottom_mm'],
            'lowest_edge_vs_initial_bottom_mm': min(x['edge_vs_initial_bottom_mm'] for x in r),
            'lowest_edge_vs_current_bottom_mm': min(x['edge_vs_current_bottom_mm'] for x in r),
            'lowest_edge_world_y_mm': min(x['edge_world_y_mm'] for x in r),
            'max_omitted_rotation_error_mm': max(abs(x['legacy_height_error_mm']) for x in r),
            'press_end_time_s': press_end_time,
            'time_edge_within_0p5mm_of_commanded_final_s': reach,
            'tracking_lag_mm_mean_during_press': sum(lag) / len(lag),
            'tracking_lag_mm_max_during_press': max(lag)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--young-scales', type=float, nargs='+', default=[1., 2.])
    parser.add_argument('--down-speed', type=float, default=.03)
    parser.add_argument('--saw', default='0:0', help='amplitude(m):frequency(Hz) during the press')
    parser.add_argument('--rest-seconds', type=float, default=.25)
    parser.add_argument('--cut-spring-scales', type=float, nargs='+', help='per world; default 1')
    parser.add_argument('--limits', type=float, nargs=6, default=[50., 50., 50., 5., 5., 5.],
                        help='sensor limits used only to report utilisation (old S0 simulation values)')
    parser.add_argument('--hold-seconds', type=float, default=.5)
    parser.add_argument('--retract', type=float, default=.03, help='upward retraction after the hold (m)')
    parser.add_argument('--sim-dt', type=float)
    parser.add_argument('--settle-seconds', type=float, default=.75,
                        help='settling with each world\'s material before the episode; 0 keeps the export')
    parser.add_argument('--friction-velocity', type=float, default=.01)
    parser.add_argument('--trace-damage', action='store_true',
                        help='Read-only observation of every material damage launch; slower')
    parser.add_argument('--wall-seconds', type=float, default=1800.,
                        help='Collection wall budget, checked between control steps')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if not np.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        parser.error('--wall-seconds must be positive and finite')
    import math
    from cutting_gpu_backend.reference import KnifeTrajectory
    from cutting_world_model.gpu_env import GPUEnvironment
    args.output.mkdir(parents=True, exist_ok=False)
    n = len(args.young_scales)
    env = GPUEnvironment(str(args.backend_root), n, scene=args.scene, sim_dt=args.sim_dt,
                         force_model='net', friction_velocity=args.friction_velocity)
    physics, material = env.physics, env.physics.material
    physics.force_limits = torch.full_like(physics.force_limits, 1e6)
    model = material.model
    scales = {k: v.clone() for k, v in material.material_scales.items()}
    scales['youngs_modulus'] = torch.tensor(args.young_scales, device=env.device).to(scales['youngs_modulus'])
    if args.cut_spring_scales:
        scales['cut_spring_stiffness'] = torch.tensor(args.cut_spring_scales, device=env.device).to(
            scales['cut_spring_stiffness'])
    every = torch.ones(n, dtype=torch.bool, device=env.device)
    material.set_material_scales(every, **scales)
    frame = env.settle(args.settle_seconds) if args.settle_seconds > 0 else env.reset(every)

    # --- mesh connectivity of one world
    particles = model.particles_per_world
    tets = model.tet_indices.reshape(-1, 4).cpu().numpy()
    tets = tets[(tets < particles).all(1)]
    label = components(particles, tets)
    names, sizes = np.unique(label, return_counts=True)
    order = np.argsort(-sizes)
    springs = model.cut_spring_indices.reshape(n, -1, 2)[0].cpu().numpy()
    edges = model.cut_edge_indices.reshape(-1, 2).cpu().numpy()
    left = label[edges[springs[:, 0], 0]]
    right = label[edges[springs[:, 1], 0]]
    static = int((model.particle_inv_mass.reshape(n, -1)[0] == 0).sum())
    connectivity = {'particles': int(particles), 'static_particles': static, 'tetrahedra': int(len(tets)),
                    'components': int(len(names)), 'component_sizes': sizes[order].tolist(),
                    'cut_springs': int(len(springs)),
                    'springs_joining_two_components': int((left != right).sum())}
    print('connectivity', json.dumps(connectivity), flush=True)
    side = torch.tensor(np.stack([label == names[order[k]] for k in range(min(2, len(names)))]), device=env.device)

    # --- knife motion: rest, press (optionally sawing), hold, retract, hold
    dt = env.metadata['environment']['agent_control_dt']
    distance = float((frame['observation'][0, 1] * env.heads.max_distance[1]).abs())
    amplitude, frequency = (float(x) for x in args.saw.split(':'))
    phases = [('rest', args.rest_seconds, 0.), ('press', distance / args.down_speed, -args.down_speed),
              ('hold', args.hold_seconds, 0.), ('retract', args.retract / args.down_speed, args.down_speed),
              ('after', args.hold_seconds, 0.)]
    times, velocity, phase_of = [0.], [], []
    for name, duration, vy in phases:
        for _ in range(int(round(duration / dt))):
            t0 = times[-1]
            t1 = t0 + dt
            vz = 0.
            if name == 'press' and amplitude > 0:
                w = 2 * math.pi * frequency
                vz = amplitude * (math.sin(w * t1) - math.sin(w * t0)) / dt
            times.append(t1)
            velocity.append([0., vy, vz])
            phase_of.append(name)
    physics.set_reference_trajectory(KnifeTrajectory(times, velocity, {'kind': 'free_object_check'}))

    def state():
        q = material.state.particle_q.reshape(n, -1, 3)
        return q, material.state.particle_qd.reshape(n, -1, 3)

    q0, _ = state()
    q0 = q0.clone()
    knife_local = model.knife_tri_vertices.reshape(-1, 3)
    knife_low = float(knife_local[:, 1].min())
    # Geometric selection of this asset's bottom edge, not a task/success threshold.
    edge_mask = torch.isclose(knife_local[:, 1], knife_local[:, 1].min(), atol=1e-7, rtol=0)
    knife0 = transformed_knife_vertices(knife_local, material.position, material.rotation)
    edge0 = knife0[:, edge_mask, 1].min(1).values.double().clone()
    initial_bottom = q0[:, :, 1].min(1).values.double()
    all_edges = model.cut_edge_indices.reshape(-1, 2)
    all_springs = model.cut_spring_indices.reshape(-1, 2)
    edge_coords = model.cut_edge_coords.reshape(-1)
    geometry = {k: [] for k in ('time_s', 'phase', 'knife_position', 'knife_rotation',
                               'spring_endpoints', 'spring_ke', 'object_bottom_y')}
    phase_codes = {name: i for i, (name, _, _) in enumerate(phases)}
    press_end_time = (int(round(args.rest_seconds / dt)) + int(round((distance / args.down_speed) / dt))) * dt
    total = int(frame['cut_spring_ke'].shape[-1])
    zero = torch.zeros(n, 6, device=env.device)
    rows = [[] for _ in range(n)]
    limits = torch.tensor(args.limits, device=env.device, dtype=torch.float64)
    peak_net = torch.zeros(n, device=env.device, dtype=torch.float64)
    peak_sensor = torch.zeros(n, 6, device=env.device, dtype=torch.float64)
    step_net = torch.zeros(n, device=env.device, dtype=torch.float64)
    step_util = torch.zeros(n, device=env.device, dtype=torch.float64)
    first_released = [None] * n

    def sample(p, active):
        # Positions, stiffness and knife pose all come from the SAME post-integrator state.
        endpoints = spring_endpoints(p.material.state.particle_q, all_edges, edge_coords,
                                     all_springs).reshape(n, -1, 2, 3)
        for key, value in {
                'time_s': p.sim_time, 'knife_position': p.material.position,
                'knife_rotation': p.material.rotation, 'spring_endpoints': endpoints,
                'spring_ke': p.material.state_field('cut_spring_ke'),
                'object_bottom_y': p.material.state.particle_q.reshape(n, -1, 3)[:, :, 1].min(1).values
        }.items():
            geometry[key].append(value.detach().cpu().numpy().copy())
        geometry['phase'].append(phase_codes[name])
        net = p.material.knife_wrench()[:, :3].double().norm(dim=-1)
        sensor = p.feedback['observation_wrench'].abs().double()
        peak_net.copy_(torch.maximum(peak_net, net))
        peak_sensor.copy_(torch.maximum(peak_sensor, sensor))
        step_net.copy_(torch.maximum(step_net, net))
        step_util.copy_(torch.maximum(step_util, (sensor / limits).amax(-1)))
        done = (p.material.state_field('cut_spring_ke') == 0).all(-1)
        for w in torch.nonzero(done).flatten().tolist():
            if first_released[w] is None:
                first_released[w] = {'sim_time_s': float(p.sim_time[w]), 'phase': name, 'step': step}

    damage_trace = None
    if args.trace_damage:
        from cut_damage_trace import DamageTrace
        damage_trace = DamageTrace(material)
    with damage_trace if damage_trace is not None else nullcontext():
        for step, name in enumerate(phase_of, 1):
            if time.monotonic() - started > args.wall_seconds:
                raise TimeoutError('Platform-check wall budget exceeded; no complete report written')
            if damage_trace is not None:
                damage_trace.phase = name
            env.needs_reset.zero_()
            step_net.zero_()
            step_util.zero_()
            frame, reward, terminated, truncated, info = env.step(zero, reference=True, sample_callback=sample)
            if damage_trace is not None:
                damage_trace.snapshot(physics.sim_time)
            q, qd = state()
            net = material.knife_wrench()
            released = (frame['cut_spring_ke'] == 0).sum(-1)
            knife_world = transformed_knife_vertices(knife_local, material.position, material.rotation)
            edge_y = knife_world[:, edge_mask, 1].min(1).values.double()
            current_bottom = q[:, :, 1].min(1).values.double()
            commanded_y = edge0 + physics.reference_trajectory.displacement(physics.sim_time.double())[:, 1]
            legacy_y = material.position[:, 1].double() + knife_low
            mean_velocity = qd.mean(1)
            rms_speed = qd.square().sum(-1).mean(1).sqrt()

            for w in range(n):
                shift = [(q[w][side[k]].mean(0) - q0[w][side[k]].mean(0)).tolist() for k in range(side.shape[0])]
                rows[w].append({'step': step, 'phase': name, 'released': int(released[w]),
                                'net_force_n': float(net[w, :3].norm()), 'net_vertical_n': float(net[w, 1]),
                                'step_peak_net_force_n': float(step_net[w]), 'step_peak_utilisation': float(step_util[w]),
                                'min_particle_y_mm': float(q[w][:, 1].min()) * 1000,
                                'sim_time_s': float(physics.sim_time[w]),
                                'edge_world_y_mm': 1000 * float(edge_y[w]),
                                'knife_mesh_min_world_y_mm': 1000 * float(knife_world[w, :, 1].min()),
                                'edge_vs_initial_bottom_mm': 1000 * float(edge_y[w] - initial_bottom[w]),
                                'edge_vs_current_bottom_mm': 1000 * float(edge_y[w] - current_bottom[w]),
                                'commanded_edge_vs_initial_bottom_mm': 1000 * float(commanded_y[w] - initial_bottom[w]),
                                'legacy_height_error_mm': 1000 * float(legacy_y[w] - edge_y[w]),
                                'knife_position_m': material.position[w].tolist(),
                                'knife_rotation_xyzw': material.rotation[w].tolist(),
                                'mean_particle_velocity_mps': mean_velocity[w].tolist(),
                                'rms_particle_speed_mps': float(rms_speed[w]),
                                'max_particle_speed_mps': float(qd[w].norm(dim=-1).max()),
                                'side_shift_mm': [[round(1000 * x, 3) for x in s] for s in shift],
                                'finite': bool(torch.isfinite(q[w]).all())})
    # Raw geometry is retained for spatial review without repeating expensive physics.
    np.savez_compressed(args.output / 'geometry.npz',
                        **{k: np.asarray(v) for k, v in geometry.items()},
                        knife_local_vertices=knife_local.cpu().numpy(),
                        edge_vertex_indices=torch.nonzero(edge_mask).flatten().cpu().numpy(),
                        initial_object_bottom_y=initial_bottom.cpu().numpy(),
                        initial_particle_positions=q0.cpu().numpy(),
                        phase_names=np.asarray(list(phase_codes)))
    report = {'measurement_revision': 'knife_pose_geometry_v2',
              'geometry_file': 'geometry.npz',
              'geometry_sampling': '2 ms, post integration; heights and spring endpoints in DiSECt world',
              'height_reference': 'initial settled object bottom and instantaneous object bottom, recorded separately',
              'command_reference': 'initial actual edge translated by the reference trajectory; fixed initial orientation',
              'seconds': time.monotonic() - started, 'git': git_revision(ROOT), 'command': sys.argv, 'scene': args.scene, 'connectivity': connectivity,
              'total_springs': total, 'force_limit_termination': 'disabled (measurement)',
              'sampling': '2 ms coupling steps for peaks; step-end values in series', 'limits': args.limits,
              'settle_seconds': args.settle_seconds, 'worlds': []}
    if damage_trace is not None:
        report['damage_trace'] = damage_trace.save(args.output / 'damage_trace.npz')
    for w in range(n):
        r = rows[w]

        def last(name):
            return [x for x in r if x['phase'] == name][-1]

        rest, press, end = last('rest'), last('press'), r[-1]
        intact = frame['cut_spring_ke'][w] > 0
        summary = {
            'young_scale': args.young_scales[w], 'finite': all(x['finite'] for x in r),
            'rest': {'side_shift_mm': rest['side_shift_mm'], 'min_particle_y_mm': rest['min_particle_y_mm'],
                     'max_particle_speed_mps': rest['max_particle_speed_mps'],
                     'rms_particle_speed_mps': rest['rms_particle_speed_mps'],
                     'mean_particle_velocity_mps': rest['mean_particle_velocity_mps'],
                     'net_force_n': rest['net_force_n']},
            'press': {'released_at_end': press['released'],
                      'step_end_max_net_force_n': max(x['net_force_n'] for x in r),
                      'side_shift_mm': press['side_shift_mm']},
            'end': {'released': end['released'], 'all_released': end['released'] == total,
                    'intact_springs': int(intact.sum()), 'side_shift_mm': end['side_shift_mm'],
                    'net_force_n': end['net_force_n']},
            'first_all_released': first_released[w],
            'knife': knife_summary(r, press_end_time),
            'remaining_connections': [
                {'spring': int(i), 'stiffness': float(frame['cut_spring_ke'][w, i]),
                 'endpoints_world_m': geometry['spring_endpoints'][-1][w, i].tolist()}
                for i in torch.nonzero(intact).flatten().tolist()],
            'peak_net_force_n_2ms': float(peak_net[w]),
            'peak_sensor_force_component_n_2ms': float(peak_sensor[w, :3].max()),
            'peak_sensor_torque_component_nm_2ms': float(peak_sensor[w, 3:].max()),
            'peak_limit_utilisation_2ms': float((peak_sensor[w] / limits).max()),
            'cut_spring_scale': (args.cut_spring_scales or [1.] * n)[w],
            'series': r}
        report['worlds'].append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'series'}), flush=True)
    (args.output / 'report.json').write_text(json.dumps(report) + '\n')


if __name__ == '__main__':
    main()
