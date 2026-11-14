"""Finite knife geometry, swept cut-interface witnesses and a nonpenetrating stop.

The stop is an execution interlock, not a board-force simulator. Coverage uses the
moving cut-interface midpoints, independent of spring stiffness/damage.
"""
import torch
from .controller.fdcc import quat_apply, quat_mul


class KnifeBoard:
    def __init__(self, material, config, enforce=True):
        self.enforce = enforce
        self.material = material
        self.config = dict(config)
        self.y = float(config['height_m'])
        self.stop_gap = float(config['stop_gap_m'])
        self.tolerance = float(config.get('coverage_tolerance_m', .0005))
        self.vertices = material.model.knife_tri_vertices.reshape(-1, 3).to(material.position)
        # pack_model repeats knife vertices for worlds; one knife geometry suffices.
        self.vertices = torch.unique(self.vertices, dim=0)
        lowest = self.vertices[:, 1].min()
        edge = self.vertices[(self.vertices[:, 1] - lowest).abs() < 1e-7]
        self.edge = torch.stack((edge[edge[:, 2].argmin()], edge[edge[:, 2].argmax()]))
        if float((self.edge[1] - self.edge[0]).norm()) <= 0:
            raise ValueError('Knife needs a finite leading edge')
        self.covered = torch.zeros_like(material.state_field('cut_spring_ke'), dtype=torch.bool)
        self.contacted = torch.zeros(material.worlds, device=material.position.device, dtype=torch.bool)
        self.min_gap = torch.full_like(material.position[:, 0], float('inf'))
        self.last_mid = self.midpoints().clone()
        self.last_footprint = self.footprint().clone()

    def transformed(self, position, rotation, vertices=None):
        v = self.vertices if vertices is None else vertices
        return position[:, None] + quat_apply(rotation[:, None].expand(-1, len(v), -1),
                                               v[None].expand(len(position), -1, -1))

    def gap(self, position=None, rotation=None):
        m = self.material
        return self.transformed(m.position if position is None else position,
                                m.rotation if rotation is None else rotation)[:, :, 1].amin(-1) - self.y

    def midpoints(self):
        m = self.material
        edges = m.model.cut_edge_indices.reshape(-1, 2).long()
        u = m.model.cut_edge_coords.reshape(-1, 1)
        x = m.state.particle_q.reshape(-1, 3)
        points = (1-u)*x[edges[:, 0]] + u*x[edges[:, 1]]
        pairs = m.model.cut_spring_indices.reshape(-1, 2).long()
        return points[pairs].mean(1).reshape(m.worlds, -1, 3)

    def footprint(self):
        particles = self.material.state_field('particle_q')
        return torch.stack((particles[:, :, 0].amin(-1), particles[:, :, 0].amax(-1)), -1)

    def reset(self, mask):
        self.covered[mask] = False
        self.contacted[mask] = False
        self.min_gap[mask] = self.gap()[mask]
        self.last_mid[mask] = self.midpoints()[mask]
        self.last_footprint[mask] = self.footprint()[mask]
        if bool((self.min_gap[mask] < self.stop_gap).any()):
            raise ValueError('Knife starts below board stop plane')

    def fraction(self, twist, duration, active):
        """Conservative continuous arc bound; never integrates an unsafe trial.

        Rotation curvature bound r*theta^2/8 bounds the deviation from the
        endpoint chord for each vertex over the exchange interval.
        """
        m = self.material
        twist = twist.to(m.position)
        omega = twist[:, 3:].norm(dim=-1)
        radius = self.vertices.norm(dim=-1).max()
        start = self.gap()
        def bound(f):
            dt = duration * f
            half = omega * dt / 2
            inc = torch.cat((twist[:, 3:] * (torch.sinc(half/torch.pi)*dt/2)[:, None],
                             torch.cos(half)[:, None]), -1)
            q = quat_mul(inc, m.rotation)
            end = self.gap(m.position + twist[:, :3]*dt[:, None], q)
            return torch.minimum(start, end) - radius*(omega*dt).square()/8
        full = torch.ones_like(start)
        hit = active & (bound(full) <= self.stop_gap)
        lo, hi = torch.zeros_like(start), full.clone()
        if bool(hit.any()):
            for _ in range(24):
                mid = (lo+hi)/2
                safe = bound(mid) > self.stop_gap
                lo = torch.where(safe, mid, lo)
                hi = torch.where(safe, hi, mid)
        return torch.where(hit, lo, full), hit

    def signed_height(self, position, rotation, points, footprint):
        edge = self.transformed(position, rotation, self.edge)
        a, b = edge[:, 0], edge[:, 1]
        dz = b[:, 2] - a[:, 2]
        valid_edge = dz.abs() > 1e-6
        u = (points[:, :, 2] - a[:, None, 2]) / torch.where(valid_edge, dz, torch.ones_like(dz))[:, None]
        line = a[:, None] + u[:, :, None]*(b-a)[:, None]
        # Coverage is of the projected intended section, not a demand that a
        # deforming free object keep its original cut plane at x=0. Reject a
        # knife outside the object's actual transverse footprint without a
        # fixed 'allowed sliding' threshold.
        xmin, xmax = footprint.unbind(-1)
        inside = ((u >= 0) & (u <= 1) & valid_edge[:, None]
                  & (line[:, :, 0] >= xmin[:, None]) & (line[:, :, 0] <= xmax[:, None]))
        # No deleted witnesses at the board: the same tolerance covers penalty-contact
        # penetration of the material, and it is reported as a geometric approximation.
        return line[:, :, 1]-points[:, :, 1], inside

    def observe(self, old_position, old_rotation, active):
        mid = self.midpoints()
        footprint = self.footprint()
        before, inside0 = self.signed_height(old_position, old_rotation, self.last_mid, self.last_footprint)
        after, inside1 = self.signed_height(self.material.position, self.material.rotation, mid, footprint)
        crossed = ((before >= -self.tolerance) & (after <= self.tolerance)
                   & (after < before) & inside0 & inside1)
        self.covered |= crossed & active[:, None]
        self.last_mid[active] = mid[active]
        self.last_footprint[active] = footprint[active]
        gap = self.gap()
        self.min_gap = torch.where(active, torch.minimum(self.min_gap, gap), self.min_gap)
        if self.enforce and bool((gap[active] < -2e-6).any()):
            raise FloatingPointError('Knife penetrated board despite execution interlock')

    def register_stop(self, limited):
        # A conservative rotation bound can shorten a step before actual contact.
        # Only actual clearance <=10 um counts as board arrival; otherwise the
        # next safe interval may continue approaching.
        self.contacted |= limited & (self.gap() <= 1e-5)

    def frame(self):
        return {'knife_board_contact': self.contacted[:, None].clone(),
                'knife_board_gap': self.gap()[:, None],
                'knife_min_board_gap': self.min_gap[:, None].clone(),
                'knife_path_progress': self.covered.double().mean(-1, keepdim=True),
                'knife_path_complete': self.covered.all(-1)[:, None]}
