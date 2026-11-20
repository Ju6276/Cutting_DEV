"""Rigid mass properties; all lengths are metres and inertias are about the COM."""
import numpy as np


def parallel_axis(r):
    r = np.asarray(r, dtype=float)
    return (r @ r) * np.eye(3) - np.outer(r, r)


def validate(mass, com, inertia):
    com, inertia = np.asarray(com, dtype=float), np.asarray(inertia, dtype=float)
    if not np.isfinite(mass) or mass <= 0 or com.shape != (3,) or inertia.shape != (3, 3):
        raise ValueError("Positive mass, COM[3] and inertia[3,3] required")
    if not np.isfinite(com).all() or not np.isfinite(inertia).all():
        raise ValueError("Non-finite mass properties")
    if not np.allclose(inertia, inertia.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Inertia must be symmetric")
    eig = np.linalg.eigvalsh(inertia)
    if eig[0] <= 0 or eig[2] > eig[:2].sum() + 1e-12:
        raise ValueError("Inertia is not physically realizable")
    return dict(mass_kg=float(mass), com_m=com.tolist(), inertia_kg_m2=inertia.tolist())


def combine(parts):
    mass = sum(p["mass_kg"] for p in parts)
    if not parts or not np.isfinite(mass) or mass <= 0:
        raise ValueError("Empty or invalid components")
    for p in parts:
        validate(p["mass_kg"], p["com_m"], p["inertia_kg_m2"])
    com = sum(p["mass_kg"] * np.asarray(p["com_m"]) for p in parts) / mass
    inertia = sum(np.asarray(p["inertia_kg_m2"]) +
                  p["mass_kg"] * parallel_axis(np.asarray(p["com_m"]) - com) for p in parts)
    return validate(mass, com, inertia)


def transform(part, rotation, translation):
    r, t = np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
    if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=1e-10) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("Proper rotation required")
    return validate(part["mass_kg"], r @ np.asarray(part["com_m"]) + t,
                    r @ np.asarray(part["inertia_kg_m2"]) @ r.T)


def mesh_mass(mesh, mass):
    """Uniform volume estimate scaled to supplied mass; never infers internal fill."""
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("A closed, consistently oriented positive-volume mesh is required")
    return validate(mass, mesh.center_mass, mesh.moment_inertia * mass / mesh.volume)

