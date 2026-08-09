"""Equation-of-state physics check: Birch-Murnaghan fit of E(V) scans.

Used to compare the final committee-mean potential against the teacher and
against the real-DFT Materials Project anchors for elemental BCC Ti, Zr, Nb.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from falearn.data import Frame
from falearn.descriptors import AcsfParams, frame_graph
from falearn.ensemble import mean_energy_forces
from falearn.structures import bcc_supercell, vegard_a


def birch_murnaghan(v: np.ndarray, e0: float, v0: float, b0: float, b0p: float) -> np.ndarray:
    """Third-order Birch-Murnaghan energy-volume relation (per atom)."""
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * (
        (eta - 1.0) ** 3 * b0p + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def fit_bm(volumes: np.ndarray, energies: np.ndarray) -> dict:
    """Fit the BM equation; returns a0 (BCC), V0, B0 in GPa, and B0'.

    Volumes and energies are per atom (A^3, eV).
    """
    v = np.asarray(volumes, dtype=float)
    e = np.asarray(energies, dtype=float)
    k0 = v[np.argmin(e)]
    p0 = (float(e.min()), float(k0), 0.6, 4.0)  # B0 in eV/A^3 ~ 96 GPa
    popt, _ = curve_fit(birch_murnaghan, v, e, p0=p0, maxfev=20000)
    e0, v0, b0, b0p = (float(x) for x in popt)
    return {
        "e0_ev_per_atom": e0,
        "v0_a3_per_atom": v0,
        "a0_bcc_a": float((2.0 * v0) ** (1.0 / 3.0)),
        "b0_gpa": b0 * 160.21766,
        "b0_prime": b0p,
    }


def eos_scan_committee(
    members,
    composition: str,
    reps: int = 2,
    n_points: int = 13,
    span: float = 0.06,
    acsf: AcsfParams | None = None,
    seed: int = 0,
) -> dict:
    """E(V) scan of an ideal BCC cell with the committee-mean potential."""
    acsf = acsf or AcsfParams()
    base = bcc_supercell(composition, reps, seed=seed)
    scales = np.linspace(1.0 - span, 1.0 + span, n_points)
    volumes, energies = [], []
    for s in scales:
        fr = Frame(
            species=base.species,
            positions=base.positions * s,
            cell=base.cell * s,
            composition=composition,
        )
        graph = frame_graph(fr, acsf)
        e_mean, _ = mean_energy_forces(members, graph)
        volumes.append(abs(np.linalg.det(fr.cell)) / fr.n_atoms)
        energies.append(float(e_mean[0]) / fr.n_atoms)
    fit = fit_bm(np.array(volumes), np.array(energies))
    fit["volumes_a3_per_atom"] = volumes
    fit["energies_ev_per_atom"] = energies
    fit["a_seed"] = vegard_a(composition)
    return fit


def eos_scan_calculator(
    calc,
    composition: str,
    reps: int = 2,
    n_points: int = 13,
    span: float = 0.06,
    seed: int = 0,
) -> dict:
    """E(V) scan of the same cells with an ASE calculator (the teacher)."""
    from falearn.teacher import frame_to_atoms

    base = bcc_supercell(composition, reps, seed=seed)
    scales = np.linspace(1.0 - span, 1.0 + span, n_points)
    volumes, energies = [], []
    for s in scales:
        fr = Frame(
            species=base.species,
            positions=base.positions * s,
            cell=base.cell * s,
            composition=composition,
        )
        atoms = frame_to_atoms(fr)
        atoms.calc = calc
        volumes.append(abs(np.linalg.det(fr.cell)) / fr.n_atoms)
        energies.append(float(atoms.get_potential_energy()) / fr.n_atoms)
    fit = fit_bm(np.array(volumes), np.array(energies))
    fit["volumes_a3_per_atom"] = volumes
    fit["energies_ev_per_atom"] = energies
    return fit
