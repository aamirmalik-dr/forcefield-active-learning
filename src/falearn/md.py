"""From-scratch molecular dynamics driven by the committee mean.

Velocity-Verlet NVE (used for the energy-drift stability check) and a BAOAB
Langevin thermostat (used for ensemble-driven candidate exploration). Forces
come from the committee-mean potential; the neighbor list is rebuilt every
step. No ASE dynamics code is used anywhere.
"""

from __future__ import annotations

import numpy as np

from falearn.data import Frame
from falearn.descriptors import AcsfParams, frame_graph
from falearn.ensemble import mean_energy_forces
from falearn.model import MemberPotential

MASSES_AMU = np.array([47.867, 91.224, 92.906])  # Ti, Zr, Nb
EV_PER_A_AMU = 9.6485332e-3  # 1 eV/(Angstrom*amu) in Angstrom/fs^2
AMU_A2_FS2_TO_EV = 103.642696  # 1 amu*(Angstrom/fs)^2 in eV
KB_EV = 8.617333e-5  # Boltzmann constant, eV/K


def _forces_fn(members: list[MemberPotential], acsf: AcsfParams):
    def compute(frame: Frame) -> tuple[float, np.ndarray]:
        graph = frame_graph(frame, acsf)
        energies, forces = mean_energy_forces(members, graph)
        return float(energies[0]), forces.numpy()

    return compute


def maxwell_velocities(
    species: np.ndarray, temperature_k: float, rng: np.random.Generator
) -> np.ndarray:
    """Maxwell-Boltzmann velocities in Angstrom/fs, COM motion removed."""
    masses = MASSES_AMU[species]
    sigma = np.sqrt(KB_EV * temperature_k / (masses * AMU_A2_FS2_TO_EV))
    vel = rng.normal(size=(len(species), 3)) * sigma[:, None]
    mom = (masses[:, None] * vel).sum(axis=0)
    return vel - mom / masses.sum()


def kinetic_energy(species: np.ndarray, velocities: np.ndarray) -> float:
    """Kinetic energy in eV."""
    masses = MASSES_AMU[species]
    return float(0.5 * (masses[:, None] * velocities**2).sum() * AMU_A2_FS2_TO_EV)


def nve_run(
    frame: Frame,
    members: list[MemberPotential],
    acsf: AcsfParams | None = None,
    temperature_k: float = 300.0,
    n_steps: int = 1500,
    dt_fs: float = 2.0,
    sample_every: int = 10,
    seed: int = 0,
    forces_fn=None,
) -> dict:
    """Velocity-Verlet NVE run recording total-energy drift.

    Args:
        frame: Starting structure (not modified).
        members: Committee whose mean drives the dynamics.
        acsf: Descriptor parameters.
        temperature_k: Initial Maxwell-Boltzmann temperature.
        n_steps: Number of integration steps.
        dt_fs: Timestep in femtoseconds.
        sample_every: Energy recording stride.
        seed: Velocity seed.
        forces_fn: Override for the force engine (used by tests with an
            analytic potential); signature frame -> (energy, forces).

    Returns:
        Dict with times (ps), total/potential/kinetic energies (eV),
        instantaneous temperature (K), and the drift rate in meV/atom/ps
        from a linear fit of total energy over time.
    """
    compute = forces_fn or _forces_fn(members, acsf or AcsfParams())
    pos = frame.positions.copy()
    vel = maxwell_velocities(frame.species, temperature_k, np.random.default_rng(seed))
    masses = MASSES_AMU[frame.species][:, None]
    work = Frame(
        species=frame.species, positions=pos, cell=frame.cell, composition=frame.composition
    )
    epot, forces = compute(work)
    times, e_tot, e_pot, e_kin, temps = [], [], [], [], []
    n_atoms = frame.n_atoms
    for step in range(n_steps + 1):
        if step % sample_every == 0:
            ekin = kinetic_energy(frame.species, vel)
            times.append(step * dt_fs / 1000.0)
            e_pot.append(epot)
            e_kin.append(ekin)
            e_tot.append(epot + ekin)
            temps.append(2.0 * ekin / (3.0 * n_atoms * KB_EV))
        if step == n_steps:
            break
        acc = forces / masses * EV_PER_A_AMU
        vel = vel + 0.5 * dt_fs * acc
        pos = pos + dt_fs * vel
        work.positions = pos
        epot, forces = compute(work)
        acc = forces / masses * EV_PER_A_AMU
        vel = vel + 0.5 * dt_fs * acc
    coeffs = np.polyfit(times, e_tot, 1)
    return {
        "time_ps": times,
        "e_total_ev": e_tot,
        "e_potential_ev": e_pot,
        "e_kinetic_ev": e_kin,
        "temperature_k": temps,
        "drift_mev_per_atom_per_ps": float(coeffs[0] * 1000.0 / n_atoms),
        "dt_fs": dt_fs,
        "n_steps": n_steps,
        "n_atoms": n_atoms,
    }


def langevin_snapshots(
    frame: Frame,
    members: list[MemberPotential],
    acsf: AcsfParams | None = None,
    temperature_k: float = 1200.0,
    n_steps: int = 120,
    dt_fs: float = 3.0,
    sample_every: int = 30,
    friction_per_fs: float = 0.02,
    seed: int = 0,
) -> list[Frame]:
    """BAOAB Langevin dynamics returning periodic snapshot frames.

    The committee mean drives the trajectory; snapshots become unlabeled
    candidates, so exploration itself costs zero teacher calls.
    """
    rng = np.random.default_rng(seed)
    compute = _forces_fn(members, acsf or AcsfParams())
    pos = frame.positions.copy()
    vel = maxwell_velocities(frame.species, temperature_k, rng)
    masses = MASSES_AMU[frame.species][:, None]
    work = Frame(
        species=frame.species, positions=pos, cell=frame.cell, composition=frame.composition
    )
    _, forces = compute(work)
    c1 = np.exp(-friction_per_fs * dt_fs)
    sigma = np.sqrt(KB_EV * temperature_k / (MASSES_AMU[frame.species][:, None] * AMU_A2_FS2_TO_EV))
    snaps: list[Frame] = []
    for step in range(1, n_steps + 1):
        vel = vel + 0.5 * dt_fs * forces / masses * EV_PER_A_AMU
        pos = pos + 0.5 * dt_fs * vel
        vel = c1 * vel + np.sqrt(1.0 - c1 * c1) * sigma * rng.normal(size=vel.shape)
        pos = pos + 0.5 * dt_fs * vel
        work.positions = pos
        _, forces = compute(work)
        vel = vel + 0.5 * dt_fs * forces / masses * EV_PER_A_AMU
        if step % sample_every == 0:
            snaps.append(
                Frame(
                    species=frame.species.copy(),
                    positions=pos.copy(),
                    cell=frame.cell.copy(),
                    composition=frame.composition,
                )
            )
    return snaps
