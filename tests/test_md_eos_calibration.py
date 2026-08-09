"""Integrator conservation, BM fit recovery, and calibration utilities."""

from __future__ import annotations

import numpy as np

from falearn.calibration import binned_reliability, label_reduction_factor
from falearn.data import Frame
from falearn.eos import birch_murnaghan, fit_bm
from falearn.md import kinetic_energy, maxwell_velocities, nve_run


def _harmonic_forces(k_spring=2.0, r0=2.9):
    """Analytic all-pairs harmonic bonds: a smooth, exactly conservative field."""

    def compute(frame: Frame):
        n = frame.n_atoms
        energy = 0.0
        forces = np.zeros((n, 3))
        for i in range(n):
            for j in range(i + 1, n):
                d = frame.positions[j] - frame.positions[i]
                r = np.linalg.norm(d)
                energy += 0.5 * k_spring * (r - r0) ** 2
                f = -k_spring * (r - r0) * d / r
                forces[j] += f
                forces[i] -= f
        return energy, forces

    return compute


def test_nve_conserves_energy_with_analytic_forces():
    rng = np.random.default_rng(0)
    cell = np.eye(3) * 20.0  # cluster in a big box, no image ambiguity
    pos = np.array(
        [[0.0, 0.0, 0.0], [2.9, 0.0, 0.0], [0.0, 2.9, 0.0], [0.0, 0.0, 2.9]], dtype=float
    )
    pos += 8.0 + rng.normal(0, 0.05, pos.shape)
    fr = Frame(species=np.array([0, 1, 2, 0]), positions=pos, cell=cell)
    run = nve_run(
        fr,
        members=None,
        temperature_k=300.0,
        n_steps=600,
        dt_fs=1.0,
        sample_every=10,
        seed=1,
        forces_fn=_harmonic_forces(),
    )
    assert abs(run["drift_mev_per_atom_per_ps"]) < 0.5
    e = np.array(run["e_total_ev"])
    assert (e.max() - e.min()) / fr.n_atoms < 5e-4  # eV/atom fluctuation band


def test_maxwell_velocities_zero_momentum_and_temperature():
    rng = np.random.default_rng(0)
    species = np.array([0, 1, 2] * 200)
    vel = maxwell_velocities(species, 500.0, rng)
    from falearn.md import KB_EV, MASSES_AMU

    mom = (MASSES_AMU[species][:, None] * vel).sum(axis=0)
    assert np.allclose(mom, 0.0, atol=1e-10)
    ekin = kinetic_energy(species, vel)
    t_inst = 2.0 * ekin / (3.0 * len(species) * KB_EV)
    assert abs(t_inst - 500.0) < 40.0


def test_bm_fit_recovers_synthetic_parameters():
    v0, b0_ev, e0, b0p = 18.0, 0.7, -6.5, 4.2
    v = np.linspace(16.0, 20.5, 15)
    e = birch_murnaghan(v, e0, v0, b0_ev, b0p)
    fit = fit_bm(v, e)
    assert abs(fit["v0_a3_per_atom"] - v0) < 1e-6
    assert abs(fit["b0_gpa"] - b0_ev * 160.21766) < 1e-3
    assert abs(fit["a0_bcc_a"] - (2 * v0) ** (1 / 3)) < 1e-6


def test_binned_reliability_perfect_correlation():
    d = np.linspace(1.0, 100.0, 64)
    e = 2.0 * d
    rel = binned_reliability(d, e, n_bins=4)
    assert rel["spearman_rho"] > 0.999
    assert rel["n_frames"] == 64
    assert all(a < b for a, b in zip(rel["bin_mean_error"], rel["bin_mean_error"][1:]))


def test_label_reduction_factor():
    budgets = np.array([60, 100, 140, 180])
    random_curve = np.array([100.0, 90.0, 80.0, 70.0])
    active_curve = np.array([95.0, 70.0, 60.0, 50.0])
    out = label_reduction_factor(budgets, active_curve, random_curve)
    assert out["reached"]
    assert out["target_error"] == 70.0
    assert 60 < out["active_budget_at_target"] <= 100
    assert out["reduction_factor"] > 1.5
    # an active curve that never reaches the target reports factor 1
    out2 = label_reduction_factor(budgets, random_curve + 10, random_curve)
    assert not out2["reached"]
    assert out2["reduction_factor"] == 1.0
