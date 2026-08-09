"""Descriptor invariances, smoothness, and force correctness."""

from __future__ import annotations

import numpy as np
import torch

from falearn.data import Frame
from falearn.descriptors import compute_descriptors, frame_graph, merge_graphs
from falearn.model import MemberPotential


def _descriptors_of(frame, acsf):
    graph = frame_graph(frame, acsf)
    return compute_descriptors(graph.positions, graph, acsf).numpy()


def test_feature_count(small_acsf):
    assert small_acsf.n_features == 3 * 8 + 6 == 30


def test_translation_invariance(tizrnb_frame, small_acsf):
    d0 = _descriptors_of(tizrnb_frame, small_acsf)
    moved = Frame(
        species=tizrnb_frame.species,
        positions=tizrnb_frame.positions + np.array([1.3, -0.7, 2.1]),
        cell=tizrnb_frame.cell,
    )
    d1 = _descriptors_of(moved, small_acsf)
    assert np.allclose(d0, d1, atol=1e-10)


def test_rotation_invariance(tizrnb_frame, small_acsf):
    d0 = _descriptors_of(tizrnb_frame, small_acsf)
    theta = 0.4
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = Frame(
        species=tizrnb_frame.species,
        positions=tizrnb_frame.positions @ rot.T,
        cell=tizrnb_frame.cell @ rot.T,
    )
    d1 = _descriptors_of(rotated, small_acsf)
    assert np.allclose(d0, d1, atol=1e-9)


def test_permutation_equivariance(tizrnb_frame, small_acsf):
    d0 = _descriptors_of(tizrnb_frame, small_acsf)
    perm = np.random.default_rng(0).permutation(tizrnb_frame.n_atoms)
    permuted = Frame(
        species=tizrnb_frame.species[perm],
        positions=tizrnb_frame.positions[perm],
        cell=tizrnb_frame.cell,
    )
    d1 = _descriptors_of(permuted, small_acsf)
    assert np.allclose(d0[perm], d1, atol=1e-10)


def test_beyond_cutoff_atom_has_no_effect(small_acsf):
    cell = np.eye(3) * 30.0
    base = np.array([[10.0, 10.0, 10.0], [12.0, 10.0, 10.0]])
    # the third atom stays beyond the cutoff of BOTH base atoms in both cases
    far_a = np.vstack([base, [12.0 + small_acsf.rc_radial + 0.5, 10.0, 10.0]])
    far_b = np.vstack([base, [12.0 + small_acsf.rc_radial + 3.0, 10.0, 10.0]])
    species = np.array([0, 1, 2])
    d_a = _descriptors_of(Frame(species=species, positions=far_a, cell=cell), small_acsf)
    d_b = _descriptors_of(Frame(species=species, positions=far_b, cell=cell), small_acsf)
    assert np.allclose(d_a[:2], d_b[:2], atol=1e-12)


def test_cutoff_smoothness_of_energy(small_acsf):
    # sliding a neighbor across the cutoff changes the energy continuously
    member = MemberPotential(small_acsf, hidden=(8,), seed=1)
    cell = np.eye(3) * 30.0
    energies = []
    for extra in np.linspace(-0.05, 0.05, 11):
        pos = np.array([[10.0, 10.0, 10.0], [10.0 + small_acsf.rc_radial + extra, 10.0, 10.0]])
        fr = Frame(species=np.array([0, 1]), positions=pos, cell=cell)
        graph = frame_graph(fr, small_acsf)
        e, _ = member.energy_forces(graph)
        energies.append(float(e[0].detach()))
    steps = np.abs(np.diff(energies))
    assert steps.max() < 5e-4  # no jump at the cutoff crossing


def test_batching_matches_single(tizrnb_frame, small_acsf):
    other = Frame(
        species=tizrnb_frame.species[::-1].copy(),
        positions=tizrnb_frame.positions + 0.05,
        cell=tizrnb_frame.cell,
    )
    g1 = frame_graph(tizrnb_frame, small_acsf)
    g2 = frame_graph(other, small_acsf)
    merged = merge_graphs([g1, g2])
    d_merged = compute_descriptors(merged.positions, merged, small_acsf).numpy()
    d1 = compute_descriptors(g1.positions, g1, small_acsf).numpy()
    d2 = compute_descriptors(g2.positions, g2, small_acsf).numpy()
    assert np.allclose(d_merged, np.vstack([d1, d2]), atol=1e-12)


def test_descriptors_differentiable(tizrnb_frame, small_acsf):
    graph = frame_graph(tizrnb_frame, small_acsf)
    pos = graph.positions.detach().clone().requires_grad_(True)
    desc = compute_descriptors(pos, graph, small_acsf)
    desc.sum().backward()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
