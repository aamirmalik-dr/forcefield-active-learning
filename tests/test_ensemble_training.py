"""Committee math, training behavior, and the reference-energy fit."""

from __future__ import annotations

import numpy as np
import torch

from falearn.descriptors import frame_graph
from falearn.ensemble import committee_forward, committee_predict, evaluate_mean, train_committee
from falearn.model import MemberPotential
from falearn.training import TrainSettings, fit_reference, make_batches


def _members(acsf, n=3):
    out = []
    for k in range(n):
        m = MemberPotential(acsf, hidden=(8,), seed=k)
        m.set_reference(np.array([-5.0, -6.0, -7.0]))
        m.eval()
        out.append(m)
    return out


def test_disagreement_matches_hand_computation(tizrnb_frame, small_acsf):
    members = _members(small_acsf)
    graph = frame_graph(tizrnb_frame, small_acsf)
    pred = committee_predict(members, [graph], chunk_size=4)
    e_all, f_all = committee_forward(members, graph)
    f_bar = f_all.mean(dim=0)
    sigma_atom = torch.sqrt(((f_all - f_bar[None]) ** 2).sum(dim=2).mean(dim=0))
    assert np.isclose(pred["disagreement_mean"][0], float(sigma_atom.mean()), atol=1e-12)
    assert np.isclose(pred["disagreement_max"][0], float(sigma_atom.max()), atol=1e-12)
    assert np.isclose(pred["e_mean"][0], float(e_all.mean()), atol=1e-12)


def test_chunking_does_not_change_predictions(labeled_batches, small_acsf):
    members = _members(small_acsf)
    graphs = [frame_graph(fr, small_acsf) for fr in labeled_batches[:6]]
    a = committee_predict(members, graphs, chunk_size=2)
    b = committee_predict(members, graphs, chunk_size=6)
    assert np.allclose(a["disagreement_mean"], b["disagreement_mean"], atol=1e-12)
    assert np.allclose(a["e_mean"], b["e_mean"], atol=1e-12)


def test_identical_members_have_zero_disagreement(tizrnb_frame, small_acsf):
    m = MemberPotential(small_acsf, hidden=(8,), seed=4)
    m.set_reference(np.array([-5.0, -6.0, -7.0]))
    m.eval()
    graph = frame_graph(tizrnb_frame, small_acsf)
    pred = committee_predict([m, m, m], [graph])
    assert pred["disagreement_mean"][0] < 1e-14
    assert pred["e_std_per_atom"][0] < 1e-14


def test_fit_reference_recovers_linear_energies():
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 10, size=(20, 3)).astype(float)
    mu_true = np.array([-4.9, -6.2, -7.1])
    energies = counts @ mu_true
    mu = fit_reference(counts, energies)
    assert np.allclose(mu, mu_true, atol=1e-10)


def test_training_reduces_error(labeled_batches, small_acsf):
    frames = labeled_batches
    graphs = [frame_graph(fr, small_acsf) for fr in frames]
    energies = np.array([fr.energy for fr in frames])
    forces = [fr.forces for fr in frames]
    counts = np.stack([fr.counts() for fr in frames])
    settings = TrainSettings(epochs=15, batch_size=8, lr=3e-3, force_weight=0.1)
    members, hist = train_committee(
        2, graphs, energies, forces, counts, settings, acsf=small_acsf, hidden=(8,), base_seed=0
    )
    metrics = evaluate_mean(members, graphs, energies, forces)
    # the synthetic oracle is exactly representable, training must get close
    assert metrics["f_mae_mev_per_a"] < 120.0
    assert metrics["e_mae_mev_per_atom"] < 40.0
    assert all(h["final_loss"] < 1.0 for h in hist)


def test_effective_epochs_rule():
    s = TrainSettings(epochs=40, target_frame_epochs=12000, max_epochs=120)
    assert s.effective_epochs(60) == 120  # capped
    assert s.effective_epochs(150) == 80
    assert s.effective_epochs(300) == 40  # floor at the base epoch count
    assert s.effective_epochs(1000) == 40
    assert TrainSettings(epochs=40).effective_epochs(60) == 40  # disabled by default


def test_make_batches_covers_all_frames(labeled_batches, small_acsf):
    graphs = [frame_graph(fr, small_acsf) for fr in labeled_batches]
    energies = np.array([fr.energy for fr in labeled_batches])
    forces = [fr.forces for fr in labeled_batches]
    batches = make_batches(graphs, energies, forces, batch_size=7, seed=0)
    total = sum(int(b.graph.n_structs) for b in batches)
    assert total == len(labeled_batches)
    total_atoms = sum(int(b.graph.positions.shape[0]) for b in batches)
    assert total_atoms == sum(fr.n_atoms for fr in labeled_batches)
