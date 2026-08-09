"""Label hiding, purchase accounting, and acquisition selection."""

from __future__ import annotations

import numpy as np
import pytest

from falearn.acquisition import select_random, select_top_disagreement
from falearn.pool import LabeledPool


def test_labels_hidden_until_purchased(labeled_batches):
    pool = LabeledPool(labeled_batches)
    # the public frames carry no labels at all
    assert all(fr.energy is None and fr.forces is None for fr in pool.frames)
    with pytest.raises(KeyError):
        pool.label(0)
    pool.purchase([0])
    energy, forces = pool.label(0)
    assert energy == labeled_batches[0].energy
    assert np.allclose(forces, labeled_batches[0].forces)
    with pytest.raises(KeyError):
        pool.label(1)


def test_double_purchase_rejected(labeled_batches):
    pool = LabeledPool(labeled_batches)
    pool.purchase([0, 1])
    with pytest.raises(ValueError):
        pool.purchase([1])


def test_purchased_labels_order_and_counts(labeled_batches):
    pool = LabeledPool(labeled_batches)
    pool.purchase([3, 1, 5])
    energies, forces, counts = pool.purchased_labels()
    assert pool.purchased == [3, 1, 5]
    assert energies[0] == labeled_batches[3].energy
    assert len(forces) == 3
    assert counts.shape == (3, 3)
    assert pool.n_purchased == 3
    assert len(pool.available()) == len(labeled_batches) - 3


def test_oracle_called_only_for_unlabeled(labeled_batches):
    calls = []

    def oracle(frame):
        calls.append(frame.group)
        return -1.0, np.zeros((frame.n_atoms, 3))

    pool = LabeledPool(labeled_batches, oracle=oracle)
    pool.purchase([0])
    assert pool.oracle_calls == 0  # stored label used, no teacher call
    fresh = labeled_batches[1]
    k = pool.add_frame(
        type(fresh)(
            species=fresh.species,
            positions=fresh.positions,
            cell=fresh.cell,
            group="TiZrNb_ensmd_r0_t0",
            composition="TiZrNb",
        )
    )
    pool.purchase([k])
    assert pool.oracle_calls == 1
    assert calls == ["TiZrNb_ensmd_r0_t0"]


def test_unlabeled_without_oracle_raises(labeled_batches):
    pool = LabeledPool(labeled_batches)
    fr = labeled_batches[0]
    k = pool.add_frame(
        type(fr)(species=fr.species, positions=fr.positions, cell=fr.cell, group="x")
    )
    with pytest.raises(KeyError):
        pool.purchase([k])


def test_select_top_disagreement():
    available = np.array([10, 20, 30, 40])
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    pick = select_top_disagreement(available, scores, 2)
    assert pick.tolist() == [20, 40]
    with pytest.raises(ValueError):
        select_top_disagreement(available, scores[:3], 2)
    # k larger than the pool truncates
    assert len(select_top_disagreement(available, scores, 99)) == 4


def test_select_top_disagreement_capped():
    from falearn.acquisition import select_top_disagreement_capped

    available = np.array([0, 1, 2, 3, 4, 5])
    scores = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    groups = ["a", "a", "a", "b", "b", "c"]
    # cap 2: the third 'a' frame (index 2) is skipped in favor of group b
    pick = select_top_disagreement_capped(available, scores, 4, groups, cap=2)
    assert pick.tolist() == [0, 1, 3, 4]
    # when the cap exhausts the ranking, the best skipped frames backfill
    pick = select_top_disagreement_capped(available, scores, 6, groups, cap=1)
    assert sorted(pick.tolist()) == [0, 1, 2, 3, 4, 5]
    assert pick.tolist()[:3] == [0, 3, 5]  # one per group first, by score
    with pytest.raises(ValueError):
        select_top_disagreement_capped(available, scores, 2, groups[:3], cap=1)


def test_select_random_no_replacement():
    rng = np.random.default_rng(0)
    available = np.arange(50)
    pick = select_random(available, 20, rng)
    assert len(pick) == 20
    assert len(set(pick.tolist())) == 20
