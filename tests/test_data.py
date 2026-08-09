"""Frame IO, leakage-safe splits, and the sibling-frame exhibit."""

from __future__ import annotations

import numpy as np

from falearn.data import (
    group_kind,
    load_frames,
    pool_test_split,
    save_frames,
    sibling_test_indices,
)
from falearn.structures import build_dataset


def test_extxyz_round_trip(tmp_path, labeled_batches):
    path = tmp_path / "frames.extxyz"
    save_frames(labeled_batches, path)
    loaded = load_frames(path)
    assert len(loaded) == len(labeled_batches)
    for a, b in zip(labeled_batches, loaded):
        assert np.allclose(a.positions, b.positions, atol=1e-7)
        assert np.allclose(a.cell, b.cell, atol=1e-7)
        assert a.group == b.group
        assert a.composition == b.composition
        assert abs(a.energy - b.energy) < 1e-6
        assert np.allclose(a.forces, b.forces, atol=1e-6)


def test_group_kind():
    assert group_kind("TiZrNb_rattle08_b0", "TiZrNb") == "rattle08"
    assert group_kind("TiNb_strain_vol_b1", "TiNb") == "strain_vol"
    assert group_kind("Zr_ensmd_r3_t1", "Zr") == "ensmd_r3_t1"


def test_pool_test_split_group_and_composition_disjoint():
    frames = build_dataset(
        ("Ti", "Nb", "TiZrNb", "Ti2ZrNb"),
        rattle_amplitudes=(0.05, 0.15),
        batches_per_kind=2,
        frames_per_batch=4,
        seed=0,
    )
    split = pool_test_split(frames, holdout_compositions=("Ti2ZrNb",), test_fraction=0.25, seed=1)
    pool_groups = {frames[k].group for k in split.pool}
    test_groups = {frames[k].group for k in split.test}
    assert pool_groups.isdisjoint(test_groups)
    assert all(frames[k].composition == "Ti2ZrNb" for k in split.transfer)
    assert all(frames[k].composition != "Ti2ZrNb" for k in split.pool + split.test)
    assert len(split.pool) + len(split.test) + len(split.transfer) == len(frames)
    # every batch kind is represented in the test set (stratification)
    test_kinds = {group_kind(frames[k].group, frames[k].composition) for k in split.test}
    assert {"rattle05", "rattle15", "strain_vol", "strain_shear"} <= test_kinds


def test_sibling_test_indices_only_unpurchased_from_purchased_groups(labeled_batches):
    frames = labeled_batches
    pool_indices = list(range(len(frames)))
    purchased = [0, 1, 2, 12, 13]  # two groups touched
    sib = sibling_test_indices(frames, pool_indices, purchased, n_per_group=2, seed=0)
    purchased_groups = {frames[k].group for k in purchased}
    assert sib  # non-empty
    for k in sib:
        assert k not in purchased
        assert frames[k].group in purchased_groups
