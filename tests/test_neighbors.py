"""Neighbor-list correctness against ASE, including hostile geometries."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list

from falearn.neighbors import neighbor_pairs, restrict_pairs, triplets_from_pairs


def _pair_set(i, j, shift):
    return {(int(a), int(b), tuple(int(x) for x in s)) for a, b, s in zip(i, j, shift)}


def _compare_with_ase(positions, cell, cutoff):
    ours = neighbor_pairs(positions, cell, cutoff)
    atoms = Atoms("H" * len(positions), positions=positions, cell=cell, pbc=True)
    ai, aj, ashift = ase_neighbor_list("ijS", atoms, cutoff)
    assert _pair_set(ours.i, ours.j, ours.shift) == _pair_set(ai, aj, ashift)


def test_matches_ase_cubic():
    rng = np.random.default_rng(0)
    cell = np.eye(3) * 6.6
    positions = rng.uniform(0, 6.6, (16, 3))
    _compare_with_ase(positions, cell, 3.2)


def test_matches_ase_sheared_cell():
    rng = np.random.default_rng(1)
    cell = np.array([[6.0, 0.0, 0.0], [3.0, 5.2, 0.0], [1.5, -2.0, 5.8]])
    frac = rng.uniform(0, 1, (12, 3))
    _compare_with_ase(frac @ cell, cell, 3.0)


def test_matches_ase_strong_shear():
    # shear comparable to the cell size stresses the image-count logic
    rng = np.random.default_rng(2)
    cell = np.array([[6.0, 0.0, 0.0], [5.5, 6.0, 0.0], [0.0, 5.5, 6.0]])
    frac = rng.uniform(0, 1, (10, 3))
    _compare_with_ase(frac @ cell, cell, 3.5)


def test_matches_ase_atoms_outside_cell():
    rng = np.random.default_rng(3)
    cell = np.eye(3) * 6.5
    positions = rng.uniform(-8.0, 15.0, (14, 3))  # far outside the box
    _compare_with_ase(positions, cell, 3.0)


def test_pair_vector_definition():
    cell = np.eye(3) * 5.0
    positions = np.array([[0.2, 0.0, 0.0], [4.9, 0.0, 0.0]])
    pairs = neighbor_pairs(positions, cell, 1.0)
    vec = positions[pairs.j] + pairs.shift @ cell - positions[pairs.i]
    dist = np.linalg.norm(vec, axis=1)
    assert np.allclose(dist, 0.3)
    assert len(pairs.i) == 2  # directed both ways


def test_restrict_pairs_and_triplets():
    cell = np.eye(3) * 20.0
    # three atoms on a line: 0-1 at 1.5, 1-2 at 1.5, 0-2 at 3.0
    positions = np.array([[5.0, 5.0, 5.0], [6.5, 5.0, 5.0], [8.0, 5.0, 5.0]])
    pairs = neighbor_pairs(positions, cell, 3.5)
    keep = restrict_pairs(pairs, positions, cell, 2.0)
    assert len(keep) == 4  # only the two short pairs, both directions
    tri = triplets_from_pairs(pairs, keep)
    # only atom 1 has two legs within 2.0
    assert set(tri.center.tolist()) == {1}
    assert len(tri.center) == 1


def test_restrict_cutoff_validation():
    cell = np.eye(3) * 5.0
    positions = np.zeros((1, 3))
    pairs = neighbor_pairs(positions, cell, 2.0)
    with pytest.raises(ValueError):
        restrict_pairs(pairs, positions, cell, 3.0)
