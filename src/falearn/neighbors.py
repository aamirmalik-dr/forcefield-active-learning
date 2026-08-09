"""Periodic neighbor lists and angular triplets, implemented from scratch.

Handles general (strained, non-orthogonal) 3x3 cells by enumerating periodic
image shifts out to the cutoff, with the image range computed from the
perpendicular face-to-face heights of the cell (columns of the inverse cell
matrix are the dual vectors). Verified against ase.neighborlist in the test
suite, including sheared cells and atoms outside the box; contains no ASE code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairList:
    """Directed periodic pair list for one structure.

    Attributes:
        i: Center atom index of each directed pair, shape (P,).
        j: Neighbor atom index of each directed pair, shape (P,).
        shift: Integer periodic image shift of atom j, shape (P, 3). The pair
            vector is positions[j] + shift @ cell - positions[i].
        cutoff: Cutoff radius in Angstrom used to build the list.
    """

    i: np.ndarray
    j: np.ndarray
    shift: np.ndarray
    cutoff: float


@dataclass
class TripletList:
    """Unordered angular triplets (j, k) around each center, as pair indices."""

    center: np.ndarray
    pair_ij: np.ndarray
    pair_ik: np.ndarray


def _image_counts(cell: np.ndarray, cutoff: float) -> tuple[int, int, int]:
    """Periodic images needed along each lattice vector to reach the cutoff.

    The perpendicular height between opposite faces along axis a is
    1 / |column a of inv(cell)|, exact for arbitrary triclinic cells.
    """
    cell = np.asarray(cell, dtype=float)
    if abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("cell is singular")
    inv = np.linalg.inv(cell)
    heights = 1.0 / np.linalg.norm(inv, axis=0)
    return tuple(int(np.ceil(cutoff / h)) for h in heights)


def neighbor_pairs(positions: np.ndarray, cell: np.ndarray, cutoff: float) -> PairList:
    """Build the directed periodic pair list within a cutoff.

    Args:
        positions: Cartesian positions, shape (N, 3), Angstrom.
        cell: Row-vector lattice matrix, shape (3, 3), Angstrom.
        cutoff: Pair cutoff radius, Angstrom.

    Returns:
        PairList with every directed pair (i, j, shift) such that
        |positions[j] + shift @ cell - positions[i]| < cutoff, excluding the
        self-pair at zero shift.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    n_atoms = positions.shape[0]
    # wrap into the cell so the image search stays bounded even for atoms
    # outside the box; shifts are corrected back to the caller's coordinates
    frac = positions @ np.linalg.inv(cell)
    wraps = np.floor(frac).astype(int)
    wrapped = (frac - wraps) @ cell
    na, nb, nc = _image_counts(cell, cutoff)
    shifts = np.stack(
        np.meshgrid(
            np.arange(-na, na + 1),
            np.arange(-nb, nb + 1),
            np.arange(-nc, nc + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    shifted = wrapped[None, :, :] + (shifts @ cell)[:, None, :]  # (S, Nj, 3)
    diff = shifted[:, None, :, :] - wrapped[None, :, None, :]  # (S, Ni, Nj, 3)
    dist2 = np.einsum("sijk,sijk->sij", diff, diff)
    within = dist2 < cutoff * cutoff
    zero_shift = np.all(shifts == 0, axis=1)
    within &= ~(zero_shift[:, None, None] & np.eye(n_atoms, dtype=bool)[None, :, :])
    s_idx, i_idx, j_idx = np.nonzero(within)
    order = np.lexsort((j_idx, i_idx))
    i_idx, j_idx, s_idx = i_idx[order], j_idx[order], s_idx[order]
    # shift in the caller's (possibly unwrapped) coordinates:
    # vec = pos[j] + S @ cell - pos[i] with S = S_wrapped - wrap_j + wrap_i
    shift = shifts[s_idx] - wraps[j_idx] + wraps[i_idx]
    return PairList(i=i_idx, j=j_idx, shift=shift, cutoff=float(cutoff))


def restrict_pairs(
    pairs: PairList, positions: np.ndarray, cell: np.ndarray, cutoff: float
) -> np.ndarray:
    """Indices of pairs whose current distance is below a smaller cutoff."""
    if cutoff > pairs.cutoff:
        raise ValueError("restricted cutoff must not exceed the pair-list cutoff")
    vec = positions[pairs.j] + pairs.shift @ cell - positions[pairs.i]
    return np.nonzero(np.linalg.norm(vec, axis=1) < cutoff)[0]


def triplets_from_pairs(pairs: PairList, keep: np.ndarray | None = None) -> TripletList:
    """Enumerate unordered neighbor triplets (j, k) around each center.

    Args:
        pairs: Directed pair list.
        keep: Optional indices into the pair list restricting which pairs
            count as angular legs (used for a shorter angular cutoff).

    Returns:
        TripletList where every unordered pair of kept legs of each center
        appears exactly once.
    """
    idx = np.arange(len(pairs.i)) if keep is None else np.asarray(keep)
    centers = pairs.i[idx]
    order = np.argsort(centers, kind="stable")
    idx = idx[order]
    centers = centers[order]
    tri_center, tri_ij, tri_ik = [], [], []
    start = 0
    while start < len(idx):
        stop = start
        while stop < len(idx) and centers[stop] == centers[start]:
            stop += 1
        legs = idx[start:stop]
        if len(legs) >= 2:
            a, b = np.triu_indices(len(legs), k=1)
            tri_center.append(np.full(len(a), centers[start]))
            tri_ij.append(legs[a])
            tri_ik.append(legs[b])
        start = stop
    if not tri_center:
        empty = np.zeros(0, dtype=int)
        return TripletList(center=empty, pair_ij=empty.copy(), pair_ik=empty.copy())
    return TripletList(
        center=np.concatenate(tri_center),
        pair_ij=np.concatenate(tri_ij),
        pair_ik=np.concatenate(tri_ik),
    )
