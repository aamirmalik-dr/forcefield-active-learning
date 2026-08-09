"""Compact atom-centered symmetry functions (ACSF) in PyTorch.

A deliberately small descriptor sized for committee retraining speed: 8
shifted-Gaussian radial G2 functions resolved by neighbor element (24
features) plus 6 element-agnostic angular G4 functions, 30 features total.
Everything is differentiable in atomic positions so forces come from autograd.
Structures pack into a flat Graph; per-frame graphs are cached once and merged
into batches, because active learning re-evaluates the same pool every round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from falearn.neighbors import neighbor_pairs, restrict_pairs, triplets_from_pairs

N_ELEMENTS = 3  # Ti, Zr, Nb


def _default_radial() -> list[tuple[float, float]]:
    """Shifted-Gaussian (rs, eta) grid covering the first three BCC shells."""
    centers = np.linspace(2.2, 5.2, 8)
    delta = centers[1] - centers[0]
    eta = 1.0 / (2.0 * delta * delta)
    return [(float(rs), float(eta)) for rs in centers]


def _default_angular() -> list[tuple[float, float, float]]:
    """(eta, zeta, lambda) G4 parameter sets, three angular resolutions."""
    return [
        (0.02, 1.0, 1.0),
        (0.02, 1.0, -1.0),
        (0.02, 4.0, 1.0),
        (0.02, 4.0, -1.0),
        (0.02, 16.0, 1.0),
        (0.02, 16.0, -1.0),
    ]


@dataclass
class AcsfParams:
    """Symmetry-function hyperparameters.

    Attributes:
        rc_radial: Radial cutoff in Angstrom.
        rc_angular: Angular cutoff in Angstrom (shorter; triplet count is
            quadratic in neighbors).
        radial: List of (rs, eta) shifted-Gaussian parameters.
        angular: List of (eta, zeta, lambda) G4 parameters (element-agnostic).
    """

    rc_radial: float = 5.5
    rc_angular: float = 4.5
    radial: list[tuple[float, float]] = field(default_factory=_default_radial)
    angular: list[tuple[float, float, float]] = field(default_factory=_default_angular)

    @property
    def n_features(self) -> int:
        return N_ELEMENTS * len(self.radial) + len(self.angular)

    def to_dict(self) -> dict:
        return {
            "rc_radial": self.rc_radial,
            "rc_angular": self.rc_angular,
            "radial": [list(p) for p in self.radial],
            "angular": [list(p) for p in self.angular],
        }

    @staticmethod
    def from_dict(d: dict) -> AcsfParams:
        return AcsfParams(
            rc_radial=float(d["rc_radial"]),
            rc_angular=float(d["rc_angular"]),
            radial=[tuple(p) for p in d["radial"]],
            angular=[tuple(p) for p in d["angular"]],
        )


@dataclass
class Graph:
    """Flat tensor view of one or more structures for descriptor evaluation.

    Atom indices of every structure are offset into one concatenated array;
    pair shift vectors are premultiplied by each structure's own cell, so
    strained and unstrained frames batch together.
    """

    positions: torch.Tensor  # (N, 3) float
    species: torch.Tensor  # (N,) long
    pair_i: torch.Tensor  # (P,) long
    pair_j: torch.Tensor  # (P,) long
    shift_vec: torch.Tensor  # (P, 3) float, constant
    tri_ij: torch.Tensor  # (T,) long indices into pairs
    tri_ik: torch.Tensor  # (T,) long indices into pairs
    struct_of_atom: torch.Tensor  # (N,) long
    n_structs: int
    n_atoms_per_struct: torch.Tensor  # (S,) long


def build_graph(frames, params: AcsfParams, dtype=torch.float64) -> Graph:
    """Pack a list of Frame objects into one flat Graph."""
    return merge_graphs([frame_graph(fr, params, dtype=dtype) for fr in frames])


def frame_graph(frame, params: AcsfParams, dtype=torch.float64) -> Graph:
    """Build the single-structure Graph for one frame (cacheable)."""
    pairs = neighbor_pairs(frame.positions, frame.cell, params.rc_radial)
    keep = restrict_pairs(pairs, frame.positions, frame.cell, params.rc_angular)
    tri = triplets_from_pairs(pairs, keep)
    n = len(frame.species)
    return Graph(
        positions=torch.tensor(frame.positions, dtype=dtype),
        species=torch.tensor(frame.species, dtype=torch.long),
        pair_i=torch.tensor(pairs.i, dtype=torch.long),
        pair_j=torch.tensor(pairs.j, dtype=torch.long),
        shift_vec=torch.tensor(pairs.shift @ frame.cell, dtype=dtype),
        tri_ij=torch.tensor(tri.pair_ij, dtype=torch.long),
        tri_ik=torch.tensor(tri.pair_ik, dtype=torch.long),
        struct_of_atom=torch.zeros(n, dtype=torch.long),
        n_structs=1,
        n_atoms_per_struct=torch.tensor([n], dtype=torch.long),
    )


def merge_graphs(graphs: list[Graph]) -> Graph:
    """Concatenate single- or multi-structure graphs with index offsets."""
    pos, spec, pi, pj, sv, tij, tik, soa, nat = [], [], [], [], [], [], [], [], []
    atom_off = 0
    pair_off = 0
    struct_off = 0
    for g in graphs:
        pos.append(g.positions)
        spec.append(g.species)
        pi.append(g.pair_i + atom_off)
        pj.append(g.pair_j + atom_off)
        sv.append(g.shift_vec)
        tij.append(g.tri_ij + pair_off)
        tik.append(g.tri_ik + pair_off)
        soa.append(g.struct_of_atom + struct_off)
        nat.append(g.n_atoms_per_struct)
        atom_off += g.positions.shape[0]
        pair_off += g.pair_i.shape[0]
        struct_off += g.n_structs
    return Graph(
        positions=torch.cat(pos),
        species=torch.cat(spec),
        pair_i=torch.cat(pi),
        pair_j=torch.cat(pj),
        shift_vec=torch.cat(sv),
        tri_ij=torch.cat(tij),
        tri_ik=torch.cat(tik),
        struct_of_atom=torch.cat(soa),
        n_structs=struct_off,
        n_atoms_per_struct=torch.cat(nat),
    )


def _cutoff_fn(r: torch.Tensor, rc: float) -> torch.Tensor:
    """Cosine cutoff, zero at and beyond rc, smooth first derivative."""
    inside = r < rc
    fc = 0.5 * (torch.cos(math.pi * r / rc) + 1.0)
    return torch.where(inside, fc, torch.zeros_like(fc))


def compute_descriptors(positions: torch.Tensor, graph: Graph, params: AcsfParams) -> torch.Tensor:
    """Evaluate ACSF descriptors for every atom in the graph.

    Args:
        positions: (N, 3) tensor; pass graph.positions (optionally with
            requires_grad set) so forces can be taken by autograd.
        graph: Packed structure graph.
        params: Symmetry-function parameters.

    Returns:
        (N, n_features) descriptor tensor, differentiable in positions.
    """
    n_atoms = positions.shape[0]
    dtype = positions.dtype
    vec = positions[graph.pair_j] + graph.shift_vec - positions[graph.pair_i]
    r = torch.linalg.norm(vec, dim=1)  # (P,)
    spec_j = graph.species[graph.pair_j]

    # --- radial G2, resolved by neighbor element ---
    n_rad = len(params.radial)
    rs = torch.tensor([p[0] for p in params.radial], dtype=dtype)
    eta = torch.tensor([p[1] for p in params.radial], dtype=dtype)
    fc_rad = _cutoff_fn(r, params.rc_radial)
    g2 = torch.exp(-eta[None, :] * (r[:, None] - rs[None, :]) ** 2) * fc_rad[:, None]
    rad_row = graph.pair_i * N_ELEMENTS + spec_j
    radial_out = torch.zeros(n_atoms * N_ELEMENTS, n_rad, dtype=dtype).index_add(0, rad_row, g2)

    # --- angular G4, element-agnostic ---
    n_ang = len(params.angular)
    ang_out = torch.zeros(n_atoms, n_ang, dtype=dtype)
    if len(graph.tri_ij) > 0:
        v_ij = vec[graph.tri_ij]
        v_ik = vec[graph.tri_ik]
        r_ij = r[graph.tri_ij]
        r_ik = r[graph.tri_ik]
        v_jk = v_ik - v_ij
        r_jk2 = (v_jk * v_jk).sum(dim=1)
        r_jk = torch.sqrt(torch.clamp(r_jk2, min=1e-12))
        cos_t = torch.clamp((v_ij * v_ik).sum(dim=1) / (r_ij * r_ik), -1.0, 1.0)
        rca = params.rc_angular
        fc3 = _cutoff_fn(r_ij, rca) * _cutoff_fn(r_ik, rca) * _cutoff_fn(r_jk, rca)
        r2sum = r_ij * r_ij + r_ik * r_ik + r_jk2
        centers = graph.pair_i[graph.tri_ij]
        feats = []
        for eta_a, zeta, lam in params.angular:
            ang = (2.0 ** (1.0 - zeta)) * (1.0 + lam * cos_t) ** zeta
            feats.append(ang * torch.exp(-eta_a * r2sum) * fc3)
        g4 = torch.stack(feats, dim=1)  # (T, n_ang)
        ang_out = ang_out.index_add(0, centers, g4)

    return torch.cat([radial_out.reshape(n_atoms, -1), ang_out], dim=1)
