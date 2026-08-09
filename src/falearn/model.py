"""The committee-member potential: compact ACSF network with autograd forces.

One member is a per-element SiLU MLP on the compact descriptor set, about 2.5k
parameters, in float64. Energies are a fixed composition-linear reference
(per-element reference energies, least-squares fit on purchased frames) plus
a learned residual. Committee members share the architecture and differ only
in initialization seed and batch order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from falearn.descriptors import N_ELEMENTS, AcsfParams, Graph, compute_descriptors


class DescriptorScaler(nn.Module):
    """Per-element feature standardization, fit once on training atoms."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_ELEMENTS, n_features, dtype=torch.float64))
        self.register_buffer("std", torch.ones(N_ELEMENTS, n_features, dtype=torch.float64))

    def fit(self, descriptors: torch.Tensor, species: torch.Tensor) -> None:
        for t in range(N_ELEMENTS):
            sel = descriptors[species == t]
            if len(sel) > 0:
                self.mean[t] = sel.mean(dim=0)
                std = sel.std(dim=0)
                # a feature constant in training (e.g. a neighbor-element
                # channel absent from the purchased compositions) gets unit
                # scale: dividing by a near-zero std would explode the first
                # time the feature is nonzero at inference
                self.std[t] = torch.where(std < 1e-6, torch.ones_like(std), std)

    def forward(self, descriptors: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        return (descriptors - self.mean[species]) / self.std[species]


class MemberPotential(nn.Module):
    """One committee member: compact ACSF descriptors into per-element MLPs.

    Args:
        acsf: Symmetry-function parameters.
        hidden: Hidden-layer widths of each element network.
        seed: Initialization seed (what distinguishes committee members).
    """

    def __init__(
        self, acsf: AcsfParams | None = None, hidden: tuple[int, ...] = (16, 16), seed: int = 0
    ) -> None:
        super().__init__()
        self.acsf = acsf or AcsfParams()
        self.hidden = tuple(hidden)
        self.seed = int(seed)
        self.scaler = DescriptorScaler(self.acsf.n_features)
        self.register_buffer("mu", torch.zeros(N_ELEMENTS, dtype=torch.float64))
        gen = torch.Generator().manual_seed(self.seed)
        self.nets = nn.ModuleList()
        for _ in range(N_ELEMENTS):
            layers: list[nn.Module] = []
            width_in = self.acsf.n_features
            for width in self.hidden:
                lin = nn.Linear(width_in, width, dtype=torch.float64)
                self._init_layer(lin, gen)
                layers.append(lin)
                layers.append(nn.SiLU())
                width_in = width
            last = nn.Linear(width_in, 1, dtype=torch.float64)
            with torch.no_grad():
                last.weight.normal_(0.0, 1e-2, generator=gen)
                last.bias.zero_()
            layers.append(last)
            self.nets.append(nn.Sequential(*layers))

    @staticmethod
    def _init_layer(lin: nn.Linear, gen: torch.Generator) -> None:
        with torch.no_grad():
            bound = 1.0 / np.sqrt(lin.in_features)
            lin.weight.uniform_(-bound, bound, generator=gen)
            lin.bias.uniform_(-bound, bound, generator=gen)

    def set_reference(self, mu: np.ndarray) -> None:
        """Set fixed per-element reference energies (eV/atom)."""
        self.mu.copy_(torch.tensor(mu, dtype=torch.float64))

    def atomic_energies(self, positions: torch.Tensor, graph: Graph) -> torch.Tensor:
        desc = compute_descriptors(positions, graph, self.acsf)
        x = self.scaler(desc, graph.species)
        out = torch.zeros(len(graph.species), dtype=positions.dtype)
        for t in range(N_ELEMENTS):
            mask = graph.species == t
            if mask.any():
                out = out.index_add(0, mask.nonzero().squeeze(1), self.nets[t](x[mask]).squeeze(1))
        return out + self.mu[graph.species]

    def forward(self, positions: torch.Tensor, graph: Graph) -> torch.Tensor:
        """Per-structure total energies in eV, shape (S,)."""
        e_atomic = self.atomic_energies(positions, graph)
        out = torch.zeros(graph.n_structs, dtype=positions.dtype)
        return out.index_add(0, graph.struct_of_atom, e_atomic)

    def energy_forces(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor]:
        """Energies (S,) and forces (N, 3) with forces from autograd."""
        pos = graph.positions.detach().clone().requires_grad_(True)
        energies = self.forward(pos, graph)
        (grad,) = torch.autograd.grad(energies.sum(), pos, create_graph=self.training)
        return energies, -grad

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_members(members: list[MemberPotential], path: str | Path, extra: dict | None = None):
    """Serialize a committee with enough metadata to reconstruct it."""
    meta = {
        "acsf": members[0].acsf.to_dict(),
        "hidden": list(members[0].hidden),
        "seeds": [m.seed for m in members],
        "extra": extra or {},
    }
    torch.save(
        {"meta_json": json.dumps(meta), "states": [m.state_dict() for m in members]},
        str(path),
    )


def load_members(path: str | Path) -> tuple[list[MemberPotential], dict]:
    """Load a committee saved by save_members; returns (members, extra)."""
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    meta = json.loads(payload["meta_json"])
    acsf = AcsfParams.from_dict(meta["acsf"])
    members = []
    for seed, state in zip(meta["seeds"], payload["states"]):
        m = MemberPotential(acsf, tuple(meta["hidden"]), seed=seed)
        m.load_state_dict(state)
        m.eval()
        members.append(m)
    return members, meta.get("extra", {})
