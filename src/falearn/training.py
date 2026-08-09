"""Fixed-budget training for committee members.

Every member trains from scratch each acquisition round with the identical
weighted energy-plus-force loss, optimizer, and epoch budget; no validation
set and no early stopping, so no labels are spent on model selection and the
active and random strategies get exactly the same training treatment.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from falearn.descriptors import Graph, compute_descriptors, merge_graphs
from falearn.model import MemberPotential


@dataclass
class TrainSettings:
    """Hyperparameters of one member training run.

    force_weight multiplies the force MSE (eV^2/A^2) against the per-atom
    energy MSE (eV^2/atom^2); energy weight is fixed at 1.

    target_frame_epochs equalizes optimization effort across acquisition
    rounds: when nonzero, the epoch count of a run on n frames becomes
    min(max_epochs, max(epochs, round(target_frame_epochs / n))), so a
    60-frame early round is not starved of optimizer steps relative to a
    300-frame final round. Both strategies get the identical rule.
    """

    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-3
    force_weight: float = 0.1
    lr_min_factor: float = 0.05
    seed: int = 0
    target_frame_epochs: int = 0
    max_epochs: int = 120

    def to_dict(self) -> dict:
        return asdict(self)

    def effective_epochs(self, n_frames: int) -> int:
        """Epoch count for a run on n_frames under the equal-effort rule."""
        if self.target_frame_epochs <= 0 or n_frames <= 0:
            return self.epochs
        return min(self.max_epochs, max(self.epochs, round(self.target_frame_epochs / n_frames)))


@dataclass
class PreparedBatch:
    """One training batch: merged graph plus label tensors."""

    graph: Graph
    energy: torch.Tensor  # (S,) eV
    forces: torch.Tensor  # (N, 3) eV/A
    n_atoms: torch.Tensor  # (S,)


def fit_reference(counts: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Least-squares per-element reference energies from total energies.

    Args:
        counts: (F, 3) per-frame element counts.
        energies: (F,) total energies in eV.

    Returns:
        (3,) per-element reference energies in eV/atom.
    """
    mu, *_ = np.linalg.lstsq(np.asarray(counts, dtype=float), np.asarray(energies), rcond=None)
    return mu


def make_batches(
    graphs: list[Graph],
    energies: np.ndarray,
    forces: list[np.ndarray],
    batch_size: int,
    seed: int = 0,
) -> list[PreparedBatch]:
    """Merge cached per-frame graphs into shuffled fixed batches with labels."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(graphs))
    batches = []
    for k in range(0, len(graphs), batch_size):
        chunk = perm[k : k + batch_size]
        graph = merge_graphs([graphs[c] for c in chunk])
        batches.append(
            PreparedBatch(
                graph=graph,
                energy=torch.tensor([energies[c] for c in chunk], dtype=torch.float64),
                forces=torch.tensor(
                    np.concatenate([forces[c] for c in chunk]), dtype=torch.float64
                ),
                n_atoms=graph.n_atoms_per_struct.to(torch.float64),
            )
        )
    return batches


def fit_scaler(member: MemberPotential, batches: list[PreparedBatch]) -> None:
    """Fit per-element descriptor standardization on the training batches."""
    descs, specs = [], []
    with torch.no_grad():
        for b in batches:
            descs.append(compute_descriptors(b.graph.positions, b.graph, member.acsf))
            specs.append(b.graph.species)
    member.scaler.fit(torch.cat(descs), torch.cat(specs))


def _batch_loss(member: MemberPotential, batch: PreparedBatch, force_weight: float) -> torch.Tensor:
    pos = batch.graph.positions.detach().clone().requires_grad_(True)
    energies = member(pos, batch.graph)
    (grad,) = torch.autograd.grad(energies.sum(), pos, create_graph=True)
    loss_f = ((-grad - batch.forces) ** 2).mean()
    loss_e = (((energies - batch.energy) / batch.n_atoms) ** 2).mean()
    return loss_e + force_weight * loss_f


def train_member(
    member: MemberPotential,
    graphs: list[Graph],
    energies: np.ndarray,
    forces: list[np.ndarray],
    counts: np.ndarray,
    settings: TrainSettings,
) -> dict:
    """Train one member for a fixed epoch budget.

    Fits the composition-linear reference and the feature scaler on the
    purchased frames only, then optimizes with Adam and cosine learning-rate
    decay to the final epoch (no early stopping).

    Returns:
        Dict with final training losses and wall time.
    """
    torch.manual_seed(settings.seed)
    member.set_reference(fit_reference(counts, energies))
    batches = make_batches(graphs, energies, forces, settings.batch_size, seed=settings.seed)
    fit_scaler(member, batches)
    epochs = settings.effective_epochs(len(graphs))
    opt = torch.optim.Adam(member.parameters(), lr=settings.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=settings.lr * settings.lr_min_factor
    )
    rng = np.random.default_rng(settings.seed + 7)
    start = time.perf_counter()
    member.train()
    last_loss = float("nan")
    for _ in range(epochs):
        total = 0.0
        for k in rng.permutation(len(batches)):
            opt.zero_grad()
            loss = _batch_loss(member, batches[k], settings.force_weight)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        sched.step()
        last_loss = total / len(batches)
    member.eval()
    return {
        "final_loss": last_loss,
        "train_seconds": time.perf_counter() - start,
        "n_frames": len(graphs),
        "epochs_used": epochs,
        "settings": settings.to_dict(),
    }
