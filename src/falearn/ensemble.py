"""Deep-ensemble committee: training, mean prediction, and force disagreement.

The committee is M compact potentials differing only in initialization seed
and batch order. The query-by-committee signal is per-atom force
disagreement: the root-mean-square deviation of member force vectors from the
committee mean, aggregated per frame. That same committee mean is the
deployed potential.
"""

from __future__ import annotations

import numpy as np
import torch

from falearn.descriptors import AcsfParams, Graph, merge_graphs
from falearn.model import MemberPotential
from falearn.training import TrainSettings, train_member


def train_committee(
    n_members: int,
    graphs: list[Graph],
    energies: np.ndarray,
    forces: list[np.ndarray],
    counts: np.ndarray,
    settings: TrainSettings,
    acsf: AcsfParams | None = None,
    hidden: tuple[int, ...] = (16, 16),
    base_seed: int = 0,
) -> tuple[list[MemberPotential], list[dict]]:
    """Train a fresh committee on the purchased frames.

    Member k gets initialization seed base_seed*100+k and the same seed for
    its batch shuffling, so members are fully reproducible and mutually
    independent given the data.

    Returns:
        (members, per-member training histories).
    """
    members, histories = [], []
    for k in range(n_members):
        seed = base_seed * 100 + k
        member = MemberPotential(acsf, hidden, seed=seed)
        s = TrainSettings(**{**settings.to_dict(), "seed": seed})
        histories.append(train_member(member, graphs, energies, forces, counts, s))
        members.append(member)
    return members, histories


def committee_forward(
    members: list[MemberPotential], graph: Graph
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacked member energies (M, S) and forces (M, N, 3) for one graph."""
    e_list, f_list = [], []
    for m in members:
        m.eval()
        energies, forces = m.energy_forces(graph)
        e_list.append(energies.detach())
        f_list.append(forces.detach())
    return torch.stack(e_list), torch.stack(f_list)


def committee_predict(
    members: list[MemberPotential], graphs: list[Graph], chunk_size: int = 48
) -> dict:
    """Committee mean predictions and disagreement over a list of frame graphs.

    Frames are merged into chunks for vectorized evaluation. The per-frame
    force disagreement is the mean over atoms of

        sigma_i = sqrt( (1/M) * sum_m |F_m,i - Fbar_i|^2 ),

    the RMS committee deviation of the force vector on atom i; the max over
    atoms and the per-atom energy disagreement are recorded alongside.

    Returns:
        Dict of numpy arrays: e_mean (F,), f_mean list of (N,3),
        disagreement_mean (F,), disagreement_max (F,), e_std_per_atom (F,).
    """
    e_mean, f_mean = [], []
    dis_mean, dis_max, e_std = [], [], []
    for k in range(0, len(graphs), chunk_size):
        chunk = graphs[k : k + chunk_size]
        graph = merge_graphs(chunk)
        e_all, f_all = committee_forward(members, graph)
        e_bar = e_all.mean(dim=0)
        f_bar = f_all.mean(dim=0)
        sigma_atom = torch.sqrt(((f_all - f_bar[None]) ** 2).sum(dim=2).mean(dim=0))  # (N,)
        n_atoms = graph.n_atoms_per_struct
        e_sigma = e_all.std(dim=0, unbiased=False) / n_atoms.to(torch.float64)
        e_mean.append(e_bar.numpy())
        e_std.append(e_sigma.numpy())
        atom_off = 0
        for s in range(graph.n_structs):
            n = int(n_atoms[s])
            f_mean.append(f_bar[atom_off : atom_off + n].numpy())
            sl = sigma_atom[atom_off : atom_off + n]
            dis_mean.append(float(sl.mean()))
            dis_max.append(float(sl.max()))
            atom_off += n
    return {
        "e_mean": np.concatenate(e_mean),
        "f_mean": f_mean,
        "disagreement_mean": np.array(dis_mean),
        "disagreement_max": np.array(dis_max),
        "e_std_per_atom": np.concatenate(e_std),
    }


def evaluate_mean(
    members: list[MemberPotential],
    graphs: list[Graph],
    energies: np.ndarray,
    forces: list[np.ndarray],
    chunk_size: int = 48,
) -> dict:
    """Errors of the committee-mean prediction against reference labels.

    Returns:
        Metrics dict (meV/atom and meV/A) plus per-frame arrays used by the
        calibration analysis: per-frame force RMSE and committee disagreement.
    """
    pred = committee_predict(members, graphs, chunk_size=chunk_size)
    n_atoms = np.array([g.positions.shape[0] for g in graphs], dtype=float)
    de = (pred["e_mean"] - np.asarray(energies)) / n_atoms
    df_all = [p - t for p, t in zip(pred["f_mean"], forces)]
    df_flat = np.concatenate([d.ravel() for d in df_all])
    f_rmse_frame = np.array([float(np.sqrt((d**2).mean())) for d in df_all])
    f_mae_frame = np.array([float(np.abs(d).mean()) for d in df_all])
    return {
        "e_mae_mev_per_atom": float(np.abs(de).mean() * 1000),
        "e_rmse_mev_per_atom": float(np.sqrt((de**2).mean()) * 1000),
        "f_mae_mev_per_a": float(np.abs(df_flat).mean() * 1000),
        "f_rmse_mev_per_a": float(np.sqrt((df_flat**2).mean()) * 1000),
        "n_structures": int(len(graphs)),
        "per_frame": {
            "f_rmse_mev_per_a": (f_rmse_frame * 1000).tolist(),
            "f_mae_mev_per_a": (f_mae_frame * 1000).tolist(),
            "disagreement_mean_mev_per_a": (pred["disagreement_mean"] * 1000).tolist(),
            "disagreement_max_mev_per_a": (pred["disagreement_max"] * 1000).tolist(),
            "e_abs_err_mev_per_atom": (np.abs(de) * 1000).tolist(),
            "e_std_mev_per_atom": (pred["e_std_per_atom"] * 1000).tolist(),
        },
    }


def mean_energy_forces(members: list[MemberPotential], graph: Graph):
    """Committee-mean energies (S,) and forces (N, 3) as torch tensors."""
    e_all, f_all = committee_forward(members, graph)
    return e_all.mean(dim=0), f_all.mean(dim=0)
