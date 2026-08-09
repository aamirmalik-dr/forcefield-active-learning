"""Uncertainty calibration: does committee disagreement predict actual error?

Works on the per-frame arrays recorded by the loop on the held-out test set:
committee force disagreement (the acquisition signal) against the actual
force RMSE of the committee-mean prediction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def binned_reliability(disagreement: np.ndarray, error: np.ndarray, n_bins: int = 8) -> dict:
    """Equal-count binning of error against disagreement.

    Args:
        disagreement: Per-frame committee disagreement (meV/A).
        error: Per-frame actual force RMSE of the committee mean (meV/A).
        n_bins: Number of equal-count bins.

    Returns:
        Dict with bin centers (median disagreement), mean/median error per
        bin, bin counts, the Spearman rank correlation, and the Pearson
        correlation of the raw pairs.
    """
    d = np.asarray(disagreement, dtype=float)
    e = np.asarray(error, dtype=float)
    if len(d) != len(e) or len(d) == 0:
        raise ValueError("disagreement and error must be equal-length, non-empty")
    order = np.argsort(d)
    d_sorted, e_sorted = d[order], e[order]
    edges = np.linspace(0, len(d), n_bins + 1).astype(int)
    centers, mean_err, med_err, counts, mean_dis = [], [], [], [], []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo:
            continue
        centers.append(float(np.median(d_sorted[lo:hi])))
        mean_dis.append(float(np.mean(d_sorted[lo:hi])))
        mean_err.append(float(np.mean(e_sorted[lo:hi])))
        med_err.append(float(np.median(e_sorted[lo:hi])))
        counts.append(int(hi - lo))
    rho, rho_p = spearmanr(d, e)
    pearson = float(np.corrcoef(d, e)[0, 1])
    return {
        "bin_median_disagreement": centers,
        "bin_mean_disagreement": mean_dis,
        "bin_mean_error": mean_err,
        "bin_median_error": med_err,
        "bin_counts": counts,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": pearson,
        "n_frames": int(len(d)),
    }


def label_reduction_factor(
    budgets: np.ndarray,
    active_curve: np.ndarray,
    random_curve: np.ndarray,
) -> dict:
    """Labels saved by active selection at the random strategy's final error.

    Interpolates the active error-versus-budget curve (log-linear in budget)
    to find the smallest budget at which the active curve reaches the random
    curve's final-budget error.

    Returns:
        Dict with the target error, the interpolated active budget, the full
        random budget, and the reduction factor (random / active). If the
        active curve never reaches the target, the factor is reported as 1.0
        with reached=False.
    """
    budgets = np.asarray(budgets, dtype=float)
    active = np.asarray(active_curve, dtype=float)
    target = float(np.asarray(random_curve, dtype=float)[-1])
    reached = active <= target
    if not reached.any():
        return {
            "target_error": target,
            "active_budget_at_target": float(budgets[-1]),
            "random_budget": float(budgets[-1]),
            "reduction_factor": 1.0,
            "reached": False,
        }
    k = int(np.argmax(reached))  # first budget index at or below target
    if k == 0:
        b_at = budgets[0]
    else:
        # linear interpolation between the bracketing budgets
        e0, e1 = active[k - 1], active[k]
        b0, b1 = budgets[k - 1], budgets[k]
        frac = (e0 - target) / max(e0 - e1, 1e-12)
        b_at = b0 + frac * (b1 - b0)
    return {
        "target_error": target,
        "active_budget_at_target": float(b_at),
        "random_budget": float(budgets[-1]),
        "reduction_factor": float(budgets[-1] / b_at),
        "reached": True,
    }
