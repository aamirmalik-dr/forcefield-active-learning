"""Acquisition strategies: query-by-committee and the random baseline.

Both strategies see the same pool and spend the same budget schedule; the only
difference is which unpurchased indices they pick each round.
"""

from __future__ import annotations

import numpy as np


def select_top_disagreement(available: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """The k available frames with the highest committee disagreement.

    Args:
        available: Unpurchased pool indices, shape (A,).
        scores: Disagreement score of each available frame, shape (A,),
            aligned with `available`.
        k: Number of frames to select.

    Returns:
        Selected pool indices, sorted by descending score.
    """
    available = np.asarray(available)
    scores = np.asarray(scores)
    if len(available) != len(scores):
        raise ValueError("available and scores must align")
    k = min(k, len(available))
    order = np.argsort(-scores, kind="stable")
    return available[order[:k]]


def select_top_disagreement_capped(
    available: np.ndarray,
    scores: np.ndarray,
    k: int,
    groups: list[str],
    cap: int,
) -> np.ndarray:
    """Top-disagreement selection with a per-generation-group cap.

    Pure top-k can spend a whole batch on near-duplicate frames from one
    rattle batch (outlier chasing). This variant walks the ranking and skips
    any frame whose generation group already contributed `cap` picks this
    round; if the cap exhausts the ranking early, the remainder is filled by
    the best skipped frames.

    Args:
        available: Unpurchased pool indices, shape (A,).
        scores: Disagreement scores aligned with `available`.
        k: Number of frames to select.
        groups: Generation-group id of each available frame, aligned.
        cap: Maximum picks per generation group per round.

    Returns:
        Selected pool indices.
    """
    available = np.asarray(available)
    scores = np.asarray(scores)
    if not (len(available) == len(scores) == len(groups)):
        raise ValueError("available, scores, and groups must align")
    k = min(k, len(available))
    order = np.argsort(-scores, kind="stable")
    picked: list[int] = []
    skipped: list[int] = []
    counts: dict[str, int] = {}
    for idx in order:
        if len(picked) == k:
            break
        g = groups[idx]
        if counts.get(g, 0) < cap:
            counts[g] = counts.get(g, 0) + 1
            picked.append(idx)
        else:
            skipped.append(idx)
    for idx in skipped:
        if len(picked) == k:
            break
        picked.append(idx)
    return available[np.array(picked, dtype=int)]


def select_random(available: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random selection from the available pool."""
    available = np.asarray(available)
    k = min(k, len(available))
    return rng.choice(available, size=k, replace=False)
