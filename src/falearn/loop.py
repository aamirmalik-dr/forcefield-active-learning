"""The active-learning loop: purchase, retrain, rank, repeat.

One run executes a fixed budget schedule (n_init random labels, then n_rounds
batches of round_size) with one of two strategies:

- "committee": rank unpurchased candidates by committee force disagreement,
  purchase the top round_size.
- "random": purchase a uniform random round_size.

Both strategies share the identical initial random draw (per seed), pool,
ensemble architecture, and training settings, so the curves differ only in
where the label budget went. Optional ensemble-driven MD exploration injects
fresh candidate snapshots into the pool each round; purchasing one of those
costs a real teacher call through the pool's oracle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from falearn.acquisition import (
    select_random,
    select_top_disagreement,
    select_top_disagreement_capped,
)
from falearn.descriptors import AcsfParams, Graph, frame_graph
from falearn.ensemble import committee_predict, evaluate_mean, train_committee
from falearn.model import MemberPotential
from falearn.pool import LabeledPool
from falearn.training import TrainSettings


@dataclass
class EvalSet:
    """A fixed labeled evaluation set (labels visible to the harness only)."""

    name: str
    graphs: list[Graph]
    energies: np.ndarray
    forces: list[np.ndarray]


@dataclass
class MdExploreSettings:
    """Ensemble-driven MD exploration of new candidates each round.

    Attributes:
        compositions: Compositions to explore.
        temperature_k: Langevin thermostat temperature.
        n_steps: MD steps per trajectory.
        sample_every: Snapshot stride (first snapshot after one stride).
        trajectories_per_comp: Independent trajectories per composition.
        reps: Supercell repetitions for the starting cells.
        min_distance: Snapshots with any pair closer than this (Angstrom)
            are discarded before they can reach the teacher.
    """

    compositions: tuple[str, ...] = ()
    temperature_k: float = 1200.0
    n_steps: int = 120
    sample_every: int = 30
    trajectories_per_comp: int = 1
    reps: int = 2
    min_distance: float = 1.6


@dataclass
class LoopSettings:
    """One active-learning run.

    Attributes:
        strategy: "committee" (pure top-disagreement), "committee_capped"
            (top-disagreement with at most group_cap picks per generation
            group per round), or "random".
        n_members: Committee size.
        hidden: Member hidden-layer widths.
        n_init: Random labels purchased before the first round (identical
            draw for both strategies at a given seed).
        round_size: Labels purchased per acquisition round.
        n_rounds: Acquisition rounds after the initial draw.
        seed: Run seed (initial draw, member inits, random purchases).
        train: Member training settings (seed field is overridden per member).
        chunk_size: Frames per merged graph during committee inference.
        explore: Optional MD exploration settings.
    """

    strategy: str = "committee"
    n_members: int = 5
    hidden: tuple[int, ...] = (16, 16)
    n_init: int = 60
    round_size: int = 40
    n_rounds: int = 6
    seed: int = 0
    train: TrainSettings = field(default_factory=TrainSettings)
    chunk_size: int = 48
    explore: MdExploreSettings | None = None
    group_cap: int = 4


class GraphCache:
    """Per-frame Graph cache keyed by pool index."""

    def __init__(self, pool: LabeledPool, acsf: AcsfParams) -> None:
        self.pool = pool
        self.acsf = acsf
        self._cache: dict[int, Graph] = {}

    def get(self, k: int) -> Graph:
        if k not in self._cache:
            self._cache[k] = frame_graph(self.pool.frames[k], self.acsf)
        return self._cache[k]

    def get_many(self, indices) -> list[Graph]:
        return [self.get(int(k)) for k in indices]


def run_loop(
    pool: LabeledPool,
    eval_sets: list[EvalSet],
    settings: LoopSettings,
    acsf: AcsfParams | None = None,
) -> tuple[dict, list[MemberPotential]]:
    """Execute one active-learning run.

    Returns:
        (history, final committee members). History carries, per round, the
        cumulative budget, metrics for every eval set, the selection metadata
        of the next purchase, and wall-clock timings.
    """
    acsf = acsf or AcsfParams()
    cache = GraphCache(pool, acsf)
    rng_run = np.random.default_rng([settings.seed, 777])
    # the initial draw is shared across strategies: seeded independently of
    # the strategy string
    rng_init = np.random.default_rng([settings.seed, 12345])
    init_pick = select_random(pool.available(), settings.n_init, rng_init)
    pool.purchase(init_pick)
    history: dict = {
        "settings": {
            "strategy": settings.strategy,
            "n_members": settings.n_members,
            "hidden": list(settings.hidden),
            "n_init": settings.n_init,
            "round_size": settings.round_size,
            "n_rounds": settings.n_rounds,
            "seed": settings.seed,
            "train": settings.train.to_dict(),
            "explore": settings.explore is not None,
            "group_cap": settings.group_cap,
        },
        "rounds": [],
        "initial_purchase": [pool.summary_row(int(k)) for k in init_pick],
    }
    members: list[MemberPotential] = []
    for r in range(settings.n_rounds + 1):
        record: dict = {"round": r, "budget": pool.n_purchased}
        energies, forces, counts = pool.purchased_labels()
        graphs = cache.get_many(pool.purchased)
        t0 = time.perf_counter()
        members, train_hist = train_committee(
            settings.n_members,
            graphs,
            energies,
            forces,
            counts,
            settings.train,
            acsf=acsf,
            hidden=settings.hidden,
            base_seed=settings.seed * 100 + r,
        )
        record["train_seconds"] = time.perf_counter() - t0
        record["member_final_losses"] = [h["final_loss"] for h in train_hist]
        record["epochs_used"] = train_hist[0]["epochs_used"]
        t0 = time.perf_counter()
        record["metrics"] = {}
        for ev in eval_sets:
            m = evaluate_mean(
                members, ev.graphs, ev.energies, ev.forces, chunk_size=settings.chunk_size
            )
            per_frame = m.pop("per_frame")
            record["metrics"][ev.name] = m
            # calibration arrays only for the primary test set, final round
            if ev.name == "test" and r == settings.n_rounds:
                history["calibration"] = per_frame
        record["eval_seconds"] = time.perf_counter() - t0
        if r < settings.n_rounds:
            if settings.explore is not None:
                record["explore"] = _explore_md(pool, members, settings, acsf, round_index=r)
            available = pool.available()
            t0 = time.perf_counter()
            if settings.strategy in ("committee", "committee_capped"):
                scores = committee_predict(
                    members, cache.get_many(available), chunk_size=settings.chunk_size
                )["disagreement_mean"]
                if settings.strategy == "committee":
                    pick = select_top_disagreement(available, scores, settings.round_size)
                else:
                    groups = [pool.frames[int(a)].group for a in available]
                    pick = select_top_disagreement_capped(
                        available, scores, settings.round_size, groups, settings.group_cap
                    )
                score_of = dict(zip(available.tolist(), scores.tolist()))
            elif settings.strategy == "random":
                pick = select_random(available, settings.round_size, rng_run)
                score_of = {}
            else:
                raise ValueError(f"unknown strategy {settings.strategy}")
            record["select_seconds"] = time.perf_counter() - t0
            record["selected"] = [
                {**pool.summary_row(int(k)), "score": score_of.get(int(k))} for k in pick
            ]
            pool.purchase(pick)
        history["rounds"].append(record)
    history["oracle_calls"] = pool.oracle_calls
    return history, members


def _explore_md(
    pool: LabeledPool,
    members: list[MemberPotential],
    settings: LoopSettings,
    acsf: AcsfParams,
    round_index: int,
) -> dict:
    """Add ensemble-MD snapshot candidates to the pool (teacher-free).

    Returns:
        Dict with the number of snapshots added and dropped by the
        overlap filter, recorded in the round history.
    """
    from falearn.md import langevin_snapshots
    from falearn.neighbors import neighbor_pairs
    from falearn.structures import bcc_supercell

    ex = settings.explore
    n_added, n_dropped = 0, 0
    for c_i, comp in enumerate(sorted(ex.compositions)):
        for t in range(ex.trajectories_per_comp):
            # one deterministic seed per (run seed, round, composition, trajectory)
            seed = int(np.random.default_rng([settings.seed, round_index, c_i, t]).integers(2**31))
            start = bcc_supercell(comp, ex.reps, seed=seed)
            snaps = langevin_snapshots(
                start,
                members,
                acsf,
                temperature_k=ex.temperature_k,
                n_steps=ex.n_steps,
                sample_every=ex.sample_every,
                seed=seed,
            )
            for fr in snaps:
                # candidates with overlapping atoms never go to the teacher,
                # the standard sanity filter of an on-policy exploration loop
                pairs = neighbor_pairs(fr.positions, fr.cell, 2.0)
                if len(pairs.i) > 0:
                    vec = fr.positions[pairs.j] + pairs.shift @ fr.cell - fr.positions[pairs.i]
                    if float(np.linalg.norm(vec, axis=1).min()) < ex.min_distance:
                        n_dropped += 1
                        continue
                fr.group = f"{comp}_ensmd_r{round_index}_t{t}"
                fr.composition = comp
                pool.add_frame(fr)
                n_added += 1
    return {"added": n_added, "dropped_overlapping": n_dropped}
