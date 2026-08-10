"""Config-driven benchmark drivers shared by the CLI and the scripts.

Each driver reads a YAML config, runs with fixed seeds, and writes one
results JSON. The active-versus-random benchmark, the MD-exploration
variant, the EOS physics check, and the NVE drift check all live here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from falearn.data import Frame, group_kind, load_frames, pool_test_split, sibling_test_indices
from falearn.descriptors import AcsfParams, frame_graph
from falearn.ensemble import evaluate_mean
from falearn.loop import EvalSet, LoopSettings, MdExploreSettings, run_loop
from falearn.model import load_members, save_members
from falearn.pool import LabeledPool
from falearn.training import TrainSettings


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _eval_set(name: str, frames: list[Frame], indices: list[int], acsf: AcsfParams) -> EvalSet:
    subset = [frames[k] for k in indices]
    return EvalSet(
        name=name,
        graphs=[frame_graph(fr, acsf) for fr in subset],
        energies=np.array([fr.energy for fr in subset]),
        forces=[fr.forces for fr in subset],
    )


def _loop_settings(cfg: dict, strategy: str, seed: int) -> LoopSettings:
    train = TrainSettings(**cfg.get("train", {}))
    explore = None
    if cfg.get("explore") and strategy == "committee":
        explore = MdExploreSettings(
            compositions=tuple(cfg["explore"]["compositions"]),
            temperature_k=float(cfg["explore"].get("temperature_k", 1200.0)),
            n_steps=int(cfg["explore"].get("n_steps", 120)),
            sample_every=int(cfg["explore"].get("sample_every", 30)),
            trajectories_per_comp=int(cfg["explore"].get("trajectories_per_comp", 1)),
            reps=int(cfg["explore"].get("reps", 2)),
            min_distance=float(cfg["explore"].get("min_distance", 1.6)),
        )
    return LoopSettings(
        strategy=strategy,
        n_members=int(cfg.get("n_members", 5)),
        hidden=tuple(cfg.get("hidden", (16, 16))),
        n_init=int(cfg.get("n_init", 60)),
        round_size=int(cfg.get("round_size", 40)),
        n_rounds=int(cfg.get("n_rounds", 6)),
        seed=seed,
        train=train,
        chunk_size=int(cfg.get("chunk_size", 48)),
        explore=explore,
        group_cap=int(cfg.get("group_cap", 4)),
    )


def run_al_benchmark(cfg: dict, oracle=None, save_committee_to: str | None = None) -> dict:
    """Active-versus-random label-budget benchmark.

    Args:
        cfg: Parsed YAML config.
        oracle: Optional teacher oracle for exploration candidates.
        save_committee_to: Where to save the final committee of the first
            committee-strategy run (overrides cfg["save_committee"]).

    Returns:
        The results dict (also written to cfg["out"]).
    """
    t_start = time.perf_counter()
    acsf = AcsfParams()
    frames = load_frames(cfg["dataset"])
    split_cfg = cfg.get("split", {})
    split = pool_test_split(
        frames,
        holdout_compositions=tuple(split_cfg.get("holdout_compositions", ("Ti2ZrNb", "TiZrNb2"))),
        test_fraction=float(split_cfg.get("test_fraction", 0.2)),
        seed=int(split_cfg.get("seed", 0)),
    )
    eval_sets = [_eval_set("test", frames, split.test, acsf)]
    if split.transfer:
        eval_sets.append(_eval_set("transfer", frames, split.transfer, acsf))
    # optional redundancy filter: cap how many whole generation groups of a
    # batch kind stay in the pool (kinds not listed keep all their groups).
    # Used by the scarce-pool benchmark to make informative frames rare.
    cap_cfg = cfg.get("pool_max_groups_per_kind")
    if cap_cfg:
        rng = np.random.default_rng(int(split_cfg.get("seed", 0)) + 1)
        groups_by_kind: dict[str, list[str]] = {}
        for k in split.pool:
            kd = group_kind(frames[k].group, frames[k].composition)
            if frames[k].group not in groups_by_kind.setdefault(kd, []):
                groups_by_kind[kd].append(frames[k].group)
        keep_groups: set[str] = set()
        for kd in sorted(groups_by_kind):
            gs = sorted(groups_by_kind[kd])
            limit = cap_cfg.get(kd)
            if limit is not None:
                rng.shuffle(gs)
                gs = gs[: int(limit)]
            keep_groups.update(gs)
        split.pool = [k for k in split.pool if frames[k].group in keep_groups]
    pool_frames = [frames[k] for k in split.pool]
    kind_counts: dict[str, int] = {}
    comp_counts: dict[str, int] = {}
    for fr in pool_frames:
        kd = group_kind(fr.group, fr.composition)
        kind_counts[kd] = kind_counts.get(kd, 0) + 1
        comp_counts[fr.composition] = comp_counts.get(fr.composition, 0) + 1
    results: dict = {
        "config": cfg,
        "n_pool": len(split.pool),
        "n_test": len(split.test),
        "n_transfer": len(split.transfer),
        "pool_kind_fractions": {k: v / len(pool_frames) for k, v in sorted(kind_counts.items())},
        "pool_composition_fractions": {
            k: v / len(pool_frames) for k, v in sorted(comp_counts.items())
        },
        "runs": [],
    }
    committee_saved = False
    save_path = save_committee_to or cfg.get("save_committee")
    for strategy in cfg.get("strategies", ("committee", "random")):
        for seed in cfg.get("seeds", (0,)):
            settings = _loop_settings(cfg, strategy, int(seed))
            pool = LabeledPool(pool_frames, oracle=oracle)
            t0 = time.perf_counter()
            history, members = run_loop(pool, eval_sets, settings, acsf=acsf)
            history["strategy"] = strategy
            history["seed"] = int(seed)
            history["wall_seconds"] = time.perf_counter() - t0
            # the split-protocol exhibit: what a sibling-frame (random-frame)
            # test would have claimed for this run's final committee
            sib_cfg = cfg.get("sibling_test")
            if sib_cfg:
                # exploration candidates sit past the original pool length and
                # have no harness labels; the sibling exhibit uses only the
                # pre-labeled pool frames
                local_purchased = [k for k in pool.purchased if k < len(pool_frames)]
                pool_local_indices = list(range(len(pool_frames)))
                sib_local = sibling_test_indices(
                    pool_frames,
                    pool_local_indices,
                    local_purchased,
                    n_per_group=int(sib_cfg.get("n_per_group", 2)),
                    seed=int(seed),
                )
                sib_frames = [pool_frames[k] for k in sib_local if k < len(pool_frames)]
                sib_frames = [fr for fr in sib_frames if fr.energy is not None]
                if sib_frames:
                    ev = EvalSet(
                        name="sibling",
                        graphs=[frame_graph(fr, acsf) for fr in sib_frames],
                        energies=np.array([fr.energy for fr in sib_frames]),
                        forces=[fr.forces for fr in sib_frames],
                    )
                    m = evaluate_mean(members, ev.graphs, ev.energies, ev.forces)
                    m.pop("per_frame")
                    history["sibling_test"] = m
            results["runs"].append(history)
            # checkpoint after every run so a crash cannot lose finished runs
            out_path = cfg.get("out")
            if out_path:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_text(json.dumps(results, indent=1))
            if strategy == "committee" and not committee_saved and save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                save_members(
                    members,
                    save_path,
                    extra={
                        "benchmark": cfg.get("name", ""),
                        "strategy": strategy,
                        "seed": int(seed),
                        "budget": pool.n_purchased,
                    },
                )
                committee_saved = True
    results["wall_seconds"] = time.perf_counter() - t_start
    out = cfg.get("out")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(results, indent=1))
    return results


def run_eos_benchmark(cfg: dict, teacher_calc=None) -> dict:
    """EOS of committed committees versus teacher and DFT anchors.

    cfg["committees"] maps names to checkpoint paths (e.g. the acquisition-
    biased active committee against the random-reference committee at the
    identical budget); the legacy single cfg["committee"] key maps to the
    name "committee". Missing checkpoint files are skipped.
    """
    from falearn.eos import eos_scan_calculator, eos_scan_committee

    committee_paths = dict(cfg.get("committees") or {})
    if cfg.get("committee"):
        committee_paths.setdefault("committee", cfg["committee"])
    loaded = {}
    out: dict = {"committee_extra": {}, "compositions": {}}
    for name, path in committee_paths.items():
        if Path(path).exists():
            members, extra = load_members(path)
            loaded[name] = members
            out["committee_extra"][name] = extra
    anchors = {}
    for path in (cfg.get("anchors_mp"), cfg.get("anchors_literature")):
        if path and Path(path).exists():
            anchors = json.loads(Path(path).read_text())
            out["anchors_source"] = anchors.get("source", str(path))
            break
    for comp in cfg.get("compositions", ("Ti", "Zr", "Nb")):
        entry: dict = {}
        for name, members in loaded.items():
            entry[name] = eos_scan_committee(
                members,
                comp,
                reps=int(cfg.get("reps", 2)),
                n_points=int(cfg.get("n_points", 13)),
                span=float(cfg.get("span", 0.06)),
                seed=int(cfg.get("seed", 0)),
            )
        if teacher_calc is not None:
            entry["teacher"] = eos_scan_calculator(
                teacher_calc,
                comp,
                reps=int(cfg.get("reps", 2)),
                n_points=int(cfg.get("n_points", 13)),
                span=float(cfg.get("span", 0.06)),
                seed=int(cfg.get("seed", 0)),
            )
        anchor = anchors.get("entries", {}).get(comp)
        if anchor:
            entry["anchor"] = anchor
        out["compositions"][comp] = entry
    dest = cfg.get("out")
    if dest:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(json.dumps(out, indent=1))
    return out


def run_nve_benchmark(cfg: dict) -> dict:
    """NVE energy-drift stability check for the committed committee."""
    from falearn.md import nve_run
    from falearn.structures import bcc_supercell

    members, extra = load_members(cfg["committee"])
    out: dict = {"committee_extra": extra, "runs": []}
    for temp in cfg.get("temperatures_k", (300.0, 900.0)):
        fr = bcc_supercell(
            cfg.get("composition", "TiZrNb"),
            int(cfg.get("reps", 2)),
            seed=int(cfg.get("seed", 0)),
        )
        run = nve_run(
            fr,
            members,
            temperature_k=float(temp),
            n_steps=int(cfg.get("n_steps", 1500)),
            dt_fs=float(cfg.get("dt_fs", 2.0)),
            sample_every=int(cfg.get("sample_every", 10)),
            seed=int(cfg.get("seed", 0)),
        )
        run["temperature_init_k"] = float(temp)
        out["runs"].append(run)
    dest = cfg.get("out")
    if dest:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(json.dumps(out, indent=1))
    return out
