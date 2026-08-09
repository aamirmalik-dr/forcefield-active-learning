"""End-to-end micro loop, benchmark driver, and CLI smoke tests."""

from __future__ import annotations

import json

import numpy as np
import yaml

from falearn.bench import run_al_benchmark
from falearn.cli import main as cli_main
from falearn.data import save_frames
from falearn.descriptors import frame_graph
from falearn.loop import EvalSet, LoopSettings, run_loop
from falearn.pool import LabeledPool
from falearn.training import TrainSettings

MICRO_TRAIN = {"epochs": 3, "batch_size": 8, "lr": 3e-3, "force_weight": 0.1}


def _micro_settings(strategy: str, seed: int = 0) -> LoopSettings:
    return LoopSettings(
        strategy=strategy,
        n_members=2,
        hidden=(8,),
        n_init=6,
        round_size=4,
        n_rounds=2,
        seed=seed,
        train=TrainSettings(**MICRO_TRAIN),
    )


def _eval_set(frames, acsf):
    return EvalSet(
        name="test",
        graphs=[frame_graph(fr, acsf) for fr in frames],
        energies=np.array([fr.energy for fr in frames]),
        forces=[fr.forces for fr in frames],
    )


def test_micro_loop_budgets_and_records(labeled_batches, small_acsf):
    pool_frames = labeled_batches[:24]
    test_frames = labeled_batches[24:]
    pool = LabeledPool(pool_frames)
    history, members = run_loop(
        pool, [_eval_set(test_frames, small_acsf)], _micro_settings("committee"), acsf=small_acsf
    )
    assert [r["budget"] for r in history["rounds"]] == [6, 10, 14]
    assert pool.n_purchased == 14
    assert len(members) == 2
    assert "calibration" in history
    assert len(history["calibration"]["f_rmse_mev_per_a"]) == len(test_frames)
    sel = history["rounds"][0]["selected"]
    assert len(sel) == 4
    assert all(s["score"] is not None for s in sel)
    # committee selection picked the highest-disagreement candidates
    scores = [s["score"] for s in sel]
    assert scores == sorted(scores, reverse=True)


def test_strategies_share_initial_draw(labeled_batches, small_acsf):
    ev = [_eval_set(labeled_batches[24:], small_acsf)]
    hist_a, _ = run_loop(
        LabeledPool(labeled_batches[:24]), ev, _micro_settings("committee", seed=3), small_acsf
    )
    hist_b, _ = run_loop(
        LabeledPool(labeled_batches[:24]), ev, _micro_settings("random", seed=3), small_acsf
    )
    init_a = [row["index"] for row in hist_a["initial_purchase"]]
    init_b = [row["index"] for row in hist_b["initial_purchase"]]
    assert init_a == init_b
    # but the purchases diverge afterwards for this seed
    later_a = {s["index"] for r in hist_a["rounds"] if "selected" in r for s in r["selected"]}
    later_b = {s["index"] for r in hist_b["rounds"] if "selected" in r for s in r["selected"]}
    assert later_a != later_b


def test_run_al_benchmark_and_cli(tmp_path, labeled_batches, monkeypatch):
    dataset = tmp_path / "micro.extxyz"
    save_frames(labeled_batches, dataset)
    cfg = {
        "name": "micro",
        "kind": "al",
        "dataset": str(dataset),
        "seeds": [0],
        "strategies": ["committee", "random"],
        "n_members": 2,
        "hidden": [8],
        "n_init": 6,
        "round_size": 4,
        "n_rounds": 1,
        "train": MICRO_TRAIN,
        "split": {"holdout_compositions": ["ZrNb"], "test_fraction": 0.5, "seed": 0},
        "sibling_test": {"n_per_group": 2},
        "out": str(tmp_path / "micro.json"),
        "save_committee": str(tmp_path / "committee.pt"),
    }
    results = run_al_benchmark(cfg)
    assert (tmp_path / "micro.json").exists()
    assert (tmp_path / "committee.pt").exists()
    assert results["n_transfer"] > 0
    for run in results["runs"]:
        assert run["rounds"][-1]["budget"] == 10
        assert "sibling_test" in run
        assert "transfer" in run["rounds"][-1]["metrics"]
    payload = json.loads((tmp_path / "micro.json").read_text())
    assert payload["config"]["name"] == "micro"

    # CLI demo consumes the same config format
    cfg_path = tmp_path / "fast.yaml"
    cfg.pop("out")
    cfg.pop("save_committee")
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert cli_main(["demo", "--config", str(cfg_path)]) == 0

    # CLI eval on the saved committee
    assert (
        cli_main(["eval", "--committee", str(tmp_path / "committee.pt"), "--frames", str(dataset)])
        == 0
    )


def test_md_exploration_adds_candidates(labeled_batches, small_acsf):
    from falearn.loop import MdExploreSettings

    pool_frames = labeled_batches[:24]

    def oracle(frame):
        return -6.0 * frame.n_atoms, np.zeros((frame.n_atoms, 3))

    pool = LabeledPool(pool_frames, oracle=oracle)
    settings = _micro_settings("committee")
    settings.explore = MdExploreSettings(
        compositions=("TiZrNb",), temperature_k=600.0, n_steps=10, sample_every=5
    )
    history, _ = run_loop(
        pool, [_eval_set(labeled_batches[24:], small_acsf)], settings, acsf=small_acsf
    )
    n_before = len(pool_frames)
    assert pool.n_frames > n_before  # exploration snapshots joined the pool
    explored = [fr for fr in pool.frames if "ensmd" in fr.group]
    assert explored
    rec = history["rounds"][0]["explore"]
    assert rec["added"] + rec["dropped_overlapping"] == 2  # 10 steps / stride 5
    # any purchased exploration frame must have gone through the oracle
    bought_explored = [k for k in pool.purchased if k >= n_before]
    assert pool.oracle_calls == len(bought_explored)


def test_loop_never_reads_hidden_labels(labeled_batches, small_acsf):
    """Instrument the private store: any read of an unpurchased label fails."""
    pool_frames = labeled_batches[:24]
    pool = LabeledPool(pool_frames)

    class GuardedStore(dict):
        def __getitem__(self, k):
            assert k in set(pool._purchased), f"hidden label {k} read before purchase"
            return dict.__getitem__(self, k)

    pool._store = GuardedStore(pool._store)
    # the run would raise inside the guard if selection, training, or
    # evaluation ever consumed a label that was not purchased first
    run_loop(
        pool,
        [_eval_set(labeled_batches[24:], small_acsf)],
        _micro_settings("committee"),
        acsf=small_acsf,
    )
    assert all(fr.energy is None and fr.forces is None for fr in pool.frames)
    assert pool.n_purchased == 14
