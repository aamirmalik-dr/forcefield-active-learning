"""Rebuild the random-selection reference committee from the recorded run.

The main benchmark logs every purchase of every run. This script replays the
random-strategy seed-0 purchase set exactly (initial draw plus all rounds),
retrains the final-round committee with the same seeds and settings, saves it
as models/committee_random.pt, and prints its test error next to the number
recorded in results/main.json as a consistency check. The checkpoint exists
so the EOS benchmark can contrast the physics of an acquisition-biased
committee against a randomly-fed one at the identical budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from falearn.bench import _eval_set, load_config
from falearn.data import load_frames, pool_test_split
from falearn.descriptors import AcsfParams, frame_graph
from falearn.ensemble import evaluate_mean, train_committee
from falearn.model import save_members
from falearn.training import TrainSettings

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    cfg = load_config(REPO / "configs" / "main.yaml")
    main_res = json.loads((REPO / "results" / "main.json").read_text())
    run = next(r for r in main_res["runs"] if r["strategy"] == "random" and r["seed"] == 0)
    purchased = [row["index"] for row in run["initial_purchase"]]
    for rec in run["rounds"]:
        purchased += [row["index"] for row in rec.get("selected", [])]
    acsf = AcsfParams()
    frames = load_frames(REPO / cfg["dataset"])
    sc = cfg["split"]
    split = pool_test_split(
        frames,
        holdout_compositions=tuple(sc["holdout_compositions"]),
        test_fraction=float(sc["test_fraction"]),
        seed=int(sc["seed"]),
    )
    pool_frames = [frames[k] for k in split.pool]
    subset = [pool_frames[k] for k in purchased]
    graphs = [frame_graph(fr, acsf) for fr in subset]
    energies = np.array([fr.energy for fr in subset])
    forces = [fr.forces for fr in subset]
    counts = np.stack([fr.counts() for fr in subset])
    settings = TrainSettings(**cfg["train"])
    # the final round of a run with seed s and n_rounds R trains with
    # base_seed s*100 + R, reproducing the recorded committee exactly
    base_seed = run["seed"] * 100 + int(cfg["n_rounds"])
    members, _ = train_committee(
        int(cfg["n_members"]),
        graphs,
        energies,
        forces,
        counts,
        settings,
        acsf=acsf,
        hidden=tuple(cfg["hidden"]),
        base_seed=base_seed,
    )
    test = _eval_set("test", frames, split.test, acsf)
    m = evaluate_mean(members, test.graphs, test.energies, test.forces)
    recorded = run["rounds"][-1]["metrics"]["test"]
    print(
        f"replayed random seed 0: F {m['f_mae_mev_per_a']:.2f} "
        f"(recorded {recorded['f_mae_mev_per_a']:.2f}), "
        f"E {m['e_mae_mev_per_atom']:.2f} "
        f"(recorded {recorded['e_mae_mev_per_atom']:.2f})"
    )
    out = REPO / "models" / "committee_random.pt"
    save_members(
        members,
        out,
        extra={"benchmark": "main", "strategy": "random", "seed": 0, "budget": len(purchased)},
    )
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
